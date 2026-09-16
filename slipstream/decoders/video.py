"""Time-based video window decoding with torchcodec.

``DecodeVideoWindow`` is a pipeline stage for a raw ``bytes`` primary field
holding one video container per record (e.g. an HEVC MP4 per clip). For every
record in the batch it decodes ``T`` frames sampled at ``rate_hz`` starting at
``t0`` seconds and returns them as ``[B, T, 3, H, W]`` uint8 together with the
frames' *true* presentation times. Source frame rates never enter the sampler:
``VideoDecoder.get_frames_played_at`` returns the frame shown at each requested
time, whatever the clip's fps.

Window start ``t0`` comes from one of two modes:

* **random-in-clip** (default): ``t0 ~ U(begin, usable_end - window_s)``,
  drawn per sample from a seed that follows the JPEG decoders' contract
  (``seed``, ``_seed_counter``), so a seeded run is reproducible and
  ``SlipstreamLoader.set_epoch`` resumes it deterministically.
* **given**: ``t0_key`` names an entry of the loader's ``sample_data`` (a
  per-sample array aligned with ``indices``), so a sampler can fix the window
  start per anchor (eval, fixed-anchor curricula, several windows per clip by
  repeating the record index with different ``t0``).

``usable_end`` is ``end_stream_seconds - end_margin_frames / fps``: the last
frames of a clip are never requested (torchcodec 0.16 + NVDEC raises
end-of-stream on them for a few percent of HEVC clips); requested times are
clamped into ``[begin, usable_end)``. On CUDA a ``RuntimeError`` is retried
once on a CPU decoder and the frames moved to the device.

Decoding runs in a persistent thread pool (torchcodec releases the GIL; one
ffmpeg thread per decoder). ``transforms`` (slipstream ``BatchAugment``
objects) are applied to the flat ``[B*T, 3, H, W]`` frames with
``seed_repeat = T``, so a window's frames share one crop / flip / colour draw.
"""

from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import torch

from slipstream.decoders.base import BatchTransform

_MOD = 2147483647


def _load_torchcodec():
    try:
        from torchcodec.decoders import VideoDecoder
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "DecodeVideoWindow needs torchcodec with a loadable libtorchcodec "
            "(pip install torchcodec; FFmpeg shared libraries must be on the loader path)."
        ) from exc
    return VideoDecoder


class DecodeVideoWindow(BatchTransform):
    """Decode ``T`` frames at ``rate_hz`` from each record's video bytes.

    Args:
        T: Frames per window.
        rate_hz: Sampling rate in Hz; ``window_s = T / rate_hz``.
        seed: Seed for the random window start (mode random-in-clip). ``None`` =
            non-reproducible (counter only), as for the JPEG decoders.
        t0_key: Name of a loader ``sample_data`` array giving the window start in
            seconds per sample. When present in the batch the random mode is off.
        device: ``"cpu"`` or ``"cuda[:N]"``. One device per stage instance; use
            one instance per GPU, never two devices in one thread.
        num_workers: Decoder threads (default: CPU count, capped at 32; 2 on CUDA).
        num_ffmpeg_threads: FFmpeg threads per decoder (default 1: parallelism
            comes from the pool).
        seek_mode: torchcodec seek mode, ``"exact"`` (default) or ``"approximate"``.
        resize: Decoder-side resize before augmentation: an int (short side) or
            ``(H, W)``. Required when a store mixes frame sizes.
        transforms: slipstream ``BatchAugment`` transforms applied to the flat
            ``[B*T, 3, H, W]`` frames with ``seed_repeat = T``.
        end_margin_frames: Frames at the end of a clip never requested (default 2).
        cpu_fallback: Retry a failed CUDA decode on the CPU (default True).
        name: Output key prefix when the batch does not carry the field name.

    Output (a dict merged into the batch by the loader):
        ``{field: [B, T, 3, H, W] uint8 (after `transforms`), field_t_sec: [B, T]
        float32 true frame times, field_t0: [B] float32, field_rec: [B] int64
        record indices}``. Frames live on ``device``; times on the CPU.
    """

    #: the stage manages `seed_repeat` of its own `transforms`; the loader must not descend into them
    owns_transforms = True

    def __init__(
        self,
        T: int,
        rate_hz: float,
        *,
        seed: int | None = None,
        t0_key: str | None = None,
        device: str | torch.device = "cpu",
        num_workers: int | None = None,
        num_ffmpeg_threads: int = 1,
        seek_mode: str = "exact",
        resize: int | tuple[int, int] | None = None,
        transforms: list[Any] | None = None,
        end_margin_frames: int = 2,
        cpu_fallback: bool = True,
        name: str = "video",
    ) -> None:
        if T < 1 or rate_hz <= 0:
            raise ValueError(f"T must be >= 1 and rate_hz > 0, got T={T}, rate_hz={rate_hz}")
        self.T = int(T)
        self.rate_hz = float(rate_hz)
        self.window_s = self.T / self.rate_hz
        self.seed = seed
        self.t0_key = t0_key
        self.device = torch.device(device)
        self.is_cuda = self.device.type == "cuda"
        if num_workers is None:
            num_workers = 2 if self.is_cuda else min(32, os.cpu_count() or 4)
        self.num_workers = max(1, int(num_workers))
        self.num_ffmpeg_threads = int(num_ffmpeg_threads)
        self.seek_mode = seek_mode
        self.resize = resize
        self.transforms = list(transforms or [])
        self.end_margin_frames = int(end_margin_frames)
        self.cpu_fallback = cpu_fallback
        self.name = name

        self._seed_counter = 0
        self._pool: ThreadPoolExecutor | None = None
        self._pool_lock = threading.Lock()
        self._VideoDecoder = None
        self._last_params: dict[str, Any] | None = None
        self._set_inner_seed_repeat()

    # ------------------------------------------------------------------ setup
    def _set_inner_seed_repeat(self) -> None:
        stack = list(self.transforms)
        while stack:
            t = stack.pop()
            if hasattr(t, "seed_repeat"):
                try:
                    t.seed_repeat = self.T
                except AttributeError:
                    pass
            inner = getattr(t, "transforms", None)
            if isinstance(inner, (list, tuple)):
                stack.extend(inner)

    def _ensure_pool(self) -> ThreadPoolExecutor:
        with self._pool_lock:
            if self._pool is None:
                self._pool = ThreadPoolExecutor(max_workers=self.num_workers, thread_name_prefix="slipstream-video")
            if self._VideoDecoder is None:
                self._VideoDecoder = _load_torchcodec()
        return self._pool

    def _decoder_transforms(self, width: int, height: int):
        if self.resize is None:
            return None
        from torchcodec.transforms import Resize
        if isinstance(self.resize, int):
            s = self.resize
            if width >= height:
                size = (s, max(2, int(round(s * width / height)) // 2 * 2))
            else:
                size = (max(2, int(round(s * height / width)) // 2 * 2), s)
        else:
            size = (int(self.resize[0]), int(self.resize[1]))
        return [Resize(size=size)]

    def _make_decoder(self, raw: bytes, device: str):
        VD = self._VideoDecoder
        kw = dict(seek_mode=self.seek_mode, num_ffmpeg_threads=self.num_ffmpeg_threads, dimension_order="NCHW")
        if device != "cpu":
            kw["device"] = device
        if self.resize is None:
            return VD(raw, **kw)
        probe = VD(raw, seek_mode="approximate", num_ffmpeg_threads=1)
        m = probe.metadata
        return VD(raw, transforms=self._decoder_transforms(m.width, m.height), **kw)

    # ------------------------------------------------------------------ decode
    def _times(self, meta, t0_given: float | None, rng: np.random.Generator | None):
        begin = float(meta.begin_stream_seconds or 0.0)
        end = float(meta.end_stream_seconds if meta.end_stream_seconds is not None else begin + (meta.duration_seconds or 0.0))
        fps = float(meta.average_fps or 30.0)
        usable_end = max(begin + 1.0 / fps, end - self.end_margin_frames / fps)
        if t0_given is not None:
            t0 = float(t0_given)
            if t0 < begin - 1e-6 or t0 + self.window_s > usable_end + 0.5 / fps:
                raise ValueError(
                    f"window [{t0:.3f}, {t0 + self.window_s:.3f}] s does not fit the clip "
                    f"(usable [{begin:.3f}, {usable_end:.3f}] s at {fps:.3g} fps)"
                )
        else:
            hi = usable_end - self.window_s
            t0 = begin if hi <= begin else float(rng.uniform(begin, hi))
        times = t0 + np.arange(self.T, dtype=np.float64) / self.rate_hz
        times = np.clip(times, begin, np.nextafter(usable_end, begin))
        return t0, times

    def _decode_one(self, raw: bytes, t0_given: float | None, seed_i: int | None):
        rng = None if t0_given is not None else np.random.default_rng(seed_i)
        device = str(self.device) if self.is_cuda else "cpu"
        try:
            dec = self._make_decoder(raw, device)
            t0, times = self._times(dec.metadata, t0_given, rng)
            fb = dec.get_frames_played_at(times.tolist())
        except RuntimeError:
            if not (self.is_cuda and self.cpu_fallback):
                raise
            dec = self._make_decoder(raw, "cpu")
            t0, times = self._times(dec.metadata, t0_given, rng)
            fb = dec.get_frames_played_at(times.tolist())
            fb_data = fb.data.to(self.device, non_blocking=True)
            return fb_data, fb.pts_seconds.to(torch.float32), t0
        return fb.data, fb.pts_seconds.to(torch.float32), t0

    def __call__(self, batch_data: dict[str, Any]) -> dict[str, Any]:
        pool = self._ensure_pool()
        data, sizes = batch_data["data"], batch_data["sizes"]
        B = len(sizes)
        field = batch_data.get("field") or self.name
        indices = batch_data.get("indices")

        # window starts: given per sample, or random with the decoders' seed contract
        t0_given = None
        sd = batch_data.get("sample_data") or {}
        if self.t0_key is not None and self.t0_key in sd:
            t0_given = np.asarray(sd[self.t0_key], dtype=np.float64)
            if len(t0_given) != B:
                raise ValueError(f"sample_data[{self.t0_key!r}] has {len(t0_given)} entries for a batch of {B}")
        self._seed_counter += 1
        base = (self.seed if self.seed is not None else 0) + B * self._seed_counter
        seeds = [(base + i) % _MOD for i in range(B)]

        raws = [bytes(data[i, : int(sizes[i])]) for i in range(B)]   # own copies: bank rows are reused
        futs = [pool.submit(self._decode_one, raws[i], None if t0_given is None else float(t0_given[i]), seeds[i])
                for i in range(B)]
        results = [f.result() for f in futs]

        frames = torch.stack([r[0] for r in results])                 # [B, T, 3, H, W] on self.device
        t_sec = torch.stack([r[1] for r in results])                  # [B, T] float32 (cpu)
        t0 = torch.tensor([r[2] for r in results], dtype=torch.float32)

        if self.transforms:
            self._set_inner_seed_repeat()                             # loader may have reset them
            flat = frames.reshape(B * self.T, *frames.shape[2:])
            for t in self.transforms:
                flat = t(flat)
            frames = flat.reshape(B, self.T, *flat.shape[1:])

        self._last_params = {"t0": t0, "t_sec": t_sec}
        out = {field: frames, f"{field}_t_sec": t_sec, f"{field}_t0": t0}
        if indices is not None:
            out[f"{field}_rec"] = torch.as_tensor(np.asarray(indices, dtype=np.int64))
        return out

    # ------------------------------------------------------------------ misc
    def shutdown(self) -> None:
        with self._pool_lock:
            if self._pool is not None:
                self._pool.shutdown(wait=False)
                self._pool = None
        for t in self.transforms:
            if hasattr(t, "shutdown"):
                t.shutdown()

    def __repr__(self) -> str:
        tr = ", ".join(type(t).__name__ for t in self.transforms)
        return (f"DecodeVideoWindow(T={self.T}, rate_hz={self.rate_hz:g}, window_s={self.window_s:g}, "
                f"seed={self.seed}, t0_key={self.t0_key!r}, device='{self.device}', workers={self.num_workers}, "
                f"resize={self.resize}, transforms=[{tr}])")
