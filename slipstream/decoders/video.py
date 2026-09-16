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
ffmpeg thread per decoder). The stage is *asynchronous*: ``submit(batch_data)``
copies the bytes out of the loader's banks and queues one decode per record,
each worker writing its ``[T, 3, H, W]`` straight into the batch's output
tensor; ``collect(pending)`` waits and applies ``transforms``. ``SlipstreamLoader``
calls ``submit`` from its prefetch thread as soon as a batch's bytes are loaded
and ``collect`` when the batch is consumed, so ``batches_ahead * batch_size``
decodes are in flight (choose ``batches_ahead >= num_workers / batch_size``).
``__call__`` is ``collect(submit(...))`` for use outside the loader.

``transforms`` (slipstream ``BatchAugment`` objects) are applied to the flat
``[B*T, 3, H, W]`` frames with ``seed_repeat = T``, so a window's frames share
one crop / flip / colour draw.
"""

from __future__ import annotations

import itertools
import os
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Sequence

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


class _PendingBatch:
    """One submitted batch: decode futures plus the output buffers the workers fill."""

    __slots__ = ("B", "T", "field", "indices", "futures", "out", "t_sec", "t0", "lock")

    def __init__(self, B: int, T: int, field: str, indices):
        self.B, self.T, self.field, self.indices = B, T, field, indices
        self.futures: list[Future] = []
        self.out: torch.Tensor | None = None            # [B, T, 3, H, W], allocated by the first finisher
        self.t_sec = torch.empty(B, T, dtype=torch.float32)
        self.t0 = torch.empty(B, dtype=torch.float32)
        self.lock = threading.Lock()


class DecodeVideoWindow(BatchTransform):
    """Decode ``T`` frames at ``rate_hz`` from each record's video bytes.

    Args:
        T: Frames per window.
        rate_hz: Sampling rate in Hz; ``window_s = T / rate_hz``.
        seed: Seed for the random window start (mode random-in-clip). ``None`` =
            non-reproducible (counter only), as for the JPEG decoders.
        t0_key: Name of a loader ``sample_data`` array giving the window start in
            seconds per sample. When present in the batch the random mode is off.
        device: ``"cpu"``, ``"cuda[:N]"``, or a list of devices. Pool threads are
            pinned to the entries round-robin, so no thread ever touches two
            devices; results are gathered on ``output_device``. Mixed lists add
            up: ``["cpu"] * 40 + ["cuda:0"] * 8 + ["cuda:1"] * 8`` with
            ``num_workers=56`` runs 40 CPU decoders and 8 NVDEC sessions per GPU.
        output_device: Where the ``[B, T, 3, H, W]`` batch lives (default: the
            first entry of ``device``).
        num_workers: Decoder threads (default: CPU count, capped at 32;
            4 per CUDA device).
        num_ffmpeg_threads: FFmpeg threads per decoder (default 1: parallelism
            comes from the pool).
        seek_mode: torchcodec seek mode, ``"exact"`` (default) or ``"approximate"``.
        resize: Decoder-side resize before augmentation: an int (short side) or
            ``(H, W)``. Required when a store mixes frame sizes.
        transforms: slipstream ``BatchAugment`` transforms applied to the flat
            ``[B*T, 3, H, W]`` frames with ``seed_repeat = T``.
        end_margin_frames: Frames at the end of a clip never requested (default 2).
        cpu_fallback: Retry a failed CUDA decode on the CPU (default True).
        reuse_output: Recycle the ``[B, T, 3, H, W]`` output tensors through a ring of
            ``ring_size`` buffers instead of allocating one per batch. Avoids a
            fresh multi-hundred-MB allocation (page faults + munmap under many
            threads) per batch. The batch's frames are then only valid until
            ``ring_size`` batches later, like the loader's own JPEG banks: do not
            keep references across batches (or clone). Default False.
        ring_size: Buffers in the ring (default 4; must exceed the loader's
            ``batches_ahead`` + 1).
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
        device: str | torch.device | Sequence[str | torch.device] = "cpu",
        output_device: str | torch.device | None = None,
        num_workers: int | None = None,
        num_ffmpeg_threads: int = 1,
        seek_mode: str = "exact",
        resize: int | tuple[int, int] | None = None,
        transforms: list[Any] | None = None,
        end_margin_frames: int = 2,
        cpu_fallback: bool = True,
        reuse_output: bool = False,
        ring_size: int = 4,
        name: str = "video",
    ) -> None:
        if T < 1 or rate_hz <= 0:
            raise ValueError(f"T must be >= 1 and rate_hz > 0, got T={T}, rate_hz={rate_hz}")
        self.T = int(T)
        self.rate_hz = float(rate_hz)
        self.window_s = self.T / self.rate_hz
        self.seed = seed
        self.t0_key = t0_key
        devs = [device] if isinstance(device, (str, torch.device)) else list(device)
        self.devices = [torch.device(d) for d in devs]
        self.device = self.devices[0]
        self.is_cuda = any(d.type == "cuda" for d in self.devices)      # any CUDA worker present
        self.output_device = torch.device(output_device) if output_device is not None else self.device
        if num_workers is None:
            n_cuda = sum(d.type == "cuda" for d in self.devices)
            n_cpu = len(self.devices) - n_cuda
            num_workers = 4 * n_cuda + (min(32, os.cpu_count() or 4) if n_cpu else 0)
        self.num_workers = max(1, int(num_workers))
        self.num_ffmpeg_threads = int(num_ffmpeg_threads)
        self.seek_mode = seek_mode
        self.resize = resize
        self.transforms = list(transforms or [])
        self.end_margin_frames = int(end_margin_frames)
        self.cpu_fallback = cpu_fallback
        self.reuse_output = reuse_output
        self.ring_size = max(2, int(ring_size))
        self._ring: list[torch.Tensor] = []
        self._ring_pos = 0
        self._ring_lock = threading.Lock()
        self.name = name

        self._seed_counter = 0
        self._pool: ThreadPoolExecutor | None = None
        self._pool_lock = threading.Lock()
        self._VideoDecoder = None
        self._last_params: dict[str, Any] | None = None
        self._tls = threading.local()                      # per-worker device
        self._thread_counter = itertools.count()
        self._set_inner_seed_repeat()

    def _init_thread(self) -> None:
        """Pin each pool thread to one device (round-robin over `devices`)."""
        k = next(self._thread_counter)
        self._tls.device = self.devices[k % len(self.devices)]

    @property
    def _thread_device(self) -> torch.device:
        return getattr(self._tls, "device", self.device)

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
                self._pool = ThreadPoolExecutor(max_workers=self.num_workers, thread_name_prefix="slipstream-video",
                                                initializer=self._init_thread)
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
        dev = self._thread_device
        device = str(dev) if dev.type == "cuda" else "cpu"
        try:
            dec = self._make_decoder(raw, device)
            t0, times = self._times(dec.metadata, t0_given, rng)
            fb = dec.get_frames_played_at(times.tolist())
        except RuntimeError:
            if not (device != "cpu" and self.cpu_fallback):
                raise
            dec = self._make_decoder(raw, "cpu")
            t0, times = self._times(dec.metadata, t0_given, rng)
            fb = dec.get_frames_played_at(times.tolist())
        return fb.data, fb.pts_seconds.to(torch.float32), t0

    def _decode_into(self, pend: _PendingBatch, i: int, raw: bytes, t0_given: float | None, seed_i: int) -> None:
        """Worker: decode record i of the batch and write it into the batch buffers."""
        data, pts, t0 = self._decode_one(raw, t0_given, seed_i)
        with pend.lock:
            if pend.out is None:
                pend.out = self._output_buffer((pend.B, pend.T, *data.shape[1:]), data.dtype)
        if tuple(data.shape) != tuple(pend.out.shape[1:]):
            raise ValueError(
                f"frame shape {tuple(data.shape[1:])} of record {i} differs from {tuple(pend.out.shape[2:])} in the "
                f"same batch; pass resize= so every clip decodes to one size"
            )
        pend.out[i].copy_(data)
        pend.t_sec[i] = pts
        pend.t0[i] = float(t0)

    def _output_buffer(self, shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
        """A [B, T, 3, H, W] output tensor: fresh, or the next slot of the reuse ring."""
        if not self.reuse_output:
            return torch.empty(shape, dtype=dtype, device=self.output_device)
        with self._ring_lock:
            if len(self._ring) < self.ring_size:
                buf = torch.empty(shape, dtype=dtype, device=self.output_device)
                self._ring.append(buf)
                return buf
            buf = self._ring[self._ring_pos]
            self._ring_pos = (self._ring_pos + 1) % self.ring_size
            if tuple(buf.shape) != tuple(shape) or buf.dtype != dtype:      # last partial batch etc.
                if buf.numel() >= int(np.prod(shape)) and buf.dtype == dtype:
                    return buf.flatten()[: int(np.prod(shape))].view(shape)
                buf = torch.empty(shape, dtype=dtype, device=self.output_device)
                self._ring[(self._ring_pos - 1) % self.ring_size] = buf
            return buf

    # ------------------------------------------------------------- async API
    def submit(self, batch_data: dict[str, Any]) -> _PendingBatch:
        """Queue the decodes for a batch (called by the loader's prefetch thread). Copies the bytes out."""
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

        pend = _PendingBatch(B, self.T, field, None if indices is None else np.asarray(indices, dtype=np.int64).copy())
        for i in range(B):
            raw = bytes(data[i, : int(sizes[i])])                     # own copy: bank rows are reused
            t0_i = None if t0_given is None else float(t0_given[i])
            pend.futures.append(pool.submit(self._decode_into, pend, i, raw, t0_i, seeds[i]))
        return pend

    def collect(self, pend: _PendingBatch) -> dict[str, Any]:
        """Wait for a submitted batch, apply `transforms`, return the output dict."""
        for f in pend.futures:
            f.result()                                                # re-raises the first worker error
        frames = pend.out
        if frames is None:                                            # B == 0
            frames = torch.empty((0, self.T, 3, 0, 0), dtype=torch.uint8, device=self.output_device)

        if self.transforms and pend.B:
            self._set_inner_seed_repeat()                             # loader may have reset them
            flat = frames.reshape(pend.B * self.T, *frames.shape[2:])
            for t in self.transforms:
                flat = t(flat)
            frames = flat.reshape(pend.B, self.T, *flat.shape[1:])

        self._last_params = {"t0": pend.t0, "t_sec": pend.t_sec}
        out = {pend.field: frames, f"{pend.field}_t_sec": pend.t_sec, f"{pend.field}_t0": pend.t0}
        if pend.indices is not None:
            out[f"{pend.field}_rec"] = torch.from_numpy(pend.indices)
        return out

    def __call__(self, batch_data: dict[str, Any]) -> dict[str, Any]:
        return self.collect(self.submit(batch_data))

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
        devs = ",".join(str(d) for d in self.devices)
        return (f"DecodeVideoWindow(T={self.T}, rate_hz={self.rate_hz:g}, window_s={self.window_s:g}, "
                f"seed={self.seed}, t0_key={self.t0_key!r}, device='{devs}', output_device='{self.output_device}', "
                f"workers={self.num_workers}, resize={self.resize}, transforms=[{tr}])")
