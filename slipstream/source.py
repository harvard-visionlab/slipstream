"""Run slipstream pipelines on batches from a live source (simulator, replay buffer) instead of a cache.

    class Replay:                                   # any object with next_batch(step) -> dict
        def next_batch(self, step):
            return {"obs": obs_uint8_NTCHW, "action": a, "reward": r, "done": d, "index": i}

    loader = SourceLoader(Replay(), seed=0, device="cuda", steps_per_epoch=1000, pipelines={"obs": [
        RandomResizedCropBatch(84, scale=(0.8, 1.0), seed=1), RandomHorizontalFlip(seed=2),
        ToFloatDiv(255.0, dtype=torch.bfloat16)]})
    for epoch in range(start, epochs):
        loader.set_epoch(epoch)
        for batch in loader: ...                     # batch["obs"]: [N, T, C, h, w] bf16

Batch format: ``obs`` [N, T, C, H, W] uint8 (CPU, pinned preferred, or CUDA); ``action`` [N, T, ...];
``reward`` / ``done`` [N, T]; ``index`` [N]; optional ``pose`` / ``egomotion``; other keys pass through.
A field with a pipeline that is 5-D ([N, T, C, H, W]) runs as [N*T, C, H, W] with ``seed_repeat = T``, so
every random augmentation draws once per sequence and all T frames share it; the result is folded back.

This is a separate path from :class:`~slipstream.loader.SlipstreamLoader`: no cache, no decoders (fused
decode stages raise), no numba. Seeding follows :mod:`slipstream.seeds` with one addition: augmentations
are reseeded before every batch with key ``(rank, epoch, step)``, so a resumed run reproduces the fresh run
exactly, even mid-epoch. The source is reseeded at each epoch start with
``source.reseed(derive_seed(source_seed, STREAM_SOURCE), (rank, epoch))`` (``source_seed`` defaults to
``seed``); unlike the cache-path shuffle the rank is
part of the key, so every rank's environments differ.

Source protocol (only ``next_batch`` is required):

- ``next_batch(step) -> dict``: the next batch; ``step`` counts batches of the current epoch (sources that
  sample a replay buffer can key their draws by it and need no state for exact resume).
- ``reseed(seed, key)``: restart the source's randomness for ``key = (rank, epoch)``.
- ``state_dict()`` / ``load_state_dict(sd)``: resumable state (keep it small; with prefetch it is captured
  after every batch so a checkpoint matches the batches actually consumed).
- ``on_policy`` (attribute, default False): the batch depends on the current policy, so it is produced inline
  on the calling thread after the previous batch was consumed (no prefetch, never stale).
- ``close()``.
"""

from __future__ import annotations

import queue
import threading
from typing import Any, Callable, Iterator, Protocol, Sequence, runtime_checkable

import torch

from slipstream.seeds import STREAM_SOURCE, derive_seed, reseed

_STOP = object()


@runtime_checkable
class BatchSource(Protocol):
    """Minimal source: ``next_batch(step) -> dict``. See the module docstring for the optional methods."""

    def next_batch(self, step: int) -> dict[str, Any]: ...


def _walk(obj: Any):
    from slipstream.loader import SlipstreamLoader
    return SlipstreamLoader._walk_transforms(obj, include_owned=True)


def _dist() -> tuple[int, int]:
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            return int(dist.get_rank()), int(dist.get_world_size())
    except Exception:                     # noqa: BLE001 - no distributed backend
        pass
    return 0, 1


class SourceLoader:
    """Iterate a :class:`BatchSource`, running per-field ``pipelines`` and ``after_batch_transforms``.

    Args:
        source: Object with ``next_batch(step)`` (see the module docstring for the optional protocol).
        pipelines: ``{field: [transform, ...]}`` or ``{field: [[view1...], [view2...]]}`` (multi-view: a
            list of results, one per sub-pipeline). Transforms are ``slipstream.transforms`` stages or any
            callable on a tensor; decoders are rejected.
        after_batch_transforms: Callables ``batch -> batch`` applied after the pipelines.
        device: Where batch tensors go (``'cpu'``, ``'cuda'``, ``'cuda:1'`` or a GPU index).
        seed: Base seed (``None``: no reseeding, transforms keep their own state).
        source_seed: Seed for the source's randomness (env resets, replay sampling); defaults to ``seed``.
            The source gets ``reseed(derive_seed(source_seed, STREAM_SOURCE), (rank, epoch))``.
        steps_per_epoch: Batches per epoch (``len(loader)``); ``None`` iterates forever.
        batches_ahead: Prefetch depth for CPU sources (0 disables the thread).
        use_threading: Prefetch CPU sources in a background thread. CUDA tensors are produced inline;
            ``on_policy`` sources are always inline.
    """

    def __init__(
        self,
        source: Any,
        *,
        pipelines: dict[str, Sequence[Any] | Callable] | None = None,
        after_batch_transforms: list[Callable] | None = None,
        device: int | str = "cpu",
        seed: int | None = None,
        source_seed: int | None = None,
        steps_per_epoch: int | None = None,
        batches_ahead: int = 2,
        use_threading: bool = True,
    ) -> None:
        if not callable(getattr(source, "next_batch", None)):
            raise TypeError("source must have a next_batch(step) method")
        if steps_per_epoch is not None and int(steps_per_epoch) <= 0:
            raise ValueError(f"steps_per_epoch must be positive or None, got {steps_per_epoch}")
        self.source = source
        self.seed = seed
        self.source_seed = seed if source_seed is None else int(source_seed)
        self.steps_per_epoch = None if steps_per_epoch is None else int(steps_per_epoch)
        self.batches_ahead = int(batches_ahead)
        self.use_threading = use_threading
        if isinstance(device, int):
            device = f"cuda:{device}"
        self.device = torch.device(device)
        self.after_batch_transforms = list(after_batch_transforms or [])

        self.pipelines: dict[str, list[Any]] = {}
        self._multi: set[str] = set()
        for name, pipe in (pipelines or {}).items():
            if isinstance(pipe, (list, tuple)) and pipe and isinstance(pipe[0], (list, tuple)):
                self.pipelines[name] = [list(p) for p in pipe]
                self._multi.add(name)
            else:
                self.pipelines[name] = list(pipe) if isinstance(pipe, (list, tuple)) else [pipe]
        from slipstream.decoders.base import BatchTransform
        for obj in _walk([list(self.pipelines.values()), self.after_batch_transforms]):
            if isinstance(obj, BatchTransform):
                raise TypeError(f"{type(obj).__name__} is a decoder stage (it reads cache bytes); SourceLoader "
                                f"batches are already frames: use slipstream.transforms stages instead")

        self.rank, self.world_size = _dist()
        self._epoch = 0
        self._step = 0
        self._source_reseeded: tuple | None = None
        self._source_state: Any = None
        self._prefetched = False               # a prefetch thread advanced the source past the consumed batches
        self._source_ptrs: set[int] = set()
        self._worker: threading.Thread | None = None
        self._stop = threading.Event()

    # ------------------------------------------------------------------ public API

    def __len__(self) -> int:
        if self.steps_per_epoch is None:
            raise TypeError("SourceLoader has no length (steps_per_epoch=None iterates forever)")
        return self.steps_per_epoch

    def set_epoch(self, epoch: int) -> None:
        """Start epoch ``epoch`` from its first batch (reseeds the source on the next iteration)."""
        self._stop_worker()
        self._epoch = int(epoch)
        self._step = 0
        self._source_reseeded = None

    @property
    def epoch(self) -> int:
        return self._epoch

    @property
    def step(self) -> int:
        """Batches of the current epoch consumed so far."""
        return self._step

    def state_dict(self) -> dict[str, Any]:
        """Position and source state as of the last consumed batch."""
        return {"epoch": self._epoch, "step": self._step, "seed": self.seed, "source_seed": self.source_seed,
                "rank": self.rank,
                "world_size": self.world_size, "source": self._current_source_state()}

    def load_state_dict(self, sd: dict[str, Any]) -> None:
        """Resume at ``sd``'s (epoch, step); augmentations continue exactly, the source as far as it can."""
        self._stop_worker()
        if sd.get("seed") != self.seed:
            raise ValueError(f"state_dict was saved with seed={sd.get('seed')}, loader has seed={self.seed}")
        if sd.get("source_seed", self.source_seed) != self.source_seed:
            raise ValueError(f"state_dict was saved with source_seed={sd.get('source_seed')}, "
                             f"loader has source_seed={self.source_seed}")
        self._epoch, self._step = int(sd["epoch"]), int(sd["step"])
        self._source_reseeded = None
        src = sd.get("source")
        if src is not None and callable(getattr(self.source, "load_state_dict", None)):
            self.source.load_state_dict(src)
            self._source_reseeded = (self._epoch, "restored")
        self._source_state = src
        self._prefetched = False

    def apply(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Run the pipelines and after_batch_transforms on ``batch`` with the transforms' current state.

        No reseeding and no step advance: for a collector that wants the training preprocessing on its own
        observations (e.g. to act on the same view). Tensors are moved to ``device`` first.
        """
        return self._transform(self._place(batch))

    def shutdown(self) -> None:
        self._stop_worker()
        close = getattr(self.source, "close", None)
        if callable(close):
            close()

    def __iter__(self) -> Iterator[dict[str, Any]]:
        self._stop_worker()
        if self._prefetched and self._source_state is not None and callable(getattr(self.source, "load_state_dict", None)):
            self.source.load_state_dict(self._source_state)     # undo batches prefetched but never consumed
        self._prefetched = False
        self._begin_source_epoch()
        inline = (getattr(self.source, "on_policy", False) or not self.use_threading or self.batches_ahead <= 0)
        produce = self._produce_inline() if inline else self._produce_threaded()
        try:
            while self.steps_per_epoch is None or self._step < self.steps_per_epoch:
                item = next(produce, _STOP)
                if item is _STOP:
                    return
                raw, state = item
                self._reseed_transforms(self._step)
                out = self._transform(raw)
                self._step += 1
                self._source_state = state
                yield out
            self._epoch += 1
            self._step = 0
        finally:
            produce.close()
            self._stop_worker()

    def __repr__(self) -> str:
        fields = ", ".join(self.pipelines) or "-"
        return (f"SourceLoader(source={type(self.source).__name__}, pipelines=[{fields}], device={self.device}, "
                f"seed={self.seed}, steps_per_epoch={self.steps_per_epoch}, epoch={self._epoch}, step={self._step})")

    # ------------------------------------------------------------------ resolution (RandomResizedCropBatch)

    def set_resolution(self, size: int) -> None:
        """Set the output size of every resizable crop stage (``RandomResizedCropBatch``) from the next batch."""
        stages = [o for o in _walk(list(self.pipelines.values())) if getattr(o, "resolution_schedulable", False)]
        if not stages:
            raise ValueError("no resizable crop stage in the pipelines (RandomResizedCropBatch)")
        for s in stages:
            s.set_size(size)
        self._resolution = int(size)

    @property
    def resolution(self) -> int | None:
        return getattr(self, "_resolution", None)

    # ------------------------------------------------------------------ internals

    def _current_source_state(self) -> Any:
        if self._source_state is None and not self._prefetched and callable(getattr(self.source, "state_dict", None)):
            return self.source.state_dict()           # nothing consumed yet: the source is where it starts
        return self._source_state                      # captured right after the last consumed batch was produced

    def _begin_source_epoch(self) -> None:
        if self._source_reseeded is not None and self._source_reseeded[0] == self._epoch:
            return                                     # already reseeded / restored for this epoch
        fn = getattr(self.source, "reseed", None)
        if self.source_seed is not None and callable(fn):
            fn(derive_seed(self.source_seed, STREAM_SOURCE), (self.rank, self._epoch))
        self._source_reseeded = (self._epoch, "reseeded")

    def _reseed_transforms(self, step: int) -> None:
        if self.seed is None:
            return
        key = (self.rank, self._epoch, step)
        seen: set[int] = set()
        for obj in _walk([list(self.pipelines.values()), self.after_batch_transforms]):
            if id(obj) not in seen:
                seen.add(id(obj))
                reseed(obj, key)

    def _set_seed_repeat(self, pipeline: Any, T: int) -> None:
        for obj in _walk(pipeline):
            if hasattr(obj, "seed_repeat"):
                try:
                    obj.seed_repeat = T
                except AttributeError:
                    pass

    def _to_device(self, v: Any) -> Any:
        return v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v

    def _run(self, pipeline: list[Any], x: Any) -> Any:
        for t in pipeline:
            x = t(x)
        return x

    def _field(self, name: str, x: Any) -> Any:
        pipe = self.pipelines[name]
        if not isinstance(x, torch.Tensor):
            return [self._run(p, x) for p in pipe] if name in self._multi else self._run(pipe, x)
        T = x.shape[1] if x.ndim == 5 else 1
        flat = x.reshape(x.shape[0] * T, *x.shape[2:]) if x.ndim == 5 else x
        self._set_seed_repeat(pipe, T)

        def fold(y):
            if x.ndim == 5 and isinstance(y, torch.Tensor) and y.ndim >= 1 and y.shape[0] == x.shape[0] * T:
                return y.reshape(x.shape[0], T, *y.shape[1:])
            return y

        if name in self._multi:
            # every view starts from the same frames: in-place stages (flip, normalize) must not leak across views
            return [fold(self._run(p, flat.clone())) for p in pipe]
        return fold(self._run(pipe, flat.clone() if self._shares(flat) else flat))

    def _shares(self, t: torch.Tensor) -> bool:
        return t.untyped_storage().data_ptr() in self._source_ptrs

    def _transform(self, batch: dict[str, Any]) -> dict[str, Any]:
        for name in self.pipelines:
            if name in batch:
                batch[name] = self._field(name, batch[name])
        for t in self.after_batch_transforms:
            batch = t(batch)
        return batch

    def _fetch(self, step: int) -> tuple[dict[str, Any], Any]:
        batch = self.source.next_batch(step)
        if not isinstance(batch, dict):
            raise TypeError(f"source.next_batch must return a dict, got {type(batch).__name__}")
        fn = getattr(self.source, "state_dict", None)
        return batch, (fn() if callable(fn) else None)

    def _place(self, batch: dict[str, Any]) -> dict[str, Any]:
        # Remember the source's own storage: tensors that stay where they are (already on device) are cloned
        # before in-place stages so a replay buffer or simulator frame buffer is never modified.
        self._source_ptrs = {v.untyped_storage().data_ptr() for v in batch.values() if isinstance(v, torch.Tensor)}
        return {k: self._to_device(v) for k, v in batch.items()}

    def _produce_inline(self):
        step = self._step
        while True:
            raw, state = self._fetch(step)
            yield self._place(raw), state
            step += 1

    def _produce_threaded(self):
        q: queue.Queue = queue.Queue(maxsize=self.batches_ahead)
        self._stop.clear()
        cuda = self.device.type == "cuda"
        start = self._step

        def work():
            step = start
            try:
                while not self._stop.is_set():
                    raw, state = self._fetch(step)
                    if cuda:
                        raw = {k: (v.pin_memory() if isinstance(v, torch.Tensor) and v.device.type == "cpu"
                                   and not v.is_pinned() else v) for k, v in raw.items()}
                    while not self._stop.is_set():
                        try:
                            q.put((raw, state), timeout=0.1)
                            break
                        except queue.Full:
                            continue
                    step += 1
            except BaseException as e:                 # noqa: BLE001 - re-raised on the consumer side
                while not self._stop.is_set():
                    try:
                        q.put(e, timeout=0.1)
                        return
                    except queue.Full:
                        continue

        self._prefetched = True
        self._worker = threading.Thread(target=work, name="slipstream-source", daemon=True)
        self._worker.start()
        while True:
            item = q.get()
            if isinstance(item, BaseException):
                raise item
            raw, state = item
            yield self._place(raw), state

    def _stop_worker(self) -> None:
        w = self._worker
        if w is None:
            return
        self._stop.set()
        w.join(timeout=5.0)
        self._worker = None
