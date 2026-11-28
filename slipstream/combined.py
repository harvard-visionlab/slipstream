"""Mix several slipstream datasets in one loader (0.19.0).

``CombinedDataset`` is a description, not a reader: pass it to :class:`SlipstreamLoader` in place of a
dataset. The loader opens (downloads, verifies, builds) each source's own slipcache exactly as for a single
dataset, then reads every batch from all of them through :class:`CombinedCache`, a view that presents the
caches as one. Nothing is copied or merged on disk, and weights can change between runs for free.

Epoch order (a pure function of ``(seed, epoch)``, like the single-dataset shuffle):

- the epoch has ``N`` samples (``epoch_length``); source ``i`` contributes exactly ``round(w_i * N)`` of them
  (largest-remainder rounding, so the counts sum to ``N``);
- each source walks its own shuffled order, keyed by ``(seed, epoch, source, pass)``; a source asked for more
  than its size starts a new pass (a fresh shuffle) when it runs out;
- the selected samples are shuffled together, so every batch mixes the sources in proportion;
- the loader then shards across ranks as usual (rank never enters the shuffle).

With ``shuffle=False`` every source is read in record order and the sources are interleaved evenly.

Batches carry the shared fields under their standardized names plus ``source_id`` (int64, position of the
source in ``datasets``) and, with a label field, ``source_label`` (the label as stored in the source) and
``label`` (the label to train on: offset by the preceding sources' class counts with ``labels="disjoint"``,
equal to ``source_label`` with ``labels="shared"``). ``batch['_indices']`` holds combined record indices:
source ``i``'s record ``r`` is ``offsets[i] + r`` (``CombinedCache.offsets``; ``CombinedCache.locate``
inverts it).
"""

from __future__ import annotations

import math
import threading
from collections.abc import Sequence
from typing import Any

import numpy as np

from slipstream.seeds import STREAM_COMBINE, _key, epoch_rng

SOURCE_ID = "source_id"
SOURCE_LABEL = "source_label"


class CombinedDataset:
    """Several datasets mixed by weight in one :class:`SlipstreamLoader`.

    Args:
        datasets: The sources (``SlipstreamDataset`` or anything a loader accepts). Each keeps its own cache.
        weights: Share of each source per epoch (normalized). Default: proportional to the sources' sizes,
            i.e. every record once per epoch.
        epoch_length: ``"shortest"``: the epoch ends when the first source has used up its records
            (``N = min_i(n_i / w_i)``, no source repeats). ``"longest"``: it ends when the last one has
            (``N = max_i(n_i / w_i)``); smaller sources start new passes. An int: exactly that many samples.
        labels: ``"disjoint"`` (each source has its own classes; ``label`` = ``source_label`` + offset) or
            ``"shared"`` (the sources' classes are the same; ``label`` = ``source_label``). Required when the
            sources have ``label_field``.
        num_classes: Class count per source, required with ``labels="disjoint"`` (the offsets are their
            running sum; a source can lack its last class, so the count is never guessed from the data).
        label_field: The label field after renaming (default ``"label"``).
        rename_fields: Per source, ``{stored_name: standard_name}`` (or None), applied before anything else:
            pipelines, ``image_field`` and ``exclude_fields`` use the standard names. Only fields every
            source has (after renaming, with the same type) are loaded.

    Example::

        combined = CombinedDataset([in10, in100], weights=[0.5, 0.5], labels="disjoint",
                                   num_classes=[10, 100], epoch_length="longest")
        loader = SlipstreamLoader(combined, batch_size=256, seed=0, pipelines=...)

    Not supported yet (raise in the loader): ``window=``, ``group=``, ``indices=``, ``sample_data=``,
    raw ``bytes`` primary fields (video).
    """

    def __init__(
        self,
        datasets: Sequence[Any],
        weights: Sequence[float] | None = None,
        *,
        epoch_length: str | int = "shortest",
        labels: str | None = None,
        num_classes: Sequence[int] | None = None,
        label_field: str = "label",
        rename_fields: Sequence[dict[str, str] | None] | None = None,
    ) -> None:
        self.datasets = list(datasets)
        k = len(self.datasets)
        if k < 1:
            raise ValueError("CombinedDataset needs at least one dataset")
        if weights is not None:
            w = np.asarray(weights, dtype=np.float64)
            if w.shape != (k,) or not np.all(np.isfinite(w)) or np.any(w <= 0):
                raise ValueError(f"weights must be {k} positive numbers, got {weights!r}")
            self.weights: np.ndarray | None = w / w.sum()
        else:
            self.weights = None                       # proportional to sizes, resolved once sizes are known
        if isinstance(epoch_length, str):
            if epoch_length not in ("shortest", "longest"):
                raise ValueError(f"epoch_length must be 'shortest', 'longest' or an int, got {epoch_length!r}")
        elif int(epoch_length) != epoch_length or epoch_length < 1:
            raise ValueError(f"epoch_length must be 'shortest', 'longest' or a positive int, got {epoch_length!r}")
        self.epoch_length = epoch_length
        if labels not in (None, "disjoint", "shared"):
            raise ValueError(f"labels must be 'disjoint' or 'shared', got {labels!r}")
        self.labels = labels
        if labels == "disjoint":
            if num_classes is None or len(num_classes) != k or any(int(c) < 1 for c in num_classes):
                raise ValueError(f"labels='disjoint' needs num_classes: one positive class count per source ({k})")
            self.num_classes: list[int] | None = [int(c) for c in num_classes]
        else:
            if num_classes is not None:
                raise ValueError("num_classes is only used with labels='disjoint'")
            self.num_classes = None
        self.label_field = label_field
        if rename_fields is None:
            rename_fields = [None] * k
        if len(rename_fields) != k:
            raise ValueError(f"rename_fields needs one entry (dict or None) per dataset ({k})")
        self.rename_fields = [dict(r or {}) for r in rename_fields]
        for i, r in enumerate(self.rename_fields):
            if len(set(r.values())) != len(r):
                raise ValueError(f"rename_fields[{i}] maps two fields to one name: {r}")

    def __len__(self) -> int:
        return len(self.datasets)

    def __repr__(self) -> str:
        w = None if self.weights is None else [round(float(x), 4) for x in self.weights]
        return (f"CombinedDataset({len(self.datasets)} datasets, weights={w}, epoch_length={self.epoch_length!r}, "
                f"labels={self.labels!r})")

    # ------------------------------------------------------------------ epoch plan
    def resolved_weights(self, sizes: Sequence[int]) -> np.ndarray:
        sizes = np.asarray(sizes, dtype=np.float64)
        return self.weights if self.weights is not None else sizes / sizes.sum()

    def epoch_size(self, sizes: Sequence[int]) -> int:
        """Samples per epoch (before sharding) for sources of these sizes."""
        if isinstance(self.epoch_length, int):
            return int(self.epoch_length)
        ratio = np.asarray(sizes, dtype=np.float64) / self.resolved_weights(sizes)
        if self.epoch_length == "shortest":
            # largest N whose rounded per-source counts never exceed a source's size
            n = int(math.floor(ratio.min() + 1e-9))
            while n > 0 and np.any(self.counts(sizes, n) > np.asarray(sizes)):
                n -= 1
            return n
        return int(math.ceil(ratio.max() - 1e-9))

    def counts(self, sizes: Sequence[int], n: int) -> np.ndarray:
        """Exact per-source sample counts for an epoch of ``n`` (largest remainder; they sum to ``n``)."""
        raw = self.resolved_weights(sizes) * n
        c = np.floor(raw).astype(np.int64)
        short = int(n - c.sum())
        if short:
            order = np.argsort(-(raw - c), kind="stable")
            c[order[:short]] += 1
        return c

    def epoch_order(self, sizes: Sequence[int], seed: int | None, epoch: int, shuffle: bool) -> np.ndarray:
        """Combined record indices of one epoch (length :meth:`epoch_size`), before sharding."""
        sizes = [int(s) for s in sizes]
        n = self.epoch_size(sizes)
        counts = self.counts(sizes, n)
        offsets = np.concatenate([[0], np.cumsum(sizes)[:-1]]).astype(np.int64)
        parts, keys = [], []
        for i, (size, c) in enumerate(zip(sizes, counts)):
            c = int(c)
            if c == 0:
                continue
            if size == 0:
                raise ValueError(f"dataset {i} is empty but is asked for {c} samples")
            passes = -(-c // size)
            if shuffle:
                local = np.concatenate([
                    (np.random.default_rng() if seed is None else
                     np.random.default_rng(np.random.SeedSequence(_key(seed, epoch, STREAM_COMBINE, i, p))))
                    .permutation(size) for p in range(passes)])[:c]
            else:
                local = np.resize(np.arange(size, dtype=np.int64), c)
            parts.append(offsets[i] + local.astype(np.int64))
            keys.append((np.arange(c) + 0.5) / c)            # even interleave for shuffle=False
        order = np.concatenate(parts) if parts else np.zeros(0, dtype=np.int64)
        if shuffle:
            epoch_rng(seed, epoch).shuffle(order)
        else:
            order = order[np.argsort(np.concatenate(keys), kind="stable")]
        return order


# ---------------------------------------------------------------------- storages
class _Routing:
    """Splits combined indices into per-source local indices and batch rows."""

    def __init__(self, offsets: np.ndarray, sizes: np.ndarray) -> None:
        self.offsets = offsets
        self.ends = offsets + sizes

    def split(self, indices: np.ndarray):
        indices = np.asarray(indices, dtype=np.int64)
        src = np.searchsorted(self.ends, indices, side="right")
        for i in np.unique(src):
            rows = np.flatnonzero(src == i)
            yield int(i), rows, indices[rows] - self.offsets[i]

    def source_of(self, indices: np.ndarray) -> np.ndarray:
        return np.searchsorted(self.ends, np.asarray(indices, dtype=np.int64), side="right").astype(np.int64)


class CombinedImageStorage:
    """The sources' ImageBytesStorages as one (same interface the loader uses)."""

    def __init__(self, field_name: str, parts: list, routing: _Routing) -> None:
        self.field_name = field_name
        self.parts = parts
        self.routing = routing
        self.max_size = max(int(p.max_size) for p in parts)
        self.num_samples = int(sum(p.num_samples for p in parts))
        self._tls = threading.local()

    def load_batch_into(self, indices, dest, sizes, heights, widths, parallel: bool = True) -> int:
        for i, rows, local in self.routing.split(indices):
            self.parts[i].load_rows_into(local, rows, dest, sizes, heights, widths, parallel=parallel)
        return len(indices)

    def load_batch(self, indices, parallel: bool = True) -> dict[str, Any]:
        n = len(indices)
        tls = self._tls
        if getattr(tls, "rows", 0) < n:
            tls.data = np.zeros((n, self.max_size), dtype=np.uint8)
            tls.sizes = np.zeros(n, dtype=np.uint64)
            tls.rows = n
        dest, sizes = tls.data[:n], tls.sizes[:n]
        heights = np.zeros(n, dtype=np.uint32)
        widths = np.zeros(n, dtype=np.uint32)
        self.load_batch_into(indices, dest, sizes, heights, widths, parallel=parallel)
        return {"data": dest, "sizes": sizes, "heights": heights, "widths": widths}


class CombinedArrayStorage:
    """Fixed-size numeric fields (NumpyStorage) of all sources as one; optional per-source offset (labels)."""

    def __init__(self, field_name: str, parts: list, routing: _Routing, add: Sequence[int] | None = None) -> None:
        self.field_name = field_name
        self.parts = parts
        self.routing = routing
        self.add = None if add is None else [int(a) for a in add]
        dtypes = [p.dtype for p in parts]
        self.dtype = np.int64 if self.add is not None else np.result_type(*dtypes)
        shapes = {tuple(p._data.shape[1:]) for p in parts}
        if len(shapes) != 1:
            raise ValueError(f"field {field_name!r} has different shapes per source: {sorted(shapes)}")
        self._shape = shapes.pop()
        self.num_samples = int(sum(p.num_samples for p in parts))

    def load_batch(self, indices, parallel: bool = True) -> dict[str, Any]:
        out = np.empty((len(indices), *self._shape), dtype=self.dtype)
        for i, rows, local in self.routing.split(indices):
            v = self.parts[i].load_batch(local)["data"]
            out[rows] = v.astype(self.dtype, copy=False) + (self.add[i] if self.add is not None else 0)
        return {"data": out}


class CombinedStringStorage:
    def __init__(self, field_name: str, parts: list, routing: _Routing) -> None:
        self.field_name = field_name
        self.parts = parts
        self.routing = routing
        self.num_samples = int(sum(p.num_samples for p in parts))

    def load_batch(self, indices, parallel: bool = True) -> dict[str, Any]:
        out: list[str] = [""] * len(indices)
        for i, rows, local in self.routing.split(indices):
            for r, v in zip(rows, self.parts[i].load_batch(local)["data"]):
                out[r] = v
        return {"data": out}


class SourceIdStorage:
    """Synthetic ``source_id`` field: the source of each combined index."""

    def __init__(self, routing: _Routing, num_samples: int) -> None:
        self.field_name = SOURCE_ID
        self.routing = routing
        self.num_samples = num_samples
        self.dtype = np.dtype(np.int64)

    def load_batch(self, indices, parallel: bool = True) -> dict[str, Any]:
        return {"data": self.routing.source_of(indices)}


# ---------------------------------------------------------------------- cache view
class CombinedCache:
    """The sources' OptimizedCaches presented as one cache to :class:`SlipstreamLoader`."""

    def __init__(self, spec: CombinedDataset, caches: list, verbose: bool = False) -> None:
        from slipstream.cache import ImageBytesStorage, NumpyStorage, StringStorage

        self.spec = spec
        self.caches = caches
        self.sizes = np.array([len(c) for c in caches], dtype=np.int64)
        self.offsets = np.concatenate([[0], np.cumsum(self.sizes)[:-1]]).astype(np.int64)
        self.num_samples = int(self.sizes.sum())
        self.routing = _Routing(self.offsets, self.sizes)
        self.cache_dir = None                              # several; see cache_dirs
        self.cache_dirs = [c.cache_dir for c in caches]

        # standard name -> stored name, per source
        self._stored: list[dict[str, str]] = []
        for i, (c, ren) in enumerate(zip(caches, spec.rename_fields)):
            missing = [f for f in ren if f not in c.field_types]
            if missing:
                raise ValueError(f"rename_fields[{i}] names fields dataset {i} does not have: {missing} "
                                 f"(it has {list(c.field_types)})")
            std = {ren.get(f, f): f for f in c.field_types}
            if len(std) != len(c.field_types):
                raise ValueError(f"rename_fields[{i}] makes two of dataset {i}'s fields share a name: {ren}")
            for name in (SOURCE_ID, SOURCE_LABEL):
                if name in std:
                    raise ValueError(f"dataset {i} has a field named {name!r}, which CombinedDataset adds; "
                                     f"rename it with rename_fields")
            self._stored.append(std)

        types = [{s: c.field_types[f] for s, f in std.items()} for c, std in zip(caches, self._stored)]
        shared = [f for f in types[0] if all(f in t for t in types[1:])]
        self.field_types: dict[str, str] = {}
        dropped = sorted(set().union(*types) - set(shared))
        for f in shared:
            ts = {t[f] for t in types}
            if len(ts) != 1:
                raise ValueError(f"field {f!r} has different types across datasets: {[t[f] for t in types]}")
            self.field_types[f] = ts.pop()
        if dropped and verbose:
            print(f"CombinedDataset: not loading fields missing from some datasets: {dropped}")

        lf = spec.label_field
        has_label = lf in self.field_types
        if spec.labels is not None and not has_label:
            raise ValueError(f"labels={spec.labels!r} but label field {lf!r} is not shared by all datasets "
                             f"(shared: {list(self.field_types)})")
        if has_label and spec.labels is None:
            raise ValueError(f"the datasets share a {lf!r} field: say how their label spaces relate with "
                             "labels='disjoint' (each source its own classes, offset) or labels='shared'")

        self.fields: dict[str, Any] = {}
        for f, t in self.field_types.items():
            parts = [c.fields[std[f]] for c, std in zip(caches, self._stored)]
            if all(isinstance(p, ImageBytesStorage) for p in parts):
                self.fields[f] = CombinedImageStorage(f, parts, self.routing)
            elif all(isinstance(p, NumpyStorage) for p in parts):
                add = None
                if f == lf and spec.labels == "disjoint":
                    add = np.concatenate([[0], np.cumsum(spec.num_classes)[:-1]])
                    for i, (p, k) in enumerate(zip(parts, spec.num_classes)):
                        lo, hi = (int(p._data.min()), int(p._data.max())) if p.num_samples else (0, -1)
                        if lo < 0 or hi >= k:
                            raise ValueError(f"dataset {i}: {lf!r} spans [{lo}, {hi}], outside num_classes={k}")
                self.fields[f] = CombinedArrayStorage(f, parts, self.routing, add=add)
            elif all(isinstance(p, StringStorage) for p in parts):
                self.fields[f] = CombinedStringStorage(f, parts, self.routing)
            else:
                raise TypeError(f"field {f!r}: unsupported storage mix {[type(p).__name__ for p in parts]}")
        self.fields[SOURCE_ID] = SourceIdStorage(self.routing, self.num_samples)
        self.field_types[SOURCE_ID] = "int"
        if has_label:
            parts = [c.fields[std[lf]] for c, std in zip(caches, self._stored)]
            self.fields[SOURCE_LABEL] = CombinedArrayStorage(SOURCE_LABEL, parts, self.routing)
            self.field_types[SOURCE_LABEL] = self.field_types[lf]

    def __len__(self) -> int:
        return self.num_samples

    def stored_name(self, source: int, field: str) -> str:
        return self._stored[source][field]

    def locate(self, indices) -> tuple[np.ndarray, np.ndarray]:
        """Combined indices -> (source ids, record indices within each source's cache)."""
        indices = np.asarray(indices, dtype=np.int64)
        src = self.routing.source_of(indices)
        return src, indices - self.offsets[src]

    def get_image_format(self, field: str) -> str:
        fmts = {c.get_image_format(self.stored_name(i, field)) for i, c in enumerate(self.caches)}
        return fmts.pop() if len(fmts) == 1 else "mixed"

    def primary_storage(self, field: str, image_format: str, verbose: bool = False) -> tuple[Any, str]:
        """The image field's storage in one effective format for every source (yuv420 siblings built as
        needed): yuv420 when asked for, or when any source stores only yuv420 (non-JPEG originals)."""
        from slipstream.cache import build_yuv420_cache, load_yuv420_cache

        stored = [c.get_image_format(self.stored_name(i, field)) for i, c in enumerate(self.caches)]
        fmt = "yuv420" if image_format == "yuv420" or "yuv420" in stored else "jpeg"
        parts = []
        for i, (c, s) in enumerate(zip(self.caches, stored)):
            name = self.stored_name(i, field)
            if fmt == "yuv420" and s != "yuv420":
                st = load_yuv420_cache(c.cache_dir, name)
                if st is None:
                    if verbose:
                        print(f"Building YUV420 cache for dataset {i} (one-time conversion from JPEG)...")
                    st = build_yuv420_cache(c.cache_dir, name, verbose=verbose)
                parts.append(st)
            else:
                parts.append(c.fields[name])
        return CombinedImageStorage(field, parts, self.routing), fmt

    def __repr__(self) -> str:
        return f"CombinedCache({len(self.caches)} caches, sizes={self.sizes.tolist()}, fields={list(self.field_types)})"
