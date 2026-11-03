# CLAUDE.md — slipstream

## Project

High-performance data loading library for PyTorch vision workloads. Namespace: `visionlab.slipstream`.

## Build & Development

```bash
uv sync --group dev              # Install dependencies
uv run pytest tests/ -m "not slow"   # core tier (~40 s Mac, ~50 s machina); see "Testing tiers"
```

**C extension** (requires system libturbojpeg: `brew install libjpeg-turbo` on macOS):

```bash
uv run python libslipstream/setup.py build_ext --inplace
```

The hatch build hook (`hatch_build.py`) fails the install when this build fails (0.9.2+);
`SLIPSTREAM_SKIP_EXT=1` skips it deliberately. `setup.py` also searches `$TURBOJPEG_ROOT`,
`$CONDA_PREFIX` and `~/.local` (`lib` and `lib64`). `slipstream/__init__.py` is a lazy namespace (PEP 562 `_LAZY` map): add new
public names to `_LAZY`, `__all__` and the `TYPE_CHECKING` block, never as eager imports. Keep
`slipstream.cli` free of torch/numba (visionlab-datasets imports it): test_import_and_build.py guards both.

**CLI** (`slipstream/cli.py`, entry point `slipstream` / `python -m slipstream`):

```bash
uv run slipstream status          # plumbing only: cache dir, permissions, S3 access, caches on disk
```

slipstream's CLI is registry-agnostic (no `sync`, no lab-dataset table). Lab-dataset status/sync
lives in `visionlab-datasets` (`visionlab-datasets status|sync`), which imports the public helpers
in `slipstream.cli` (`inspect_dir`, `check_s3`, `remote_listing`, `find_other_caches`, `fmt_bytes`,
`dir_bytes`, `configure_color`, `print_line`, glyphs, `DirAccess`, `S3Info`). Keep those
signatures stable.

**s5cmd** (for S3 remote cache):

```bash
uv tool install s5cmd    # Recommended
```

Do NOT use `pip install s5cmd` — that installs a broken 2018 Python wrapper, not the modern Go binary.

---

## Key Directives

### DO NOT run benchmarks directly (one exception: the release gate)

Benchmark scripts output progress bars that consume excessive tokens. Instead:

1. Prepare code changes
2. Ask the user to run benchmarks
3. User will paste the results back

**Release gate (approved by George 2026-09-30):** before every tag, run
`uv run python -m benchmarks.release_gate --host machina --append-md --wait 30` yourself on **machina** (clean checkout of
the commit to tag, inside `jupyter-grez72` as jovyan; see the workstation skill). It is quiet (~30 lines),
refuses to start on a busy machine (tell peers not to use machina meanwhile; send "gate done" right after), saves
`benchmarks/results/gate-full/machina/v<version>.json`, and exits 1 on a >10% drop vs the previous version or
threaded < 75% of simple. Do not tag on a FAIL; commit the results file + BENCHMARKS.md with the release.
Since 0.14.1 it times **full passes** over imagenet1k val (~3 min): per step the cache is evicted from the page
cache, then 3 epochs; cold = epoch 1, warm = best of 2-3, both gated (run-to-run noise measured at <= 5%).
Speed is the top priority after correctness (George 2026-11-03): no release may regress either number.

### Testing tiers (George 2026-11-03: don't run slow suites a change can't affect)

- `uv run python scripts/affected_tests.py [BASE]` (BASE = latest tag by default) lists the changed files, the
  test files that reference a changed module (targeted run while developing), and the tiers needed before tagging.
- **Core tier, before every tag:** `pytest tests/ -m "not slow"` on the Mac and on machina (~50 s there).
- **Full tier** (`pytest tests/`, ~4.5 min on machina) only when the script reports a full-tier trigger: cache
  build / readers / S3 sync / integrity / identity / backends / C extension / dependencies (`pyproject.toml`,
  `uv.lock`; a numba bump once cost 40% throughput). The slow files are listed in `tests/conftest.py` (`SLOW_FILES`).
- The speed gate runs before every tag regardless of tier.

### DO NOT add materialization ops to CPU benchmarks

Never add `.sum()`, `.item()`, etc. to benchmark loops. On CPU, operations are synchronous.

- **CPU**: `img.shape[0]` is sufficient
- **GPU**: Use `torch.cuda.synchronize()`, NOT `.sum()`

### DO NOT use synthetic data for verification tests

Always use real datasets. Synthetic data tests can pass while real data fails due to edge cases in actual image files.

**Real datasets for testing:**

- **FFCV**: `s3://visionlab-datasets/imagenet1k/pre-processed/s256-l512-jpgbytes-q100-ffcv/imagenet1k-s256-l512-jpg-q100-cs100-val-7ac6386e.ffcv`
- **LitData**: `s3://visionlab-datasets/imagenet1k/pre-processed/s256-l512-jpgbytes-q100-streaming/val/`
- **ImageFolder**: `s3://visionlab-datasets/imagenet1k-raw/val.tar.gz`

Tests requiring S3 access must be marked `@pytest.mark.s3`.

### Numba thread-safety in the prefetch worker

Numba's workqueue threading layer is NOT reentrant. The prefetch worker thread must use `parallel=False` for ALL Numba calls (both `load_batch` for image bytes and other fields). Only the main thread may use `parallel=True` (for decoders).

### Seeding (0.9.0+, see `slipstream/seeds.py`)

Every seeded stream is a pure function of `(seed, rank, epoch)`. The loader calls `seeds.reseed` on every
decoder / transform / after_batch_transform at each epoch start and in `set_epoch`; decoders key per-sample
draws with `sample_seeds(..., key=self._seed_key)`, transforms seed their generator with `stream_seed(self)`.
New seeded code must follow this (never `seed + offset` arithmetic, never a raw `manual_seed(self.seed)`).
The rank is the global rank and is never part of the shuffle seed.
`SourceLoader` (`slipstream/source.py`, live sources) is a separate path: it reseeds augmentations per batch
with `(rank, epoch, step)` and the source with `(rank, epoch)`; never route it through `loader.py`/`cache.py`.

### Pipeline seed conventions

Seed stream labels in `pipelines/_common.py` (values from lrm-ssl; hashed with the base seed via
`derive_seed(base, offset, view)`, no longer added):

- crop=1234, flip=1111, color/jitter=2222, gray=3333, solar=4444, blur=5555
- Global crops: scale=(0.30, 1.0), size=224
- Local crops: scale=(0.05, 0.30), size=98 (0.4375 x global_size)
- Symmetric augmentations: all crops get identical transforms

---

## Commit Messages

Use conventional commits: `feat:`, `fix:`, `refactor:`, `docs:`, `test:`
