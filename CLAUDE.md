# CLAUDE.md — slipstream

## Project

High-performance data loading library for PyTorch vision workloads. Namespace: `visionlab.slipstream`.

## Build & Development

```bash
uv sync --group dev              # Install dependencies
uv run pytest tests/ -v          # Run tests
```

**C extension** (requires system libturbojpeg: `brew install libjpeg-turbo` on macOS):

```bash
uv run python libslipstream/setup.py build_ext --inplace
```

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

### DO NOT run benchmarks directly

Benchmark scripts output progress bars that consume excessive tokens. Instead:

1. Prepare code changes
2. Ask the user to run benchmarks
3. User will paste the results back

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

### Pipeline seed conventions (must match lrm-ssl yaml)

Seed offsets defined in `pipelines/_common.py`:

- crop=1234, flip=1111, color/jitter=2222, gray=3333, solar=4444, blur=5555
- Global crops: scale=(0.30, 1.0), size=224
- Local crops: scale=(0.05, 0.30), size=98 (0.4375 x global_size)
- Symmetric augmentations: all crops get identical transforms

---

## Commit Messages

Use conventional commits: `feat:`, `fix:`, `refactor:`, `docs:`, `test:`
