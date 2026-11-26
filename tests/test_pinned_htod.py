"""0.18.0: a decoder followed by ToTorchImage(<cuda>) writes into pinned memory and the copy runs async on a
side stream. Output must be byte-identical to the synchronous path: real data, past buffer reuse, partial last
batch, 2 epochs, with the GPU deliberately backed up so copies lag. Needs CUDA (machina)."""

import numpy as np
import pytest
import torch

from slipstream import SlipstreamLoader
from slipstream.decoders import DecodeCenterCrop, DecodeRandomResizedCrop
from slipstream.transforms import ToTorchImage
from tests.test_ssl_params import _cache

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _run(fmt, make, pinned, dtype=torch.float32, check_count=True):
    from slipstream import SlipstreamDataset
    path = _cache(fmt)
    if path is None:
        pytest.skip(f"no real {fmt} val cache")
    orig = SlipstreamLoader._enable_pinned_outputs
    if not pinned:
        SlipstreamLoader._enable_pinned_outputs = lambda self: None
    try:
        loader = SlipstreamLoader(SlipstreamDataset(input_dir=str(path)), batch_size=96, shuffle=True, seed=0,
                                  drop_last=False, verbose=False, exclude_fields=["path"], image_format=fmt,
                                  indices=np.arange(500),
                                  pipelines={"image": [make(), ToTorchImage("cuda", dtype=dtype)]})
    finally:
        SlipstreamLoader._enable_pinned_outputs = orig
    from slipstream import _pinned
    n0 = _pinned.async_copies
    dec = loader.pipelines["image"][0]._decoder
    a = torch.randn(2048, 2048, device="cuda")
    out = []
    try:
        for _ in range(2):
            for b in loader:
                for _ in range(10):
                    a @ a                                  # GPU busy: the next decode races the copy
                out.append(b["image"].cpu())
    finally:
        loader.shutdown()
    n_async = _pinned.async_copies - n0
    if check_count:                                  # the async path really ran (counter is process-wide)
        assert n_async == (len(out) if pinned else 0), (n_async, len(out))
    return out, dec.pin_output


@pytest.mark.parametrize("fmt", ["jpeg", "yuv420"])
@pytest.mark.parametrize("make", [lambda: DecodeRandomResizedCrop(size=64, seed=1), lambda: DecodeCenterCrop(64)],
                         ids=["rrc", "center"])
def test_pinned_async_matches_sync(fmt, make):
    ref, p0 = _run(fmt, make, pinned=False)
    got, p1 = _run(fmt, make, pinned=True)
    assert (p0, p1) == (False, True)
    assert len(ref) == len(got) and ref[-1].shape[0] == 500 % 96     # partial last batch
    for r, g in zip(ref, got):
        assert torch.equal(r, g)


def test_pinned_bf16_matches_sync():
    make = lambda: DecodeRandomResizedCrop(size=64, seed=1)  # noqa: E731
    ref, _ = _run("jpeg", make, pinned=False, dtype=torch.bfloat16)
    got, _ = _run("jpeg", make, pinned=True, dtype=torch.bfloat16)
    assert all(torch.equal(r, g) for r, g in zip(ref, got))


def test_not_pinned_without_cuda_consumer():
    """Raw decoder output (no ToTorchImage on CUDA next) stays pageable: user code may copy it itself."""
    from slipstream import SlipstreamDataset
    path = _cache("jpeg")
    if path is None:
        pytest.skip("no real jpeg val cache")
    for pipe in ([DecodeRandomResizedCrop(size=64, seed=1)],
                 [DecodeRandomResizedCrop(size=64, seed=1), ToTorchImage("cpu")]):
        loader = SlipstreamLoader(SlipstreamDataset(input_dir=str(path)), batch_size=64, verbose=False,
                                  exclude_fields=["path"], pipelines={"image": pipe})
        assert not pipe[0]._decoder.pin_output
        loader.shutdown()


def test_two_loaders_two_threads():
    """fence() is global: two pinned loaders iterated concurrently must not drop each other's in-flight copies."""
    import threading
    make = lambda: DecodeRandomResizedCrop(size=64, seed=1)  # noqa: E731
    ref, _ = _run("jpeg", make, pinned=False)
    results, errors = {}, []

    def work(k):
        try:
            results[k], _ = _run("jpeg", make, pinned=True, check_count=False)
        except BaseException as e:  # noqa: BLE001
            errors.append(e)
    ts = [threading.Thread(target=work, args=(k,)) for k in range(2)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert not errors, errors
    for k in range(2):
        assert all(torch.equal(r, g) for r, g in zip(ref, results[k]))


def test_presets_get_pinned():
    """The shipped pipelines (decoder then ToTorchImage on CUDA) take the async path."""
    from slipstream import SlipstreamDataset
    from slipstream.pipelines import supervised_train
    path = _cache("jpeg")
    if path is None:
        pytest.skip("no real jpeg val cache")
    pipes = supervised_train(64, device="cuda")
    loader = SlipstreamLoader(SlipstreamDataset(input_dir=str(path)), batch_size=64, verbose=False,
                              exclude_fields=["path"], pipelines=pipes)
    try:
        decs = [getattr(s, "_decoder", None) for pl in loader.pipelines.values() if isinstance(pl, list) for s in pl]
        assert any(getattr(d, "pin_output", False) for d in decs), [type(s).__name__ for pl in
                                                                     loader.pipelines.values() for s in pl]
    finally:
        loader.shutdown()


def test_shutdown_drops_pinned_buffers_and_loader_still_iterates():
    """0.18.1: shutdown() releases the decoders' pinned buffers; the next epoch reallocates them."""
    from slipstream import SlipstreamDataset
    path = _cache("jpeg")
    if path is None:
        pytest.skip("no real jpeg val cache")
    loader = SlipstreamLoader(SlipstreamDataset(input_dir=str(path)), batch_size=64, shuffle=False, seed=0,
                              verbose=False, exclude_fields=["path"], image_format="jpeg", indices=np.arange(128),
                              pipelines={"image": [DecodeCenterCrop(64), ToTorchImage("cuda")]})
    dec = loader.pipelines["image"][0]._decoder
    first = [b["image"].cpu() for b in loader]
    it = iter(loader)
    next(it)                                         # abandoned mid-epoch: worker + queued batches alive
    loader.shutdown()
    assert dec.pin_output and dec._dest_buffer is None and dec._chw_buffer is None
    second = [b["image"].cpu() for b in loader]
    loader.shutdown()
    assert all(torch.equal(a, b) for a, b in zip(first, second, strict=True))


def test_exit_hook_releases_pinned_memory(tmp_path):
    """0.18.1: a process that leaves a pinned loader mid-epoch (no shutdown) frees its pinned memory in the
    atexit hook, prints one line, and exits cleanly."""
    import subprocess
    import sys
    if _cache("jpeg") is None:
        pytest.skip("no real jpeg val cache")
    script = tmp_path / "exit_pinned.py"
    script.write_text(
        "import numpy as np\n"
        "from slipstream import SlipstreamDataset, SlipstreamLoader\n"
        "from slipstream.decoders import DecodeRandomResizedCrop\n"
        "from slipstream.transforms import ToTorchImage\n"
        "from tests.test_ssl_params import _cache\n"
        "ds = SlipstreamDataset(input_dir=str(_cache('jpeg')))\n"
        "loader = SlipstreamLoader(ds, batch_size=256, verbose=False, exclude_fields=['path'],\n"
        "    image_format='jpeg', indices=np.arange(2048),\n"
        "    pipelines={'image': [DecodeRandomResizedCrop(size=224, seed=1), ToTorchImage('cuda')]})\n"
        "it = iter(loader)\n"
        "for _ in range(3):\n"
        "    b = next(it)\n"
        "print('done', b['image'].shape, flush=True)\n")
    r = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, timeout=120,
                       cwd=str(__import__("pathlib").Path(__file__).parents[1]))
    assert r.returncode == 0, r.stderr[-2000:]
    assert "done" in r.stdout
    assert "slipstream: released pinned host memory at exit" in r.stderr, r.stderr[-2000:]
