"""0.9.2: cheap `import slipstream`, and a visible decoder build status."""

import subprocess
import sys

from slipstream import cli


def _run(code: str) -> str:
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip()


def test_import_does_not_load_torchvision():
    # torchvision pulls in torchvision.models + torch._dynamo (~2-7 s on a cluster filesystem).
    assert _run("import sys, slipstream; print('torchvision' in sys.modules)") == "False"


def test_import_loads_no_heavy_dependency():
    # Whole namespace is lazy (PEP 562): names load on first access.
    out = _run("import sys, slipstream; print([m for m in ('torch', 'numba', 'litdata', 'torchvision') if m in sys.modules])")
    assert out == "[]"


def test_every_public_name_resolves():
    out = _run(
        "import slipstream\n"
        "bad = [n for n in slipstream.__all__ if getattr(slipstream, n, None) is None and n != 'prep']\n"
        "import slipstream.decoders as d\n"
        "print(bad, slipstream.decoders is d, slipstream.prep.__name__, 'SlipstreamLoader' in dir(slipstream))"
    )
    assert out == "[] True slipstream.prep True"


def test_lazy_imagefolder_exports_resolve():
    out = _run(
        "import sys, slipstream\n"
        "from slipstream import SlipstreamImageFolder, open_imagefolder\n"
        "from slipstream.readers import SlipstreamImageFolder as S2\n"
        "from slipstream.readers.imagefolder import SlipstreamImageFolder as S3\n"
        "print(SlipstreamImageFolder is S2 is S3, callable(open_imagefolder), 'torchvision' in sys.modules)"
    )
    assert out == "True True True"


def test_unknown_attribute_still_raises():
    assert _run(
        "import slipstream\ntry:\n    slipstream.nope\nexcept AttributeError:\n    print('raised')"
    ) == "raised"


def test_decoder_status_reported():
    dec = cli.check_decoder()
    assert set(dec) == {"available", "path", "error"}
    status = {"cache": {"access": {"exists": True, "readable": True, "traversable": True, "writable": True},
                        "path": "/x"},
              "s3": {"s5cmd_error": None, "credentials_found": True, "checked": False},
              "decoder": {"available": False, "path": None, "error": "not built"}}
    assert any("decoder not built" in m for m in cli.problems(status))
    status["decoder"] = {"available": True, "path": "/lib.so", "error": None}
    assert not any("decoder" in m for m in cli.problems(status))
    del status["decoder"]                          # dicts built by other tools (no decoder key) still work
    assert cli.problems(status) == []


def test_cli_import_is_light():
    # visionlab-datasets imports slipstream.cli for `visionlab-datasets status`.
    out = _run("import sys, slipstream.cli; print([m for m in ('torch', 'numba', 'litdata', 'torchvision') if m in sys.modules])")
    assert out == "[]"
