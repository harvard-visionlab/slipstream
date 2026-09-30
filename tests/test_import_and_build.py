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


def test_build_hook_always_relinks(monkeypatch):
    # 0.9.3: setuptools skipped relinking an up-to-date .so (uv's cached git checkout), keeping an
    # earlier TURBOJPEG_ROOT's rpath. The hook must force a rebuild every time.
    import importlib.util
    import types
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]

    class _Hook:                                  # stand-in for hatchling's BuildHookInterface
        def __init__(self):
            self.root = str(root)
            quiet = lambda *a, **k: None
            self.app = types.SimpleNamespace(display_info=quiet, display_success=quiet, display_warning=quiet)

    for name in ("hatchling", "hatchling.builders", "hatchling.builders.hooks",
                 "hatchling.builders.hooks.plugin", "hatchling.builders.hooks.plugin.interface"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    sys.modules["hatchling.builders.hooks.plugin.interface"].BuildHookInterface = _Hook
    spec = importlib.util.spec_from_file_location("hatch_build", root / "hatch_build.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    calls = []
    monkeypatch.setattr(mod.subprocess, "check_call", lambda cmd, **kw: calls.append(cmd))
    monkeypatch.delenv("SLIPSTREAM_SKIP_EXT", raising=False)
    mod.LibslipstreamBuildHook().initialize("standard", {})
    assert calls and calls[0][-3:] == ["build_ext", "--inplace", "--force"]


def test_loader_prefers_this_interpreters_build():
    # 0.9.4: with stale builds for other Pythons next to ours, the loader picked the first glob match.
    import shutil
    import sysconfig
    from pathlib import Path

    import pytest

    from slipstream.decoders.numba_decoder import _find_library

    libdir = Path(__file__).resolve().parents[1] / "libslipstream"
    own = libdir / f"_libslipstream{sysconfig.get_config_var('EXT_SUFFIX')}"
    if not own.exists():
        pytest.skip("decoder not built for this interpreter")
    stale = [libdir / f"_libslipstream.cpython-{v}-stale-test.so" for v in ("30", "99", "0")]
    try:
        for p in stale:
            shutil.copy(own, p)
        assert _find_library() == own
    finally:
        for p in stale:
            p.unlink(missing_ok=True)
