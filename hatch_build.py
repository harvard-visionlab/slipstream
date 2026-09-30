"""Hatch build hook to compile the libslipstream C++ extension.

This hook runs automatically during `pip install` / `uv sync` to build
the C++ extension (TurboJPEG + stb_image_resize2) without requiring
a manual `python libslipstream/setup.py build_ext --inplace` step.

Requires system libturbojpeg:
  - macOS: brew install libjpeg-turbo
  - Ubuntu: apt-get install libturbojpeg0-dev
  - no root (e.g. a cluster): conda-forge libjpeg-turbo, or a cmake install into ~/.local;
    point TURBOJPEG_ROOT at a non-standard prefix.

A failed build fails the install (a slipstream without the decoder only breaks later, at the first
batch). Set SLIPSTREAM_SKIP_EXT=1 to install without it on purpose, e.g. on a machine that
only uses the CLI helpers; `slipstream status` reports whether the decoder is available.
"""

import os
import subprocess
import sys
import sysconfig
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class LibslipstreamBuildHook(BuildHookInterface):
    PLUGIN_NAME = "libslipstream"

    def initialize(self, version, build_data):
        """Build the C++ extension before packaging."""
        root = Path(self.root)
        libdir = root / "libslipstream"
        setup_py = libdir / "setup.py"

        if not setup_py.exists():
            self.app.display_warning(
                f"libslipstream/setup.py not found at {setup_py}, skipping C++ build"
            )
            return

        if os.environ.get("SLIPSTREAM_SKIP_EXT", "").strip().lower() in ("1", "true", "yes"):
            self.app.display_warning(
                "SLIPSTREAM_SKIP_EXT is set: skipping the libslipstream C++ build. "
                "NumbaBatchDecoder and every JPEG/YUV decode pipeline will be unavailable."
            )
            return

        self.app.display_info("Building libslipstream C++ extension...")

        try:
            subprocess.check_call(
                # --force: always recompile and relink. Otherwise setuptools sees an up-to-date .so
                # (e.g. in uv's cached git checkout) and keeps the rpath / libturbojpeg of an earlier
                # build, silently ignoring a changed TURBOJPEG_ROOT.
                [sys.executable, str(setup_py), "build_ext", "--inplace", "--force"],
                cwd=str(libdir),
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            )
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            detail = f"exit code {e.returncode}" if isinstance(e, subprocess.CalledProcessError) else str(e)
            raise RuntimeError(
                f"Failed to build the libslipstream C++ extension ({detail}); without it "
                "NumbaBatchDecoder and every decode pipeline fail. It needs the TurboJPEG API "
                "library (libturbojpeg + turbojpeg.h), not just libjpeg:\n"
                "  macOS:  brew install libjpeg-turbo\n"
                "  Ubuntu: apt-get install libturbojpeg0-dev\n"
                "  no root (cluster): conda install -c conda-forge libjpeg-turbo (in an active env), "
                "or build libjpeg-turbo with cmake into ~/.local, or set TURBOJPEG_ROOT=<prefix>\n"
                "To install without the decoder on purpose (CLI-only machines): SLIPSTREAM_SKIP_EXT=1"
            ) from e
        # The wheel carries a compiled extension for this interpreter: tag it per interpreter/platform
        # (e.g. cp311-cp311-macosx_11_0_arm64), not py3-none-any. Otherwise uv caches one wheel per
        # commit and installs a cpython-312 build into a 3.10 venv. Ship only this interpreter's
        # build: other versions' .so files can sit in the same (shared uv git) checkout.
        so = libdir / f"_libslipstream{sysconfig.get_config_var('EXT_SUFFIX') or '.so'}"
        if not so.exists():
            raise RuntimeError(f"libslipstream build finished but {so.name} is missing")
        build_data["pure_python"] = False
        build_data["infer_tag"] = True
        if version != "editable":                 # editable installs load it from the source tree
            build_data["force_include"][str(so)] = f"libslipstream/{so.name}"
        self.app.display_success(f"libslipstream C++ extension built successfully ({so.name})")
