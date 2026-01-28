"""Hatch build hook to compile the libslipstream C++ extension.

This hook runs automatically during `pip install` / `uv sync` to build
the C++ extension (TurboJPEG + stb_image_resize2) without requiring
a manual `python libslipstream/setup.py build_ext --inplace` step.

Requires system libturbojpeg:
  - macOS: brew install libjpeg-turbo
  - Ubuntu: apt-get install libturbojpeg0-dev
"""

import os
import subprocess
import sys
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

        self.app.display_info("Building libslipstream C++ extension...")

        try:
            subprocess.check_call(
                [sys.executable, str(setup_py), "build_ext", "--inplace"],
                cwd=str(libdir),
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            )
            self.app.display_success("libslipstream C++ extension built successfully")
        except subprocess.CalledProcessError as e:
            self.app.display_warning(
                f"Failed to build libslipstream C++ extension (exit code {e.returncode}). "
                "The NumbaBatchDecoder will not be available. "
                "Ensure libturbojpeg is installed:\n"
                "  macOS: brew install libjpeg-turbo\n"
                "  Ubuntu: apt-get install libturbojpeg0-dev"
            )
        except FileNotFoundError:
            self.app.display_warning(
                "Python executable not found for C++ extension build. "
                "Run manually: uv run python libslipstream/setup.py build_ext --inplace"
            )
