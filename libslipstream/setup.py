"""Build script for libslipstream C++ extension.

Build with:
    cd libslipstream && python setup.py build_ext --inplace

Or from project root:
    uv run python libslipstream/setup.py build_ext --inplace

For OpenCV support (enables resize_crop):
    USE_OPENCV=1 python setup.py build_ext --inplace
"""

import os
import sys
from pathlib import Path
from setuptools import Extension, setup

# Change to libslipstream directory so paths work correctly
script_dir = Path(__file__).parent.absolute()
os.chdir(script_dir)

# Check for OpenCV support
use_opencv = os.environ.get("USE_OPENCV", "0") == "1"

# Base configuration
sources = ["libslipstream.cpp"]
include_dirs = ["/usr/local/include", "/usr/include"]
library_dirs = ["/usr/local/lib", "/usr/lib", "/usr/lib/x86_64-linux-gnu"]
libraries = ["turbojpeg"]
extra_compile_args = ["-std=c++11", "-O3", "-fPIC"]
extra_link_args = ["-Wl,-rpath,/usr/local/lib"]
define_macros = []

# Add OpenCV if requested
if use_opencv:
    libraries.append("opencv_core")
    libraries.append("opencv_imgproc")
    define_macros.append(("USE_OPENCV", "1"))
    print("Building with OpenCV support (resize_crop enabled)")
else:
    print("Building without OpenCV (resize_crop disabled)")

# Linux-specific paths (libjpeg-turbo from package or manual install)
if sys.platform == "linux":
    include_dirs.extend([
        "/usr/libjpeg-turbo/include",
        "/opt/libjpeg-turbo/include",
    ])
    library_dirs.extend([
        "/usr/libjpeg-turbo/lib64",
        "/usr/libjpeg-turbo/lib",
        "/opt/libjpeg-turbo/lib64",
        "/opt/libjpeg-turbo/lib",
    ])
    extra_link_args.extend([
        "-Wl,-rpath,/usr/libjpeg-turbo/lib64",
        "-Wl,-rpath,/opt/libjpeg-turbo/lib64",
    ])

# macOS-specific paths
elif sys.platform == "darwin":
    # Homebrew paths for libjpeg-turbo
    homebrew_prefix = os.environ.get("HOMEBREW_PREFIX", "/opt/homebrew")
    include_dirs.extend([
        f"{homebrew_prefix}/include",
        f"{homebrew_prefix}/opt/jpeg-turbo/include",
    ])
    library_dirs.extend([
        f"{homebrew_prefix}/lib",
        f"{homebrew_prefix}/opt/jpeg-turbo/lib",
    ])
    extra_link_args = [f"-Wl,-rpath,{homebrew_prefix}/lib"]

    if use_opencv:
        include_dirs.append(f"{homebrew_prefix}/opt/opencv/include/opencv4")
        library_dirs.append(f"{homebrew_prefix}/opt/opencv/lib")

# Define the C++ extension
libslipstream = Extension(
    "_libslipstream",
    sources=sources,
    include_dirs=include_dirs,
    library_dirs=library_dirs,
    libraries=libraries,
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
    define_macros=define_macros,
)

setup(
    name="libslipstream",
    version="0.1.0",
    description="Fast parallel JPEG decoding with TurboJPEG",
    ext_modules=[libslipstream],
    # Prevent auto-discovery of packages
    py_modules=[],
    packages=[],
)
