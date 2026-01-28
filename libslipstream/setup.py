"""Build script for libslipstream C++ extension.

Build with:
    cd libslipstream && python setup.py build_ext --inplace

Or from project root:
    uv run python libslipstream/setup.py build_ext --inplace

For OpenCV support (enables resize_crop):
    USE_OPENCV=1 python setup.py build_ext --inplace
"""

import os
import subprocess
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
    # Try pkg-config for OpenCV paths (works on most Linux systems)
    try:
        cv_cflags = subprocess.check_output(
            ["pkg-config", "--cflags", "opencv4"], stderr=subprocess.DEVNULL
        ).decode().strip().split()
        cv_libs = subprocess.check_output(
            ["pkg-config", "--libs-only-L", "opencv4"], stderr=subprocess.DEVNULL
        ).decode().strip().split()
        for flag in cv_cflags:
            if flag.startswith("-I"):
                include_dirs.append(flag[2:])
        for flag in cv_libs:
            if flag.startswith("-L"):
                library_dirs.append(flag[2:])
        print(f"Building with OpenCV support (pkg-config found)")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Building with OpenCV support (pkg-config not found, using default paths)")
else:
    print("Building without OpenCV (stb_image_resize2 fallback)")

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
