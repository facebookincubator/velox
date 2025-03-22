#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import os
import re
import shutil
import subprocess
import sys
from distutils.command.clean import clean as clean_orig
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build import build as build_orig
from setuptools.command.build_ext import build_ext

ROOT_DIR = Path(__file__).parent.resolve()


class BuildCommand(build_orig):
    def initialize_options(self):
        super().initialize_options()
        self.build_base = "_build"


class CMakeBuild(build_ext):
    def run(self):
        try:
            import pyarrow  # noqa: F401
        except ImportError:
            raise RuntimeError("pyarrow must be installed before building pyvelox.")

        try:
            subprocess.check_output(["cmake", "--version"])
        except OSError:
            raise RuntimeError("CMake is not available.")

        super().run()

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        # Allow using a pre-built Velox library (for CI and development) e.g. 'VELOX_BUILD_DIR=_build/velox/debug'
        # The build in question must have been built with 'VELOX_BUILD_PYTHON_PACKAGE=ON' and the same python version.
        if "VELOX_BUILD_DIR" in os.environ:
            velox_dir = os.path.abspath(os.environ["VELOX_BUILD_DIR"])
            if not os.path.isdir(extdir):
                os.symlink(velox_dir, os.path.dirname(extdir), target_is_directory=True)
            print(f"Using pre-built Velox library from {velox_dir}")
            return

        # required for auto-detection of auxiliary "native" libs
        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep

        if "DEBUG" in os.environ:
            cfg = "Debug" if os.environ["DEBUG"] == "1" else "MinSizeRel"
        else:
            cfg = "Debug" if self.debug else "MinSizeRel"

        exec_path = sys.executable

        cmake_args = [
            "-DVELOX_BUILD_TESTING=OFF",
            "-DVELOX_MONO_LIBRARY=ON",
            "-DVELOX_BUILD_SHARED=ON",
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DCMAKE_BUILD_TYPE={cfg}",
            f"-DCMAKE_INSTALL_PREFIX={extdir}",
            "-DCMAKE_VERBOSE_MAKEFILE=ON",
            "-DVELOX_BUILD_PYTHON_PACKAGE=ON",
            f"-DPYTHON_EXECUTABLE={exec_path}",
        ]

        build_args = []

        if "CMAKE_GENERATOR" not in os.environ:
            cmake_args += ["-GNinja"]

        if "CMAKE_BUILD_PARALLEL_LEVEL" not in os.environ and getattr(
            self, "parallel", None
        ):
            build_args += [f"-j{self.parallel}"]

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        subprocess.check_call(
            ["cmake", str(ROOT_DIR)] + cmake_args, cwd=self.build_temp
        )
        subprocess.check_call(
            ["cmake", "--build", "."] + build_args, cwd=self.build_temp
        )


class CleanCommand(clean_orig):
    def run(self):
        super().run()
        for path in (ROOT_DIR / "pyvelox").rglob("*.so"):
            print(f"removing '{path}'")
            path.unlink()
        build_dirs = [ROOT_DIR / "_build"]
        for path in build_dirs:
            if path.exists():
                print(f"removing '{path}' (and everything under it)")
                shutil.rmtree(str(path), ignore_errors=True)


setup(
    ext_modules=[Extension(name="pyvelox.legacy", sources=[])],
    cmdclass={
        "build_ext": CMakeBuild,
        "clean": CleanCommand,
        "build": BuildCommand,
    },
)
