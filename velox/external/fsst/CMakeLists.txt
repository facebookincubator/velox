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

# FSST (https://github.com/cwida/fsst), MIT licensed, vendored at commit
# 50dd308befd5121bd1121428a833f42cce6e1f9d. Upstream has never cut a release, so
# there is no version to track and nothing to find with find_package(). Only the
# sources that make up its `fsst` library are kept here; the command line tool,
# the 12-bit variant and 58MB of conference material are omitted.
#
# This replaces the upstream CMakeLists, which cannot be used as-is: it declares
# cmake_minimum_required(VERSION 3.0), which CMake 4 rejects outright, and it
# appends -march=native whenever the compiler accepts it, which would override
# the architecture flags Velox derives from CPU_TARGET and produce binaries that
# fault on other hosts.

# Deliberately an OBJECT library rather than velox_add_library. Nimble renames
# FSST's exported symbols by compiling these sources with
# -Dfsst_create=nimble_fsst_create and friends, which has to apply to this
# target alone. Under VELOX_MONO_LIBRARY, velox_add_library would fold the
# sources into the single `velox` target, where those definitions would leak
# into every other translation unit.
#
# The upstream sources also define C++ helpers with external linkage. Those
# helpers live in the nimble_fsst_internal namespace to avoid colliding with
# DuckDB's separately vendored copy of FSST in the same process.
#
# A plain static library would keep that isolation but leave a gap for installed
# consumers: CMake does not merge static archives, so libfsst.a would stay
# outside libvelox.a and go uninstalled, leaving fsst_* undefined downstream. An
# OBJECT library's objects are compiled into whichever target links it, so FSST
# ends up inside libvelox for both the static and the shared build.
add_library(fsst OBJECT libfsst.cpp fsst_avx512.cpp)

# An OBJECT library that a target links must belong to the same export set as
# that target, or install(EXPORT velox_targets) fails to configure. This adds no
# installed artifact; it only registers the target.
install(TARGETS fsst EXPORT velox_targets)

# fsst.h sits next to the sources and is included as <fsst.h> by consumers.
# Deliberately not SYSTEM: DuckDB vendors a header of the same name and, when
# resolved BUNDLED, contributes it as a plain -I. -isystem is searched after
# every -I, so a SYSTEM export here loses and Nimble compiles against DuckDB's
# FSST, leaving the nimble_fsst_* renames undefined.
target_include_directories(fsst PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")

# Upstream compiles this one file at -O1 in release builds. The AVX-512 kernels
# are heavily unrolled by the .inc files, and -O3 makes compilation
# disproportionately slow for no measurable gain.
if(CMAKE_BUILD_TYPE STREQUAL "Release")
  set_source_files_properties(fsst_avx512.cpp PROPERTIES COMPILE_OPTIONS -O1)
endif()
