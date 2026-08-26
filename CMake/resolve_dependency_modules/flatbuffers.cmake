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
include_guard(GLOBAL)

# Held at the version cuDF expects; do not bump on its own. cuDF resolves
# FlatBuffers through rapids_cpm_find(flatbuffers 24.3.25), which calls
# find_package() before falling back to its own download, and FlatBuffers'
# config-version file reports every newer version as compatible. Any later
# version visible to that find_package() is therefore used in place of 24.3.25,
# and cuDF's checked-in generated headers static_assert on major 24 / minor 3.
# Nimble itself only requires 22.9.4 or later, for build_flatbuffers().
set(VELOX_FLATBUFFERS_VERSION 24.3.25)
set(
  VELOX_FLATBUFFERS_BUILD_SHA256_CHECKSUM
  4157c5cacdb59737c5d627e47ac26b140e9ee28b1102f812b36068aab728c1ed
)
string(
  CONCAT
  VELOX_FLATBUFFERS_SOURCE_URL
  "https://github.com/google/flatbuffers/archive/refs/tags/"
  "v${VELOX_FLATBUFFERS_VERSION}.tar.gz"
)

velox_resolve_dependency_url(FLATBUFFERS)

message(STATUS "Building FlatBuffers from source")

# The flatc code generator is required, not just the runtime library: Nimble
# generates C++ headers from .fbs schemas at build time via build_flatbuffers().
# This version exports it as the plain `flatc` target, which build_flatbuffers()
# falls back to on its own, so there is no need to set
# FLATBUFFERS_FLATC_EXECUTABLE here. FlatBuffers does set that variable itself,
# but only in its own directory scope, so it never reaches Velox.
set(FLATBUFFERS_BUILD_FLATC ON)
set(FLATBUFFERS_BUILD_FLATLIB ON)
set(FLATBUFFERS_BUILD_SHAREDLIB OFF)
set(FLATBUFFERS_BUILD_FLATHASH OFF)
set(FLATBUFFERS_BUILD_TESTS OFF)
set(FLATBUFFERS_INSTALL OFF)

FetchContent_Declare(
  flatbuffers
  URL ${VELOX_FLATBUFFERS_SOURCE_URL}
  URL_HASH ${VELOX_FLATBUFFERS_BUILD_SHA256_CHECKSUM}
  OVERRIDE_FIND_PACKAGE
  SYSTEM
  EXCLUDE_FROM_ALL
)

FetchContent_MakeAvailable(flatbuffers)

# Consumers written against a system FlatBuffers use FLATBUFFERS_INCLUDE_DIR,
# which only FindFlatBuffers.cmake defines. Export it for the bundled build too
# so the same target_include_directories() call works either way.
set(
  FLATBUFFERS_INCLUDE_DIR
  "${flatbuffers_SOURCE_DIR}/include"
  CACHE INTERNAL
  "FlatBuffers include directory"
)
