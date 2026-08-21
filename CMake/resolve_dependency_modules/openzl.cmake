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

# Pinned to a raw commit rather than to the v0.2.0 tag, which predates both the
# cross-platform zstd/xxhash dependency handling OpenZL needs to configure
# cleanly here and the current descriptor API. The revision tracks what fbcode
# builds Nimble against, so that velox/dwio/nimble compiles identically in both
# trees; a pin older than fbcode's OpenZL breaks the internal build. OpenZL
# ships from fbcode to GitHub, so there is no tag to follow. Keep in sync with
# OPENZL_VERSION in scripts/setup-versions.sh, which installs the same revision
# for builds that resolve OpenZL as a system package.
set(VELOX_OPENZL_VERSION 7340a712cce1b8331bec3467600dba99a562e052)
set(
  VELOX_OPENZL_BUILD_SHA256_CHECKSUM
  ace6a975c3bb28da0f211c86b3f172b427acec4a2044ab637e66649cb022a953
)
string(
  CONCAT
  VELOX_OPENZL_SOURCE_URL
  "https://github.com/facebook/openzl/archive/"
  "${VELOX_OPENZL_VERSION}.tar.gz"
)

velox_resolve_dependency_url(OPENZL)

message(STATUS "Building OpenZL from source")

# Only the core `openzl` and `openzl_cpp` libraries are consumed. Everything
# else OpenZL can build pulls in dependencies Velox does not otherwise need.
set(OPENZL_BUILD_CLI OFF)
set(OPENZL_BUILD_EXAMPLES OFF)
set(OPENZL_BUILD_TOOLS OFF)
set(OPENZL_BUILD_CUSTOM_PARSERS OFF)
set(OPENZL_BUILD_TESTS OFF)
set(OPENZL_BUILD_BENCHMARKS OFF)
set(OPENZL_BUILD_PYTHON_EXT OFF)
set(OPENZL_INSTALL OFF)

# OpenZL hard-codes CMAKE_CXX_STANDARD to 17, but Velox builds with C++20. Its
# C++ bindings select between std:: types and internal polyfills (poly::span,
# poly::source_location, poly::string_view, ...) at compile time via
# feature-test macros in openzl/cpp/include/openzl/cpp/detail/Portability.hpp.
# If openzl_cpp is compiled as C++17 while Velox consumes its headers as C++20,
# the same poly:: aliases resolve to different underlying types on either side
# of the ABI, producing mismatched mangled symbols and undefined-reference link
# errors. The patch makes OpenZL honor an externally provided standard, so this
# build inherits Velox's C++20 and scripts/setup-common.sh can install a
# matching copy. The tarball is not a git repository, hence the `git init`.
FetchContent_Declare(
  openzl
  URL ${VELOX_OPENZL_SOURCE_URL}
  URL_HASH ${VELOX_OPENZL_BUILD_SHA256_CHECKSUM}
  PATCH_COMMAND git init -q && git apply ${CMAKE_CURRENT_LIST_DIR}/openzl/openzl-cxx-standard.patch
  OVERRIDE_FIND_PACKAGE
  SYSTEM
  EXCLUDE_FROM_ALL
)

# OpenZL's C++ bindings brace initialize several descriptor structs without
# naming every member. TREAT_WARNINGS_AS_ERRORS puts -Werror into the global
# CMAKE_CXX_FLAGS, which FetchContent subprojects inherit, so those become build
# failures. Velox holds its own code to that standard, not its third party
# dependencies. Relax just this warning while OpenZL is configured and restore
# the flags afterwards, as duckdb and geos already do. Only the C++ bindings are
# affected; -Werror is never added to CMAKE_C_FLAGS.
set(PREVIOUS_CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-missing-field-initializers")

FetchContent_MakeAvailable(openzl)

set(CMAKE_CXX_FLAGS ${PREVIOUS_CMAKE_CXX_FLAGS})
