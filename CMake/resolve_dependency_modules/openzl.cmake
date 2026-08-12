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

# Pinned to the commit the standalone Nimble repository carries as its openzl
# submodule, rather than to the v0.2.0 tag: Nimble is validated against this
# revision, and v0.2.0 predates the cross-platform zstd/xxhash dependency
# handling that OpenZL needs to configure cleanly here.
set(VELOX_OPENZL_VERSION 6b48fa4868160ed1e5c78ac422639615dd0dcf28)
set(
  VELOX_OPENZL_BUILD_SHA256_CHECKSUM
  59bd4d05cc8a34a8d92863e42846af4bf99772da4452f5d59cde1229cb7d89d4
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

FetchContent_MakeAvailable(openzl)
