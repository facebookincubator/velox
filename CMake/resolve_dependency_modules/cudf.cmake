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

# 3.30.4 is the minimum version required by cudf
cmake_minimum_required(VERSION 3.30.4)

set(VELOX_rapids_cmake_VERSION 25.06)
set(VELOX_rapids_cmake_BUILD_SHA256_CHECKSUM
    812cef3478ef2ee02742d8cba68ab754603eb21e1333f1be03b91e7820ca0e27)
set(VELOX_rapids_cmake_SOURCE_URL
    "https://github.com/rapidsai/rapids-cmake/archive/4f203ce0126c91ff57289dfc70555f29cd81a8ee.tar.gz"
)
velox_resolve_dependency_url(rapids_cmake)

set(VELOX_rmm_VERSION 25.06)
set(VELOX_rmm_BUILD_SHA256_CHECKSUM
    d2cabadc6c484620a7aaff533494920f4e8c8c027ae717cae83110883e3cb378)
set(VELOX_rmm_SOURCE_URL
    "https://github.com/rapidsai/rmm/archive/c7a331432f003bcaa8cc45d7bbcbd21a2876565f.tar.gz"
)
velox_resolve_dependency_url(rmm)

set(VELOX_kvikio_VERSION 25.06)
set(VELOX_kvikio_BUILD_SHA256_CHECKSUM
    bdf756557ea6608ec5a00cde5130ec22b6059f96208c269a83c49917967804f8)
set(VELOX_kvikio_SOURCE_URL
    "https://github.com/rapidsai/kvikio/archive/9f143867a41c56d6df4c58572311c9bee004285a.tar.gz"
)
velox_resolve_dependency_url(kvikio)

set(VELOX_cudf_VERSION 25.06)
set(VELOX_cudf_BUILD_SHA256_CHECKSUM
    39e1b32c8491bb84f52388a3ae9cd28b1e82ce81f311e6678067741f9d0ac01d)
set(VELOX_cudf_SOURCE_URL
    "https://github.com/rapidsai/cudf/archive/191620472f3d3daeadf32003c37ee99eaa4773a9.tar.gz"
)
velox_resolve_dependency_url(cudf)

# Use block so we don't leak variables
block(SCOPE_FOR VARIABLES)
# Setup libcudf build to not have testing components
set(BUILD_TESTS OFF)
set(CUDF_BUILD_TESTUTIL OFF)
set(BUILD_SHARED_LIBS ON)

FetchContent_Declare(
  rapids-cmake
  URL ${VELOX_rapids_cmake_SOURCE_URL}
  URL_HASH ${VELOX_rapids_cmake_BUILD_SHA256_CHECKSUM}
  UPDATE_DISCONNECTED 1)

FetchContent_Declare(
  rmm
  URL ${VELOX_rmm_SOURCE_URL}
  URL_HASH ${VELOX_rmm_BUILD_SHA256_CHECKSUM}
  UPDATE_DISCONNECTED 1)

FetchContent_Declare(
  kvikio
  URL ${VELOX_kvikio_SOURCE_URL}
  URL_HASH ${VELOX_kvikio_BUILD_SHA256_CHECKSUM}
  SOURCE_SUBDIR cpp
  UPDATE_DISCONNECTED 1)

FetchContent_Declare(
  cudf
  URL ${VELOX_cudf_SOURCE_URL}
  URL_HASH ${VELOX_cudf_BUILD_SHA256_CHECKSUM}
  SOURCE_SUBDIR cpp
  UPDATE_DISCONNECTED 1)

FetchContent_MakeAvailable(cudf)

# cudf sets all warnings as errors, and therefore fails to compile with velox
# expanded set of warnings. We selectively disable problematic warnings just for
# cudf
target_compile_options(
  cudf PRIVATE -Wno-non-virtual-dtor -Wno-missing-field-initializers
               -Wno-deprecated-copy)

unset(BUILD_SHARED_LIBS)
endblock()
