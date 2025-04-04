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

set(VELOX_rapids_cmake_VERSION 25.04)
set(VELOX_rapids_cmake_BUILD_SHA256_CHECKSUM
    458c14eaff9000067b32d65c8c914f4521090ede7690e16eb57035ce731386db)
set(VELOX_rapids_cmake_SOURCE_URL
    "https://github.com/rapidsai/rapids-cmake/archive/7828fc8ff2e9f4fa86099f3c844505c2f47ac672.tar.gz"
)
velox_resolve_dependency_url(rapids_cmake)

set(VELOX_rmm_VERSION 25.04)
set(VELOX_rmm_BUILD_SHA256_CHECKSUM
    17aa9cf50e37ac0058bd09cb05f01e0c1b788ba5ce3e77fc9f7e386fab54397a)
set(VELOX_rmm_SOURCE_URL
    "https://github.com/rapidsai/rmm/archive/7529f921a0bea3587e357be89d127797a4acea37.tar.gz"
)
velox_resolve_dependency_url(rmm)

set(VELOX_kvikio_VERSION 25.04)
set(VELOX_kvikio_BUILD_SHA256_CHECKSUM
    a39ba878ddc7bdd065bb7e4ecf04fd7944c0d51c8ebf8a49b6ee24dacebeb021)
set(VELOX_kvikio_SOURCE_URL
    "https://github.com/rapidsai/kvikio/archive/0b90bb84872fd2f4709d116d9d60d3741ef577a2.tar.gz"
)
velox_resolve_dependency_url(kvikio)

set(VELOX_cudf_VERSION 25.06)
set(VELOX_cudf_BUILD_SHA256_CHECKSUM
    b9e9ebf7593571940aa25320466278e48fc3aea4e6795ebf63ffa41155f8f218)
set(VELOX_cudf_SOURCE_URL
    "https://github.com/rapidsai/cudf/archive/52a7f51d1e845d0fb4faf4577aa6af8fcae7e1fb.tar.gz"
)
velox_resolve_dependency_url(cudf)

# Use block so we don't leak variables
block(SCOPE_FOR VARIABLES)
# Setup libcudf build to not have testing components
set(BUILD_TESTS OFF)
set(CUDF_BUILD_TESTUTIL OFF)
set(BUILD_SHARED_LIBS ON)

# cudf sets all warnings as errors, and therefore fails to compile with velox
# expanded set of warnings. We selectively disable problematic warnings just for
# cudf
string(
  APPEND CMAKE_CXX_FLAGS
  " -Wno-non-virtual-dtor -Wno-missing-field-initializers -Wno-deprecated-copy")

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
unset(BUILD_SHARED_LIBS)
endblock()
