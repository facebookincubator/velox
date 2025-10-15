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

set(VELOX_rapids_cmake_VERSION 25.10)
set(
  VELOX_rapids_cmake_BUILD_SHA256_CHECKSUM
  2acb16519b146021e230875ca13fcae96cbc77d5c27f9550cec64e69a63c16b5
)
set(
  VELOX_rapids_cmake_SOURCE_URL
  "https://github.com/rapidsai/rapids-cmake/archive/84f8cf8386ac56e3f4f9400f44e752345d8c2997.tar.gz"
)
velox_resolve_dependency_url(rapids_cmake)

set(VELOX_rmm_VERSION 25.10)
set(
  VELOX_rmm_BUILD_SHA256_CHECKSUM
  c1aab4e77e6a161c8d6a4afad0013f71dad1fdc0f4f866c29c67d54b4339d2bb
)
set(
  VELOX_rmm_SOURCE_URL
  "https://github.com/rapidsai/rmm/archive/271346cb2d20a425735cd02b5d11c36c82615815.tar.gz"
)
velox_resolve_dependency_url(rmm)

set(VELOX_kvikio_VERSION 25.10)
set(
  VELOX_kvikio_BUILD_SHA256_CHECKSUM
  c63c076d724a5f0f9c17cda86a413f7b339132afcb164aba31bcbfa3dd7a7605
)
set(
  VELOX_kvikio_SOURCE_URL
  "https://github.com/rapidsai/kvikio/archive/9ac0c317a352315bc82d925e09a6c82684ce3695.tar.gz"
)
velox_resolve_dependency_url(kvikio)

set(VELOX_cudf_VERSION 25.10 CACHE STRING "cudf version")

set(
  VELOX_cudf_BUILD_SHA256_CHECKSUM
  cf28398a3f397ddab7cec8711d960e549b04f7f423b832ca4dea155a68a7184a
)
set(
  VELOX_cudf_SOURCE_URL
  "https://github.com/rapidsai/cudf/archive/43505bb975f46ce4e140fdaf55192611d6e231ac.tar.gz"
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
    UPDATE_DISCONNECTED 1
  )

  FetchContent_Declare(
    rmm
    URL ${VELOX_rmm_SOURCE_URL}
    URL_HASH ${VELOX_rmm_BUILD_SHA256_CHECKSUM}
    SOURCE_SUBDIR
    cpp
    UPDATE_DISCONNECTED 1
  )

  FetchContent_Declare(
    kvikio
    URL ${VELOX_kvikio_SOURCE_URL}
    URL_HASH ${VELOX_kvikio_BUILD_SHA256_CHECKSUM}
    SOURCE_SUBDIR
    cpp
    UPDATE_DISCONNECTED 1
  )

  FetchContent_Declare(
    cudf
    URL ${VELOX_cudf_SOURCE_URL}
    URL_HASH ${VELOX_cudf_BUILD_SHA256_CHECKSUM}
    SOURCE_SUBDIR
    cpp
    UPDATE_DISCONNECTED 1
  )

  FetchContent_MakeAvailable(cudf)

  # cudf sets all warnings as errors, and therefore fails to compile with velox
  # expanded set of warnings. We selectively disable problematic warnings just for
  # cudf
  target_compile_options(
    cudf
    PRIVATE -Wno-non-virtual-dtor -Wno-missing-field-initializers -Wno-deprecated-copy -Wno-restrict
  )

  unset(BUILD_SHARED_LIBS)
endblock()
