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

# rapids_cmake commit cce9b03 from 2026-03-20
set(VELOX_rapids_cmake_VERSION 26.06)
set(VELOX_rapids_cmake_COMMIT cce9b03b6e30fbb40d092ed1053342fa34ae2a2c)
set(
  VELOX_rapids_cmake_BUILD_SHA256_CHECKSUM
  e29b092312912476d28f7144b1dbe5435066bacd97eb416ccedc271a9c927414
)
set(
  VELOX_rapids_cmake_SOURCE_URL
  "https://github.com/rapidsai/rapids-cmake/archive/${VELOX_rapids_cmake_COMMIT}.tar.gz"
)
velox_resolve_dependency_url(rapids_cmake)

# rmm commit 86a91a3 from 2026-03-18
set(VELOX_rmm_VERSION 26.06)
set(VELOX_rmm_COMMIT 86a91a364481e2ca7ab030da4caca1179860d4e5)
set(
  VELOX_rmm_BUILD_SHA256_CHECKSUM
  87efb89bd7b5fad9b0c46e8764f5c75025889b92828b4faf8ce39aade4b85395
)
set(VELOX_rmm_SOURCE_URL "https://github.com/rapidsai/rmm/archive/${VELOX_rmm_COMMIT}.tar.gz")
velox_resolve_dependency_url(rmm)

# kvikio commit afeb69e from 2026-03-18
set(VELOX_kvikio_VERSION 26.06)
set(VELOX_kvikio_COMMIT afeb69e791ab512a0a2fbdb05a19be9318e1db84)
set(
  VELOX_kvikio_BUILD_SHA256_CHECKSUM
  6ddf5f1e230f165bcac77227368f52455b6b398992f9413c46da1de71ea237ca
)
set(
  VELOX_kvikio_SOURCE_URL
  "https://github.com/rapidsai/kvikio/archive/${VELOX_kvikio_COMMIT}.tar.gz"
)
velox_resolve_dependency_url(kvikio)

# cudf commit da748f4 from 2026-03-20
set(VELOX_cudf_VERSION 26.06 CACHE STRING "cudf version")
set(VELOX_cudf_COMMIT da748f44c8d44e6203beeacf3b3d846c40a37e31)
set(
  VELOX_cudf_BUILD_SHA256_CHECKSUM
  92d74e2f2a78dbf64956ea281fcf0df6a8a28ff865392bc079d79cb65fa24c7c
)
set(VELOX_cudf_SOURCE_URL "https://github.com/rapidsai/cudf/archive/${VELOX_cudf_COMMIT}.tar.gz")
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
  unset(BUILD_TESTING CACHE)
endblock()
