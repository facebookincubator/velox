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

# rapids_cmake commit 7ece71c from 2026-02-04
set(VELOX_rapids_cmake_VERSION 26.04)
set(VELOX_rapids_cmake_COMMIT 7ece71c2f94fb0ed402d567b457ce54ecb859695)
set(
  VELOX_rapids_cmake_BUILD_SHA256_CHECKSUM
  02abaa8580c30a0b01eb142d5cd58b5acc85005bf58f5360f4a62efbd6e4635a
)
set(
  VELOX_rapids_cmake_SOURCE_URL
  "https://github.com/rapidsai/rapids-cmake/archive/${VELOX_rapids_cmake_COMMIT}.tar.gz"
)
velox_resolve_dependency_url(rapids_cmake)

# rmm commit f225c62 from 2026-02-10
set(VELOX_rmm_VERSION 26.04)
set(VELOX_rmm_COMMIT f225c620fc177cbdd807dc67bfb53fa4b4272e9f)
set(
  VELOX_rmm_BUILD_SHA256_CHECKSUM
  c6a5b4855802d7c17c0a0a978f2643792b9e17a6b773dd6bbb2f99b16496b38a
)
set(VELOX_rmm_SOURCE_URL "https://github.com/rapidsai/rmm/archive/${VELOX_rmm_COMMIT}.tar.gz")
velox_resolve_dependency_url(rmm)

# kvikio commit 62b3d6f from 2026-02-10
set(VELOX_kvikio_VERSION 26.04)
set(VELOX_kvikio_COMMIT 62b3d6fa4ba1fc518b0069a8214b1283e63df019)
set(
  VELOX_kvikio_BUILD_SHA256_CHECKSUM
  b590a3b6f2d1fa0b36e07d26361d0a5f3cbcce25582f8ee07bcfcf5796ea37b7
)
set(
  VELOX_kvikio_SOURCE_URL
  "https://github.com/rapidsai/kvikio/archive/${VELOX_kvikio_COMMIT}.tar.gz"
)
velox_resolve_dependency_url(kvikio)

# cudf commit e6ba1fe from 2026-02-11
set(VELOX_cudf_VERSION 26.04 CACHE STRING "cudf version")
set(VELOX_cudf_COMMIT e6ba1feee8f056ce2e245f771403cfa5f598d813)
set(
  VELOX_cudf_BUILD_SHA256_CHECKSUM
  19137d306db0ddbf4eebb4333e0257de4815563039b1e805829ce2dcc525c3f5
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
