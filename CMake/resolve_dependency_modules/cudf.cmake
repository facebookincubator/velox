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

# rapids_cmake commit a9d2afb from 2026-01-05
set(VELOX_rapids_cmake_VERSION 26.02)
set(VELOX_rapids_cmake_COMMIT a9d2afb645aaed0ae9bff0b074613a1f09081416)
set(
  VELOX_rapids_cmake_BUILD_SHA256_CHECKSUM
  955784e2c752fbf1d7110734ccef79c1949dbee664be70e2fad275b11d881318
)
set(
  VELOX_rapids_cmake_SOURCE_URL
  "https://github.com/rapidsai/rapids-cmake/archive/${VELOX_rapids_cmake_COMMIT}.tar.gz"
)
velox_resolve_dependency_url(rapids_cmake)

# rmm commit d3cd8b9 from 2026-01-07
set(VELOX_rmm_VERSION 26.02)
set(VELOX_rmm_COMMIT d3cd8b9b7d87d0f3a6ff174fc5251a53686f9fec)
set(
  VELOX_rmm_BUILD_SHA256_CHECKSUM
  ad845b8fb4fb5d977a2d919b27023dfa57b311beb2405d777d42f5b9b059fe67
)
set(VELOX_rmm_SOURCE_URL "https://github.com/rapidsai/rmm/archive/${VELOX_rmm_COMMIT}.tar.gz")
velox_resolve_dependency_url(rmm)

# kvikio commit 1141512 from 2026-01-05
set(VELOX_kvikio_VERSION 26.02)
set(VELOX_kvikio_COMMIT 11415120c7e019c3fcf2ec2201d6e03502cbc645)
set(
  VELOX_kvikio_BUILD_SHA256_CHECKSUM
  5c333da29eceda6ed056f8dd8f198234cc1717dd6548d05fbd5aae1dcd8678e0
)
set(
  VELOX_kvikio_SOURCE_URL
  "https://github.com/rapidsai/kvikio/archive/${VELOX_kvikio_COMMIT}.tar.gz"
)
velox_resolve_dependency_url(kvikio)

# cudf commit 3f85f62 from 2026-01-09
set(VELOX_cudf_VERSION 26.02 CACHE STRING "cudf version")
set(VELOX_cudf_COMMIT 3f85f626633ca4202941cc3bf3112bcd319eab8e)
set(
  VELOX_cudf_BUILD_SHA256_CHECKSUM
  96b54c2b33281f58183978429933740869ef384d2687308699a257b05076d4fd
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
