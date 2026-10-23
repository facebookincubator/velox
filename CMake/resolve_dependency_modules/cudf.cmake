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

# 4.0 is the minimum version required by cudf
cmake_minimum_required(VERSION 4.0)

# rapids_cmake commit 8c5dde7 from 2026-10-22 (main branch)
set(VELOX_rapids_cmake_VERSION 26.12)
set(VELOX_rapids_cmake_COMMIT 8c5dde7fa22db63b243cb4723ccefea7adf8d8d2)
set(
  VELOX_rapids_cmake_BUILD_SHA256_CHECKSUM
  5faef3f99c348fc35a8318cb018d90a6621814ee65cdc600a7518b0728a3e3d7
)
set(
  VELOX_rapids_cmake_SOURCE_URL
  "https://github.com/rapidsai/rapids-cmake/archive/${VELOX_rapids_cmake_COMMIT}.tar.gz"
)
velox_resolve_dependency_url(rapids_cmake)

# rmm commit 9d84148 from 2026-10-19 (main branch)
set(VELOX_rmm_VERSION 26.12)
set(VELOX_rmm_COMMIT 9d8414834dd331a0e5449329b458d4c2ffee58eb)
set(
  VELOX_rmm_BUILD_SHA256_CHECKSUM
  e2190bed6538b46f5c4087b2309d61130005fb8b67c78868b9512f788ab03741
)
set(VELOX_rmm_SOURCE_URL "https://github.com/rapidsai/rmm/archive/${VELOX_rmm_COMMIT}.tar.gz")
velox_resolve_dependency_url(rmm)

# kvikio commit 5c2b975 from 2026-10-20 (main branch)
set(VELOX_kvikio_VERSION 26.12)
set(VELOX_kvikio_COMMIT 5c2b9755b182de46e5838c6a5aabf5899a2938b2)
set(
  VELOX_kvikio_BUILD_SHA256_CHECKSUM
  aa3f85fe42463044a3e7212d8cea615ed1bf3b7991d802d6c404dd9c68627a31
)
set(
  VELOX_kvikio_SOURCE_URL
  "https://github.com/rapidsai/kvikio/archive/${VELOX_kvikio_COMMIT}.tar.gz"
)
velox_resolve_dependency_url(kvikio)

# cudf commit aaad800 from 2026-10-23 (aaad80039628604d5da216f95de2535dc2e1bb81 branch)
set(VELOX_cudf_VERSION 26.12 CACHE STRING "cudf version")
set(VELOX_cudf_COMMIT aaad80039628604d5da216f95de2535dc2e1bb81)
set(
  VELOX_cudf_BUILD_SHA256_CHECKSUM
  02feae1ddbdcdfeff9dc6c4efbc5b0ef27e8fcf767dea91beed5d2fcc310fae0
)
set(VELOX_cudf_SOURCE_URL "https://github.com/rapidsai/cudf/archive/${VELOX_cudf_COMMIT}.tar.gz")
velox_resolve_dependency_url(cudf)

# Probe for a system UCX install. The variables are used only to gate ucxx
# fetching below; nothing in Velox links against UCX directly yet.
find_library(UCX_LIBRARY NAMES ucp)
find_path(UCX_INCLUDE_DIR NAMES ucp/api/ucp.h)
if(UCX_LIBRARY AND UCX_INCLUDE_DIR)
  set(UCX_FOUND TRUE)
else()
  set(UCX_FOUND FALSE)
endif()
if(UCX_FOUND)
  message(STATUS "Found UCX: ${UCX_LIBRARY} (headers: ${UCX_INCLUDE_DIR}) -- ucxx will be fetched")
  # ucxx commit 86378fa from 2026-10-20 (main branch)
  set(VELOX_ucxx_VERSION 0.53)
  set(VELOX_ucxx_COMMIT 86378fa658b8d9601cf3640de8f9354baca36d6d)
  set(
    VELOX_ucxx_BUILD_SHA256_CHECKSUM
    0c8de5bd7676188ca7e6727a0005fdc7465145defda70d3b0d3a284e9a65473f
  )
  set(VELOX_ucxx_SOURCE_URL "https://github.com/rapidsai/ucxx/archive/${VELOX_ucxx_COMMIT}.tar.gz")
  velox_resolve_dependency_url(ucxx)
else()
  message(STATUS "UCX not found -- ucxx will not be fetched")
endif()

# Use block so we don't leak variables
block(SCOPE_FOR VARIABLES)
  # Setup libcudf build to not have testing components
  set(BUILD_TESTS OFF)
  set(CUDF_BUILD_TESTUTIL OFF)
  set(CUDF_BUILD_STREAMS_TEST_UTIL OFF)
  # Keep spdlog/nvcomp shared to avoid multiple copies of spdlog in the final binary.
  set(CUDF_BUILD_STATIC_DEPS OFF)
  set(BUILD_SHARED_LIBS ON)
  set(KvikIO_BUILD_NSYS_PLUGIN OFF)

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

  if(UCX_FOUND)
    FetchContent_Declare(
      ucxx
      URL ${VELOX_ucxx_SOURCE_URL}
      URL_HASH ${VELOX_ucxx_BUILD_SHA256_CHECKSUM}
      SOURCE_SUBDIR
      cpp
      UPDATE_DISCONNECTED 1
    )
  endif()

  FetchContent_MakeAvailable(cudf)

  if(UCX_FOUND)
    FetchContent_MakeAvailable(ucxx)
  endif()

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
