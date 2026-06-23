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

# rapids_cmake commit 01775db from 2026-06-11
set(VELOX_rapids_cmake_VERSION 26.08)
set(VELOX_rapids_cmake_COMMIT 01775dba8b91385fa516545f82667e959d777d4f)
set(
  VELOX_rapids_cmake_BUILD_SHA256_CHECKSUM
  3182880408b5ddeaf2bd9b55aba4a6d0cd44ad3ad3f72f8a79413f3892b956c6
)
set(
  VELOX_rapids_cmake_SOURCE_URL
  "https://github.com/rapidsai/rapids-cmake/archive/${VELOX_rapids_cmake_COMMIT}.tar.gz"
)
velox_resolve_dependency_url(rapids_cmake)

# rmm commit 9ad6c33 from 2026-06-11
set(VELOX_rmm_VERSION 26.08)
set(VELOX_rmm_COMMIT 9ad6c33d02913fe384c78b16c02b6ace2ac94a6d)
set(
  VELOX_rmm_BUILD_SHA256_CHECKSUM
  f54fd7204ff04222f42f69781b9e3f32347757f74c8b446d18151ddd6e17cfe0
)
set(VELOX_rmm_SOURCE_URL "https://github.com/rapidsai/rmm/archive/${VELOX_rmm_COMMIT}.tar.gz")
velox_resolve_dependency_url(rmm)

# kvikio commit 3f967c5 from 2026-06-11
set(VELOX_kvikio_VERSION 26.08)
set(VELOX_kvikio_COMMIT 3f967c5e38fad9ea73633e733c0cb29b58b8a0d8)
set(
  VELOX_kvikio_BUILD_SHA256_CHECKSUM
  12ab7b37f392b7cdd1c2e35aacdc7aa9fbd4a859a560414de7275446923f9c73
)
set(
  VELOX_kvikio_SOURCE_URL
  "https://github.com/rapidsai/kvikio/archive/${VELOX_kvikio_COMMIT}.tar.gz"
)
velox_resolve_dependency_url(kvikio)

# cudf commit 6de5991 from 2026-06-11
set(VELOX_cudf_VERSION 26.08 CACHE STRING "cudf version")
set(VELOX_cudf_COMMIT 6de59911ffb3c9461080a3d22df73dfcbefde9f1)
set(
  VELOX_cudf_BUILD_SHA256_CHECKSUM
  99194cc68c22100c9464b8a09512cfb5bf948499a1f9a3ede679b245af850c44
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
  # ucxx commit 0c59e80 from 2026-06-11 (release/0.50 branch)
  set(VELOX_ucxx_VERSION 0.51)
  set(VELOX_ucxx_COMMIT 0c59e804d0aa0f3ec15965005a2ce4b747792cd6)
  set(
    VELOX_ucxx_BUILD_SHA256_CHECKSUM
    c8682404e96d0248a05a04894cee0aadac0bbeb81efe6b7a2521a9e587d43da2
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
  set(BUILD_SHARED_LIBS ON)

  # TODO(mh,bd): Remove this once we have a permanent solution for the spdlog/fmt
  # incompatibility.

  # cuDF (via rapids_logger) pins spdlog 1.14.1, which is incompatible with
  # the fmt 11.2.0 that Velox builds. Override the rapids-cmake/CPM spdlog
  # version to 1.15.3, which is fmt 11.2 compatible.
  # RAPIDS_CMAKE_CPM_OVERRIDE_VERSION_FILE is honored by every rapids_cpm_init,
  # so the override applies before rapids_logger fetches spdlog.
  set(RAPIDS_CMAKE_CPM_OVERRIDE_VERSION_FILE "${CMAKE_CURRENT_LIST_DIR}/cudf-cpm-overrides.json")

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
