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

# rapids_cmake commit 6c639ca from 2026-06-03
set(VELOX_rapids_cmake_VERSION 26.06)
set(VELOX_rapids_cmake_COMMIT 6c639ca87831fd0b04c8d65450373eb362d9bc1d)
set(
  VELOX_rapids_cmake_BUILD_SHA256_CHECKSUM
  5620a0c8482d98950de8c1be9f4747c6bedb07f369562ac79c1a055caf731f83
)
set(
  VELOX_rapids_cmake_SOURCE_URL
  "https://github.com/rapidsai/rapids-cmake/archive/${VELOX_rapids_cmake_COMMIT}.tar.gz"
)
velox_resolve_dependency_url(rapids_cmake)

# rmm commit f3310cb from 2026-06-04
set(VELOX_rmm_VERSION 26.06)
set(VELOX_rmm_COMMIT f3310cb85b3fe15fd21f550adbbf7eeb1e374588)
set(
  VELOX_rmm_BUILD_SHA256_CHECKSUM
  d2d78501e54e17119f4f0c4028e90684f356be557ee06bdb73bad75a3bb3ffeb
)
set(VELOX_rmm_SOURCE_URL "https://github.com/rapidsai/rmm/archive/${VELOX_rmm_COMMIT}.tar.gz")
velox_resolve_dependency_url(rmm)

# kvikio commit 5eb6c5d from 2026-06-03
set(VELOX_kvikio_VERSION 26.06)
set(VELOX_kvikio_COMMIT 5eb6c5d8b6e544cead3bdc336516927adb12612a)
set(
  VELOX_kvikio_BUILD_SHA256_CHECKSUM
  cb6c954edeb9b1f3226ac741b1736a43449a2731231e64a3e467f32dc8b8a5b2
)
set(
  VELOX_kvikio_SOURCE_URL
  "https://github.com/rapidsai/kvikio/archive/${VELOX_kvikio_COMMIT}.tar.gz"
)
velox_resolve_dependency_url(kvikio)

# cudf commit 33320d6 from 2026-06-04
set(VELOX_cudf_VERSION 26.06 CACHE STRING "cudf version")
set(VELOX_cudf_COMMIT 33320d64c94a64c94bccc5e2c522721e4d275858)
set(
  VELOX_cudf_BUILD_SHA256_CHECKSUM
  b3d855a70e62435e038f39559cc2d99711737e1f8db2b91c9792408bfdfe4614
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
  # ucxx commit dc57333 from 2026-06-09 (release/0.50 branch)
  set(VELOX_ucxx_VERSION 0.50)
  set(VELOX_ucxx_COMMIT dc573338cdc651ed520bd4f3de8b350462c20bed)
  set(
    VELOX_ucxx_BUILD_SHA256_CHECKSUM
    0f9ab9f4124766259f3ab06dc5fad0e5fe100724390eaba17112afc75444af68
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
