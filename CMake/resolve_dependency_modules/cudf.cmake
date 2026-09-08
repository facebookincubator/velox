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

# rapids_cmake commit 9c0829e from 2026-07-23 (release/26.08 branch)
set(VELOX_rapids_cmake_VERSION 26.08)
set(VELOX_rapids_cmake_COMMIT 9c0829ec73702b3df8a5c2ec43f6aaabe5f1e5ec)
set(
  VELOX_rapids_cmake_BUILD_SHA256_CHECKSUM
  3582f621f3b3d63952aafd716985901a075f2ad85602073973f3619761723962
)
set(
  VELOX_rapids_cmake_SOURCE_URL
  "https://github.com/rapidsai/rapids-cmake/archive/${VELOX_rapids_cmake_COMMIT}.tar.gz"
)
velox_resolve_dependency_url(rapids_cmake)

# rmm commit 1a39f9e from 2026-07-24 (release/26.08 branch)
set(VELOX_rmm_VERSION 26.08)
set(VELOX_rmm_COMMIT 1a39f9e81c467b1a9522a4dcec8b5581ae165fef)
set(
  VELOX_rmm_BUILD_SHA256_CHECKSUM
  d8b82bc491a2c5b093cdfb4e1fb99630ccf16d1cbd69bc74451fbb56efc3e68c
)
set(VELOX_rmm_SOURCE_URL "https://github.com/rapidsai/rmm/archive/${VELOX_rmm_COMMIT}.tar.gz")
velox_resolve_dependency_url(rmm)

# kvikio commit 93606c0 from 2026-07-23 (release/26.08 branch)
set(VELOX_kvikio_VERSION 26.08)
set(VELOX_kvikio_COMMIT 93606c074f3d863a7052af25afae569f72cb3304)
set(
  VELOX_kvikio_BUILD_SHA256_CHECKSUM
  882e1b7c8950c0bf3520c3c317b0d85da2c91b66d90f3ff44ae81a60166979f3
)
set(
  VELOX_kvikio_SOURCE_URL
  "https://github.com/rapidsai/kvikio/archive/${VELOX_kvikio_COMMIT}.tar.gz"
)
velox_resolve_dependency_url(kvikio)

# cudf commit 5beaa59 from 2026-07-27 (release/26.08 branch)
set(VELOX_cudf_VERSION 26.08 CACHE STRING "cudf version")
set(VELOX_cudf_COMMIT 5beaa5954688fcb12236ffb434e192ea2c77db30)
set(
  VELOX_cudf_BUILD_SHA256_CHECKSUM
  1fc77d1ddf97ede67b783bbabacf69d658254be30b2859299f4255525443b1d3
)
set(VELOX_cudf_SOURCE_URL "https://github.com/rapidsai/cudf/archive/${VELOX_cudf_COMMIT}.tar.gz")
velox_resolve_dependency_url(cudf)

# Probe for a system UCX install, to pick the default for
# VELOX_ENABLE_UCX_EXCHANGE below. velox_ucx_exchange runs its own
# find_package(ucx REQUIRED); this probe only decides whether we opt in by
# default and whether ucxx is fetched.
find_library(UCX_LIBRARY NAMES ucp)
find_path(UCX_INCLUDE_DIR NAMES ucp/api/ucp.h)
if(UCX_LIBRARY AND UCX_INCLUDE_DIR)
  set(UCX_FOUND TRUE)
else()
  set(UCX_FOUND FALSE)
endif()

# Whether to build the experimental UCX GPU exchange transport
# (velox/experimental/ucx-exchange) and the cuDF-side registration that selects
# it. Defaults to whether a system UCX was found, which reproduces the earlier
# implicit behaviour, but can be forced either way from the command line --
# -DVELOX_ENABLE_UCX_EXCHANGE=OFF is how the no-UCX configuration is exercised
# on a host that does have UCX. Declared here rather than next to the other
# options because the default depends on the probe above; cache variables are
# global, so every subdirectory sees it. Requires VELOX_ENABLE_CUDF, since this
# file is only reached when cuDF is enabled and the transport links cudf::cudf.
option(
  VELOX_ENABLE_UCX_EXCHANGE
  "Build the experimental UCX GPU exchange transport. Requires a system UCX install."
  ${UCX_FOUND}
)
if(VELOX_ENABLE_UCX_EXCHANGE AND NOT UCX_FOUND)
  message(
    FATAL_ERROR
    "VELOX_ENABLE_UCX_EXCHANGE=ON but no system UCX was found (need libucp and ucp/api/ucp.h)."
  )
endif()

if(VELOX_ENABLE_UCX_EXCHANGE)
  message(
    STATUS
    "UCX exchange enabled with ${UCX_LIBRARY} (headers: ${UCX_INCLUDE_DIR}) -- ucxx will be fetched"
  )
  # ucxx commit b7faed1 from 2026-07-23 (release/0.51 branch)
  set(VELOX_ucxx_VERSION 0.51)
  set(VELOX_ucxx_COMMIT b7faed1a2e8038f63676183cdb056c3b69daa15d)
  set(
    VELOX_ucxx_BUILD_SHA256_CHECKSUM
    3eb5ff5459dde31edf344f24f0b3086550be70961038b69229b05973b6f37524
  )
  set(VELOX_ucxx_SOURCE_URL "https://github.com/rapidsai/ucxx/archive/${VELOX_ucxx_COMMIT}.tar.gz")
  velox_resolve_dependency_url(ucxx)
else()
  message(STATUS "UCX exchange disabled -- ucxx will not be fetched")
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

  if(VELOX_ENABLE_UCX_EXCHANGE)
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

  if(VELOX_ENABLE_UCX_EXCHANGE)
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
