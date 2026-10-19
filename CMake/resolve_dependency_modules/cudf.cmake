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

# rapids_cmake commit 037d4c5 from 2026-10-09 (main branch)
set(VELOX_rapids_cmake_VERSION 26.12)
set(VELOX_rapids_cmake_COMMIT 037d4c5489cc13f5b378b54197238a6e2cdbff17)
set(
  VELOX_rapids_cmake_BUILD_SHA256_CHECKSUM
  946ebc43d81dc7359e973df54bc0749f281a0b65b32a299b00ee6a66507b0a04
)
set(
  VELOX_rapids_cmake_SOURCE_URL
  "https://github.com/rapidsai/rapids-cmake/archive/${VELOX_rapids_cmake_COMMIT}.tar.gz"
)
velox_resolve_dependency_url(rapids_cmake)

# rmm commit 7e29fc0 from 2026-10-15 (main branch)
set(VELOX_rmm_VERSION 26.12)
set(VELOX_rmm_COMMIT 7e29fc098edc5d4d8f186c0da2be277410839a05)
set(
  VELOX_rmm_BUILD_SHA256_CHECKSUM
  e1cb8d9f3483bebaea805817182da1b9a2c315a2cd71dd6496d0577381fec5b8
)
set(VELOX_rmm_SOURCE_URL "https://github.com/rapidsai/rmm/archive/${VELOX_rmm_COMMIT}.tar.gz")
velox_resolve_dependency_url(rmm)

# kvikio commit 90b116c from 2026-10-13 (main branch)
set(VELOX_kvikio_VERSION 26.12)
set(VELOX_kvikio_COMMIT 90b116c67316eedd6b561c0ac7bcdfd09e536be9)
set(
  VELOX_kvikio_BUILD_SHA256_CHECKSUM
  df6e4ad41b956a6fa9378aed901f85daafac7c521d913661da8d7fc477d9e0aa
)
set(
  VELOX_kvikio_SOURCE_URL
  "https://github.com/rapidsai/kvikio/archive/${VELOX_kvikio_COMMIT}.tar.gz"
)
velox_resolve_dependency_url(kvikio)

# cudf commit c9086e0 from 2026-10-15 (main branch)
set(VELOX_cudf_VERSION 26.12 CACHE STRING "cudf version")
set(VELOX_cudf_COMMIT c9086e0afbf0cb11194a12bb2233b80139eabd09)
set(
  VELOX_cudf_BUILD_SHA256_CHECKSUM
  65c4df018b62ced84ebd74235f70abd053fbcae342e2eb720bc0571023dba72f
)
set(VELOX_cudf_SOURCE_URL "https://github.com/rapidsai/cudf/archive/${VELOX_cudf_COMMIT}.tar.gz")
velox_resolve_dependency_url(cudf)

# Probe for a system UCX install, to pick the default for
# VELOX_ENABLE_UCX_EXCHANGE below. velox_ucx_exchange runs its own
# find_package(ucx REQUIRED); this probe only decides the default and fails the
# configure before cuDF is fetched when the option is on without UCX.
find_library(UCX_LIBRARY NAMES ucp)
find_path(UCX_INCLUDE_DIR NAMES ucp/api/ucp.h)
if(UCX_LIBRARY AND UCX_INCLUDE_DIR)
  set(UCX_FOUND TRUE)
else()
  set(UCX_FOUND FALSE)
endif()

# Whether to build the experimental UCX GPU exchange transport
# (velox/experimental/ucx-exchange) and the cuDF-side registration that selects
# it. Defaults, on the first configure of a build directory, to whether a system
# UCX was found; the value is cached after that, so a UCX installed later needs
# -DVELOX_ENABLE_UCX_EXCHANGE=ON, and -DVELOX_ENABLE_UCX_EXCHANGE=OFF builds
# without it on a host that has UCX. Declared here, next to the ucxx fetch it
# controls, rather than with the other options. This file only runs for a
# bundled cuDF, so the option exists only then; cache variables are global, so
# every subdirectory sees it.
option(
  VELOX_ENABLE_UCX_EXCHANGE
  "Build the experimental UCX GPU exchange transport. Requires a system UCX install."
  ${UCX_FOUND}
)

if(VELOX_ENABLE_UCX_EXCHANGE)
  if(NOT UCX_FOUND)
    message(
      FATAL_ERROR
      "VELOX_ENABLE_UCX_EXCHANGE=ON but no system UCX was found (need libucp and ucp/api/ucp.h)."
    )
  endif()
  message(
    STATUS
    "UCX exchange enabled with ${UCX_LIBRARY} (headers: ${UCX_INCLUDE_DIR}) -- ucxx will be fetched"
  )
  # ucxx commit f3e85e5 from 2026-10-14 (main branch)
  set(VELOX_ucxx_VERSION 0.53)
  set(VELOX_ucxx_COMMIT f3e85e580fb4349240917f85f5a6260c7f3b94f6)
  set(
    VELOX_ucxx_BUILD_SHA256_CHECKSUM
    63c5f8a9e7a84bab932b52d1dad82619c684a0383b3318914eba029d9b2fa9cd
  )
  set(VELOX_ucxx_SOURCE_URL "https://github.com/rapidsai/ucxx/archive/${VELOX_ucxx_COMMIT}.tar.gz")
  velox_resolve_dependency_url(ucxx)
else()
  if(UCX_FOUND)
    message(
      STATUS
      "UCX found, but VELOX_ENABLE_UCX_EXCHANGE is OFF -- pass -DVELOX_ENABLE_UCX_EXCHANGE=ON to build the UCX exchange"
    )
  endif()
  message(STATUS "UCX exchange disabled -- ucxx will not be fetched")
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

  FetchContent_MakeAvailable(cudf)

  # cuDF does not use ucxx, and ucxx takes rapids-cmake from the declaration
  # above, so ucxx is resolved on its own after cuDF.
  if(VELOX_ENABLE_UCX_EXCHANGE)
    FetchContent_Declare(
      ucxx
      URL ${VELOX_ucxx_SOURCE_URL}
      URL_HASH ${VELOX_ucxx_BUILD_SHA256_CHECKSUM}
      SOURCE_SUBDIR
      cpp
      UPDATE_DISCONNECTED 1
    )
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
