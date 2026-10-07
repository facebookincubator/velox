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

# rapids_cmake commit 008faba from 2026-10-01 (main branch)
set(VELOX_rapids_cmake_VERSION 26.12)
set(VELOX_rapids_cmake_COMMIT 008fabaad824ff09281b29e9a570580487e45f05)
set(
  VELOX_rapids_cmake_BUILD_SHA256_CHECKSUM
  ae971124426e74151a7c9757cb5fdd50972df0e4ae3c506be5e3e91a990d685e
)
set(
  VELOX_rapids_cmake_SOURCE_URL
  "https://github.com/rapidsai/rapids-cmake/archive/${VELOX_rapids_cmake_COMMIT}.tar.gz"
)
velox_resolve_dependency_url(rapids_cmake)

# rmm commit 2d64592 from 2026-10-01 (main branch)
set(VELOX_rmm_VERSION 26.12)
set(VELOX_rmm_COMMIT 2d645925a1aea7ce8809aa57e6d78e32bdb0e36b)
set(
  VELOX_rmm_BUILD_SHA256_CHECKSUM
  1905a24d0d14570746002dcd322ac76a5975b9eefbb5ac13e454b439026b9dd5
)
set(VELOX_rmm_SOURCE_URL "https://github.com/rapidsai/rmm/archive/${VELOX_rmm_COMMIT}.tar.gz")
velox_resolve_dependency_url(rmm)

# kvikio commit 257d08b from 2026-10-01 (main branch)
set(VELOX_kvikio_VERSION 26.12)
set(VELOX_kvikio_COMMIT 257d08b41ef4e566cba6f5f64cb365e681961a3b)
set(
  VELOX_kvikio_BUILD_SHA256_CHECKSUM
  15117616e0f7228e2324e0e64c21f3fa9305ffbe6b92d842f967e9b2ceb47bfc
)
set(
  VELOX_kvikio_SOURCE_URL
  "https://github.com/rapidsai/kvikio/archive/${VELOX_kvikio_COMMIT}.tar.gz"
)
velox_resolve_dependency_url(kvikio)

# cudf commit af62255 from 2026-10-02 (main branch)
set(VELOX_cudf_VERSION 26.12 CACHE STRING "cudf version")
set(VELOX_cudf_COMMIT af6225524d845b2d2db2ace06ea1031226ef8ed1)
set(
  VELOX_cudf_BUILD_SHA256_CHECKSUM
  6cb59cfa7315c4cb97114f0018b48f7a94df81ecbf692627a6d81632873bfd9f
)
set(VELOX_cudf_SOURCE_URL "https://github.com/rapidsai/cudf/archive/${VELOX_cudf_COMMIT}.tar.gz")
velox_resolve_dependency_url(cudf)

# Whether to build the experimental UCX GPU exchange transport
# (velox/experimental/ucx-exchange) and the cuDF-side registration that selects
# it. Off by default because no CI job builds it, so a dependency bump that
# breaks it would otherwise break the default cuDF build on every host with a
# system UCX. Requires a system UCX install. Declared here, next to the ucxx
# fetch it controls, rather than with the other options. This file only runs
# for a bundled cuDF, so the option exists only then; cache variables are
# global, so every subdirectory sees it.
option(
  VELOX_ENABLE_UCX_EXCHANGE
  "Build the experimental UCX GPU exchange transport. Requires a system UCX install."
  OFF
)

if(VELOX_ENABLE_UCX_EXCHANGE)
  # velox_ucx_exchange runs its own find_package(ucx REQUIRED); this probe fails
  # the configure before cuDF is fetched and built.
  find_library(UCX_LIBRARY NAMES ucp)
  find_path(UCX_INCLUDE_DIR NAMES ucp/api/ucp.h)
  if(NOT UCX_LIBRARY OR NOT UCX_INCLUDE_DIR)
    message(
      FATAL_ERROR
      "VELOX_ENABLE_UCX_EXCHANGE=ON but no system UCX was found (need libucp and ucp/api/ucp.h)."
    )
  endif()
  message(
    STATUS
    "UCX exchange enabled with ${UCX_LIBRARY} (headers: ${UCX_INCLUDE_DIR}) -- ucxx will be fetched"
  )
  # ucxx commit 7ecd4f5 from 2026-09-29 (main branch)
  set(VELOX_ucxx_VERSION 0.53)
  set(VELOX_ucxx_COMMIT 7ecd4f55ce9a833b3f23c85a574d07db8f98e0b8)
  set(
    VELOX_ucxx_BUILD_SHA256_CHECKSUM
    8e3ab889d8a4610b2859d3b36d981eac8b0eb68034ebf4ec9fdba645c255159c
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
