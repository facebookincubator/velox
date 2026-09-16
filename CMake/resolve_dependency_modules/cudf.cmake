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

# rapids_cmake commit 5df8fd1 from 2026-09-08 (release/26.10 branch)
set(VELOX_rapids_cmake_VERSION 26.10)
set(VELOX_rapids_cmake_COMMIT 5df8fd1ea26515b6b50fa94844ef1855e457048c)
set(
  VELOX_rapids_cmake_BUILD_SHA256_CHECKSUM
  f7cb91451aeae915f066907f9ae4eb555348fb62db1857c205a67691816c78c3
)
set(
  VELOX_rapids_cmake_SOURCE_URL
  "https://github.com/rapidsai/rapids-cmake/archive/${VELOX_rapids_cmake_COMMIT}.tar.gz"
)
velox_resolve_dependency_url(rapids_cmake)

# rmm commit 9a693e0 from 2026-09-10 (release/26.10 branch)
set(VELOX_rmm_VERSION 26.10)
set(VELOX_rmm_COMMIT 9a693e042004e1da9d2e2db018ce2c2963437d59)
set(
  VELOX_rmm_BUILD_SHA256_CHECKSUM
  98916c2801fd9ad72eba8bb95ac45813ac933a4877d2d43445fa174402949063
)
set(VELOX_rmm_SOURCE_URL "https://github.com/rapidsai/rmm/archive/${VELOX_rmm_COMMIT}.tar.gz")
velox_resolve_dependency_url(rmm)

# kvikio commit 3ea0db0 from 2026-09-10 (release/26.10 branch)
set(VELOX_kvikio_VERSION 26.10)
set(VELOX_kvikio_COMMIT 3ea0db06a308925097e6fe84628f5888efca13b8)
set(
  VELOX_kvikio_BUILD_SHA256_CHECKSUM
  b2c8418ef8eba3f08c4dcb859b3df44711b5ae5cbbf85fc8c5c09992bfb1f9bc
)
set(
  VELOX_kvikio_SOURCE_URL
  "https://github.com/rapidsai/kvikio/archive/${VELOX_kvikio_COMMIT}.tar.gz"
)
velox_resolve_dependency_url(kvikio)

# cudf commit 456580f from 2026-09-11 (release/26.10 branch)
set(VELOX_cudf_VERSION 26.10 CACHE STRING "cudf version")
set(VELOX_cudf_COMMIT 456580fcdd726380dcb3de6b9686e7d47d6dd0a2)
set(
  VELOX_cudf_BUILD_SHA256_CHECKSUM
  160ff8be040c434c9f51fe3e41f08d26421c41710a8adabe559b1f994ad06959
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
  # ucxx v0.51.01, released 2026-08-12
  set(VELOX_ucxx_VERSION 0.51)
  set(VELOX_ucxx_COMMIT 2c309cab417a6f12d5891a423211d7cd993ab3ef)
  set(
    VELOX_ucxx_BUILD_SHA256_CHECKSUM
    76c4a59dd3f47d3062040279a90ce42fc82e955094453a0a15baec627b579afb
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
