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

# rapids_cmake commit a11c837 from 2026-11-03 (main branch)
set(VELOX_rapids_cmake_VERSION 26.12)
set(VELOX_rapids_cmake_COMMIT a11c837f5025b46e2b17507112493732650e7abb)
set(
  VELOX_rapids_cmake_BUILD_SHA256_CHECKSUM
  5cff9e667342007c7a62af5c100ab74720d3c74786d6c1665699727ac28ebefa
)
set(
  VELOX_rapids_cmake_SOURCE_URL
  "https://github.com/rapidsai/rapids-cmake/archive/${VELOX_rapids_cmake_COMMIT}.tar.gz"
)
velox_resolve_dependency_url(rapids_cmake)

# rmm commit cd54ddc from 2026-11-03 (main branch)
set(VELOX_rmm_VERSION 26.12)
set(VELOX_rmm_COMMIT cd54ddc5a7905df21b02ac88497d6dfda26b4249)
set(
  VELOX_rmm_BUILD_SHA256_CHECKSUM
  6b7bdbdc096014a222c4410394e8e2fa4b19e0a328ff06d7707dbe514dcecbef
)
set(VELOX_rmm_SOURCE_URL "https://github.com/rapidsai/rmm/archive/${VELOX_rmm_COMMIT}.tar.gz")
velox_resolve_dependency_url(rmm)

# kvikio commit 6ddda9a from 2026-11-03 (main branch)
set(VELOX_kvikio_VERSION 26.12)
set(VELOX_kvikio_COMMIT 6ddda9a1ad59e105723fab9348b3d4d262a56b76)
set(
  VELOX_kvikio_BUILD_SHA256_CHECKSUM
  e4869aa236d12e5b151fbeff43e92d2c457505fa030484ce6258ff8db2e1f26e
)
set(
  VELOX_kvikio_SOURCE_URL
  "https://github.com/rapidsai/kvikio/archive/${VELOX_kvikio_COMMIT}.tar.gz"
)
velox_resolve_dependency_url(kvikio)

# cudf commit 6c7d2bd from 2026-11-05 (6c7d2bd4cad79590c4a586209bd7aa72159d9e86 branch)
set(VELOX_cudf_VERSION 26.12 CACHE STRING "cudf version")
set(VELOX_cudf_COMMIT 6c7d2bd4cad79590c4a586209bd7aa72159d9e86)
set(
  VELOX_cudf_BUILD_SHA256_CHECKSUM
  0802aad1c23d5f30497ea2b0bf45b86e519d192a9d1e4758392c1b83080ea228
)
set(VELOX_cudf_SOURCE_URL "https://github.com/rapidsai/cudf/archive/${VELOX_cudf_COMMIT}.tar.gz")
velox_resolve_dependency_url(cudf)

# Probe for a system UCX, to pick the default for VELOX_ENABLE_UCX_EXCHANGE
# below and to fail the configure before cuDF is fetched when the option is on
# without UCX. It is the lookup ucxx and velox_ucx_exchange make later, for
# UCX's CMake package, so the probe succeeds exactly when they will. GLOBAL
# creates ucx::ucp as a global target here: ucxx reuses that target and would
# otherwise try to make it global from its own directory, which CMake rejects.
find_package(ucx QUIET GLOBAL)
if(ucx_FOUND)
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
  "Build the experimental UCX GPU exchange transport. Requires a system UCX with its CMake package."
  ${UCX_FOUND}
)

if(VELOX_ENABLE_UCX_EXCHANGE)
  if(NOT UCX_FOUND)
    message(
      FATAL_ERROR
      "VELOX_ENABLE_UCX_EXCHANGE=ON but no system UCX was found (need UCX's CMake package, ucx-config.cmake; set ucx_DIR to its directory if it is not on the search path). Install UCX or pass -DVELOX_ENABLE_UCX_EXCHANGE=OFF."
    )
  endif()
  message(
    STATUS
    "UCX exchange enabled with UCX ${ucx_VERSION} from ${ucx_DIR} -- ucxx will be fetched"
  )
  # ucxx commit 2bbaa76 from 2026-11-03 (main branch)
  set(VELOX_ucxx_VERSION 0.53)
  set(VELOX_ucxx_COMMIT 2bbaa76a06e9133989991164d6dc11462906c17a)
  set(
    VELOX_ucxx_BUILD_SHA256_CHECKSUM
    3e75db06f047f16720eec264403d1a2beebfbbff24aa45acc695b26cc8b94335
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
