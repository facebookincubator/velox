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

# rapids_cmake commit 5f60b4d from 2026-05-20
set(VELOX_rapids_cmake_VERSION 26.08)
set(VELOX_rapids_cmake_COMMIT 5f60b4de46c8256e78ea742c7f759e6ded45f503)
set(
  VELOX_rapids_cmake_BUILD_SHA256_CHECKSUM
  2fb74390e20dc411a907cbf7c25107e4ca4071e1e88ad3e598c2fe46c718b56a
)
set(
  VELOX_rapids_cmake_SOURCE_URL
  "https://github.com/rapidsai/rapids-cmake/archive/${VELOX_rapids_cmake_COMMIT}.tar.gz"
)
velox_resolve_dependency_url(rapids_cmake)

# rmm commit 25e06c4 from 2026-05-20
set(VELOX_rmm_VERSION 26.08)
set(VELOX_rmm_COMMIT 25e06c4713893b0240666d008288cd7ef8760350)
set(
  VELOX_rmm_BUILD_SHA256_CHECKSUM
  33584dadc3c5aab2ac10c1771d850889dbd7538da678e4b6cd1fdfe1973ddecb
)
set(VELOX_rmm_SOURCE_URL "https://github.com/rapidsai/rmm/archive/${VELOX_rmm_COMMIT}.tar.gz")
velox_resolve_dependency_url(rmm)

# kvikio commit 81eb154 from 2026-05-21
set(VELOX_kvikio_VERSION 26.08)
set(VELOX_kvikio_COMMIT 81eb15445ade6a4df3c1895789c5b191dfe11c36)
set(
  VELOX_kvikio_BUILD_SHA256_CHECKSUM
  853a05d99ade4fa1655424877d15f50cb52f649ba53c74d7e4003d08eee3ba56
)
set(
  VELOX_kvikio_SOURCE_URL
  "https://github.com/rapidsai/kvikio/archive/${VELOX_kvikio_COMMIT}.tar.gz"
)
velox_resolve_dependency_url(kvikio)

# cudf commit 0d8d49b from 2026-05-21
set(VELOX_cudf_VERSION 26.08 CACHE STRING "cudf version")
set(VELOX_cudf_COMMIT 0d8d49b7c75812b293232abc6e07ffb240b106f5)
set(
  VELOX_cudf_BUILD_SHA256_CHECKSUM
  874f1d6692de26f271b0b7e26b3ff562b8239b1872a0ab7744f1d3859f6e16f5
)
set(VELOX_cudf_SOURCE_URL "https://github.com/rapidsai/cudf/archive/${VELOX_cudf_COMMIT}.tar.gz")
velox_resolve_dependency_url(cudf)

# Use block so we don't leak variables
block(SCOPE_FOR VARIABLES)
  # Setup libcudf build to not have testing components
  set(BUILD_TESTS OFF)
  set(CUDF_BUILD_TESTUTIL OFF)
  set(CUDF_BUILD_STREAMS_TEST_UTIL OFF)
  set(BUILD_SHARED_LIBS ON)

  # rapids_logger bundles spdlog 1.14.1 which is incompatible with system fmt
  # >= 11. Force rapids_logger to bundle its own fmt instead of linking
  # against the external one.
  set(RAPIDS_LOGGER_HIDE_ALL_SPDLOG_SYMBOLS OFF)
  set(RAPIDS_LOGGER_FMT_OPTION BUNDLED)

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
