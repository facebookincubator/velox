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

# rapids_cmake commit fa303cb from 2026-03-24
set(VELOX_rapids_cmake_VERSION 26.06)
set(VELOX_rapids_cmake_COMMIT fa303cb883f0e127fb2bb950d303626239050964)
set(
  VELOX_rapids_cmake_BUILD_SHA256_CHECKSUM
  633616ce36fa21097483e793caa0dd94b355ea3735b6cb2a83e6f0fc10866bbd
)
set(
  VELOX_rapids_cmake_SOURCE_URL
  "https://github.com/rapidsai/rapids-cmake/archive/${VELOX_rapids_cmake_COMMIT}.tar.gz"
)
velox_resolve_dependency_url(rapids_cmake)

# rmm commit ad99c11 from 2026-03-23
set(VELOX_rmm_VERSION 26.06)
set(VELOX_rmm_COMMIT ad99c114b62b9e1c8277563fe353ffb80589c84b)
set(
  VELOX_rmm_BUILD_SHA256_CHECKSUM
  baf203f4579bd778118360839bad57836aae4b07e482bec486ce5a850d92199d
)
set(VELOX_rmm_SOURCE_URL "https://github.com/rapidsai/rmm/archive/${VELOX_rmm_COMMIT}.tar.gz")
velox_resolve_dependency_url(rmm)

# kvikio commit b2bbfcc from 2026-03-24
set(VELOX_kvikio_VERSION 26.06)
set(VELOX_kvikio_COMMIT b2bbfcc3147fbadcdaf0e3f4b9737d9dd4bf76a0)
set(
  VELOX_kvikio_BUILD_SHA256_CHECKSUM
  d805843c9534a29815a66a1f047d4cac17cc6654da324a1f2a615330a8106ca1
)
set(
  VELOX_kvikio_SOURCE_URL
  "https://github.com/rapidsai/kvikio/archive/${VELOX_kvikio_COMMIT}.tar.gz"
)
velox_resolve_dependency_url(kvikio)

# cudf commit b593be9 from 2026-03-24
set(VELOX_cudf_VERSION 26.06 CACHE STRING "cudf version")
set(VELOX_cudf_COMMIT b593be9ab0bf144997efce09aaf9946f05113a39)
set(
  VELOX_cudf_BUILD_SHA256_CHECKSUM
  8f42f98a160388f45384f4ffa5f7c565c0532e6294dea1491b875cdfd28a70ec
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
