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

# rapids_cmake commit 1a871c4 from 2026-02-24
set(VELOX_rapids_cmake_VERSION 26.04)
set(VELOX_rapids_cmake_COMMIT 1a871c436dfa25d6e6990e72cec989a0fb1c5e66)
set(
  VELOX_rapids_cmake_BUILD_SHA256_CHECKSUM
  6345f9f386c10b75faa7f2c03fcde509d5e157c0619ee5ab16797ea3baaf19ac
)
set(
  VELOX_rapids_cmake_SOURCE_URL
  "https://github.com/rapidsai/rapids-cmake/archive/${VELOX_rapids_cmake_COMMIT}.tar.gz"
)
velox_resolve_dependency_url(rapids_cmake)

# rmm commit 3ff7e2a from 2026-02-24
set(VELOX_rmm_VERSION 26.04)
set(VELOX_rmm_COMMIT 3ff7e2a6c0eaee239d6a85013a87eb1f74832b70)
set(
  VELOX_rmm_BUILD_SHA256_CHECKSUM
  8ceccaf3a094dc9ceec49d517eb48c243089d10cb3d6c183bb2cfe8d65f5f184
)
set(VELOX_rmm_SOURCE_URL "https://github.com/rapidsai/rmm/archive/${VELOX_rmm_COMMIT}.tar.gz")
velox_resolve_dependency_url(rmm)

# kvikio commit d4162f9 from 2026-02-24
set(VELOX_kvikio_VERSION 26.04)
set(VELOX_kvikio_COMMIT d4162f96081f1cd4fa96048718ceb969947687e0)
set(
  VELOX_kvikio_BUILD_SHA256_CHECKSUM
  e5ebdac096942469b6369238aeba0e20a1f11210521db417f6847826def5bee0
)
set(
  VELOX_kvikio_SOURCE_URL
  "https://github.com/rapidsai/kvikio/archive/${VELOX_kvikio_COMMIT}.tar.gz"
)
velox_resolve_dependency_url(kvikio)

# cudf commit 0fcbe99 from 2026-02-24
set(VELOX_cudf_VERSION 26.04 CACHE STRING "cudf version")
set(VELOX_cudf_COMMIT 0fcbe99451966c4c731dc74d9f0dd709e29ff8ff)
set(
  VELOX_cudf_BUILD_SHA256_CHECKSUM
  f499a67dbf3ae327361d98b3b6b33802fe2bcc6012459cf166cacfa3be899228
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
