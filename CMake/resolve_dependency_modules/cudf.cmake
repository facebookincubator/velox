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

# rapids_cmake commit 935fccb from 2026-03-12
set(VELOX_rapids_cmake_VERSION 26.04)
set(VELOX_rapids_cmake_COMMIT 935fccbe19a292e22a19a4632f121c18f0ef21e8)
set(
  VELOX_rapids_cmake_BUILD_SHA256_CHECKSUM
  aea8c63e5edb660c262c08867752f6eb80d8c833ac2cc1cbd6eb3c7806bbfa7c
)
set(
  VELOX_rapids_cmake_SOURCE_URL
  "https://github.com/rapidsai/rapids-cmake/archive/${VELOX_rapids_cmake_COMMIT}.tar.gz"
)
velox_resolve_dependency_url(rapids_cmake)

# rmm commit 22d812c from 2026-03-12
set(VELOX_rmm_VERSION 26.04)
set(VELOX_rmm_COMMIT 22d812c2137f4eeaa14696981487e4744bbb0c65)
set(
  VELOX_rmm_BUILD_SHA256_CHECKSUM
  62063c497731ceccd1f16994a3b944c3933b050da7fd788a0a6ad48a74ae66c3
)
set(VELOX_rmm_SOURCE_URL "https://github.com/rapidsai/rmm/archive/${VELOX_rmm_COMMIT}.tar.gz")
velox_resolve_dependency_url(rmm)

# kvikio commit 745bab7 from 2026-03-12
set(VELOX_kvikio_VERSION 26.04)
set(VELOX_kvikio_COMMIT 745bab7ecdf1ffbf3cee71f0df394f2e38127140)
set(
  VELOX_kvikio_BUILD_SHA256_CHECKSUM
  b12463d1d06fe5fcaa802d65302fdee23035f1dbd32d58058c9e4661e4b26bdd
)
set(
  VELOX_kvikio_SOURCE_URL
  "https://github.com/rapidsai/kvikio/archive/${VELOX_kvikio_COMMIT}.tar.gz"
)
velox_resolve_dependency_url(kvikio)

# cudf commit 3d344d7 from 2026-03-12
set(VELOX_cudf_VERSION 26.04 CACHE STRING "cudf version")
set(VELOX_cudf_COMMIT 3d344d7816695d10de585b8e1171778f264053f4)
set(
  VELOX_cudf_BUILD_SHA256_CHECKSUM
  2b8e6a136d63dbb92f9b16b412ce47a9eb7418290d614aa76c969bd5cbe473f0
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
