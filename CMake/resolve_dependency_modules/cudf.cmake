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

# rapids_cmake commit d79e071 from 2026-05-01
set(VELOX_rapids_cmake_VERSION 26.06)
set(VELOX_rapids_cmake_COMMIT d79e071f805e709771b80008d50a8b3a5bed93ca)
set(
  VELOX_rapids_cmake_BUILD_SHA256_CHECKSUM
  d0f9eea4feaef1cc325e86eac787052ec951659fdcf21abdfb06efc337a63179
)
set(
  VELOX_rapids_cmake_SOURCE_URL
  "https://github.com/rapidsai/rapids-cmake/archive/${VELOX_rapids_cmake_COMMIT}.tar.gz"
)
velox_resolve_dependency_url(rapids_cmake)

# rmm commit 2357ddd from 2026-05-04
set(VELOX_rmm_VERSION 26.06)
set(VELOX_rmm_COMMIT 2357ddddcff042ba378e8de6f89e4a995a23b2db)
set(
  VELOX_rmm_BUILD_SHA256_CHECKSUM
  61492c2da88e7f6a6a4edc7101cce4698c156704d135db0b80119f4b9a2c575c
)
set(VELOX_rmm_SOURCE_URL "https://github.com/rapidsai/rmm/archive/${VELOX_rmm_COMMIT}.tar.gz")
velox_resolve_dependency_url(rmm)

# kvikio commit 247b64e from 2026-05-02
set(VELOX_kvikio_VERSION 26.06)
set(VELOX_kvikio_COMMIT 247b64e97ecb7cb9ccb06ab123aea87ac571c5b4)
set(
  VELOX_kvikio_BUILD_SHA256_CHECKSUM
  6419490de95e412cefdbafffc73fac6b2162bc2200076611883fe41637028198
)
set(
  VELOX_kvikio_SOURCE_URL
  "https://github.com/rapidsai/kvikio/archive/${VELOX_kvikio_COMMIT}.tar.gz"
)
velox_resolve_dependency_url(kvikio)

# cudf commit d09d10d from 2026-05-04
set(VELOX_cudf_VERSION 26.06 CACHE STRING "cudf version")
set(VELOX_cudf_COMMIT d09d10d14d3ed932b8de93638809101af5c7fec3)
set(
  VELOX_cudf_BUILD_SHA256_CHECKSUM
  5042ec46beb8260eb60d13b9cd44f26357b9756628f7d58659c77e88c67e15d5
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
