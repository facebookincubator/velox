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

set(VELOX_rapids_cmake_VERSION 25.08)
set(VELOX_rapids_cmake_BUILD_SHA256_CHECKSUM
    f0e5484d49d9365b84eb41799db4fb1c5c8532f251ace734ead168bbbc1111c7)
set(VELOX_rapids_cmake_SOURCE_URL
    "https://github.com/rapidsai/rapids-cmake/archive/492d474a62a91dd61bd6b91994eec2125e12fab8.tar.gz"
)
velox_resolve_dependency_url(rapids_cmake)

set(VELOX_rmm_VERSION 25.08)
set(VELOX_rmm_BUILD_SHA256_CHECKSUM
    9dc014c44191b0a23430f1bd7ebe38663643cdbbedd47a5a8e2079fdbbd4eb2e)
set(VELOX_rmm_SOURCE_URL
    "https://github.com/rapidsai/rmm/archive/a68fdfe26e0d1e2e37feddf53eda1b7b5044f9f4.tar.gz"
)
velox_resolve_dependency_url(rmm)

set(VELOX_kvikio_VERSION 25.08)
set(VELOX_kvikio_BUILD_SHA256_CHECKSUM
    8a5251ee1dff576578b0f96130ef87bfd8693de4769c08d33f203b6ea6750bab)
set(VELOX_kvikio_SOURCE_URL
    "https://github.com/rapidsai/kvikio/archive/54c420a652bf29e3c7de9ac2e2e19af07de7c256.tar.gz"
)
velox_resolve_dependency_url(kvikio)

set(VELOX_cudf_VERSION
    25.08
    CACHE STRING "cudf version")

set(VELOX_cudf_BUILD_SHA256_CHECKSUM
    01d8bd30e8b953b97c71adb2bfc9d83be440b3df6f822f92af95a471ebf001da)
set(VELOX_cudf_SOURCE_URL
    "https://github.com/rapidsai/cudf/archive/da6ce2a96b616891d66c4f41ade9db4047eb4b3f.tar.gz"
)
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
  UPDATE_DISCONNECTED 1)

FetchContent_Declare(
  rmm
  URL ${VELOX_rmm_SOURCE_URL}
  URL_HASH ${VELOX_rmm_BUILD_SHA256_CHECKSUM}
  UPDATE_DISCONNECTED 1)

FetchContent_Declare(
  kvikio
  URL ${VELOX_kvikio_SOURCE_URL}
  URL_HASH ${VELOX_kvikio_BUILD_SHA256_CHECKSUM}
  SOURCE_SUBDIR cpp
  UPDATE_DISCONNECTED 1)

FetchContent_Declare(
  cudf
  URL ${VELOX_cudf_SOURCE_URL}
  URL_HASH ${VELOX_cudf_BUILD_SHA256_CHECKSUM}
  SOURCE_SUBDIR cpp
  UPDATE_DISCONNECTED 1)

FetchContent_MakeAvailable(cudf)

# cudf sets all warnings as errors, and therefore fails to compile with velox
# expanded set of warnings. We selectively disable problematic warnings just for
# cudf
target_compile_options(
  cudf PRIVATE -Wno-non-virtual-dtor -Wno-missing-field-initializers
               -Wno-deprecated-copy -Wno-restrict)

unset(BUILD_SHARED_LIBS)
endblock()
