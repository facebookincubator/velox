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

set(VELOX_DUCKDB_VERSION 1.1.1)
set(VELOX_DUCKDB_BUILD_SHA256_CHECKSUM
    a764cef80287ccfd8555884d8facbe962154e7c747043c0842cd07873b4d6752)
set(VELOX_DUCKDB_SOURCE_URL
    "https://github.com/duckdb/duckdb/archive/refs/tags/v${VELOX_DUCKDB_VERSION}.tar.gz"
)

velox_resolve_dependency_url(DUCKDB)

message(STATUS "Building DuckDB from source")
# We need remove-ccache.patch to remove adding ccache to the build command
# twice. Velox already does this. We need fix-duckdbversion.patch as DuckDB
# tries to infer the version via a git commit hash or git tag. This inference
# can lead to errors when building in another git project such as Prestissimo.
FetchContent_Declare(
  duckdb
  URL ${VELOX_DUCKDB_SOURCE_URL}
  URL_HASH ${VELOX_DUCKDB_BUILD_SHA256_CHECKSUM}
  PATCH_COMMAND
    git apply ${CMAKE_CURRENT_LIST_DIR}/duckdb/remove-ccache.patch && git apply
    ${CMAKE_CURRENT_LIST_DIR}/duckdb/fix-duckdbversion.patch && git apply
    ${CMAKE_CURRENT_LIST_DIR}/duckdb/re2.patch)

set(BUILD_UNITTESTS OFF)
set(ENABLE_SANITIZER OFF)
set(ENABLE_UBSAN OFF)
set(BUILD_SHELL OFF)
set(EXPORT_DLL_SYMBOLS OFF)
set(PREVIOUS_BUILD_TYPE ${CMAKE_BUILD_TYPE})
set(CMAKE_BUILD_TYPE Release)
set(PREVIOUS_CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-non-virtual-dtor")

FetchContent_MakeAvailable(duckdb)

if("${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU")
  target_compile_options(duckdb_catalog PRIVATE -Wno-nonnull-compare)
endif()

set(CMAKE_CXX_FLAGS ${PREVIOUS_CMAKE_CXX_FLAGS})
set(CMAKE_BUILD_TYPE ${PREVIOUS_BUILD_TYPE})
