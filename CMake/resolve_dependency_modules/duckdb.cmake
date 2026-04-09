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

set(VELOX_DUCKDB_VERSION 0.8.1)
set(
  VELOX_DUCKDB_BUILD_SHA256_CHECKSUM
  a0674f7e320dc7ebcf51990d7fc1c0e7f7b2c335c08f5953702b5285e6c30694
)
set(
  VELOX_DUCKDB_SOURCE_URL
  "https://github.com/duckdb/duckdb/archive/refs/tags/v${VELOX_DUCKDB_VERSION}.tar.gz"
)
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)

velox_resolve_dependency_url(DUCKDB)

message(STATUS "Building DuckDB from source")
# DuckDB FSST libfsst.hpp typedefs u8/u32 from stdint types without <cstdint>.
# GCC 14+ may not pull those typedefs in via other headers. We patch after extract.
# Use a CMake script (not git apply): FetchContent's patch step cwd is unreliable,
# and unified diffs break on whitespace/CRLF differences in the license block.
if(DEFINED FETCHCONTENT_BASE_DIR)
  cmake_path(SET _velox_duckdb_src NORMALIZE
             "${FETCHCONTENT_BASE_DIR}/duckdb-src")
else()
  cmake_path(SET _velox_duckdb_src NORMALIZE
             "${CMAKE_BINARY_DIR}/_deps/duckdb-src")
endif()
FetchContent_Declare(
  duckdb
  URL ${VELOX_DUCKDB_SOURCE_URL}
  URL_HASH ${VELOX_DUCKDB_BUILD_SHA256_CHECKSUM}
  PATCH_COMMAND
    ${CMAKE_COMMAND}
    -DVELOX_DUCKDB_SRC=${_velox_duckdb_src}
    -P ${CMAKE_CURRENT_LIST_DIR}/duckdb/patch_fsst_cstdint.cmake
)

# DuckDB uses git commands to retrieve version information during the build,
# which works with git clone. To prevent incorrectly using the parent project's
# git version when building from a tarball, we define GIT_COMMIT_HASH to skip
# that.
set(GIT_COMMIT_HASH "6536a77")
set(BUILD_UNITTESTS OFF)
set(BUILD_TESTING OFF)
set(ENABLE_SANITIZER OFF)
set(ENABLE_UBSAN OFF)
set(BUILD_SHELL OFF)
set(EXPORT_DLL_SYMBOLS OFF)
set(PREVIOUS_BUILD_TYPE ${CMAKE_BUILD_TYPE})
set(CMAKE_BUILD_TYPE Release)
set(PREVIOUS_CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-non-virtual-dtor")
# Clang17 requires this. See issue #13215.
if("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 17.0.0)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-missing-template-arg-list-after-template-kw")
endif()

FetchContent_MakeAvailable(duckdb)

if("${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU")
  target_compile_options(duckdb_catalog PRIVATE -Wno-nonnull-compare)
endif()

set(CMAKE_CXX_FLAGS ${PREVIOUS_CMAKE_CXX_FLAGS})
set(CMAKE_BUILD_TYPE ${PREVIOUS_BUILD_TYPE})
# Some DuckDB third-party package sets this flags. We cannot control that.
unset(BUILD_TESTING)
