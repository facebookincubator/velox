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

find_program(RAGEL ragel)
if(${RAGEL} STREQUAL "RAGEL-NOTFOUND")
  message(STATUS "Building colm from source")
  # Install colm, required by ragel build.
  FetchContent_Declare(
    colm
    URL https://github.com/adrian-thurston/colm/archive/refs/tags/0.14.7.tar.gz
    URL_HASH
      SHA256=06c8296cab3c660dcb0b150d5b58c10707278d34a35fe664f8ed05f4606fc079)
  FetchContent_GetProperties(colm)
  if(NOT colm_POPULATED)
    FetchContent_Populate(colm)
  endif()
  execute_process(
    COMMAND bash -c "./autogen.sh && ./configure && make && make install"
    WORKING_DIRECTORY ${colm_SOURCE_DIR}
    RESULT_VARIABLE result
    OUTPUT_VARIABLE output
    ERROR_VARIABLE output)

  message(STATUS "Building ragel from source")
  # Install ragel, required by hyperscan build.
  FetchContent_Declare(
    ragel
    URL https://github.com/adrian-thurston/ragel/archive/refs/tags/7.0.4.tar.gz
    URL_HASH
      SHA256=0f7c3866f82ba2552f1ae1f03b94170121a0ff8bac92c8e22c531d732fd20581)
  FetchContent_GetProperties(ragel)
  if(NOT ragel_POPULATED)
    FetchContent_Populate(ragel)
  endif()
  execute_process(
    COMMAND
      bash -c
      "./autogen.sh && ./configure --with-colm=/usr/local/ --disable-manual && make && make install"
    WORKING_DIRECTORY ${ragel_SOURCE_DIR}
    RESULT_VARIABLE result
    OUTPUT_VARIABLE output
    ERROR_VARIABLE output)
  if(result)
    message(FATAL_ERROR "Failed to build and install ragel: ${output}")
  endif()
else()
  message(STATUS "Using existing ragel for building hyperscan.")
endif()

if(DEFINED ENV{VELOX_HYPERSCAN_URL})
  set(VELOX_HYPERSCAN_SOURCE_URL "$ENV{VELOX_HYPERSCAN_URL}")
else()
  set(VELOX_HYPERSCAN_VERSION v5.4.2)
  set(VELOX_HYPERSCAN_SOURCE_URL
      "https://github.com/intel/hyperscan/archive/refs/tags/${VELOX_HYPERSCAN_VERSION}.tar.gz"
  )
  set(VELOX_HYPERSCAN_BUILD_SHA256_CHECKSUM
      32b0f24b3113bbc46b6bfaa05cf7cf45840b6b59333d078cc1f624e4c40b2b99)
endif()

message(STATUS "Building hyperscan from source")
FetchContent_Declare(
  hyperscan
  URL ${VELOX_HYPERSCAN_SOURCE_URL}
  URL_HASH SHA256=${VELOX_HYPERSCAN_BUILD_SHA256_CHECKSUM})

set(CMAKE_CXX_STANDARD_BACKUP ${CMAKE_CXX_STANDARD})
# C++ 17 is not supported.
set(CMAKE_CXX_STANDARD 11)
set(BUILD_EXAMPLES FALSE)
set(BUILD_AVX512 ON)
FetchContent_MakeAvailable(hyperscan)
set(CMAKE_CXX_STANDARD ${CMAKE_CXX_STANDARD_BACKUP})
