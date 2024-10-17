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
set_target_properties(
  hs
  PROPERTIES INTERFACE_INCLUDE_DIRECTORIES ${hyperscan_SOURCE_DIR}/src)
set(CMAKE_CXX_STANDARD ${CMAKE_CXX_STANDARD_BACKUP})
