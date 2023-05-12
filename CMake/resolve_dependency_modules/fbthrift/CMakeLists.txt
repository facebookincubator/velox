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

set(VELOX_FBTHRIFT_VERSION "2023.05.01.00")
set(VELOX_FBTHRIFT_BUILD_SHA256_CHECKSUM 
    7d6b9272c637a44767cb30e75d6f703fdfd6966bababa61fca773e68d299bfe0)
set(VELOX_FBTHRIFT_SOURCE_URL 
    "https://github.com/facebook/fbthrift/archive/refs/tags/v${VELOX_FBTHRIFT_VERSION}.tar.gz") 

resolve_dependency_url(FBTHRIFT)
message(STATUS "SEEE: ${Boost_INCLUDE_DIR}")
message(STATUS "SEEE: ${folly_DIR}")
message(STATUS "SEEE: ${folly_BINARY_DIR}")

message(STATUS "Building FBThrift from source")

FetchContent_Declare(
  fbthrift
  GIT_REPOSITORY https://github.com/facebook/fbthrift
  GIT_TAG v2022.11.14.00
  #PATCH_COMMAND git apply ${CMAKE_CURRENT_LIST_DIR}/fbthrift-no-export.patch
)

set_source(fizz)
resolve_dependency(fizz)

set_source(wangle)
resolve_dependency(wangle)

set(enable_tests OFF)
set(BUILD_SHARED_LIBS ON)

add_compile_options(-w)
FetchContent_MakeAvailable(fbthrift)
