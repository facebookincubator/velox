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

set(VELOX_WANGLE_VERSION "2023.05.08.00")
set(VELOX_WANGLE_BUILD_SHA256_CHECKSUM 
    e43d9035cf5e403287c2c7ba2b4b29f52afeae7bc50abb8dbb2b62cdfd9638da)
set(VELOX_WANGLE_SOURCE_URL 
    "https://github.com/facebook/wangle/archive/refs/tags/v${VELOX_WANGLE_VERSION}.tar.gz")

resolve_dependency_url(WANGLE)

message(STATUS "Building Wangle from source")
FetchContent_Declare(
  wangle
  URL ${VELOX_WANGLE_SOURCE_URL}
  URL_HASH ${VELOX_WANGLE_BUILD_SHA256_CHECKSUM}
  OVERRIDE_FIND_PACKAGE)

FetchContent_MakeAvailable(wangle)
list(APPEND CMAKE_PREFIX_PATH ${wangle_BINARY_DIR})

add_library(wangle INTERFACE)
add_library(wangle::wangle ALIAS wangle)
