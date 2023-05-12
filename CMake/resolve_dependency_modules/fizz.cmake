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

set(VELOX_FIZZ_VERSION "2023.05.08.00")
set(VELOX_FIZZ_BUILD_SHA256_CHECKSUM 
    ad1bad61b1079aa1f5b73682672145c8bec558fa91b3511253fc9673b84b3e4b)
set(VELOX_FIZZ_SOURCE_URL 
    "https://github.com/facebookincubator/fizz/archive/refs/tags/v${VELOX_FIZZ_VERSION}.tar.gz")

resolve_dependency_url(FIZZ)

message(STATUS "Building Fizz from source ${VELOX_FIZZ_SOURCE_URL} - ${VELOX_FIZZ_BUILD_SHA256_CHECKSUM}")
FetchContent_Declare(
  fizz
  URL ${VELOX_FIZZ_SOURCE_URL}
  URL_HASH ${VELOX_FIZZ_BUILD_SHA256_CHECKSUM}
  OVERRIDE_FIND_PACKAGE)

FetchContent_MakeAvailable(fizz)
list(APPEND CMAKE_PREFIX_PATH ${fizz_BINARY_DIR})
