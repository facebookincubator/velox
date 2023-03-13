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

if(DEFINED ENV{VELOX_SIMDJSON_URL})
  set(SIMDJSON_SOURCE_URL "$ENV{VELOX_SIMDJSON_URL}")
else()
  set(VELOX_SIMDJSON_BUILD_VERSION 3.1.5)
  string(CONCAT SIMDJSON_SOURCE_URL
                "https://github.com/simdjson/simdjson/archive/refs/tags/"
                "v${VELOX_SIMDJSON_BUILD_VERSION}.tar.gz")
  set(VELOX_SIMDJSON_BUILD_SHA256_CHECKSUM
      5b916be17343324426fc467a4041a30151e481700d60790acfd89716ecc37076)
endif()

message(STATUS "Building simdjson from source")

FetchContent_Declare(
  simdjson
  URL ${SIMDJSON_SOURCE_URL}
  URL_HASH SHA256=${VELOX_SIMDJSON_BUILD_SHA256_CHECKSUM})

FetchContent_MakeAvailable(simdjson)
include_directories(${simdjson_SOURCE_DIR}/include)
