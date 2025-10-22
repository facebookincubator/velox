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

set(VELOX_SIMDJSON_VERSION 3.13.0)
set(
  VELOX_SIMDJSON_BUILD_SHA256_CHECKSUM
  07a1bb3587aac18fd6a10a83fe4ab09f1100ab39f0cb73baea1317826b9f9e0d
)
set(
  VELOX_SIMDJSON_SOURCE_URL
  "https://github.com/simdjson/simdjson/archive/refs/tags/v${VELOX_SIMDJSON_VERSION}.tar.gz"
)

velox_resolve_dependency_url(SIMDJSON)

message(STATUS "Building simdjson from source")

FetchContent_Declare(
  simdjson
  URL ${VELOX_SIMDJSON_SOURCE_URL}
  URL_HASH ${VELOX_SIMDJSON_BUILD_SHA256_CHECKSUM}
)

if(${VELOX_SIMDJSON_SKIPUTF8VALIDATION})
  set(SIMDJSON_SKIPUTF8VALIDATION ON)
endif()

FetchContent_MakeAvailable(simdjson)
