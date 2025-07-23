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

set(VELOX_NLOHMANN_JSON_BUILD_VERSION 3.11.3)
set(VELOX_NLOHMANN_JSON_BUILD_SHA256_CHECKSUM
    0d8ef5af7f9794e3263480193c491549b2ba6cc74bb018906202ada498a79406)
set(VELOX_NLOHMANN_JSON_SOURCE_URL
    "https://github.com/nlohmann/json/archive/refs/tags/v${VELOX_NLOHMANN_JSON_BUILD_VERSION}.tar.gz"
)

velox_resolve_dependency_url(NLOHMANN_JSON)

message(STATUS "Building nlohmann_json from source")

FetchContent_Declare(
  nlohmann_json
  URL ${VELOX_NLOHMANN_JSON_SOURCE_URL}
  URL_HASH ${VELOX_NLOHMANN_JSON_BUILD_SHA256_CHECKSUM}
  OVERRIDE_FIND_PACKAGE)

set(JSON_BuildTests
    OFF
    CACHE INTERNAL "")

FetchContent_MakeAvailable(nlohmann_json)
