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

set(VELOX_ABSL_BUILD_VERSION 20250512.1)
set(
  VELOX_ABSL_BUILD_SHA256_CHECKSUM
  9b7a064305e9fd94d124ffa6cc358592eb42b5da588fb4e07d09254aa40086db
)
string(
  CONCAT
  VELOX_ABSL_SOURCE_URL
  "https://github.com/abseil/abseil-cpp/archive/refs/tags/"
  "${VELOX_ABSL_BUILD_VERSION}.tar.gz"
)

velox_resolve_dependency_url(ABSL)

message(STATUS "Building Abseil from source")

FetchContent_Declare(
  absl
  URL ${VELOX_ABSL_SOURCE_URL}
  URL_HASH ${VELOX_ABSL_BUILD_SHA256_CHECKSUM}
  OVERRIDE_FIND_PACKAGE
  EXCLUDE_FROM_ALL
  SYSTEM
)

set(ABSL_BUILD_TESTING OFF)
set(ABSL_PROPAGATE_CXX_STD ON)
set(ABSL_ENABLE_INSTALL ON)
FetchContent_MakeAvailable(absl)
