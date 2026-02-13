# Copyright (c) Meta Platforms, Inc. and affiliates.
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

set(VELOX_CURL_VERSION 8.4.0)
string(REPLACE "." "_" VELOX_CURL_VERSION_UNDERSCORES ${VELOX_CURL_VERSION})
set(
  VELOX_CURL_BUILD_SHA256_CHECKSUM
  16c62a9c4af0f703d28bda6d7bbf37ba47055ad3414d70dec63e2e6336f2a82d
)
string(
  CONCAT
  VELOX_CURL_SOURCE_URL
  "https://github.com/curl/curl/releases/download/"
  "curl-${VELOX_CURL_VERSION_UNDERSCORES}/curl-${VELOX_CURL_VERSION}.tar.xz"
)

set(BUILD_TESTING OFF)
set(BUILD_SHARED_LIBS ON)

velox_resolve_dependency_url(CURL)

message(STATUS "Building CURL from source")

FetchContent_Declare(curl URL ${VELOX_CURL_SOURCE_URL} URL_HASH ${VELOX_CURL_BUILD_SHA256_CHECKSUM})

FetchContent_MakeAvailable(curl)

# Curl uses CMake option for BUILD_TESTING and BUILD_SHARED_LIBS
# See CMake option semantics.
unset(BUILD_TESTING CACHE)
unset(BUILD_SHARED_LIBS CACHE)
