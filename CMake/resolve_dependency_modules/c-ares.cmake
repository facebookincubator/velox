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

set(VELOX_CARES_BUILD_VERSION cares-1_17_2)
set(VELOX_CARES_BUILD_SHA256_CHECKSUM
    444b7dbf33cb6e8a764d069e3485cbaaa28386d60eace6988275c3cb9534d300)
string(CONCAT VELOX_CARES_SOURCE_URL
              "https://github.com/c-ares/c-ares/archive/refs/tags/"
              "${VELOX_CARES_BUILD_VERSION}.tar.gz")

resolve_dependency_url(CARES)

message(STATUS "Building C-ARES from source")

FetchContent_Declare(
  c-ares
  URL ${VELOX_CARES_SOURCE_URL}
  URL_HASH ${VELOX_CARES_BUILD_SHA256_CHECKSUM}
  OVERRIDE_FIND_PACKAGE EXCLUDE_FROM_ALL SYSTEM)

set(CARES_STATIC ON)
set(CARES_INSTALL ON)
set(CARES_SHARED OFF)
FetchContent_MakeAvailable(c-ares)
if(NOT TARGET c-ares::cares)
  add_library(c-ares::cares ALIAS c-ares)
endif()
