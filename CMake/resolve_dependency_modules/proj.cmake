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

set(VELOX_PROJ_BUILD_VERSION 9.3.0)
set(
  VELOX_PROJ_BUILD_SHA256_CHECKSUM
  f48c334eaf56c38d681bcfa37f188f422a562f45a66e6e646a79b4249641ecdb
)
string(
  CONCAT
  VELOX_PROJ_SOURCE_URL
  "https://github.com/OSGeo/PROJ/archive/refs/tags/"
  "${VELOX_PROJ_BUILD_VERSION}.tar.gz"
)

velox_resolve_dependency_url(PROJ)

message(STATUS "Building PROJ from source")

FetchContent_Declare(
  proj
  URL ${VELOX_PROJ_SOURCE_URL}
  URL_HASH ${VELOX_PROJ_BUILD_SHA256_CHECKSUM}
  OVERRIDE_FIND_PACKAGE
  SYSTEM
  EXCLUDE_FROM_ALL
)

# PROJ build configuration options
set(BUILD_SHARED_LIBS OFF)
set(BUILD_TESTING OFF)
set(ENABLE_TIFF OFF)
set(ENABLE_CURL OFF)
set(BUILD_APPS OFF)
set(BUILD_PROJ ON)
set(BUILD_PROJINFO ON)
set(BUILD_PROJSYNC ON)

FetchContent_MakeAvailable(proj)

# Create alias target for consistency
if(NOT TARGET PROJ::proj)
  add_library(PROJ::proj ALIAS proj)
endif()
