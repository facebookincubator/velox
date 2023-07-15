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

set(VELOX_SIMDJSON_VERSION 3.1.5)
set(VELOX_SIMDJSON_BUILD_SHA256_CHECKSUM
    5b916be17343324426fc467a4041a30151e481700d60790acfd89716ecc37076)
set(VELOX_SIMDJSON_SOURCE_URL
    "https://github.com/simdjson/simdjson/archive/refs/tags/v${VELOX_SIMDJSON_VERSION}.tar.gz"
)

resolve_dependency_url(SIMDJSON)

message(STATUS "Building simdjson from source")

FetchContent_Declare(
  simdjson
  URL ${VELOX_SIMDJSON_SOURCE_URL}
  URL_HASH ${VELOX_SIMDJSON_BUILD_SHA256_CHECKSUM}
  SOURCE_DIR ${CMAKE_BINARY_DIR}/_deps/simdjson-src/simdjson)

# When simdjson is built with cmake it places simdjson/singleheader/simdjson.h
# at the root so files simply depend on simdjson.h.  This breaks building with
# systems other than cmake. This hack makes it so that simdjson.h can
# consistently be found at simdjson/singleheader/simdjson.h.
FetchContent_Populate(simdjson)

add_library(simdjson INTERFACE)
target_sources(simdjson
               INTERFACE ${simdjson_SOURCE_DIR}/singleheader/simdjson.cpp)
target_include_directories(simdjson INTERFACE ${simdjson_SOURCE_DIR}/..)
