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

set(VELOX_ANTLR4_RUNTIME_VERSION 4.13.2)
set(VELOX_ANTLR4_RUNTIME_BUILD_SHA256_CHECKSUM
    9f18272a9b32b622835a3365f850dd1063d60f5045fb1e12ce475ae6e18a35bb)
set(VELOX_ANTLR4_RUNTIME_SOURCE_URL
    "https://github.com/antlr/antlr4/archive/refs/tags/${VELOX_ANTLR4_RUNTIME_VERSION}.tar.gz"
)

velox_resolve_dependency_url(ANTLR4_RUNTIME)

message(STATUS "Building antlr4-runtime from source")

FetchContent_Declare(
  antlr4-runtime
  URL ${VELOX_ANTLR4_RUNTIME_SOURCE_URL}
  URL_HASH ${VELOX_ANTLR4_RUNTIME_BUILD_SHA256_CHECKSUM}
  SOURCE_SUBDIR runtime/Cpp OVERRIDE_FIND_PACKAGE)

set(ANTLR4_INSTALL
    ON
    CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(antlr4-runtime)
