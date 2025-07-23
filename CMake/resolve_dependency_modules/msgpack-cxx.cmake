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

set(VELOX_MSGPACK_BUILD_VERSION cpp-7.0.0)
set(VELOX_MSGPACK_BUILD_SHA256_CHECKSUM
    070881ebea9208cf7e731fd5a46a11404025b2f260ab9527e32dfcb7c689fbfc)
set(VELOX_MSGPACK_SOURCE_URL
    "https://github.com/msgpack/msgpack-c/archive/refs/tags/${VELOX_MSGPACK_BUILD_VERSION}.tar.gz"
)

velox_resolve_dependency_url(MSGPACK)

message(STATUS "Building msgpack-cxx from source")

FetchContent_Declare(
  msgpack-cxx
  URL ${VELOX_MSGPACK_SOURCE_URL}
  URL_HASH ${VELOX_MSGPACK_BUILD_SHA256_CHECKSUM}
  OVERRIDE_FIND_PACKAGE EXCLUDE_FROM_ALL SYSTEM)

set(MSGPACK_USE_BOOST OFF)

FetchContent_MakeAvailable(msgpack-cxx)
