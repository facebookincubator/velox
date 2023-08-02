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

set(VELOX_QATZSTD_BUILD_VERSION 0.0.1)
set(VELOX_QATZSTD_BUILD_SHA256_CHECKSUM
    c15aa561d5139d49f896315cbd38d02a51b636e448abfa58e4d3a011fe0424f2)
string(CONCAT VELOX_QATZSTD_SOURCE_URL
              "https://github.com/intel/QAT-ZSTD-Plugin/archive/refs/tags/"
              "v${VELOX_QATZSTD_BUILD_VERSION}.tar.gz")

resolve_dependency_url(QATZSTD)

message(STATUS "Building QATZSTD from source")

FetchContent_Declare(
  qatzstd
  URL ${VELOX_QATZSTD_SOURCE_URL}
  URL_HASH ${VELOX_QATZSTD_BUILD_SHA256_CHECKSUM})

FetchContent_MakeAvailable(qatzstd)
