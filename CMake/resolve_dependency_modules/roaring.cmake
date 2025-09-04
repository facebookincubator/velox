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

set(VELOX_ROARING_VERSION 2.0.4)
set(
  VELOX_ROARING_BUILD_SHA256_CHECKSUM
  3c962c196ba28abf2639067f2e2fd25879744ba98152a4e0e74556ca515eda33
)
set(
  VELOX_ROARING_SOURCE_URL
  "https://github.com/RoaringBitmap/CRoaring/archive/refs/tags/v${VELOX_ROARING_VERSION}.tar.gz"
)

velox_resolve_dependency_url(ROARING)

set(ENABLE_ROARING_TESTS OFF)

FetchContent_Declare(
  roaring
  URL ${VELOX_ROARING_SOURCE_URL}
  URL_HASH ${VELOX_ROARING_BUILD_SHA256_CHECKSUM}
)

FetchContent_MakeAvailable(roaring)
