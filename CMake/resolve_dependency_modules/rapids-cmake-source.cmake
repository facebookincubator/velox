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

# rapids_cmake commit 037d4c5 from 2026-10-09 (main branch)
set(VELOX_rapids_cmake_COMMIT 037d4c5489cc13f5b378b54197238a6e2cdbff17)
set(
  VELOX_rapids_cmake_BUILD_SHA256_CHECKSUM
  946ebc43d81dc7359e973df54bc0749f281a0b65b32a299b00ee6a66507b0a04
)
set(
  VELOX_rapids_cmake_SOURCE_URL
  "https://github.com/rapidsai/rapids-cmake/archive/${VELOX_rapids_cmake_COMMIT}.tar.gz"
)
velox_resolve_dependency_url(rapids_cmake)

FetchContent_Declare(
  rapids-cmake
  URL ${VELOX_rapids_cmake_SOURCE_URL}
  URL_HASH ${VELOX_rapids_cmake_BUILD_SHA256_CHECKSUM}
  UPDATE_DISCONNECTED 1
)
