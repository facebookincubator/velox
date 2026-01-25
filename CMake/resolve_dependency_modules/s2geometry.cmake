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

# This creates a separate scope so any changed variables don't affect
# the rest of the build.
block()
  set(VELOX_S2GEOMETRY_BUILD_VERSION 0.11.1)
  set(
    VELOX_S2GEOMETRY_BUILD_SHA256_CHECKSUM
    bdbeb8ebdb88fa934257caf81bb44b55711617a3ab4fdec2c3cfd6cc31b61734
  )
  string(
    CONCAT
    VELOX_S2GEOMETRY_SOURCE_URL
    "https://github.com/google/s2geometry/archive/refs/tags/"
    "v${VELOX_S2GEOMETRY_BUILD_VERSION}.tar.gz"
  )

  velox_resolve_dependency_url(S2GEOMETRY)

  FetchContent_Declare(
    s2geometry
    URL ${VELOX_S2GEOMETRY_SOURCE_URL}
    URL_HASH ${VELOX_S2GEOMETRY_BUILD_SHA256_CHECKSUM}
    OVERRIDE_FIND_PACKAGE
    SYSTEM
    EXCLUDE_FROM_ALL
  )

  list(APPEND CMAKE_MODULE_PATH "${s2geometry_SOURCE_DIR}/cmake")
  set(BUILD_SHARED_LIBS OFF)
  set(BUILD_TESTING OFF)
  set(CMAKE_BUILD_TYPE Release)

  FetchContent_MakeAvailable(s2geometry)

  add_library(s2geometry::s2geometry ALIAS s2)
endblock()
