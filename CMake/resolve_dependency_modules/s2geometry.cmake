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

list(PREPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/s2geometry)

# This creates a separate scope so any changed variables don't affect
# the rest of the build.
block()
  # s2geometry needs absl.
  if(NOT TARGET absl::base)
    velox_set_source(absl)
    velox_resolve_dependency(absl)
  endif()

  set(VELOX_S2GEOMETRY_BUILD_VERSION 0.12.0)
  set(
    VELOX_S2GEOMETRY_BUILD_SHA256_CHECKSUM
    c09ec751c3043965a0d441e046a73c456c995e6063439a72290f661c1054d611
  )
  string(
    CONCAT
    VELOX_S2GEOMETRY_SOURCE_URL
    "https://github.com/google/s2geometry/archive/refs/tags/"
    "v${VELOX_S2GEOMETRY_BUILD_VERSION}.tar.gz"
  )

  velox_resolve_dependency_url(S2GEOMETRY)

  set(
    VELOX_S2GEOMETRY_PATCHES
    ${CMAKE_CURRENT_LIST_DIR}/s2geometry/s2geometry-gcc12-max.patch
  )
  # On Apple platforms folly defines `nallocx` as a null function pointer
  # because weak symbols cannot be left undefined in Mach-O. That data symbol
  # wins over s2's weak function definition, so s2's call jumps into __DATA and
  # crashes with SIGBUS.
  if(APPLE)
    list(
      APPEND
      VELOX_S2GEOMETRY_PATCHES
      ${CMAKE_CURRENT_LIST_DIR}/s2geometry/s2geometry-apple-nallocx.patch
    )
  endif()

  FetchContent_Declare(
    s2geometry
    URL ${VELOX_S2GEOMETRY_SOURCE_URL}
    URL_HASH ${VELOX_S2GEOMETRY_BUILD_SHA256_CHECKSUM}
    OVERRIDE_FIND_PACKAGE
    SYSTEM
    EXCLUDE_FROM_ALL
    PATCH_COMMAND git apply ${VELOX_S2GEOMETRY_PATCHES}
  )

  list(APPEND CMAKE_MODULE_PATH "${s2geometry_SOURCE_DIR}/cmake")
  set(BUILD_SHARED_LIBS OFF)
  set(BUILD_TESTING OFF)
  set(BUILD_TESTS OFF)
  set(CMAKE_BUILD_TYPE Release)

  FetchContent_MakeAvailable(s2geometry)

  # Clang does not enable C++14 sized deallocation by default, unlike GCC.
  # s2geometry's port.h uses ::operator delete(ptr, size) which requires it.
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(s2 PRIVATE -fsized-deallocation)
  endif()

  add_library(s2::s2 ALIAS s2)
endblock()
