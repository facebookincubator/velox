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
project(Folly)
cmake_minimum_required(VERSION 3.28)

velox_set_source(FastFloat)
velox_resolve_dependency(FastFloat CONFIG REQUIRED)

set(VELOX_FOLLY_BUILD_VERSION v2026.09.07.00)
set(
  VELOX_FOLLY_BUILD_SHA256_CHECKSUM
  f114713009738faa49961aeaa050592a09a4c8a7b46ba9a0780d4aaba468173b
)
set(
  VELOX_FOLLY_SOURCE_URL
  "https://github.com/facebook/folly/releases/download/${VELOX_FOLLY_BUILD_VERSION}/folly-${VELOX_FOLLY_BUILD_VERSION}.tar.gz"
)

velox_resolve_dependency_url(FOLLY)

message(STATUS "Building Folly from source")

set(folly_patches ${CMAKE_CURRENT_LIST_DIR}/folly-no-export.patch)
if(WIN32)
  list(APPEND folly_patches ${CMAKE_CURRENT_LIST_DIR}/folly-windows.patch)
  if(NOT gflags_SOURCE STREQUAL "BUNDLED")
    list(PREPEND folly_patches --exclude=CMake/folly-deps.cmake)
  endif()
endif()

if(gflags_SOURCE STREQUAL "BUNDLED")
  set(glog_patch && git apply ${CMAKE_CURRENT_LIST_DIR}/folly-gflags-glog.patch)
  if(WIN32)
    set(glog_patch "")
  endif()
  # Together with the patch applied above prevents folly from test compiling a
  # snippet to find the right namespace (which would fail because gflags isn't
  # built yet)
  set(FOLLY_UNUSUAL_GFLAGS_NAMESPACE OFF)
  set(FOLLY_GFLAGS_NAMESPACE gflags)
endif()

FetchContent_Declare(
  folly
  URL ${VELOX_FOLLY_SOURCE_URL}
  URL_HASH ${VELOX_FOLLY_BUILD_SHA256_CHECKSUM}
  PATCH_COMMAND git apply ${folly_patches} ${glog_patch}
  OVERRIDE_FIND_PACKAGE
  SYSTEM
  EXCLUDE_FROM_ALL
)

set(BUILD_SHARED_LIBS ${VELOX_BUILD_SHARED})

# Let Folly probe for native 128-bit integers on Windows.
if(NOT WIN32)
  set(FOLLY_HAVE_INT128_T ON)
endif()

FetchContent_MakeAvailable(folly)
if(WIN32)
  target_link_libraries(folly_deps INTERFACE Boost::headers)
endif()

if(gflags_SOURCE STREQUAL "BUNDLED")
  add_dependencies(folly glog::glog gflags::gflags fmt::fmt)
endif()
