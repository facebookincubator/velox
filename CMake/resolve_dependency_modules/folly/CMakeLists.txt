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

set(VELOX_FOLLY_BUILD_VERSION v2026.01.05.00)
set(
  VELOX_FOLLY_BUILD_SHA256_CHECKSUM
  8b41494b664fcde3d02f652d6a27381ec5e938ef58cf986d8ce4958d424af101
)
set(
  VELOX_FOLLY_SOURCE_URL
  "https://github.com/facebook/folly/releases/download/${VELOX_FOLLY_BUILD_VERSION}/folly-${VELOX_FOLLY_BUILD_VERSION}.tar.gz"
)

velox_resolve_dependency_url(FOLLY)

message(STATUS "Building Folly from source")

if(gflags_SOURCE STREQUAL "BUNDLED")
  set(glog_patch && git apply ${CMAKE_CURRENT_LIST_DIR}/folly-gflags-glog.patch)
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
  PATCH_COMMAND git apply ${CMAKE_CURRENT_LIST_DIR}/folly-no-export.patch ${glog_patch}
  OVERRIDE_FIND_PACKAGE
  SYSTEM
  EXCLUDE_FROM_ALL
)

set(BUILD_SHARED_LIBS ${VELOX_BUILD_SHARED})

# Enable INT128 support
set(FOLLY_HAVE_INT128_T ON)

FetchContent_MakeAvailable(folly)

# folly_base is OBJECT; with CMake 4.x, PRIVATE links to IMPORTED targets may not
# put INTERFACE_INCLUDE_DIRECTORIES on compile lines. Mirror includes explicitly.
function(_velox_folly_base_use_dependency dep)
  if(NOT TARGET ${dep})
    return()
  endif()
  target_link_libraries(folly_base PRIVATE ${dep})
  get_target_property(_inc ${dep} INTERFACE_INCLUDE_DIRECTORIES)
  if(_inc AND NOT _inc STREQUAL "_inc-NOTFOUND")
    target_include_directories(folly_base SYSTEM PRIVATE ${_inc})
  endif()
endfunction()

if(TARGET folly_base)
  if(TARGET fmt::fmt)
    _velox_folly_base_use_dependency(fmt::fmt)
  elseif(TARGET fmt::fmt-header-only)
    _velox_folly_base_use_dependency(fmt::fmt-header-only)
  endif()
  _velox_folly_base_use_dependency(glog::glog)
  _velox_folly_base_use_dependency(double-conversion::double-conversion)
  # CMake 4.x + OBJECT libs: IMPORTED INTERFACE includes still missing on some
  # generators; Conda prefix has fmt, glog, double-conversion in one tree.
  if(DEFINED ENV{CONDA_PREFIX} AND EXISTS "$ENV{CONDA_PREFIX}/include")
    target_include_directories(folly_base SYSTEM PRIVATE "$ENV{CONDA_PREFIX}/include")
  endif()
endif()

# Folly::folly is not valid for FC but we want to match FindFolly
add_library(Folly::folly ALIAS folly)
add_library(Folly::follybenchmark ALIAS follybenchmark)

if(gflags_SOURCE STREQUAL "BUNDLED")
  add_dependencies(folly glog::glog gflags::gflags fmt::fmt)
endif()
