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

set(VELOX_GFLAGS_VERSION 2.3.0)
set(
  VELOX_GFLAGS_BUILD_SHA256_CHECKSUM
  f619a51371f41c0ad6837b2a98af9d4643b3371015d873887f7e8d3237320b2f
)
string(
  CONCAT
  VELOX_GFLAGS_SOURCE_URL
  "https://github.com/gflags/gflags/archive/refs/tags/"
  "v${VELOX_GFLAGS_VERSION}.tar.gz"
)

velox_resolve_dependency_url(GFLAGS)

message(STATUS "Building gflags from source")
FetchContent_Declare(
  gflags
  URL ${VELOX_GFLAGS_SOURCE_URL}
  URL_HASH ${VELOX_GFLAGS_BUILD_SHA256_CHECKSUM}
  PATCH_COMMAND git apply ${CMAKE_CURRENT_LIST_DIR}/gflags/gflags-config.patch
  OVERRIDE_FIND_PACKAGE
  EXCLUDE_FROM_ALL
  SYSTEM
)

# glog relies on the old `google` namespace
set(GFLAGS_NAMESPACE "google;gflags")

# Always build both the static and shared targets, unconditionally, tied to
# neither VELOX_BUILD_SHARED nor VELOX_GFLAGS_TYPE:
#
# Static: The pre-installed folly in the velox-dev:ubuntu-22.04 image was
# linked against gflags_static and its exported config (folly-targets.cmake)
# keeps `gflags_static` in INTERFACE_LINK_LIBRARIES. When VELOX_BUILD_SHARED=ON,
# cmake_dependent_option forces VELOX_BUILD_STATIC=OFF, so without this
# override BUNDLED gflags would only emit the shared variant and CMake would
# fall back to a literal `-lgflags_static` that the linker can't resolve.
#
# Shared: symmetrically, the pre-installed FBThrift's exported
# FBThriftTargets.cmake (FBThrift::concurrency, FBThrift::thriftfrozen2) keeps
# a literal `gflags_shared` in INTERFACE_LINK_LIBRARIES too. VELOX_GFLAGS_TYPE
# defaults to "shared", but that COMPONENTS request only reaches a real
# find_package(gflags) on the SYSTEM/AUTO path - this BUNDLED module doesn't
# consume it, and used to key GFLAGS_BUILD_SHARED_LIBS off VELOX_BUILD_SHARED
# (OFF by default) instead, so `gflags_shared` never got built and any target
# pulling in FBThrift::thriftcpp2 failed at link time with
# `cannot find -lgflags_shared`.
#
# gflags is tiny, so building both unconditionally is cheap and keeps both
# legacy names resolvable regardless of which one a dependency's exported
# config happens to reference.
set(GFLAGS_BUILD_STATIC_LIBS ON)
set(GFLAGS_BUILD_SHARED_LIBS ON)

set(GFLAGS_BUILD_gflags_LIB ON)
set(GFLAGS_BUILD_gflags_nothreads_LIB ON)
set(GFLAGS_IS_SUBPROJECT ON)

# Workaround for https://github.com/gflags/gflags/issues/277
unset(BUILD_SHARED_LIBS)
if(DEFINED CACHE{BUILD_SHARED_LIBS})
  set(CACHED_BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS})
  unset(BUILD_SHARED_LIBS CACHE)
endif()

FetchContent_MakeAvailable(gflags)

# Workaround for https://github.com/gflags/gflags/issues/277
if(DEFINED CACHED_BUILD_SHARED_LIBS)
  set(
    BUILD_SHARED_LIBS
    ${CACHED_BUILD_SHARED_LIBS}
    CACHE BOOL
    "Restored after setting up gflags"
    FORCE
  )
endif()

# This causes find_package(gflags) in other dependencies to search in the build
# directory and prevents the system gflags from being found when they don't use
# the target directly (like folly).
set(gflags_FOUND TRUE)
set(gflags_LIBRARY gflags::gflags)
set(gflags_INCLUDE_DIR)
