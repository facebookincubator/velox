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

set(VELOX_FOLLY_BUILD_VERSION v2024.04.01.00)
set(VELOX_FOLLY_BUILD_SHA256_CHECKSUM
    e0c8386eea1a317ff17f1edc36c29362db94aeca8689659f1c530ea777779409)
set(VELOX_FOLLY_SOURCE_URL
    "https://github.com/facebook/folly/releases/download/${VELOX_FOLLY_BUILD_VERSION}/folly-${VELOX_FOLLY_BUILD_VERSION}.tar.gz"
)

resolve_dependency_url(FOLLY)

message(STATUS "Building Folly from source")

if(gflags_SOURCE STREQUAL "BUNDLED")
  set(glog_patch && git apply ${CMAKE_CURRENT_LIST_DIR}/folly-gflags-glog.patch)
endif()

FetchContent_Declare(
  folly
  URL ${VELOX_FOLLY_SOURCE_URL}
  URL_HASH ${VELOX_FOLLY_BUILD_SHA256_CHECKSUM}
  PATCH_COMMAND git apply ${CMAKE_CURRENT_LIST_DIR}/folly-no-export.patch
                ${glog_patch} OVERRIDE_FIND_PACKAGE SYSTEM EXCLUDE_FROM_ALL)

if(ON_APPLE_M1)
  # folly will wrongly assume x86_64 if this is not set
  set(CMAKE_LIBRARY_ARCHITECTURE aarch64)
endif()

set(BUILD_SHARED_LIBS ${VELOX_BUILD_SHARED})

# Enable INT128 support
set(FOLLY_HAVE_INT128_T ON)
FetchContent_MakeAvailable(folly)

# Folly::folly is not valid for FC but we want to match FindFolly
add_library(Folly::folly ALIAS folly)
add_library(Folly::follybenchmark ALIAS follybenchmark)

if(${gflags_SOURCE} STREQUAL "BUNDLED")
  add_dependencies(folly glog gflags::gflags fmt::fmt)
endif()
