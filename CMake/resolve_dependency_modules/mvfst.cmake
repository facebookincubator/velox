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

set(VELOX_MVFST_BUILD_VERSION v2026.01.05.00)
set(
  VELOX_MVFST_BUILD_SHA256_CHECKSUM
  761b4504e542dcf536f5692c387de200acdd287e5ae249afb724963475cd69ca
)
set(
  VELOX_MVFST_SOURCE_URL
  "https://github.com/facebook/mvfst/archive/refs/tags/${VELOX_MVFST_BUILD_VERSION}.tar.gz"
)

velox_resolve_dependency_url(MVFST)

FetchContent_Declare(
  mvfst
  URL ${VELOX_MVFST_SOURCE_URL}
  URL_HASH ${VELOX_MVFST_BUILD_SHA256_CHECKSUM}
  OVERRIDE_FIND_PACKAGE
  SYSTEM
  EXCLUDE_FROM_ALL
)

block()
  FetchContent_GetProperties(folly SOURCE_DIR FOLLY_INCLUDE_DIR)
  set(FOLLY_LIBRARIES Folly::folly)
  FetchContent_GetProperties(fizz SOURCE_DIR FIZZ_INCLUDE_DIR)
  set(FIZZ_LIBRARIES fizz::fizz)
  set(BUILD_TESTS OFF)
  set(BUILD_SHARED_LIBS OFF)
  velox_fetchcontent_makeavailable_without_install(mvfst)
endblock()

if(NOT TARGET mvfst_server)
  message(FATAL_ERROR "mvfst did not define the required mvfst_server target")
endif()
if(NOT TARGET mvfst_server_async_tran)
  message(FATAL_ERROR "mvfst did not define the required mvfst_server_async_tran target")
endif()
if(NOT TARGET mvfst::mvfst_server)
  add_library(mvfst::mvfst_server ALIAS mvfst_server)
endif()
if(NOT TARGET mvfst::mvfst_server_async_tran)
  add_library(mvfst::mvfst_server_async_tran ALIAS mvfst_server_async_tran)
endif()

set(mvfst_FOUND true)
