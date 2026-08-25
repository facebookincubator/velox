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

set(VELOX_WANGLE_BUILD_VERSION v2026.01.05.00)
set(
  VELOX_WANGLE_BUILD_SHA256_CHECKSUM
  bd20611ac5e40b03ba2c2a6107f064fc32a502d623eaeb914d80c283b6bd5ca7
)
set(
  VELOX_WANGLE_SOURCE_URL
  "https://github.com/facebook/wangle/archive/refs/tags/${VELOX_WANGLE_BUILD_VERSION}.tar.gz"
)

velox_resolve_dependency_url(WANGLE)

FetchContent_Declare(
  wangle
  URL ${VELOX_WANGLE_SOURCE_URL}
  URL_HASH ${VELOX_WANGLE_BUILD_SHA256_CHECKSUM}
  SOURCE_SUBDIR
  wangle
  OVERRIDE_FIND_PACKAGE
  SYSTEM
  EXCLUDE_FROM_ALL
)

block()
  FetchContent_GetProperties(folly SOURCE_DIR FOLLY_INCLUDE_DIR)
  set(FOLLY_LIBRARIES Folly::folly)
  set(BUILD_TESTS OFF)
  set(BUILD_SHARED_LIBS OFF)
  velox_fetchcontent_makeavailable_without_install(wangle)
endblock()

if(NOT TARGET wangle)
  message(FATAL_ERROR "Wangle did not define the required wangle target")
endif()
if(NOT TARGET wangle::wangle)
  add_library(wangle::wangle ALIAS wangle)
endif()

block()
  FetchContent_GetProperties(wangle SOURCE_DIR _wangle_include_dir)
  foreach(_property IN ITEMS INCLUDE_DIRECTORIES INTERFACE_INCLUDE_DIRECTORIES)
    get_target_property(_include_directories wangle ${_property})
    string(
      REPLACE
      "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/..>"
      "$<BUILD_INTERFACE:${_wangle_include_dir}>"
      _include_directories
      "${_include_directories}"
    )
    set_property(TARGET wangle PROPERTY ${_property} "${_include_directories}")
  endforeach()
endblock()

set(wangle_FOUND TRUE)
