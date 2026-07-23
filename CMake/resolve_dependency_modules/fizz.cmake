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

set(VELOX_FIZZ_BUILD_VERSION v2026.01.05.00)
set(
  VELOX_FIZZ_BUILD_SHA256_CHECKSUM
  a389a46460a7231e120d37e99130f304ce5efdf5c50e1bf6a2d69e85311a87d1
)
set(
  VELOX_FIZZ_SOURCE_URL
  "https://github.com/facebookincubator/fizz/archive/refs/tags/${VELOX_FIZZ_BUILD_VERSION}.tar.gz"
)

velox_resolve_dependency_url(FIZZ)

FetchContent_Declare(
  fizz
  URL ${VELOX_FIZZ_SOURCE_URL}
  URL_HASH ${VELOX_FIZZ_BUILD_SHA256_CHECKSUM}
  SOURCE_SUBDIR
  fizz
  OVERRIDE_FIND_PACKAGE
  SYSTEM
  EXCLUDE_FROM_ALL
)

block()
  FetchContent_GetProperties(folly SOURCE_DIR FOLLY_INCLUDE_DIR)
  set(FOLLY_LIBRARIES Folly::folly)
  set(BUILD_TESTS OFF)
  set(BUILD_SHARED_LIBS OFF)
  velox_fetchcontent_makeavailable_without_install(fizz)
endblock()

if(NOT TARGET fizz)
  message(FATAL_ERROR "Fizz did not define the required fizz target")
endif()
if(NOT TARGET fizz::fizz)
  add_library(fizz::fizz ALIAS fizz)
endif()

FetchContent_GetProperties(fizz SOURCE_DIR FIZZ_INCLUDE_DIR)
block()
  get_filename_component(_fizz_incorrect_include_dir "${CMAKE_SOURCE_DIR}/.." ABSOLUTE)
  foreach(_property IN ITEMS INCLUDE_DIRECTORIES INTERFACE_INCLUDE_DIRECTORIES)
    get_target_property(_include_directories fizz ${_property})
    string(
      REPLACE
      "$<BUILD_INTERFACE:${_fizz_incorrect_include_dir}>"
      "$<BUILD_INTERFACE:${FIZZ_INCLUDE_DIR}>"
      _include_directories
      "${_include_directories}"
    )
    set_property(TARGET fizz PROPERTY ${_property} "${_include_directories}")
  endforeach()
endblock()

set(FIZZ_LIBRARIES fizz::fizz)
set(Fizz_FOUND TRUE)
set(fizz_FOUND TRUE)
