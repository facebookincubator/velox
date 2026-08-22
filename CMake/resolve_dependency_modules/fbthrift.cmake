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

set(VELOX_FBTHRIFT_BUILD_VERSION v2026.01.05.00)
set(
  VELOX_FBTHRIFT_BUILD_SHA256_CHECKSUM
  c266851c7a7c3b6798973250669ac713a2f838203882e312501dc390d36c3f89
)
set(
  VELOX_FBTHRIFT_SOURCE_URL
  "https://github.com/facebook/fbthrift/archive/refs/tags/${VELOX_FBTHRIFT_BUILD_VERSION}.tar.gz"
)

velox_resolve_dependency_url(FBTHRIFT)

message(STATUS "Building FBThrift from source")
FetchContent_Declare(
  FBThrift
  URL ${VELOX_FBTHRIFT_SOURCE_URL}
  URL_HASH ${VELOX_FBTHRIFT_BUILD_SHA256_CHECKSUM}
  PATCH_COMMAND
    ${CMAKE_COMMAND} -DSOURCE_DIR=<SOURCE_DIR>
    -DPATCH_FILE=${CMAKE_CURRENT_LIST_DIR}/fbthrift/compactv1-protocol-refiller.patch -P
    ${CMAKE_CURRENT_LIST_DIR}/ApplyPatch.cmake COMMAND ${CMAKE_COMMAND} -DSOURCE_DIR=<SOURCE_DIR>
    -DPATCH_FILE=${CMAKE_CURRENT_LIST_DIR}/fbthrift/cmake-generated-tcc.patch -P
    ${CMAKE_CURRENT_LIST_DIR}/ApplyPatch.cmake
  OVERRIDE_FIND_PACKAGE
  SYSTEM
  EXCLUDE_FROM_ALL
)

block()
  set(enable_tests OFF)
  set(BUILD_TESTS OFF)
  set(BUILD_SHARED_LIBS OFF)
  set(thriftpy3 OFF)
  velox_fetchcontent_makeavailable_without_install(FBThrift)
endblock()

FetchContent_GetProperties(FBThrift SOURCE_DIR FBTHRIFT_INCLUDE_DIR)
if(NOT IS_DIRECTORY "${FBTHRIFT_INCLUDE_DIR}/thrift")
  message(
    FATAL_ERROR
    "Bundled FBThrift include root does not contain thrift/: ${FBTHRIFT_INCLUDE_DIR}"
  )
endif()
if(NOT EXISTS "${FBTHRIFT_INCLUDE_DIR}/thrift/annotation/thrift.thrift")
  message(
    FATAL_ERROR
    "Bundled FBThrift is missing thrift/annotation/thrift.thrift under: ${FBTHRIFT_INCLUDE_DIR}"
  )
endif()

if(NOT TARGET thriftcpp2)
  message(FATAL_ERROR "FBThrift did not define the required thriftcpp2 target")
endif()
if(NOT TARGET thriftmetadata)
  message(FATAL_ERROR "FBThrift did not define the required thriftmetadata target")
endif()
if(NOT TARGET FBThrift::thriftcpp2)
  add_library(FBThrift::thriftcpp2 ALIAS thriftcpp2)
endif()
if(NOT TARGET FBThrift::thriftmetadata)
  add_library(FBThrift::thriftmetadata ALIAS thriftmetadata)
endif()

if(NOT TARGET thrift1)
  message(FATAL_ERROR "FBThrift did not define the required thrift1 target")
endif()
if(NOT TARGET FBThrift::thrift1)
  add_executable(FBThrift::thrift1 ALIAS thrift1)
endif()

set(FBThrift_FOUND true)
