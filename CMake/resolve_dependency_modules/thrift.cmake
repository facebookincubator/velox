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

# This creates a separate scope so any changed variables don't affect the rest
# of the build.
block()

set(VELOX_THRIFT_VERSION 0.22.0)
set(VELOX_THRIFT_BUILD_SHA256_CHECKSUM
    794a0e455787960d9f27ab92c38e34da27e8deeda7a5db0e59dc64a00df8a1e5)
# This is a mirror URL.
# https://www.apache.org/dyn/closer.lua/thrift/${VELOX_THRIFT_VERSION}/thrift-${VELOX_THRIFT_VERSION}.tar.gz?action=download
# is the official original URL. But it may cause many timeouts when more newer
# Apache Thrift is released. Because https://www.apache.org/dyn/closer.lua uses
# limited HTTP service (archive.apache.org) for old releases.
#
# For stable download, we use a mirror URL here.
set(VELOX_THRIFT_SOURCE_URL
    "https://packages.apache.org/artifactory/arrow/thirdparty/thrift-${VELOX_THRIFT_VERSION}.tar.gz"
)

velox_resolve_dependency_url(THRIFT)

message(STATUS "Building Apache Thrift from source")
FetchContent_Declare(
  thrift
  URL ${VELOX_THRIFT_SOURCE_URL}
  URL_HASH ${VELOX_THRIFT_BUILD_SHA256_CHECKSUM}
  OVERRIDE_FIND_PACKAGE EXCLUDE_FROM_ALL SYSTEM
  # We can remove this once https://github.com/apache/thrift/pull/3187/ is
  # resolved.
  PATCH_COMMAND git apply ${CMAKE_CURRENT_LIST_DIR}/thrift/3187.patch)

set(BUILD_COMPILER OFF)
set(BUILD_EXAMPLES OFF)
set(BUILD_TUTORIALS OFF)
string(APPEND CMAKE_CXX_FLAGS " -Wno-error")
set(CMAKE_UNITY_BUILD OFF)
set(WITH_AS3 OFF)
set(WITH_CPP ON)
set(WITH_C_GLIB OFF)
set(WITH_JAVA OFF)
set(WITH_JAVASCRIPT OFF)
set(WITH_LIBEVENT OFF)
set(WITH_MT OFF)
set(WITH_NODEJS OFF)
set(WITH_PYTHON OFF)
set(WITH_QT5 OFF)
set(WITH_ZLIB OFF)

if(Boost_SOURCE STREQUAL "BUNDLED")
  set(Boost_INCLUDE_DIRS)
  foreach(component algorithm locale tokenizer)
    list(APPEND Boost_INCLUDE_DIRS
         $<TARGET_PROPERTY:Boost::${component},INTERFACE_INCLUDE_DIRECTORIES>)
  endforeach()
endif()

# Apache Thrift may change CMAKE_DEBUG_POSTFIX. So we'll restore the original
# CMAKE_DEBUG_POSTFIX later.
set(CMAKE_DEBUG_POSTFIX_KEEP ${CMAKE_DEBUG_POSTFIX})
FetchContent_MakeAvailable(thrift)
# Apache Thrift may change CMAKE_DEBUG_POSTFIX. So we restore
# CMAKE_DEBUG_POSTFIX.
set(CMAKE_DEBUG_POSTFIX
    ${CMAKE_DEBUG_POSTFIX_KEEP}
    CACHE BOOL "" FORCE)

target_include_directories(
  thrift INTERFACE $<BUILD_LOCAL_INTERFACE:${thrift_BINARY_DIR}>
                   $<BUILD_LOCAL_INTERFACE:${thrift_SOURCE_DIR}/lib/cpp/src>)
if(Boost_SOURCE STREQUAL "BUNDLED")
  target_link_libraries(
    thrift
    PUBLIC $<BUILD_LOCAL_INTERFACE:Boost::headers>
    PRIVATE $<BUILD_LOCAL_INTERFACE:Boost::locale>)
endif()

add_library(thrift::thrift INTERFACE IMPORTED)
target_link_libraries(
  thrift::thrift
  INTERFACE thrift)

endblock()
