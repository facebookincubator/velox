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

set(VELOX_INTELIAA_VERSION 1.3.0)
set(VELOX_INTELIAA_BUILD_SHA256_CHECKSUM
    c3eba4d04a9d7aabcf26c9eaf81f6e9b26d19cb1b87a4a5f197a652cfa98f310)
set(VELOX_INTELIAA_SOURCE_URL
    "https://github.com/intel/qpl/archive/refs/tags/v${VELOX_INTELIAA_VERSION}.tar.gz"
)

resolve_dependency_url(INTELIAA)

message(STATUS "Building Intel IAA from source")

set(QPL_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/qpl_ep/install")
set(QPL_STATIC_LIB_NAME
    ${CMAKE_STATIC_LIBRARY_PREFIX}qpl${CMAKE_STATIC_LIBRARY_SUFFIX})
set(QPL_STATIC_LIB "${QPL_PREFIX}/lib/${QPL_STATIC_LIB_NAME}")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ldl -laccel-config -L/usr/lib64")

set(QPL_CMAKE_ARGS
    -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
    -DCMAKE_INSTALL_LIBDIR=${QPL_PREFIX}/lib
    -DCMAKE_INSTALL_PREFIX=${QPL_PREFIX}
    -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}
    -DQPL_BUILD_TESTS=OFF
    -DQPL_BUILD_EXAMPLES=OFF
    -DQPL_LIB=ON)

ExternalProject_Add(
  intel_iaa
  URL ${VELOX_INTELIAA_SOURCE_URL}
  URL_HASH ${VELOX_INTELIAA_BUILD_SHA256_CHECKSUM}
  BUILD_BYPRODUCTS "${QPL_STATIC_LIB}"
  CMAKE_ARGS ${QPL_CMAKE_ARGS})

file(MAKE_DIRECTORY "${QPL_PREFIX}/include")

add_library(iaa::iaa UNKNOWN IMPORTED)
set(QPL_LIBRARIES ${QPL_STATIC_LIB})
set(QPL_INCLUDE_DIRS "${QPL_PREFIX}/include")
set_target_properties(
  iaa::iaa PROPERTIES IMPORTED_LOCATION ${QPL_LIBRARIES}
                      INTERFACE_INCLUDE_DIRECTORIES ${QPL_INCLUDE_DIRS})

add_dependencies(iaa::iaa intel_iaa-build)
