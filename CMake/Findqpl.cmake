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
include(ExternalProject)

macro(build_qpl)
  message(STATUS "Building QPL from source")
  set(QPL_BUILD_VERSION "v1.3.1")
  set(QPL_BUILD_SHA256_CHECKSUM
      "6ae537f9b84c222212e1ca8edaa275d1e1923d179a691353da38856ed8f4e5c4")
  set(QPL_SOURCE_URL
      "https://github.com/intel/qpl/archive/refs/tags/v1.3.1.tar.gz")
  set(QPL_LIB_NAME "qpl")

  set(QPL_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/qpl_ep-install")
  set(QPL_SOURCE_DIR "${QPL_PREFIX}/src/qpl_ep")
  set(QPL_INCLUDE_DIR "${QPL_PREFIX}/include")
  set(QPL_LIB_DIR "${QPL_PREFIX}/lib")
  set(QPL_STATIC_LIB_NAME
      "${CMAKE_STATIC_LIBRARY_PREFIX}${QPL_LIB_NAME}${CMAKE_STATIC_LIBRARY_SUFFIX}${QPL_STATIC_LIB_MAJOR_VERSION}"
  )
  set(QPL_STATIC_LIB_TARGETS "${QPL_LIB_DIR}/${QPL_STATIC_LIB_NAME}")
  ExternalProject_Add(
    qpl_ep
    PREFIX ${QPL_PREFIX}
    URL ${QPL_SOURCE_URL}
    URL_HASH "SHA256=${QPL_BUILD_SHA256_CHECKSUM}"
    SOURCE_DIR ${QPL_SOURCE_DIR}
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${QPL_PREFIX}
               -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} -DQPL_BUILD_TESTS=OFF
               -DLOG_HW_INIT=ON
    BUILD_BYPRODUCTS ${QPL_STATIC_LIB_TARGETS})

  # The include directory must exist before it is referenced by a target.
  file(MAKE_DIRECTORY "${QPL_INCLUDE_DIR}")

  add_library(qpl::qpl STATIC IMPORTED)
  set_target_properties(
    qpl::qpl
    PROPERTIES IMPORTED_LOCATION "${QPL_LIB_DIR}/${QPL_STATIC_LIB_NAME}"
               INTERFACE_INCLUDE_DIRECTORIES "${QPL_INCLUDE_DIR}")

  add_dependencies(qpl::qpl qpl_ep)
endmacro()

build_qpl()
