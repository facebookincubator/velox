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

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
                      "${CMAKE_CURRENT_SOURCE_DIR}/cmake_modules/")

include(ExternalProject)

if(VELOX_ENABLE_ARROW)
  find_package(Thrift)
  if(Thrift_FOUND)
    set(THRIFT_SOURCE "SYSTEM")
  else()
    set(THRIFT_SOURCE "BUNDLED")
  endif()
  if(DEFINED ENV{ARROW_EP_PATH})
    message(
      STATUS
        "Arrow path is set: ${ARROW_EP_PATH}, trying to use existing arrow libraries."
    )
    set(ARROW_LIB_PATH $ENV{ARROW_EP_PATH}/install/lib/)
    find_library(ARROW_LIB libarrow.a PATHS ${ARROW_LIB_PATH})
    find_library(PARQUET_LIB libparquet.a PATHS ${ARROW_LIB_PATH})
    find_library(ARROW_TESTING_LIB libarrow_testing.a PATHS ${ARROW_LIB_PATH})
    if("${ARROW_LIB}" STREQUAL "ARROW_LIB-NOTFOUND"
       OR "${PARQUET_LIB}" STREQUAL "PARQUET_LIB-NOTFOUND"
       OR "${ARROW_TESTING_LIB}" STREQUAL "ARROW_TESTING_LIB-NOTFOUND")
      message(FATAL_ERROR "Arrow libraries not found in ${ARROW_LIB_PATH}")
    endif()

    add_library(thrift STATIC IMPORTED GLOBAL)
    if(NOT Thrift_FOUND)
      set(THRIFT_ROOT $ENV{ARROW_EP_PATH}/src/arrow_ep-build/thrift_ep-install)
      find_library(THRIFT_LIB thrift PATHS ${THRIFT_ROOT}/lib)
      if("${THRIFT_LIB}" STREQUAL "THRIFT_LIB-NOTFOUND")
        message(FATAL_ERROR "Thrift library not found in ${THRIFT_ROOT}/lib")
      endif()
      set(THRIFT_INCLUDE_DIR ${THRIFT_ROOT}/include)
    endif()
    set_property(TARGET thrift PROPERTY INTERFACE_INCLUDE_DIRECTORIES
                                        ${THRIFT_INCLUDE_DIR})
    set_property(TARGET thrift PROPERTY IMPORTED_LOCATION ${THRIFT_LIB})

    add_library(arrow STATIC IMPORTED GLOBAL)
    add_library(parquet STATIC IMPORTED GLOBAL)
    add_library(arrow_testing STATIC IMPORTED GLOBAL)

    set_target_properties(
      arrow arrow_testing parquet PROPERTIES INTERFACE_INCLUDE_DIRECTORIES
                                             $ENV{ARROW_EP_PATH}/install/include)
    set_target_properties(arrow PROPERTIES IMPORTED_LOCATION ${ARROW_LIB})
    set_property(TARGET arrow PROPERTY INTERFACE_LINK_LIBRARIES ${RE2} thrift)
    set_target_properties(parquet PROPERTIES IMPORTED_LOCATION ${PARQUET_LIB})
    set_target_properties(arrow_testing PROPERTIES IMPORTED_LOCATION
                                                   ${ARROW_TESTING_LIB})

    message(
      STATUS
        "Using existing arrow libraries: ${ARROW_LIB}, ${PARQUET_LIB}, ${ARROW_TESTING_LIB}"
    )
    return()
  endif()
  set(ARROW_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/arrow_ep")
  set(ARROW_CMAKE_ARGS
      -DARROW_PARQUET=ON
      -DARROW_FILESYSTEM=ON
      -DARROW_PROTOBUF_USE_SHARED=OFF
      -DARROW_WITH_THRIFT=ON
      -DARROW_WITH_LZ4=ON
      -DARROW_WITH_SNAPPY=ON
      -DARROW_WITH_ZLIB=ON
      -DARROW_WITH_ZSTD=ON
      -DARROW_JEMALLOC=OFF
      -DARROW_SIMD_LEVEL=NONE
      -DARROW_RUNTIME_SIMD_LEVEL=NONE
      -DARROW_WITH_UTF8PROC=OFF
      -DARROW_TESTING=ON
      -DCMAKE_INSTALL_PREFIX=${ARROW_PREFIX}/install
      -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
      -DARROW_BUILD_STATIC=ON
      -DThrift_SOURCE=${THRIFT_SOURCE})
  set(ARROW_LIBDIR ${ARROW_PREFIX}/install/${CMAKE_INSTALL_LIBDIR})

  add_library(thrift STATIC IMPORTED GLOBAL)
  if(NOT Thrift_FOUND)
    set(THRIFT_ROOT ${ARROW_PREFIX}/src/arrow_ep-build/thrift_ep-install)
    set(THRIFT_LIB ${THRIFT_ROOT}/lib/libthrift.a)

    file(MAKE_DIRECTORY ${THRIFT_ROOT}/include)
    set(THRIFT_INCLUDE_DIR ${THRIFT_ROOT}/include)
  endif()

  set_property(TARGET thrift PROPERTY INTERFACE_INCLUDE_DIRECTORIES
                                      ${THRIFT_INCLUDE_DIR})
  set_property(TARGET thrift PROPERTY IMPORTED_LOCATION ${THRIFT_LIB})

  set(VELOX_ARROW_BUILD_VERSION 15.0.0)
  set(VELOX_ARROW_BUILD_SHA256_CHECKSUM
      01dd3f70e85d9b5b933ec92c0db8a4ef504a5105f78d2d8622e84279fb45c25d)
  set(VELOX_ARROW_SOURCE_URL
      "https://archive.apache.org/dist/arrow/arrow-${VELOX_ARROW_BUILD_VERSION}/apache-arrow-${VELOX_ARROW_BUILD_VERSION}.tar.gz"
  )

  resolve_dependency_url(ARROW)

  ExternalProject_Add(
    arrow_ep
    PREFIX ${ARROW_PREFIX}
    URL ${VELOX_ARROW_SOURCE_URL}
    PATCH_COMMAND patch -p1 < ${CMAKE_CURRENT_SOURCE_DIR}/modify_arrow.patch
    URL_HASH ${VELOX_ARROW_BUILD_SHA256_CHECKSUM}
    SOURCE_SUBDIR cpp
    CMAKE_ARGS ${ARROW_CMAKE_ARGS}
    BUILD_BYPRODUCTS ${ARROW_LIBDIR}/libarrow.a ${ARROW_LIBDIR}/libparquet.a
                     ${ARROW_LIBDIR}/libarrow_testing.a ${THRIFT_LIB})
  add_library(arrow STATIC IMPORTED GLOBAL)
  add_library(arrow_testing STATIC IMPORTED GLOBAL)
  add_library(parquet STATIC IMPORTED GLOBAL)
  add_dependencies(arrow arrow_ep)
  add_dependencies(arrow_testing arrow)
  add_dependencies(parquet arrow)
  file(MAKE_DIRECTORY ${ARROW_PREFIX}/install/include)
  set_target_properties(
    arrow arrow_testing parquet PROPERTIES INTERFACE_INCLUDE_DIRECTORIES
                                           ${ARROW_PREFIX}/install/include)
  set_target_properties(arrow PROPERTIES IMPORTED_LOCATION
                                         ${ARROW_LIBDIR}/libarrow.a)
  set_property(TARGET arrow PROPERTY INTERFACE_LINK_LIBRARIES ${RE2} thrift)
  set_target_properties(
    arrow_testing PROPERTIES IMPORTED_LOCATION
                             ${ARROW_LIBDIR}/libarrow_testing.a)
  set_target_properties(parquet PROPERTIES IMPORTED_LOCATION
                                           ${ARROW_LIBDIR}/libparquet.a)

endif()
