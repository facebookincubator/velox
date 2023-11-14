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

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake_modules/")

include(ExternalProject)

if(VELOX_ENABLE_ARROW)
  find_package(Arrow)

  add_library(thrift STATIC IMPORTED GLOBAL)
  add_library(arrow STATIC IMPORTED GLOBAL)
  add_library(arrow_testing STATIC IMPORTED GLOBAL)
  add_library(parquet STATIC IMPORTED GLOBAL)

  if(NOT Arrow_FOUND)
    find_package(Thrift)
    if(Thrift_FOUND)
      set(THRIFT_SOURCE "SYSTEM")
    else()
      set(THRIFT_SOURCE "BUNDLED")
    endif()
    set(ARROW_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/arrow_ep")
    set(ARROW_CMAKE_ARGS
        -DARROW_PARQUET=ON
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
        -DThrift_SOURCE=${THRIFT_SOURCE}
    )
    set(ARROW_LIBDIR ${ARROW_PREFIX}/install/${CMAKE_INSTALL_LIBDIR})
    file(MAKE_DIRECTORY ${ARROW_INCLUDE_DIR})
    set(ARROW_INCLUDE_DIR ${ARROW_PREFIX}/install/include)
    set(ARROW_STATIC_LIB ${ARROW_LIBDIR}/libarrow.a)
    set(PARQUET_STATIC_LIB ${ARROW_LIBDIR}/libparquet.a)

    if(NOT Thrift_FOUND)
      set(THRIFT_ROOT ${ARROW_PREFIX}/src/arrow_ep-build/thrift_ep-install)
      set(THRIFT_LIB ${THRIFT_ROOT}/lib/libthrift.a)

      file(MAKE_DIRECTORY ${THRIFT_ROOT}/include)
      set(THRIFT_INCLUDE_DIR ${THRIFT_ROOT}/include)
    endif()

    set(VELOX_ARROW_BUILD_VERSION 13.0.0)
    set(VELOX_ARROW_BUILD_SHA256_CHECKSUM 35dfda191262a756be934eef8afee8d09762cad25021daa626eb249e251ac9e6)
    set(VELOX_ARROW_SOURCE_URL
        "https://archive.apache.org/dist/arrow/arrow-${VELOX_ARROW_BUILD_VERSION}/apache-arrow-${VELOX_ARROW_BUILD_VERSION}.tar.gz"
    )

    resolve_dependency_url(ARROW)

    ExternalProject_Add(
      arrow_ep
      PREFIX ${ARROW_PREFIX}
      URL ${VELOX_ARROW_SOURCE_URL}
      URL_HASH ${VELOX_ARROW_BUILD_SHA256_CHECKSUM}
      SOURCE_SUBDIR cpp
      CMAKE_ARGS ${ARROW_CMAKE_ARGS}
      BUILD_BYPRODUCTS
        ${ARROW_LIBDIR}/libarrow.a
        ${ARROW_LIBDIR}/libparquet.a
        ${ARROW_LIBDIR}/libarrow_testing.a
        ${THRIFT_LIB}
    )
    add_dependencies(arrow arrow_ep)
    set_target_properties(arrow_testing PROPERTIES IMPORTED_LOCATION ${ARROW_LIBDIR}/libarrow_testing.a)
  else() # NOT Arrow_FOUND
    find_package(Thrift)
    if(NOT Thrift_FOUND)
      set(ARROW_THRIFT_BUILD_VERSION 0.19.0)
      set(THRIFT_SOURCE_URL
          "https://downloads.apache.org/thrift/${ARROW_THRIFT_BUILD_VERSION}/thrift-${ARROW_THRIFT_BUILD_VERSION}.tar.gz"
      )
      set(ARROW_THRIFT_BUILD_SHA256_CHECKSUM d49c896c2724a78701e05cfccf6cf70b5db312d82a17efe951b441d300ccf275)
      set(THRIFT_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/thrift_ep-install")
      set(THRIFT_INCLUDE_DIR ${THRIFT_PREFIX}/include)
      file(MAKE_DIRECTORY ${THRIFT_INCLUDE_DIR})
      set(THRIFT_CMAKE_ARGS
          -DCMAKE_INSTALL_PREFIX=${THRIFT_PREFIX}
          -DCMAKE_INSTALL_RPATH=${THRIFT_PREFIX}/lib
          -DBUILD_STATIC_LIBS=ON
          -DBUILD_TESTING=OFF
          # Work around https://gitlab.kitware.com/cmake/cmake/issues/18865
          -DBoost_NO_BOOST_CMAKE=ON
          -DBUILD_COMPILER=OFF
          -DBUILD_EXAMPLES=OFF
          -DBUILD_TUTORIALS=OFF
          -DCMAKE_DEBUG_POSTFIX=
          -DWITH_AS3=OFF
          -DWITH_CPP=ON
          -DWITH_C_GLIB=OFF
          -DWITH_JAVA=OFF
          -DWITH_JAVASCRIPT=OFF
          -DWITH_LIBEVENT=OFF
          -DWITH_NODEJS=OFF
          -DWITH_PYTHON=OFF
          -DWITH_QT5=OFF
          -DWITH_ZLIB=OFF
      )
      set(THRIFT_LIB ${THRIFT_PREFIX}/lib/libthrift.a)

      ExternalProject_Add(
        thrift_ep
        PREFIX ${THRIFT_PREFIX}
        URL ${THRIFT_SOURCE_URL}
        URL_HASH "SHA256=${ARROW_THRIFT_BUILD_SHA256_CHECKSUM}"
        BUILD_BYPRODUCTS ${THRIFT_LIB}
        CMAKE_ARGS ${THRIFT_CMAKE_ARGS}
      )
      add_dependencies(thrift thrift_ep)
    endif()
    find_package(Parquet CONFIG REQUIRED)
  endif() # NOT Arrow_FOUND
  add_dependencies(arrow_testing arrow)
  add_dependencies(parquet arrow)
  set_property(TARGET arrow PROPERTY INTERFACE_LINK_LIBRARIES ${RE2} thrift)

  set_property(TARGET thrift PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${THRIFT_INCLUDE_DIR})
  set_property(TARGET thrift PROPERTY IMPORTED_LOCATION ${THRIFT_LIB})
  set_target_properties(arrow arrow_testing parquet PROPERTIES INTERFACE_INCLUDE_DIRECTORIES ${ARROW_INCLUDE_DIR})
  set_target_properties(arrow PROPERTIES IMPORTED_LOCATION ${ARROW_STATIC_LIB})
  set_target_properties(parquet PROPERTIES IMPORTED_LOCATION ${PARQUET_STATIC_LIB})
endif() # VELOX_ENABLE_ARROW
