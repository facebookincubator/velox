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

include(ExternalProject)

macro(build_qatzip)
  message(STATUS "Building QATzip from source")
  set(QATZIP_BUILD_VERSION "v1.1.2")
  set(QATZIP_BUILD_SHA256_CHECKSUM
      "31419fa4b42d217b3e55a70a34545582cbf401a4f4d44738d21b4a3944b1e1ef")
  set(QATZIP_SOURCE_URL
      "https://github.com/intel/QATzip/archive/refs/tags/${QATZIP_BUILD_VERSION}.tar.gz"
  )
  set(QATZIP_LIB_NAME "qatzip")

  set(QATZIP_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/qatzip_ep-install")
  set(QATZIP_SOURCE_DIR "${QATZIP_PREFIX}/src/qatzip_ep")
  set(QATZIP_INCLUDE_DIR "${QATZIP_SOURCE_DIR}/include")
  set(QATZIP_STATIC_LIB_NAME
      "${CMAKE_STATIC_LIBRARY_PREFIX}${QATZIP_LIB_NAME}${CMAKE_STATIC_LIBRARY_SUFFIX}"
  )
  set(QATZIP_STATIC_LIB_TARGETS
      "${QATZIP_SOURCE_DIR}/src/.libs/${QATZIP_STATIC_LIB_NAME}")
  set(QATZIP_CONFIGURE_ARGS "--prefix=${QATZIP_PREFIX}" "--with-pic"
                            "--with-ICP_ROOT=${ICP_ROOT}")

  ExternalProject_Add(
    qatzip_ep
    PREFIX ${QATZIP_PREFIX}
    URL ${QATZIP_SOURCE_URL}
    URL_HASH "SHA256=${QATZIP_BUILD_SHA256_CHECKSUM}"
    SOURCE_DIR ${QATZIP_SOURCE_DIR}
    CONFIGURE_COMMAND ${CMAKE_COMMAND} -E env QZ_ROOT=${QATZIP_SOURCE_DIR}
                      ./configure ${QATZIP_CONFIGURE_ARGS}
    BUILD_COMMAND ${MAKE_PROGRAM} all
    BUILD_BYPRODUCTS ${QATZIP_STATIC_LIB_TARGETS}
    BUILD_IN_SOURCE 1)

  ExternalProject_Add_Step(
    qatzip_ep pre-configure
    COMMAND ./autogen.sh
    DEPENDEES download
    DEPENDERS configure
    WORKING_DIRECTORY ${QATZIP_SOURCE_DIR})

  # The include directory must exist before it is referenced by a target.
  file(MAKE_DIRECTORY "${QATZIP_INCLUDE_DIR}")

  set(QATZIP_LINK_LIBRARIES
      ZLIB::ZLIB lz4::lz4 "${UDEV_LIBRARY}" "${USDM_DRV_LIBRARY}"
      "${QAT_S_LIBRARY}" Threads::Threads)

  add_library(qatzip::qatzip STATIC IMPORTED)
  set_target_properties(
    qatzip::qatzip
    PROPERTIES IMPORTED_LOCATION "${QATZIP_STATIC_LIB_TARGETS}"
               INTERFACE_INCLUDE_DIRECTORIES "${QATZIP_INCLUDE_DIR}"
               INTERFACE_LINK_LIBRARIES "${QATZIP_LINK_LIBRARIES}")

  add_dependencies(qatzip::qatzip qatzip_ep)
endmacro()

set(ICP_ROOT $ENV{ICP_ROOT})
set(THREADS_PREFER_PTHREAD_FLAG ON)

find_package(Threads REQUIRED)
find_program(MAKE_PROGRAM make REQUIRED)

find_library(UDEV_LIBRARY REQUIRED NAMES udev)
find_library(
  USDM_DRV_LIBRARY REQUIRED
  NAMES usdm_drv_s
  PATHS "${ICP_ROOT}/build"
  NO_DEFAULT_PATH)
find_library(
  QAT_S_LIBRARY REQUIRED
  NAMES qat_s
  PATHS "${ICP_ROOT}/build"
  NO_DEFAULT_PATH)

message(STATUS "Found udev: ${UDEV_LIBRARY}")
message(STATUS "Found usdm_drv: ${USDM_DRV_LIBRARY}")
message(STATUS "Found qat_s: ${QAT_S_LIBRARY}")

build_qatzip()
