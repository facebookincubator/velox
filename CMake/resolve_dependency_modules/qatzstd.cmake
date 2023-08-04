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

include(Findzstd)

set(VELOX_QATZSTD_BUILD_VERSION 0.0.1)
set(VELOX_QATZSTD_BUILD_SHA256_CHECKSUM
    c15aa561d5139d49f896315cbd38d02a51b636e448abfa58e4d3a011fe0424f2)
string(CONCAT VELOX_QATZSTD_SOURCE_URL
              "https://github.com/intel/QAT-ZSTD-Plugin/archive/refs/tags/"
              "v${VELOX_QATZSTD_BUILD_VERSION}.tar.gz")

resolve_dependency_url(QATZSTD)

message(STATUS "Building QATZSTD from source")

ProcessorCount(NUM_JOBS)
set_with_default(NUM_JOBS NUM_THREADS ${NUM_JOBS})
find_program(MAKE_PROGRAM make REQUIRED)

set(QATZSTD_SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/_deps/qatzstd-src")
set(QATZSTD_INCLUDE_DIR "${QATZSTD_SOURCE_DIR}/src")
set(QATZSTD_STATIC_LIB_TARGETS "${QATZSTD_SOURCE_DIR}/src/libqatseqprod.a")
set(QATZSTD_MAKE_ARGS "ENABLE_USDM_DRV=1" "ZSTDLIB=${ZSTD_INCLUDE_DIR}")

ExternalProject_Add(
  qatzstd
  URL ${VELOX_QATZSTD_SOURCE_URL}
  URL_HASH ${VELOX_QATZSTD_BUILD_SHA256_CHECKSUM}
  SOURCE_DIR ${QATZSTD_SOURCE_DIR}
  BINARY_DIR ${QATZSTD_BINARY_DIR}
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ${MAKE_PROGRAM} ${QATZSTD_MAKE_ARGS}
  INSTALL_COMMAND ""
  BUILD_BYPRODUCTS ${QATZSTD_STATIC_LIB_TARGETS}
  BUILD_IN_SOURCE 1)

add_library(qatzstd::qatzstd STATIC IMPORTED)

# The include directory must exist before it is referenced by a target.
file(MAKE_DIRECTORY "${QATZSTD_INCLUDE_DIR}")

find_library(
  USDM_DRV_LIBRARY REQUIRED
  NAMES usdm_drv_s
  PATHS "$ENV{ICP_ROOT}/build"
  NO_DEFAULT_PATH)
find_library(
  QAT_S_LIBRARY REQUIRED
  NAMES qat_s
  PATHS "$ENV{ICP_ROOT}/build"
  NO_DEFAULT_PATH)

message(STATUS "Found usdm_drv: ${USDM_DRV_LIBRARY}")
message(STATUS "Found qat_s: ${QAT_S_LIBRARY}")

set(QATZSTD_INCLUDE_DIRS "${QATZSTD_INCLUDE_DIR}" "${ZSTD_INCLUDE_DIR}")

set(QATZSTD_LINK_LIBRARIES "${ZSTD_LIBRARY}" "${USDM_DRV_LIBRARY}"
                           "${QAT_S_LIBRARY}")

set_target_properties(
  qatzstd::qatzstd
  PROPERTIES IMPORTED_LOCATION "${QATZSTD_STATIC_LIB_TARGETS}"
             INTERFACE_INCLUDE_DIRECTORIES "${QATZSTD_INCLUDE_DIRS}"
             INTERFACE_LINK_LIBRARIES "${QATZSTD_LINK_LIBRARIES}")

add_dependencies(qatzstd::qatzstd qatzstd)

# We have to manually create these files and folders so that the checks for the
# files made at configure time (before icu is built) pass set(QATZSTD_DIR
# ${CMAKE_CURRENT_BINARY_DIR}/_deps/qatzstd) set(QATZSTD_LIBRARIES
# ${QATZSTD_DIR}/lib)

# file(MAKE_DIRECTORY ${QATZSTD_INCLUDE_DIRS}) file(MAKE_DIRECTORY
# ${QATZSTD_LIBRARIES})

# add_library(QATZSTD::QATZSTD SHARED IMPORTED)
# add_dependencies(QATZSTD::QATZSTD QATZSTD-build)

# set_target_properties( QATZSTD::QATZSTD PROPERTIES IMPORTED_LOCATION
# ${QATZSTD_LIBRARY} INTERFACE_SYSTEM_INCLUDE_DIRECTORIES
# ${QATZSTD_INCLUDE_DIRS}) target_link_libraries(QATZSTD::QATZSTD INTERFACE
# QATZSTD::QATZSTD)

# FetchContent_Declare( qatzstd URL ${VELOX_QATZSTD_SOURCE_URL} URL_HASH
# ${VELOX_QATZSTD_BUILD_SHA256_CHECKSUM} BUILD_COMMAND "make ENABLE_USDM_DRV=1"
# )

# FetchContent_MakeAvailable(qatzstd)

# add_library(qatzstd INTERFACE) # set(qatzstd_LIBRARIES
# ${qatzstd_BINARY_DIR}/libqatseqprod.a) set(qatzstd_INCLUDE_DIRS
# ${qatzstd_SOURCE_DIR}/src)
