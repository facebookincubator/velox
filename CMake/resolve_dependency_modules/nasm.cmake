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

if(DEFINED ENV{VELOX_NASM_URL})
  set(NASM_SOURCE_URL "$ENV{VELOX_NASM_URL}")
else()
  set(VELOX_NASM_BUILD_VERSION 2.15.05)
  string(CONCAT NASM_SOURCE_URL
                "https://github.com/netwide-assembler/nasm/archive/refs/tags/"
                "nasm-${VELOX_NASM_BUILD_VERSION}.tar.gz")

  set(VELOX_NASM_BUILD_SHA256_CHECKSUM
      f575c516b5c1c28d5d64efdc26ed52b137ad36bfbd2d855d4782f050ca964245)
endif()

message(STATUS "Building NASM from source")

ProcessorCount(NUM_JOBS)
set_with_default(NUM_JOBS NUM_THREADS ${NUM_JOBS})
find_program(MAKE_PROGRAM make REQUIRED)

# set(NASM_CFG --disable-tests --disable-samples)
set(HOST_ENV_CMAKE
    ${CMAKE_COMMAND}
    -E
    env
    CC="${CMAKE_C_COMPILER}"
    CXX="${CMAKE_CXX_COMPILER}"
    CFLAGS="${CMAKE_C_FLAGS}"
    CXXFLAGS="${CMAKE_CXX_FLAGS}"
    LDFLAGS="${CMAKE_MODULE_LINKER_FLAGS}")

message(STATUS ${CMAKE_CURRENT_BINARY_DIR})
set(NASM_DIR ${CMAKE_CURRENT_BINARY_DIR}/nasm)
set(NASM_INCLUDE_DIRS ${NASM_DIR}/include)
set(NASM_LIBRARIES ${NASM_DIR}/lib)

# We can not use FetchContent as NASM does not use cmake
ExternalProject_Add(
  nasm
  URL ${NASM_SOURCE_URL}
  # URL_HASH SHA256=${VELOX_NASM_BUILD_SHA256_CHECKSUM}
  PREFIX ${CMAKE_CURRENT_BINARY_DIR}/_deps/nasm-src
  SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/_deps/nasm-src
  # BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/NASM-bld
  CONFIGURE_COMMAND <SOURCE_DIR>/autogen.sh <SOURCE_DIR>/configure
                    --prefix=${NASM_DIR} --libdir=${NASM_LIBRARIES}
  BUILD_COMMAND ${MAKE_PROGRAM} -j ${NUM_JOBS}
  INSTALL_COMMAND ${HOST_ENV_CMAKE} ${MAKE_PROGRAM} install)

# add_library(NASM::NASM UNKNOWN IMPORTED) add_dependencies(NASM::NASM
# NASM-builcd d) set_target_properties( NASM::NASM PROPERTIES
# INTERFACE_INCLUDE_DIRECTORIES ${NASM_INCLUDE_DIRS} INTERFACE_LINK_LIBRARIES
# ${NASM_LIBRARIES})

# We have to keep the FindNASM.cmake in a subfolder to prevent it from
# overriding the system provided one when NASM_SOURCE=SYSTEM
list(PREPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/NASM)
