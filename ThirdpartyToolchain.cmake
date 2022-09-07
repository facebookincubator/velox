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

# MODULE:   ThirdpartyToolchain
#
# PROVIDES: resolve_dependency( DEPENDENCY_NAME dependencyName [REQUIRED_VERSION
# required version] ... )
#
# Provides the ability to resolve third party dependencies. If the dependency is
# already available in the system it will be used.
#
# The DEPENDENCY_NAME argument is required. The dependencyName value will be
# used to search for the installed dependencies Config file and thus this name
# should match find_package() standards.
#
# EXAMPLE USAGE: # Download and setup or use already installed dependency.
# include(ThirdpartyToolchain) resolve_dependency(folly)
#
# ========================================================================================

add_custom_target(toolchain)
include(ExternalProject)

# =====================================FOLLY==============================================

if(DEFINED ENV{VELOX_FOLLY_URL})
  set(FOLLY_SOURCE_URL "$ENV{VELOX_FOLLY_URL}")
else()
  set(VELOX_FOLLY_BUILD_VERSION v2022.07.11.00)
  set(FOLLY_SOURCE_URL
      "https://github.com/facebook/folly/archive/${VELOX_FOLLY_BUILD_VERSION}.tar.gz"
  )
  set(VELOX_FOLLY_BUILD_SHA256_CHECKSUM
      b6cc4082afd1530fdb8d759bc3878c1ea8588f6d5bc9eddf8e1e8abe63f41735)
endif()

macro(build_folly)
  message(STATUS "Building Folly from source")
  set(FOLLY_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/folly-install")
  set(FOLLY_INCLUDE_DIR "${FOLLY_PREFIX}/include")
  file(MAKE_DIRECTORY ${FOLLY_INCLUDE_DIR})

  set(FOLLY_STATIC_LIB
      "${FOLLY_PREFIX}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}folly${CMAKE_STATIC_LIBRARY_SUFFIX}"
  )
  set(FOLLY_BENCHMARK_STATIC_LIB
      "${FOLLY_PREFIX}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}follybenchmark${CMAKE_STATIC_LIBRARY_SUFFIX}"
  )
  if(NOT SCRIPT_CXX_FLAGS)
    get_script_cxx_flags()
  endif()
  set(FOLLY_CMAKE_ARGS
      "-DCMAKE_CXX_FLAGS=${SCRIPT_CXX_FLAGS}" -DCMAKE_CXX_STANDARD=17
      -DCMAKE_POSITION_INDEPENDENT_CODE=ON
      "-DCMAKE_INSTALL_PREFIX=${FOLLY_PREFIX}")

  ExternalProject_Add(
    folly
    URL ${FOLLY_SOURCE_URL}
    URL_HASH "SHA256=${VELOX_FOLLY_BUILD_SHA256_CHECKSUM}"
    CMAKE_ARGS ${FOLLY_CMAKE_ARGS}
    BUILD_BYPRODUCTS "${FOLLY_STATIC_LIB}" "${FOLLY_BENCHMARK_STATIC_LIB}")

  # TODO: FIX HOW TO LINK FOLLY REQUIRED LIBRARIES

  # LIBUNWIND INTERFACE_LINK_LIBRARIES REQUIRED
  find_library(LIBUNWIND_LIBRARY NAMES unwind)
  if(LIBUNWIND_LIBRARY)
    list(APPEND FOLLY_LINK_LIBRARIES ${LIBUNWIND_LIBRARY})
  endif()

  # OPENSSL INTERFACE_LINK_LIBRARIES REQUIRED
  find_package(OpenSSL 1.1.1 MODULE REQUIRED)
  if(OPENSSL_FOUND)
    list(APPEND FOLLY_LINK_LIBRARIES ${OPENSSL_LIBRARIES})
  endif()

  # BOOST INTERFACE_LINK_LIBRARIES REQUIRED
  find_package(
    Boost 1.51.0 MODULE
    COMPONENTS context filesystem program_options regex system thread
    REQUIRED)
  list(APPEND FOLLY_LINK_LIBRARIES ${Boost_LIBRARIES})

  # ZSTD INTERFACE_LINK_LIBRARIES REQUIRED
  find_library(ZSTD_LIBRARY_RELEASE NAMES zstd zstd_static)
  if(ZSTD_LIBRARY_RELEASE)
    list(APPEND FOLLY_LINK_LIBRARIES ${ZSTD_LIBRARY_RELEASE})
  endif()

  # LZ4 INTERFACE_LINK_LIBRARIES REQUIRED
  find_library(LZ4_LIBRARY_RELEASE NAMES lz4)
  if(LZ4_LIBRARY_RELEASE)
    list(APPEND FOLLY_LINK_LIBRARIES ${LZ4_LIBRARY_RELEASE})
  endif()

  # LIBLZMA INTERFACE_LINK_LIBRARIES REQUIRED
  find_package(LibLZMA MODULE)
  set(FOLLY_HAVE_LIBLZMA ${LIBLZMA_FOUND})
  if(LIBLZMA_FOUND)
    list(APPEND FOLLY_LINK_LIBRARIES ${LIBLZMA_LIBRARIES})
  endif()

  # ZLIB INTERFACE_LINK_LIBRARIES REQUIRED
  find_package(ZLIB MODULE)
  if(ZLIB_FOUND)
    list(APPEND FOLLY_LINK_LIBRARIES ${ZLIB_LIBRARIES})
  endif()

  add_dependencies(toolchain folly)
  add_library(Folly::follybenchmark STATIC IMPORTED)
  add_library(Folly::folly STATIC IMPORTED)

  set_target_properties(
    Folly::follybenchmark
    PROPERTIES IMPORTED_LOCATION "${FOLLY_BENCHMARK_STATIC_LIB}"
               INTERFACE_INCLUDE_DIRECTORIES "${FOLLY_INCLUDE_DIR}"
               INTERFACE_LINK_LIBRARIES "${FOLLY_LINK_LIBRARIES}")
  set_target_properties(
    Folly::folly
    PROPERTIES IMPORTED_LOCATION "${FOLLY_STATIC_LIB}"
               INTERFACE_INCLUDE_DIRECTORIES "${FOLLY_INCLUDE_DIR}"
               INTERFACE_LINK_LIBRARIES "${FOLLY_LINK_LIBRARIES}")

  add_dependencies(Folly::folly folly)

  set(FOLLY_LIBRARIES Folly::folly)
endmacro()

# ===================================END FOLLY===============================

macro(build_dependency DEPENDENCY_NAME)
  if("${DEPENDENCY_NAME}" STREQUAL "folly")
    build_folly()
  else()
    message(
      FATAL_ERROR "Unknown thirdparty dependency to build: ${DEPENDENCY_NAME}")
  endif()
endmacro()

macro(resolve_dependency DEPENDENCY_NAME)
  set(options)
  set(one_value_args REQUIRED_VERSION)
  set(multi_value_args)
  cmake_parse_arguments(ARG "${options}" "${one_value_args}"
                        "${multi_value_args}" ${ARGN})
  if(ARG_UNPARSED_ARGUMENTS)
    message(
      SEND_ERROR "Error: unrecognized arguments: ${ARG_UNPARSED_ARGUMENTS}")
  endif()
  set(PACKAGE_NAME ${DEPENDENCY_NAME})
  set(FIND_PACKAGE_ARGUMENTS ${PACKAGE_NAME})
  if(ARG_REQUIRED_VERSION)
    list(APPEND FIND_PACKAGE_ARGUMENTS ${ARG_REQUIRED_VERSION})
  endif()
  if(${DEPENDENCY_NAME}_SOURCE STREQUAL "AUTO")
    find_package(${FIND_PACKAGE_ARGUMENTS})
    if(${${PACKAGE_NAME}_FOUND})
      set(${DEPENDENCY_NAME}_SOURCE "SYSTEM")
    else()
      build_dependency(${DEPENDENCY_NAME})
      set(${DEPENDENCY_NAME}_SOURCE "BUNDLED")
    endif()
  endif()
endmacro()
