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
# PROVIDES:
#   resolve_dependency( DEPENDENCY_NAME dependencyName
#                    [REQUIRED_VERSION required version]
#                    ...
#   )
#
#       Provides the ability to resolve third party dependencies. If the dependency is
#       already available in the system it will be used.
#
#       The DEPENDENCY_NAME argument is required. The dependencyName value will be used to search for
#       the installed dependencies Config file and thus this name should match find_package()
#       standards.
#
# EXAMPLE USAGE:
#   # Download and setup or use already installed dependency.
#   include(ThirdpartyToolchain)
#   resolve_dependency(folly)
#
#
#========================================================================================


include(FetchContent)

#=====================================FOLLY==============================================

if(DEFINED ENV{VELOX_FOLLY_URL})
  set(FOLLY_SOURCE_URL "$ENV{VELOX_FOLLY_URL}")
else()
  set(VELOX_FOLLY_BUILD_VERSION v2022.07.11.00)
  set(FOLLY_SOURCE_URL
           "https://github.com/facebook/folly/archive/${VELOX_FOLLY_BUILD_VERSION}.tar.gz")
  set(VELOX_FOLLY_BUILD_SHA256_CHECKSUM b6cc4082afd1530fdb8d759bc3878c1ea8588f6d5bc9eddf8e1e8abe63f41735)
endif()

macro(build_folly)
  message(STATUS "Building Folly from source")
  FetchContent_Declare(
      folly
      URL ${FOLLY_SOURCE_URL}
      URL_HASH SHA256=${VELOX_FOLLY_BUILD_SHA256_CHECKSUM}
    )
  if(NOT folly_POPULATED)
    # Fetch the content using previously declared details
    FetchContent_Populate(folly)
  endif()
  add_subdirectory(${folly_SOURCE_DIR} ${folly_BINARY_DIR})

  # Currently failing OpenSSL due to deprecation warning being used on folly.
  # The above does not fix the issue.
  find_package(OpenSSL 1.1.1 MODULE REQUIRED)
  set(FOLLY_LIBRARIES ${OPENSSL_LIBRARIES})
  list(APPEND FOLLY_LIBRARIES folly)
endmacro()
#===================================END FOLLY============================================

macro(build_dependency DEPENDENCY_NAME)
  if("${DEPENDENCY_NAME}" STREQUAL "folly")
    build_folly()
  else()
    message(FATAL_ERROR "Unknown thirdparty dependency to build: ${DEPENDENCY_NAME}")
  endif()
endmacro()

macro(resolve_dependency DEPENDENCY_NAME)
  set(options)
  set(one_value_args REQUIRED_VERSION)
  set(multi_value_args)
  cmake_parse_arguments(ARG
                        "${options}"
                        "${one_value_args}"
                        "${multi_value_args}"
                        ${ARGN})
  if(ARG_UNPARSED_ARGUMENTS)
    message(SEND_ERROR "Error: unrecognized arguments: ${ARG_UNPARSED_ARGUMENTS}")
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