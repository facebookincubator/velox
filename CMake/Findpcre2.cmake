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

# Locate a system PCRE2 installation (8-bit code unit width) and expose it
# under the canonical target name `pcre2-8::pcre2-8` used by the
# velox/external/regex_compat module.

find_package(PCRE2 QUIET CONFIG COMPONENTS 8BIT)
if(PCRE2_FOUND)
  if(NOT TARGET pcre2-8::pcre2-8 AND TARGET PCRE2::8BIT)
    add_library(pcre2-8::pcre2-8 ALIAS PCRE2::8BIT)
  endif()
  message(STATUS "Found PCRE2 via CMake.")
  return()
endif()

if(TARGET pcre2-8::pcre2-8)
  message(STATUS "PCRE2 target already defined.")
  return()
endif()

find_package(PkgConfig REQUIRED)
pkg_check_modules(PCRE2_8 QUIET libpcre2-8)
if(PCRE2_8_FOUND)
  add_library(pcre2-8::pcre2-8 INTERFACE IMPORTED)
  set_property(TARGET pcre2-8::pcre2-8 PROPERTY INTERFACE_INCLUDE_DIRECTORIES
               "${PCRE2_8_INCLUDE_DIRS}")
  set_property(TARGET pcre2-8::pcre2-8 PROPERTY INTERFACE_LINK_LIBRARIES
               "${PCRE2_8_LDFLAGS}")
  set_property(TARGET pcre2-8::pcre2-8 PROPERTY INTERFACE_COMPILE_DEFINITIONS
               "PCRE2_CODE_UNIT_WIDTH=8")
  set(pcre2_FOUND TRUE)
  message(STATUS "Found PCRE2 via pkg-config.")
  return()
endif()

if(pcre2_FIND_REQUIRED)
  message(FATAL_ERROR "Failed to find PCRE2.")
elseif(NOT pcre2_FIND_QUIETLY)
  message(WARNING "Failed to find PCRE2.")
endif()
