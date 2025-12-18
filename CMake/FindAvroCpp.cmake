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

include(FindPackageHandleStandardArgs)

find_path(AVROCPP_INCLUDE_DIRS NAMES avro/DataFile.hh)

find_library(AVROCPP_SHARED_LIB NAMES avrocpp)
find_library(AVROCPP_STATIC_LIB NAMES avrocpp_s)

if(AVROCPP_SHARED_LIB)
  set(AVROCPP_LIBRARIES "${AVROCPP_SHARED_LIB}")
  set(_AVROCPP_LIBTYPE SHARED)
elseif(AVROCPP_STATIC_LIB)
  set(AVROCPP_LIBRARIES "${AVROCPP_STATIC_LIB}")
  set(_AVROCPP_LIBTYPE STATIC)
endif()

find_package_handle_standard_args(AvroCpp REQUIRED_VARS AVROCPP_INCLUDE_DIRS AVROCPP_LIBRARIES)

if(AvroCpp_FOUND)
  if(NOT TARGET AvroCpp::avrocpp)
    add_library(AvroCpp::avrocpp ${_AVROCPP_LIBTYPE} IMPORTED)
    set_target_properties(
      AvroCpp::avrocpp
      PROPERTIES
        IMPORTED_LOCATION "${AVROCPP_LIBRARIES}"
        INTERFACE_INCLUDE_DIRECTORIES "${AVROCPP_INCLUDE_DIRS}"
    )
  endif()
endif()

mark_as_advanced(AVROCPP_INCLUDE_DIRS AVROCPP_SHARED_LIB AVROCPP_STATIC_LIB AVROCPP_LIBRARIES)
