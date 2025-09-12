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

find_package(PkgConfig REQUIRED)

pkg_search_module(Roaring_PC QUIET roaring)

find_library(Roaring_LIBRARY NAMES roaring HINTS ${Roaring_PC_LIBDIR} ${Roaring_PC_LIBRARY_DIRS})

find_path(
  Roaring_INCLUDE_DIR
  NAMES "roaring/roaring.h"
  HINTS ${Roaring_PC_INCLUDEDIR}/include ${Roaring_PC_INCLUDE_DIRS}
)

mark_as_advanced(Roaring_LIBRARY Roaring_INCLUDE_DIR)

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(
  Roaring
  REQUIRED_VARS Roaring_LIBRARY Roaring_INCLUDE_DIR
  VERSION_VAR Roaring_PC_VERSION
)

set(Roaring_LIBRARIES ${Roaring_LIBRARY})
set(Roaring_INCLUDE_DIRS ${Roaring_INCLUDE_DIR})

if(Roaring_FOUND AND NOT (TARGET Roaring::roaring))
  add_library(Roaring::roaring UNKNOWN IMPORTED)

  set_target_properties(
    Roaring::roaring
    PROPERTIES
      IMPORTED_LOCATION ${Roaring_LIBRARY}
      INTERFACE_INCLUDE_DIRECTORIES ${Roaring_INCLUDE_DIRS}
  )
endif()
