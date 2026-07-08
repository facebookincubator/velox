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
#
# - Try to find lz4
# Once done, this will define
#
# LZ4_FOUND - system has Glog
# LZ4_INCLUDE_DIRS - deprecated
# LZ4_LIBRARIES -  deprecated
# lz4::lz4 will be defined based on CMAKE_FIND_LIBRARY_SUFFIXES priority

include(FindPackageHandleStandardArgs)
include(SelectLibraryConfigurations)

find_library(LZ4_LIBRARY_RELEASE lz4 PATHS ${LZ4_LIBRARYDIR})
find_library(LZ4_LIBRARY_DEBUG lz4d PATHS ${LZ4_LIBRARYDIR})

find_path(LZ4_INCLUDE_DIR lz4.h PATHS ${LZ4_INCLUDEDIR})

select_library_configurations(LZ4)

find_package_handle_standard_args(lz4 DEFAULT_MSG LZ4_LIBRARY LZ4_INCLUDE_DIR)

mark_as_advanced(LZ4_LIBRARY LZ4_INCLUDE_DIR)

# Handle case where LZ4_LIBRARY might be a list of optimized/debug libraries (vcpkg)
if(LZ4_LIBRARY_RELEASE)
  get_filename_component(liblz4_ext ${LZ4_LIBRARY_RELEASE} EXT)
elseif(LZ4_LIBRARY_DEBUG)
  get_filename_component(liblz4_ext ${LZ4_LIBRARY_DEBUG} EXT)
elseif(LZ4_LIBRARY)
  # LZ4_LIBRARY might be a list with generator expressions, extract just the first actual path
  if(LZ4_LIBRARY MATCHES "optimized;([^;]+)")
    get_filename_component(liblz4_ext "${CMAKE_MATCH_1}" EXT)
  elseif(LZ4_LIBRARY MATCHES "debug;([^;]+)")
    get_filename_component(liblz4_ext "${CMAKE_MATCH_1}" EXT)
  else()
    # Single library path
    get_filename_component(liblz4_ext ${LZ4_LIBRARY} EXT)
  endif()
else()
  if(MSVC)
    set(liblz4_ext ".lib")
  else()
    set(liblz4_ext ".so")
  endif()
endif()

if(liblz4_ext STREQUAL ".a" OR (WIN32 AND liblz4_ext STREQUAL ".lib"))
  set(liblz4_type STATIC)
else()
  set(liblz4_type SHARED)
endif()

if(NOT TARGET lz4::lz4)
  add_library(lz4::lz4 ${liblz4_type} IMPORTED)
  set_target_properties(lz4::lz4 PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${LZ4_INCLUDE_DIR}")
  set_target_properties(lz4::lz4 PROPERTIES IMPORTED_LINK_INTERFACE_LANGUAGES "C")
  # Set per-config locations for multi-config generators (Visual Studio).
  # LZ4_LIBRARIES from select_library_configurations is a mixed list
  # ("optimized;...;debug;...") that doesn't work as IMPORTED_LOCATION.
  # Always set a fallback IMPORTED_LOCATION for configs without per-config locations.
  if(LZ4_LIBRARY_RELEASE)
    set_target_properties(lz4::lz4 PROPERTIES
      IMPORTED_LOCATION "${LZ4_LIBRARY_RELEASE}"
      IMPORTED_LOCATION_RELEASE "${LZ4_LIBRARY_RELEASE}")
  endif()
  if(LZ4_LIBRARY_DEBUG)
    set_target_properties(lz4::lz4 PROPERTIES IMPORTED_LOCATION_DEBUG "${LZ4_LIBRARY_DEBUG}")
  endif()
  if(NOT LZ4_LIBRARY_RELEASE AND NOT LZ4_LIBRARY_DEBUG)
    set_target_properties(lz4::lz4 PROPERTIES IMPORTED_LOCATION "${LZ4_LIBRARIES}")
  endif()
endif()
