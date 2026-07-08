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
# - Try to find lzo2
# Once done, this will define
#
# LZO2_FOUND - system has Glog
# LZO2_INCLUDE_DIRS - deprecated
# LZO2_LIBRARIES -  deprecated
# lzo2::lzo2 will be defined based on CMAKE_FIND_LIBRARY_SUFFIXES priority

include(FindPackageHandleStandardArgs)
include(SelectLibraryConfigurations)

find_library(LZO2_LIBRARY_RELEASE lzo2 PATHS $LZO2_LIBRARYDIR})
find_library(LZO2_LIBRARY_DEBUG lzo2d PATHS ${LZO2_LIBRARYDIR})

find_path(LZO2_INCLUDE_DIR lzo/lzo1a.h PATHS ${LZO2_INCLUDEDIR})

select_library_configurations(LZO2)

find_package_handle_standard_args(lzo2 DEFAULT_MSG LZO2_LIBRARY LZO2_INCLUDE_DIR)

mark_as_advanced(LZO2_LIBRARY LZO2_INCLUDE_DIR)

# Handle case where LZO2_LIBRARY might be a list of optimized/debug libraries (vcpkg)
if(LZO2_LIBRARY_RELEASE)
  get_filename_component(liblzo2_ext ${LZO2_LIBRARY_RELEASE} EXT)
elseif(LZO2_LIBRARY_DEBUG)
  get_filename_component(liblzo2_ext ${LZO2_LIBRARY_DEBUG} EXT)
elseif(LZO2_LIBRARY)
  # LZO2_LIBRARY might be a list with generator expressions, extract just the first actual path
  if(LZO2_LIBRARY MATCHES "optimized;([^;]+)")
    get_filename_component(liblzo2_ext "${CMAKE_MATCH_1}" EXT)
  elseif(LZO2_LIBRARY MATCHES "debug;([^;]+)")
    get_filename_component(liblzo2_ext "${CMAKE_MATCH_1}" EXT)
  else()
    # Single library path
    get_filename_component(liblzo2_ext ${LZO2_LIBRARY} EXT)
  endif()
else()
  if(MSVC)
    set(liblzo2_ext ".lib")
  else()
    set(liblzo2_ext ".so")
  endif()
endif()

if(liblzo2_ext STREQUAL ".a" OR (WIN32 AND liblzo2_ext STREQUAL ".lib"))
  set(liblzo2_type STATIC)
else()
  set(liblzo2_type SHARED)
endif()

if(NOT TARGET lzo2::lzo2)
  add_library(lzo2::lzo2 ${liblzo2_type} IMPORTED)
  set_target_properties(lzo2::lzo2 PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${LZO2_INCLUDE_DIR}")
  set_target_properties(lzo2::lzo2 PROPERTIES IMPORTED_LINK_INTERFACE_LANGUAGES "C")
  # Always set a fallback IMPORTED_LOCATION for configs without per-config locations.
  # On vcpkg, lzo2 debug lib keeps the same name (lzo2.lib, not lzo2d.lib), so
  # LZO2_LIBRARY_DEBUG may be empty. The fallback ensures Debug config still works.
  if(LZO2_LIBRARY_RELEASE)
    set_target_properties(lzo2::lzo2 PROPERTIES
      IMPORTED_LOCATION "${LZO2_LIBRARY_RELEASE}"
      IMPORTED_LOCATION_RELEASE "${LZO2_LIBRARY_RELEASE}")
  endif()
  if(LZO2_LIBRARY_DEBUG)
    set_target_properties(lzo2::lzo2 PROPERTIES IMPORTED_LOCATION_DEBUG "${LZO2_LIBRARY_DEBUG}")
  endif()
  if(NOT LZO2_LIBRARY_RELEASE AND NOT LZO2_LIBRARY_DEBUG)
    set_target_properties(lzo2::lzo2 PROPERTIES IMPORTED_LOCATION "${LZO2_LIBRARIES}")
  endif()
endif()
