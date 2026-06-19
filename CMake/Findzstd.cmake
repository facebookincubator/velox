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
# - Try to find zstd
# Once done, this will define
#
# ZSTD_FOUND - system has Glog
# ZSTD_INCLUDE_DIRS - deprecated
# ZSTD_LIBRARIES -  deprecated
# zstd::zstd will be defined based on CMAKE_FIND_LIBRARY_SUFFIXES priority

include(FindPackageHandleStandardArgs)
include(SelectLibraryConfigurations)

find_library(ZSTD_LIBRARY_RELEASE zstd PATHS $ZSTD_LIBRARYDIR})
find_library(ZSTD_LIBRARY_DEBUG zstdd PATHS ${ZSTD_LIBRARYDIR})

find_path(ZSTD_INCLUDE_DIR zstd.h PATHS ${ZSTD_INCLUDEDIR})

select_library_configurations(ZSTD)

find_package_handle_standard_args(zstd DEFAULT_MSG ZSTD_LIBRARY ZSTD_INCLUDE_DIR)

mark_as_advanced(ZSTD_LIBRARY ZSTD_INCLUDE_DIR)

# Handle case where ZSTD_LIBRARY might be a list of optimized/debug libraries (vcpkg)
if(ZSTD_LIBRARY_RELEASE)
  get_filename_component(libzstd_ext ${ZSTD_LIBRARY_RELEASE} EXT)
elseif(ZSTD_LIBRARY_DEBUG)
  get_filename_component(libzstd_ext ${ZSTD_LIBRARY_DEBUG} EXT)
elseif(ZSTD_LIBRARY)
  # ZSTD_LIBRARY might be a list with generator expressions, extract just the first actual path
  if(ZSTD_LIBRARY MATCHES "optimized;([^;]+)")
    get_filename_component(libzstd_ext "${CMAKE_MATCH_1}" EXT)
  elseif(ZSTD_LIBRARY MATCHES "debug;([^;]+)")
    get_filename_component(libzstd_ext "${CMAKE_MATCH_1}" EXT)
  else()
    # Single library path
    get_filename_component(libzstd_ext ${ZSTD_LIBRARY} EXT)
  endif()
else()
  if(MSVC)
    set(libzstd_ext ".lib")
  else()
    set(libzstd_ext ".so")
  endif()
endif()

if(libzstd_ext STREQUAL ".a" OR (WIN32 AND libzstd_ext STREQUAL ".lib"))
  set(libzstd_type STATIC)
else()
  set(libzstd_type SHARED)
endif()

if(NOT TARGET zstd::zstd)
  add_library(zstd::zstd ${libzstd_type} IMPORTED)
  set_target_properties(zstd::zstd PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${ZSTD_INCLUDE_DIR}")
  set_target_properties(zstd::zstd PROPERTIES IMPORTED_LINK_INTERFACE_LANGUAGES "C")
  # Always set a fallback IMPORTED_LOCATION for configs without per-config locations.
  # On vcpkg, zstd debug lib keeps the same name (zstd.lib, not zstdd.lib), so
  # ZSTD_LIBRARY_DEBUG may be empty. The fallback ensures Debug config still works.
  if(ZSTD_LIBRARY_RELEASE)
    set_target_properties(zstd::zstd PROPERTIES
      IMPORTED_LOCATION "${ZSTD_LIBRARY_RELEASE}"
      IMPORTED_LOCATION_RELEASE "${ZSTD_LIBRARY_RELEASE}")
  endif()
  if(ZSTD_LIBRARY_DEBUG)
    set_target_properties(zstd::zstd PROPERTIES IMPORTED_LOCATION_DEBUG "${ZSTD_LIBRARY_DEBUG}")
  endif()
  if(NOT ZSTD_LIBRARY_RELEASE AND NOT ZSTD_LIBRARY_DEBUG)
    set_target_properties(zstd::zstd PROPERTIES IMPORTED_LOCATION "${ZSTD_LIBRARIES}")
  endif()
endif()
