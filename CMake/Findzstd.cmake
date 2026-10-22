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

# Prefer zstd's own CMake package. It defines zstd::libzstd_shared and
# zstd::libzstd_static, which other packages such as Arrow link to.
#
# This creates a separate scope so any changed variables don't affect
# the rest of the build.
block(PROPAGATE zstd_FOUND zstd_VERSION)
  set(find_package_args)
  if(zstd_FIND_VERSION)
    list(APPEND find_package_args ${zstd_FIND_VERSION})
  endif()
  # Neither REQUIRED nor a "not found" message is wanted because we fall back to
  # find_library() below.
  list(APPEND find_package_args QUIET)
  find_package(zstd ${find_package_args} CONFIG)
  # A package that only redirects find_package(zstd), such as one created by
  # CPM, may define no targets.
  if(NOT TARGET zstd::libzstd_shared AND NOT TARGET zstd::libzstd_static)
    set(zstd_FOUND FALSE)
  endif()
endblock()

if(zstd_FOUND)
  if(NOT TARGET zstd::zstd)
    # Prefer the shared library like find_library() below.
    if(TARGET zstd::libzstd_shared)
      add_library(zstd::zstd ALIAS zstd::libzstd_shared)
    else()
      add_library(zstd::zstd ALIAS zstd::libzstd_static)
    endif()
  endif()
  return()
endif()

include(FindPackageHandleStandardArgs)
include(SelectLibraryConfigurations)

find_library(ZSTD_LIBRARY_RELEASE zstd PATHS ${ZSTD_LIBRARYDIR})
find_library(ZSTD_LIBRARY_DEBUG zstdd PATHS ${ZSTD_LIBRARYDIR})

find_path(ZSTD_INCLUDE_DIR zstd.h PATHS ${ZSTD_INCLUDEDIR})

select_library_configurations(ZSTD)

find_package_handle_standard_args(zstd DEFAULT_MSG ZSTD_LIBRARY ZSTD_INCLUDE_DIR)

mark_as_advanced(ZSTD_LIBRARY ZSTD_INCLUDE_DIR)

get_filename_component(libzstd_ext ${ZSTD_LIBRARY} EXT)
if(libzstd_ext STREQUAL ".a")
  set(libzstd_type STATIC)
else()
  set(libzstd_type SHARED)
endif()

if(NOT TARGET zstd::zstd)
  add_library(zstd::zstd ${libzstd_type} IMPORTED)
  set_target_properties(zstd::zstd PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${ZSTD_INCLUDE_DIR}")
  set_target_properties(
    zstd::zstd
    PROPERTIES IMPORTED_LINK_INTERFACE_LANGUAGES "C" IMPORTED_LOCATION "${ZSTD_LIBRARIES}"
  )
endif()
