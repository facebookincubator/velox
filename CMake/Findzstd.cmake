# Copyright (c) Facebook, Inc. and its affiliates.
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

find_package_handle_standard_args(zstd DEFAULT_MSG ZSTD_LIBRARY
                                  ZSTD_INCLUDE_DIR)

mark_as_advanced(ZSTD_LIBRARY ZSTD_INCLUDE_DIR)

get_filename_component(libzstd_ext ${ZSTD_LIBRARY} EXT)
if(libzstd_ext STREQUAL ".a")
  set(libzstd_type STATIC)
else()
  set(libzstd_type SHARED)
endif()

if(NOT TARGET zstd::zstd)
  add_library(zstd::zstd ${libzstd_type} IMPORTED)
  set_target_properties(zstd::zstd PROPERTIES INTERFACE_INCLUDE_DIRECTORIES
                                              "${ZSTD_INCLUDE_DIR}")
  set_target_properties(
    zstd::zstd PROPERTIES IMPORTED_LINK_INTERFACE_LANGUAGES "C"
                          IMPORTED_LOCATION "${ZSTD_LIBRARIES}")
endif()
