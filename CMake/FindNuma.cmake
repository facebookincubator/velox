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
# - Try to find libnuma
# Once done, this will define
#
# Numa_FOUND - system has libnuma
# Numa::numa - imported target for the numa library and headers

include(FindPackageHandleStandardArgs)

find_library(NUMA_LIBRARY numa PATHS ${NUMA_LIBRARYDIR})
find_path(NUMA_INCLUDE_DIR numa.h PATHS ${NUMA_INCLUDEDIR})

find_package_handle_standard_args(Numa DEFAULT_MSG NUMA_LIBRARY NUMA_INCLUDE_DIR)

mark_as_advanced(NUMA_LIBRARY NUMA_INCLUDE_DIR)

if(Numa_FOUND AND NOT TARGET Numa::numa)
  add_library(Numa::numa UNKNOWN IMPORTED)
  set_target_properties(
    Numa::numa
    PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${NUMA_INCLUDE_DIR}"
      IMPORTED_LINK_INTERFACE_LANGUAGES "C"
      IMPORTED_LOCATION "${NUMA_LIBRARY}"
  )
endif()
