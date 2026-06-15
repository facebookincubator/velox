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

velox_add_library(
  velox_experimental_cxl
  CxlMemoryAllocator.cpp
  CxlMemoryResource.cpp
  HEADERS
  CxlMemoryAllocator.h
  CxlMemoryResource.h
)

# libnuma backs CxlMemoryAllocator's page binding (numa_tonode_memory). The
# parent CMakeLists only builds this subdirectory on Linux.
# TODO: Promote this to CMake/FindNuma.cmake and add libnuma-dev / numactl-devel
# to the setup scripts.
find_library(NUMA_LIBRARY numa)
if(NOT NUMA_LIBRARY)
  message(
    FATAL_ERROR
    "libnuma not found; install libnuma-dev (Debian/Ubuntu) or "
    "numactl-devel (RHEL/CentOS)."
  )
endif()

# Route the numa link through velox_link_libraries: with VELOX_MONO_LIBRARY the
# velox_experimental_cxl target is an alias to velox, on which a raw
# target_link_libraries() is rejected.
velox_link_libraries(velox_experimental_cxl velox_memory ${NUMA_LIBRARY})

# CxlHashAggregation pulls in the exec layer (HashAggregation, DriverAdapter),
# so it lives in its own target to keep the allocator/resource library free of
# a velox_exec dependency.
velox_add_library(
  velox_experimental_cxl_exec
  CxlHashTable.cpp
  CxlHashAggregation.cpp
  HEADERS
  CxlHashTable.h
  CxlHashAggregation.h
  CxlRelocationMap.h
)
velox_link_libraries(velox_experimental_cxl_exec velox_experimental_cxl velox_exec)

if(VELOX_BUILD_TESTING)
  add_subdirectory(tests)
endif()

if(${VELOX_ENABLE_BENCHMARKS})
  add_subdirectory(benchmarks)
endif()
