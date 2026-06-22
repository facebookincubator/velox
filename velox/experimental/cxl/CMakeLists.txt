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

# Provides the libnuma-backed allocator and resource that register the CXL
# memory tier. The relocate path is compiled into core velox_exec under
# VELOX_ENABLE_CXL and selected at runtime by a customPool("cxl") tier; the tag
# constant (CxlResourceTag.h) is shared with core.
velox_add_library(
  velox_experimental_cxl
  CxlMemoryAllocator.cpp
  CxlMemoryResource.cpp
  HEADERS
  CxlMemoryAllocator.h
  CxlMemoryResource.h
  CxlResourceTag.h
)

# libnuma backs CxlMemoryAllocator's page binding (numa_tonode_memory). The
# parent CMakeLists only builds this subdirectory on Linux, and the setup
# scripts install libnuma-dev (Debian/Ubuntu) / numactl-devel (RHEL/CentOS).
# TODO: Promote this to CMake/FindNuma.cmake.
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

if(VELOX_BUILD_TESTING)
  add_subdirectory(tests)
endif()

if(${VELOX_ENABLE_BENCHMARKS})
  add_subdirectory(benchmarks)
endif()
