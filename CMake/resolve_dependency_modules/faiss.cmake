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

# Prevents this CMake file from being included more than once in the same build, avoiding duplicate definitions.
include_guard(GLOBAL)

# FAISS Configuration
# Defines the version and SHA256 checksum for the FAISS source archive to ensure integrity.
set(VELOX_FAISS_BUILD_VERSION 1.11.0)
set(VELOX_FAISS_BUILD_SHA256_CHECKSUM
    c5d517da6deb6a6d74290d7145331fc7474426025e2d826fa4a6d40670f4493c)

# Builds the URL to download the FAISS source tarball for the specified version.
string(CONCAT VELOX_FAISS_SOURCE_URL
    "https://github.com/facebookresearch/faiss/archive/refs/tags/v${VELOX_FAISS_BUILD_VERSION}.tar.gz")

# Resolve the dependency URL for FAISS
velox_resolve_dependency_url(FAISS)

# Only apply patches if they exist
set(_faiss_patch_command "")
if(EXISTS "${CMAKE_CURRENT_LIST_DIR}/faiss/faiss-cmakelists.patch")
  list(APPEND _faiss_patch_command git apply "${CMAKE_CURRENT_LIST_DIR}/faiss/faiss-cmakelists.patch")
endif()
if(EXISTS "${CMAKE_CURRENT_LIST_DIR}/faiss/faiss-build.patch")
  list(APPEND _faiss_patch_command git apply "${CMAKE_CURRENT_LIST_DIR}/faiss/faiss-build.patch")
endif()

# Enable OpenMP for FAISS
message(STATUS "APPLE is: ${APPLE}")

set(FAISS_ENABLE_OPENMP ON)
# Use HOMEBREW_PREFIX environment variable for libomp paths
if(DEFINED ENV{HOMEBREW_PREFIX})
  set(HOMEBREW_PREFIX $ENV{HOMEBREW_PREFIX})
else()
  message(FATAL_ERROR "HOMEBREW_PREFIX environment variable is not set!")
endif()
# Set architecture-specific flags if needed
set(ARCH_FLAGS "")
if(DEFINED ENV{CPU_TARGET} AND "$ENV{CPU_TARGET}" STREQUAL "arm64")
  message(STATUS "Detected Apple Silicon (arm64) via CPU_TARGET")
  set(ARCH_FLAGS "-mcpu=apple-m1+crc")
else()
  message(STATUS "Detected non-arm64 architecture via CPU_TARGET")
endif()
# Only apply these flags on Apple platforms
if(APPLE)
  string(APPEND CMAKE_CXX_FLAGS
    " ${ARCH_FLAGS} -isystem ${HOMEBREW_PREFIX}/include -Xpreprocessor -fopenmp -I${HOMEBREW_PREFIX}/opt/libomp/include"
  )
  string(APPEND CMAKE_EXE_LINKER_FLAGS
    " -L${HOMEBREW_PREFIX}/opt/libomp/lib -lomp"
  )
  # Print the values of the OpenMP variables
  message("++OpenMP_CXX_FLAGS: ${OpenMP_CXX_FLAGS}")
  message("++OpenMP_CXX_LIB_NAMES: ${OpenMP_CXX_LIB_NAMES}")
  message("++OpenMP_omp_LIBRARY: ${OpenMP_omp_LIBRARY}")
endif()
find_package(OpenMP REQUIRED)


# Tells CMake to fetch FAISS from the specified URL, verify its checksum, and apply any patches if needed.
FetchContent_Declare(
  faiss
  URL ${VELOX_FAISS_SOURCE_URL}
  URL_HASH ${VELOX_FAISS_BUILD_SHA256_CHECKSUM}
  PATCH_COMMAND ${_faiss_patch_command}
)

# Append FAISS cmake module path
list(APPEND CMAKE_MODULE_PATH "${faiss_SOURCE_DIR}/cmake")

# Set build options
set(BUILD_SHARED_LIBS ${VELOX_BUILD_SHARED})
set(CMAKE_BUILD_TYPE Release)
set(PREVIOUS_CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -Wno-nonnull ")

# Disables developer-specific warnings in FAISS.
set(FAISS_BUILD_DEVELOPER OFF)

# Add specific compiler flags for GNU
if("${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -Wno-dangling-pointer")
endif()

# Make FAISS available
FetchContent_MakeAvailable(faiss)

# Create an alias for the FAISS library
add_library(FAISS::faiss ALIAS faiss)

# Reset build options
unset(BUILD_SHARED_LIBS)
set(CMAKE_CXX_FLAGS ${PREVIOUS_CMAKE_CXX_FLAGS})
set(CMAKE_BUILD_TYPE ${PREVIOUS_BUILD_TYPE})
