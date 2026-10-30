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
include_guard(GLOBAL)

if(VELOX_ENABLE_FAISS_GPU)
  # Released faiss builds against cuVS 26.06 at most. This main commit builds
  # against the 26.1x cuVS that matches cudf.cmake.
  # faiss commit d4a6548 from 2026-10-27 (main branch)
  set(VELOX_FAISS_BUILD_VERSION d4a6548c9a354bee366c14f39409d399f75c497f)
  set(
    VELOX_FAISS_BUILD_SHA256_CHECKSUM
    4c6d2355a7ebf777a4c876e803696117946fe4ae72105fa8b4bccff3fe30a589
  )
  set(
    VELOX_FAISS_SOURCE_URL
    "https://github.com/facebookresearch/faiss/archive/${VELOX_FAISS_BUILD_VERSION}.tar.gz"
  )
else()
  set(VELOX_FAISS_BUILD_VERSION 1.11.0)
  set(
    VELOX_FAISS_BUILD_SHA256_CHECKSUM
    c5d517da6deb6a6d74290d7145331fc7474426025e2d826fa4a6d40670f4493c
  )
  set(
    VELOX_FAISS_SOURCE_URL
    "https://github.com/facebookresearch/faiss/archive/refs/tags/v${VELOX_FAISS_BUILD_VERSION}.tar.gz"
  )
endif()

velox_resolve_dependency_url(FAISS)

# We need these hints for macos to build.
if(CMAKE_SYSTEM_NAME MATCHES "Darwin")
  message(STATUS "Detected Apple platform")
  execute_process(
    COMMAND brew --prefix libomp
    RESULT_VARIABLE BREW_LIBOMP_RESULT
    OUTPUT_VARIABLE BREW_LIBOMP_PREFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  if(BREW_LIBOMP_RESULT EQUAL 0 AND EXISTS "${BREW_LIBOMP_PREFIX}")
    list(APPEND CMAKE_PREFIX_PATH "${BREW_LIBOMP_PREFIX}")
  endif()

  execute_process(
    COMMAND brew --prefix openblas
    RESULT_VARIABLE BREW_OPENBLAS_RESULT
    OUTPUT_VARIABLE BREW_OPENBLAS_PREFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  if(BREW_OPENBLAS_RESULT EQUAL 0 AND EXISTS "${BREW_OPENBLAS_PREFIX}")
    list(APPEND CMAKE_PREFIX_PATH "${BREW_OPENBLAS_PREFIX}")
  endif()
endif()

FetchContent_Declare(
  faiss
  URL ${VELOX_FAISS_SOURCE_URL}
  URL_HASH ${VELOX_FAISS_BUILD_SHA256_CHECKSUM}
  BINARY_DIR
  ${FETCHCONTENT_BASE_DIR}/faiss-build
  SYSTEM
  EXCLUDE_FROM_ALL
)

if(VELOX_ENABLE_FAISS_GPU)
  # faiss downloads its own RAPIDS.cmake unless this file exists. Point it at
  # the rapids-cmake that cuDF already populated so all RAPIDS packages agree.
  FetchContent_GetProperties(rapids-cmake SOURCE_DIR VELOX_RAPIDS_CMAKE_SOURCE_DIR)
  if(NOT VELOX_RAPIDS_CMAKE_SOURCE_DIR)
    message(FATAL_ERROR "GPU faiss must be resolved after cuDF.")
  endif()
  file(
    WRITE
    ${FETCHCONTENT_BASE_DIR}/faiss-build/FAISS_RAPIDS.cmake
    "include(\"${VELOX_RAPIDS_CMAKE_SOURCE_DIR}/RAPIDS.cmake\")\n"
  )
  # faiss/gpu links OpenMP::OpenMP_CXX, which faiss only creates in a sibling
  # directory. An installed cuVS package would create it at the top level.
  find_package(OpenMP REQUIRED COMPONENTS CXX)
endif()

# Set build options
block()
  set(BUILD_SHARED_LIBS OFF)
  set(BUILD_TESTING OFF)
  set(CMAKE_BUILD_TYPE Release)
  set(FAISS_ENABLE_PYTHON OFF)
  set(FAISS_ENABLE_GPU_TESTS OFF)
  set(FAISS_ENABLE_C_API OFF)
  set(FAISS_ENABLE_EXTRAS OFF)
  if(VELOX_ENABLE_FAISS_GPU)
    set(FAISS_ENABLE_GPU ON)
    set(FAISS_ENABLE_CUVS ON)
    set(FAISS_ENABLE_ROCM OFF)
  else()
    set(FAISS_ENABLE_GPU OFF)
  endif()
  # Make FAISS available
  FetchContent_MakeAvailable(faiss)
  add_library(FAISS::faiss ALIAS faiss)
  unset(BUILD_TESTING CACHE)
  unset(BUILD_SHARED_LIBS CACHE)
endblock()
