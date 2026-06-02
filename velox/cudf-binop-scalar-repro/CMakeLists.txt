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

# cuDF requires CMake >= 3.30.4; Velox currently allows 3.28+ at the root.
cmake_minimum_required(VERSION 3.30.4)

# Standalone: cmake -S . -B build -DCMAKE_PREFIX_PATH=/opt/velox-build/release
# In-tree:   cmake --build <velox-build> --target cudf_nullmax_scalar_repro
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(cudf_binop_scalar_repro LANGUAGES CXX CUDA)

  set(CMAKE_CXX_STANDARD 20)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  cmake_policy(SET CMP0104 NEW)

  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    message(
      FATAL_ERROR
      "Standalone configure requires -DCMAKE_CUDA_ARCHITECTURES=... (e.g. 80)"
    )
  endif()

  find_package(CUDAToolkit REQUIRED)
  find_package(cudf REQUIRED)
endif()

if(NOT TARGET cudf::cudf)
  message(
    FATAL_ERROR
    "cudf::cudf not found. Build from the Velox tree with VELOX_ENABLE_CUDF=ON, "
    "or pass -DCMAKE_PREFIX_PATH=<velox-build-dir> for a standalone configure."
  )
endif()

add_executable(cudf_nullmax_scalar_repro main.cpp)

set_target_properties(
  cudf_nullmax_scalar_repro
  PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
             CUDA_STANDARD 20
             CUDA_STANDARD_REQUIRED ON
)

target_link_libraries(
  cudf_nullmax_scalar_repro
  PRIVATE cudf::cudf CUDA::cudart
)

target_compile_options(
  cudf_nullmax_scalar_repro
  PRIVATE -Wno-missing-field-initializers
)
