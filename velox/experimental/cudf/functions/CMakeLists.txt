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

# Header-only interface library that provides the gpu_shadows/ include
# path to any target that links against it.

add_library(velox_cudf_gpu_shadows INTERFACE)

target_include_directories(
  velox_cudf_gpu_shadows
  BEFORE
  INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/gpu_shadows
)

# GPU function registry, dispatch, and expression evaluator (host-side)
add_library(
  velox_cudf_gpu_registry
  GpuFunctionRegistry.cpp
  CudfFallbackFunction.cpp
  GpuFunctionDispatch.cpp
  GpuExprEvaluator.cpp
)

target_link_libraries(
  velox_cudf_gpu_registry
  PUBLIC
  cudf::cudf
)

# Presto function registrations (CUDA, compiled with nvcc)
add_library(velox_cudf_gpu_presto_functions GpuPrestoFunctions.cu)

set_target_properties(
  velox_cudf_gpu_presto_functions PROPERTIES
  CUDA_STANDARD 20
  CUDA_SEPARABLE_COMPILATION ON
)

target_include_directories(
  velox_cudf_gpu_presto_functions BEFORE PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/gpu_shadows
)

target_compile_options(
  velox_cudf_gpu_presto_functions PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)

target_link_libraries(
  velox_cudf_gpu_presto_functions
  PUBLIC
  velox_cudf_gpu_registry
  cudf::cudf
)
