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

# GPU function registry and dispatch (host-side, compiled with g++)
add_library(
  velox_cudf_gpu_registry
  GpuFunctionRegistry.cpp
  CudfFallbackFunction.cpp
  GpuFunctionDispatch.cpp
)

target_link_libraries(
  velox_cudf_gpu_registry
  PUBLIC
  cudf::cudf
)
