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
# path to any target that links against it. The shadow path is added
# BEFORE the regular include search order so that, when nvcc compiles a
# Velox simple function header, lightweight GPU-compatible stubs resolve
# in place of the real Velox / Folly headers that pull in host-only
# dependencies.
#
# Consumers add the shadow path via `target_include_directories(... BEFORE
# PRIVATE ...)` rather than relying on INTERFACE inheritance, because the
# directory-level `include_directories(.)` in the velox root would otherwise win
# the ordering and the real headers would resolve first.
add_library(velox_cudf_gpu_shadows INTERFACE)

target_include_directories(
  velox_cudf_gpu_shadows
  BEFORE
  INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/gpu_shadows
)

# Device side of the simple-function bridge. Isolated in its own target because
# it is the only place the shadow include path is active: inside it, names such
# as StringView resolve to GPU substitutes with different layouts. Its exported
# surface is GpuSimpleFunctionLaunch.h, which names only cudf, rmm, and standard
# library types so that nothing shadowed reaches a real-Velox translation unit.
set(VELOX_CUDF_GPU_SIMPLE_FUNCTION_SOURCES GpuPrestoFunctions.cu)
if(${VELOX_ENABLE_SPARK_FUNCTIONS})
  list(APPEND VELOX_CUDF_GPU_SIMPLE_FUNCTION_SOURCES GpuSparkFunctions.cu)
endif()
add_library(
  velox_cudf_gpu_simple_functions ${VELOX_CUDF_GPU_SIMPLE_FUNCTION_SOURCES}
)

target_include_directories(
  velox_cudf_gpu_simple_functions
  BEFORE
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/gpu_shadows
)

# --diag-error=20011 promotes "calling a __host__ function from a __host__
# __device__ function" from a warning to an error. nvcc only warns when the
# call sits in a template instantiated for both, and then emits a kernel that
# silently computes garbage rather than failing to launch -- so without this,
# a simple function whose body reaches a host-only helper compiles clean and
# returns wrong answers. Every diagnosis of that failure mode has cost hours;
# the compiler can just tell us.
target_compile_options(
  velox_cudf_gpu_simple_functions
  PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    $<$<COMPILE_LANGUAGE:CUDA>:--diag-error=20011>
)

# Whole-program device compilation. The kernels live in
# GpuSimpleFunctionAdapter.cuh and are instantiated here, so no device function
# in this library calls one in another translation unit -- the host/device
# boundary is GpuLaunchFn, an ordinary function pointer. Separable compilation
# would only add a device-link step, and having two consumers each perform it
# left GpuPrestoFunctions' __cudaRegisterLinkedBinary unresolved at link time.
set_target_properties(
  velox_cudf_gpu_simple_functions
  PROPERTIES CUDA_SEPARABLE_COMPILATION OFF POSITION_INDEPENDENT_CODE ON
)

target_link_libraries(velox_cudf_gpu_simple_functions PRIVATE cudf::cudf velox_cudf_gpu_shadows)

# Host side. Deliberately a separate target with no shadow include path, so it
# sees real Velox headers and can name Velox types. It reaches the device code
# above only through GpuLaunchFn, a function pointer over cudf and rmm types.
# Keeping the two apart is what makes the boundary a build-level guarantee
# rather than a convention.
add_library(velox_cudf_gpu_function_bridge GpuFunctionRegistry.cpp GpuSfiExpression.cpp)

# cudf is PUBLIC because GpuFunctionRegistry.h -- a header consumers include --
# names cudf::column_view and cudf::data_type in GpuLaunchFn. Linking it
# privately compiled here but left every consumer without cudf's include path.
target_link_libraries(
  velox_cudf_gpu_function_bridge
  PUBLIC velox_cudf_gpu_simple_functions cudf::cudf
  PRIVATE velox_cudf_expression velox_cudf_expression_registry
)
