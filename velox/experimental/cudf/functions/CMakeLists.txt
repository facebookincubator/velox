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

# Puts gpu_shadows/ ahead of the regular include path, so that nvcc compiling a
# Velox simple-function header resolves GPU-compatible stand-ins in place of
# headers that reach host-only dependencies. A target under the Velox root's
# include_directories(.) must add the path itself with
# target_include_directories(... BEFORE PRIVATE ...), or the real headers win.
#
# Some shadows redefine a real Velox name, such as bits::countBits, so a
# shadowed entity must not cross between shadow-compiled .cu and real-Velox
# .cpp translation units. Pass POD types, such as cudf::column_view and
# primitive scalars, instead.
#
# TODO: Wrap each shadow in an inline namespace so such a mismatch fails at
# link time.
add_library(velox_cudf_gpu_shadows INTERFACE)

target_include_directories(
  velox_cudf_gpu_shadows
  BEFORE
  INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/gpu_shadows
)

# Device side of the simple-function bridge: the only target with the shadow
# include path. It exposes only cudf, rmm and standard library types, so no
# shadowed type reaches a real-Velox translation unit.
set(VELOX_CUDF_GPU_SIMPLE_FUNCTION_SOURCES GpuPrestoFunctions.cu)
if(${VELOX_ENABLE_SPARK_FUNCTIONS})
  list(APPEND VELOX_CUDF_GPU_SIMPLE_FUNCTION_SOURCES GpuSparkFunctions.cu)
endif()
add_library(velox_cudf_gpu_simple_functions ${VELOX_CUDF_GPU_SIMPLE_FUNCTION_SOURCES})

target_include_directories(
  velox_cudf_gpu_simple_functions
  BEFORE
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/gpu_shadows
)

# --diag-error=20011 makes calling a __host__ function from a __host__
# __device__ one an error. nvcc otherwise only warns when the call sits in a
# template instantiated for both, and the kernel computes garbage.
target_compile_options(
  velox_cudf_gpu_simple_functions
  PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    $<$<COMPILE_LANGUAGE:CUDA>:--diag-error=20011>
)

# Whole-program device compilation: every kernel is instantiated in the
# translation unit that launches it, and the host reaches them only through
# GpuLaunchFn. Separable compilation would add a device-link step that each
# consumer would have to repeat.
set_target_properties(
  velox_cudf_gpu_simple_functions
  PROPERTIES CUDA_SEPARABLE_COMPILATION OFF POSITION_INDEPENDENT_CODE ON
)

# Host half of the session time zone, where tz::TimeZone and QueryConfig are
# reachable; initialize() calls it through the POD-only GpuTimeZone.h.
add_library(velox_cudf_gpu_time_zone GpuTimeZone.cpp)

target_link_libraries(velox_cudf_gpu_time_zone PRIVATE cudf::cudf velox_core velox_type_tz)

target_link_libraries(
  velox_cudf_gpu_simple_functions
  PRIVATE cudf::cudf velox_cudf_gpu_shadows velox_cudf_gpu_time_zone
)

# Host side, without the shadow include path, so it sees real Velox headers. It
# reaches the device code above only through GpuLaunchFn, a function pointer
# over cudf and rmm types.
add_library(velox_cudf_gpu_function_bridge GpuFunctionRegistry.cpp GpuSfiExpression.cpp)

# cudf is PUBLIC because GpuFunctionRegistry.h, which consumers include, names
# cudf types in GpuLaunchFn.
target_link_libraries(
  velox_cudf_gpu_function_bridge
  PUBLIC velox_cudf_gpu_simple_functions cudf::cudf
  PRIVATE velox_cudf_expression velox_cudf_expression_registry
)
