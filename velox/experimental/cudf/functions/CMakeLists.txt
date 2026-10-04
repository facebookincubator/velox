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

# Compiles the CUDA sources of a target against the GPU shadows: everything
# such a target needs, so that no consumer repeats it.
#
# Puts gpu_shadows/ ahead of the Velox source root, so that nvcc compiling a
# Velox simple-function header finds a GPU-compatible stand-in where one exists
# and the real header otherwise. The shadow wins because it comes first in the
# include search, so the real header is never opened. Every Velox target
# inherits the root's include_directories(.), so the path is added BEFORE, on
# the target itself; an INTERFACE target cannot do that.
#
# --expt-relaxed-constexpr lets device code call constexpr functions that carry
# no annotation, such as the Exceptions.h shadow's useArgs and the masks in the
# BitUtil.h shadow. --diag-error=20011 makes a call from a __host__ __device__
# function into a __host__ one an error. nvcc otherwise only warns, for a
# template and a plain function alike, and the kernel computes garbage.
#
# Some shadows redefine a real Velox name, such as bits::countBits, so a
# shadowed entity must not cross between shadow-compiled .cu and real-Velox
# .cpp translation units. Pass POD types, such as cudf::column_view and
# primitive scalars, instead.
#
# TODO: Wrap each shadow in an inline namespace so such a mismatch fails at
# link time.
function(velox_cudf_use_gpu_shadows target)
  target_include_directories(
    ${target}
    BEFORE
    PRIVATE ${CMAKE_CURRENT_FUNCTION_LIST_DIR}/gpu_shadows
  )
  target_compile_options(
    ${target}
    PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
      $<$<COMPILE_LANGUAGE:CUDA>:--diag-error=20011>
  )
endfunction()

# Device side of the simple-function bridge: the only target with the shadow
# include path. It exposes only cudf, rmm and standard library types, so no
# shadowed type reaches a real-Velox translation unit.
set(VELOX_CUDF_GPU_SIMPLE_FUNCTION_SOURCES GpuPrestoFunctions.cu)
if(${VELOX_ENABLE_SPARK_FUNCTIONS})
  list(APPEND VELOX_CUDF_GPU_SIMPLE_FUNCTION_SOURCES GpuSparkFunctions.cu)
endif()
add_library(velox_cudf_gpu_simple_functions ${VELOX_CUDF_GPU_SIMPLE_FUNCTION_SOURCES})

velox_cudf_use_gpu_shadows(velox_cudf_gpu_simple_functions)

# Whole-program device compilation: every kernel is instantiated in the
# translation unit that launches it, and the host reaches them only through
# GpuLaunchFn. Separable compilation would add a device-link step that each
# consumer would have to repeat.
set_target_properties(
  velox_cudf_gpu_simple_functions
  PROPERTIES CUDA_SEPARABLE_COMPILATION OFF POSITION_INDEPENDENT_CODE ON
)

# Host half of the shadow-compiled functions, compiled against real Velox:
# what initialize() and a check failing on the host reach that needs
# tz::TimeZone, QueryConfig, the unit parser or VeloxException. The shadow
# side calls it through the POD-only GpuTimeZone.h, GpuTimeZoneDatabase.h,
# GpuDateTimeUnit.h and GpuCheckFailure.h.
add_library(
  velox_cudf_gpu_sfi_host
  GpuCheckFailure.cpp
  GpuDateTimeUnit.cpp
  GpuTimeZone.cpp
  GpuTimeZoneDatabase.cpp
)

target_link_libraries(
  velox_cudf_gpu_sfi_host
  PRIVATE cudf::cudf velox_core velox_exception velox_functions_lib velox_type_tz
)

# velox_type_tz is named directly: the shadow side links to the real string
# overloads of tz::getTimeZoneID().
target_link_libraries(
  velox_cudf_gpu_simple_functions
  PRIVATE cudf::cudf velox_cudf_gpu_sfi_host velox_type_tz
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
