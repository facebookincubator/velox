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
