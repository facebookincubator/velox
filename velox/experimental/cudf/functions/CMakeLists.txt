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
