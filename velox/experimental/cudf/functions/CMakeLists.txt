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
# Currently no target in PR 1+2 consumes this INTERFACE library; the
# shadow compile test in tests/CMakeLists.txt adds the shadow path
# directly via `target_include_directories(... BEFORE PRIVATE ...)`
# because directory-level `include_directories(.)` in the velox root
# would otherwise win over INTERFACE inheritance. Future PRs
# (PR 3+ GpuSimpleFunctionAdapter, PR 7+ Presto function registrations)
# will link against this INTERFACE library from CUDA libraries that
# don't carry that ordering wrinkle.
add_library(velox_cudf_gpu_shadows INTERFACE)

target_include_directories(
  velox_cudf_gpu_shadows
  BEFORE
  INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/gpu_shadows
)
