# Copyright (c) Meta Platforms, Inc. and affiliates.
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

# Header-only helper shared by the per-directory benchmark binaries. It has no
# sources of its own; the target exists so the header is owned by the build
# graph rather than showing up as untracked.
velox_add_library(velox_dwio_nimble_benchmark_suite INTERFACE HEADERS BenchmarkSuite.h)
