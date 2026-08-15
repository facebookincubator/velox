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

# ---------------------------------------------------------------------------
# Shared library used by all drivers in this benchmark suite.
# Contains the gflag definitions (BenchCommonFlags.cpp) so that each driver
# links exactly one copy; headers are exposed via PUBLIC include dirs.
# ---------------------------------------------------------------------------
add_library(nimble_mlidc_common BenchCommonFlags.cpp)

target_include_directories(
  nimble_mlidc_common
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(
  nimble_mlidc_common
  PUBLIC
    gflags
)

# ---------------------------------------------------------------------------
# bench_compression: encodes a column of ML IDs under every candidate
# encoding and reports compressed size + round-trip latency.
# ---------------------------------------------------------------------------
add_executable(nimble_mlidc_compression bench_compression.cpp)

target_link_libraries(
  nimble_mlidc_compression
  nimble_mlidc_common
  nimble_encodings
  nimble_encodings_tests_utils
  nimble_common
  Folly::folly
  velox_memory
  gflags
)

# ---------------------------------------------------------------------------
# Future driver targets (placeholders)
# ---------------------------------------------------------------------------
# add_executable(nimble_mlidc_gather      bench_gather.cpp)
# target_link_libraries(nimble_mlidc_gather ...)
#
# add_executable(nimble_mlidc_skip        bench_skip.cpp)
# target_link_libraries(nimble_mlidc_skip ...)
