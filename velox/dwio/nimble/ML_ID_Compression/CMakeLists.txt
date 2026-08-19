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
  openzl_cpp
  Folly::folly
  velox_memory
  gflags
)

# ---------------------------------------------------------------------------
# bench_decode_bulk: full-materialization decode throughput.
# ---------------------------------------------------------------------------
add_executable(nimble_mlidc_decode_bulk bench_decode_bulk.cpp)

target_link_libraries(
  nimble_mlidc_decode_bulk
  nimble_mlidc_common
  nimble_encodings
  nimble_encodings_tests_utils
  nimble_common
  openzl_cpp
  Folly::folly
  velox_memory
  gflags
)

# ---------------------------------------------------------------------------
# bench_decode_range: contiguous-range decode throughput heatmap.
# ---------------------------------------------------------------------------
add_executable(nimble_mlidc_decode_range bench_decode_range.cpp)

target_link_libraries(
  nimble_mlidc_decode_range
  nimble_mlidc_common
  nimble_encodings
  nimble_encodings_tests_utils
  nimble_common
  Folly::folly
  velox_memory
  gflags
)

# ---------------------------------------------------------------------------
# bench_encode: encode throughput per encoder/dataset pair.
# ---------------------------------------------------------------------------
add_executable(nimble_mlidc_encode bench_encode.cpp)

target_link_libraries(
  nimble_mlidc_encode
  nimble_mlidc_common
  nimble_encodings
  nimble_encodings_tests_utils
  nimble_common
  openzl_cpp
  Folly::folly
  velox_memory
  gflags
)

# ---------------------------------------------------------------------------
# bench_decode_point: single-probe point-lookup latency.
# ---------------------------------------------------------------------------
add_executable(nimble_mlidc_decode_point bench_decode_point.cpp)

target_link_libraries(
  nimble_mlidc_decode_point
  nimble_mlidc_common
  nimble_encodings
  nimble_encodings_tests_utils
  nimble_common
  Folly::folly
  velox_memory
  gflags
)

# ---------------------------------------------------------------------------
# bench_decode_gather: gather-pattern decode throughput.
# ---------------------------------------------------------------------------
add_executable(nimble_mlidc_decode_gather bench_decode_gather.cpp)

target_link_libraries(
  nimble_mlidc_decode_gather
  nimble_mlidc_common
  nimble_encodings
  nimble_encodings_tests_utils
  nimble_common
  Folly::folly
  velox_memory
  gflags
)

# ---------------------------------------------------------------------------
# bench_costmodel_oracle: validates SubIntSplit's DP cost model against an
# oracle that actually encodes each candidate bit range.
# ---------------------------------------------------------------------------
add_executable(nimble_mlidc_costmodel_oracle bench_costmodel_oracle.cpp)

target_link_libraries(
  nimble_mlidc_costmodel_oracle
  nimble_mlidc_common
  nimble_encodings
  nimble_encodings_tests_utils
  nimble_common
  Folly::folly
  velox_memory
  gflags
)

# ---------------------------------------------------------------------------
# bench_index_oracle: sweeps FPE index types, Pareto frontier, scalarised oracle.
# ---------------------------------------------------------------------------
add_executable(nimble_mlidc_index_oracle bench_index_oracle.cpp)

target_link_libraries(
  nimble_mlidc_index_oracle
  nimble_mlidc_common
  nimble_encodings
  nimble_encodings_tests_utils
  nimble_common
  Folly::folly
  velox_memory
  gflags
)

# ---------------------------------------------------------------------------
# bench_ablation: progressive encoding-set restriction for SIS sections.
# ---------------------------------------------------------------------------
add_executable(nimble_mlidc_ablation bench_ablation.cpp)

target_link_libraries(
  nimble_mlidc_ablation
  nimble_mlidc_common
  nimble_encodings
  nimble_encodings_tests_utils
  nimble_common
  Folly::folly
  velox_memory
  gflags
)

# ---------------------------------------------------------------------------
# bench_smoke: encode + decode correctness check (no timing).
# ---------------------------------------------------------------------------
add_executable(nimble_mlidc_smoke bench_smoke.cpp)

target_link_libraries(
  nimble_mlidc_smoke
  nimble_mlidc_common
  nimble_encodings
  nimble_encodings_tests_utils
  nimble_common
  Folly::folly
  velox_memory
  gflags
)
