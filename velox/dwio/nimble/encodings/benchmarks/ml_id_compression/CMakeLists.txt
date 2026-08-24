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

# Benchmark drivers for ML ID columns. Every driver sweeps the same encoder set
# (see BenchCommon.h), which includes the experimental SubIntSplit and
# FrequencyPartition encodings, so this directory is entered only when
# NIMBLE_ENABLE_EXPERIMENTAL_ENCODINGS is on. The sources compile to a stub main
# without it, but there is no reason to build that.
#
# BenchCommon.h includes encodings/tests/TestUtils.h for test::Encoder, but does
# not link nimble_encodings_tests_utils: that target is only defined under
# NIMBLE_BUILD_TESTING, which does not follow the gate above. test::Encoder is a
# header-only template and the header pulls in nothing but encoding headers, so
# nimble_encodings alone satisfies it. The only out-of-line symbol in that
# library, TestUtils::getRawDataSize, is unused here.

# ---------------------------------------------------------------------------
# Shared library used by all drivers in this suite. Contains the gflag
# definitions (MlIdBenchmarkFlags.cpp) so that each driver links exactly one
# copy. The headers need no include directory of their own: they are included
# by their path from the repository root, which is already on every target's
# include path.
# ---------------------------------------------------------------------------
add_library(nimble_ml_id_benchmark_common MlIdBenchmarkFlags.cpp)

target_link_libraries(
  nimble_ml_id_benchmark_common
  PUBLIC
    gflags
)

# ---------------------------------------------------------------------------
# MlIdCompressionBenchmark: encodes a column of ML IDs under every candidate
# encoding and reports compressed size + round-trip latency.
# ---------------------------------------------------------------------------
add_executable(nimble_ml_id_compression_benchmark MlIdCompressionBenchmark.cpp)

target_link_libraries(
  nimble_ml_id_compression_benchmark
  nimble_ml_id_benchmark_common
  nimble_encodings
  nimble_common
  openzl_cpp
  Folly::folly
  velox_memory
  gflags
)

# ---------------------------------------------------------------------------
# MlIdDecodeBulkBenchmark: full-materialization decode throughput.
# ---------------------------------------------------------------------------
add_executable(nimble_ml_id_decode_bulk_benchmark MlIdDecodeBulkBenchmark.cpp)

target_link_libraries(
  nimble_ml_id_decode_bulk_benchmark
  nimble_ml_id_benchmark_common
  nimble_encodings
  nimble_common
  openzl_cpp
  Folly::folly
  velox_memory
  gflags
)

# ---------------------------------------------------------------------------
# MlIdDecodeRangeBenchmark: contiguous-range decode throughput heatmap.
# ---------------------------------------------------------------------------
add_executable(nimble_ml_id_decode_range_benchmark MlIdDecodeRangeBenchmark.cpp)

target_link_libraries(
  nimble_ml_id_decode_range_benchmark
  nimble_ml_id_benchmark_common
  nimble_encodings
  nimble_common
  Folly::folly
  velox_memory
  gflags
)

# ---------------------------------------------------------------------------
# MlIdEncodeBenchmark: encode throughput per encoder/dataset pair.
# ---------------------------------------------------------------------------
add_executable(nimble_ml_id_encode_benchmark MlIdEncodeBenchmark.cpp)

target_link_libraries(
  nimble_ml_id_encode_benchmark
  nimble_ml_id_benchmark_common
  nimble_encodings
  nimble_common
  openzl_cpp
  Folly::folly
  velox_memory
  gflags
)

# ---------------------------------------------------------------------------
# MlIdDecodePointBenchmark: single-probe point-lookup latency.
# ---------------------------------------------------------------------------
add_executable(nimble_ml_id_decode_point_benchmark MlIdDecodePointBenchmark.cpp)

target_link_libraries(
  nimble_ml_id_decode_point_benchmark
  nimble_ml_id_benchmark_common
  nimble_encodings
  nimble_common
  Folly::folly
  velox_memory
  gflags
)

# ---------------------------------------------------------------------------
# MlIdDecodeGatherBenchmark: gather-pattern decode throughput.
# ---------------------------------------------------------------------------
add_executable(nimble_ml_id_decode_gather_benchmark MlIdDecodeGatherBenchmark.cpp)

target_link_libraries(
  nimble_ml_id_decode_gather_benchmark
  nimble_ml_id_benchmark_common
  nimble_encodings
  nimble_common
  Folly::folly
  velox_memory
  gflags
)

# ---------------------------------------------------------------------------
# MlIdCostModelOracleBenchmark: validates SubIntSplit's DP cost model against an
# oracle that actually encodes each candidate bit range.
# ---------------------------------------------------------------------------
add_executable(nimble_ml_id_cost_model_oracle_benchmark MlIdCostModelOracleBenchmark.cpp)

target_link_libraries(
  nimble_ml_id_cost_model_oracle_benchmark
  nimble_ml_id_benchmark_common
  nimble_encodings
  nimble_common
  Folly::folly
  velox_memory
  gflags
)

# ---------------------------------------------------------------------------
# MlIdIndexOracleBenchmark: sweeps FPE index types, Pareto frontier, scalarised
# oracle.
# ---------------------------------------------------------------------------
add_executable(nimble_ml_id_index_oracle_benchmark MlIdIndexOracleBenchmark.cpp)

target_link_libraries(
  nimble_ml_id_index_oracle_benchmark
  nimble_ml_id_benchmark_common
  nimble_encodings
  nimble_common
  Folly::folly
  velox_memory
  gflags
)

# ---------------------------------------------------------------------------
# MlIdAblationBenchmark: progressive encoding-set restriction for SIS sections.
# ---------------------------------------------------------------------------
add_executable(nimble_ml_id_ablation_benchmark MlIdAblationBenchmark.cpp)

target_link_libraries(
  nimble_ml_id_ablation_benchmark
  nimble_ml_id_benchmark_common
  nimble_encodings
  nimble_common
  Folly::folly
  velox_memory
  gflags
)

# ---------------------------------------------------------------------------
# MlIdSmokeBenchmark: encode + decode correctness check (no timing).
# ---------------------------------------------------------------------------
add_executable(nimble_ml_id_smoke_benchmark MlIdSmokeBenchmark.cpp)

target_link_libraries(
  nimble_ml_id_smoke_benchmark
  nimble_ml_id_benchmark_common
  nimble_encodings
  nimble_common
  Folly::folly
  velox_memory
  gflags
)
