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

target_link_libraries(nimble_ml_id_benchmark_common PUBLIC gflags)

# Header-only pieces of the suite, attached to the shared library so they are
# tracked by a target.
velox_add_test_headers(
  nimble_ml_id_benchmark_common
  AblationPolicy.h
  Axes.h
  CachePolicy.h
  DriverSweep.h
  ElemType.h
  GatherTraceGen.h
  MeasureLoop.h
  OpenZLBenchTarget.h
  PointTraceGen.h
  ResultWriter.h
  SelectiveTraceGen.h
  SubstreamCompression.h
  TimingStats.h
)
# ---------------------------------------------------------------------------
# Every driver in this suite links the same set: the shared flag library, the
# encodings under test, and the benchmark harness. Only whether the driver
# exercises OpenZL varies, so the link set is declared once here rather than
# repeated per target. Pass OPENZL for the drivers that sweep it as a
# black-box codec.
# ---------------------------------------------------------------------------
function(nimble_add_ml_id_benchmark NAME SOURCE)
  cmake_parse_arguments(ARG "OPENZL" "" "" ${ARGN})

  add_executable(${NAME} ${SOURCE})

  target_link_libraries(
    ${NAME}
    nimble_ml_id_benchmark_common
    nimble_encodings
    nimble_common
    Folly::folly
    velox_memory
    gflags
  )

  if(ARG_OPENZL)
    target_link_libraries(${NAME} openzl_cpp)
  endif()
endfunction()

# MlIdCompressionBenchmark: encodes a column of ML IDs under every candidate
# encoding and reports compressed size + round-trip latency.
nimble_add_ml_id_benchmark(
  nimble_ml_id_compression_benchmark MlIdCompressionBenchmark.cpp OPENZL
)

# MlIdDecodeBulkBenchmark: full-materialization decode throughput.
nimble_add_ml_id_benchmark(
  nimble_ml_id_decode_bulk_benchmark MlIdDecodeBulkBenchmark.cpp OPENZL
)

# MlIdDecodeRangeBenchmark: contiguous-range decode across (start, length).
nimble_add_ml_id_benchmark(
  nimble_ml_id_decode_range_benchmark MlIdDecodeRangeBenchmark.cpp OPENZL
)

# MlIdEncodeBenchmark: encode-side throughput and selection cost.
nimble_add_ml_id_benchmark(
  nimble_ml_id_encode_benchmark MlIdEncodeBenchmark.cpp OPENZL
)

# MlIdDecodePointBenchmark: single-row random access latency.
nimble_add_ml_id_benchmark(
  nimble_ml_id_decode_point_benchmark MlIdDecodePointBenchmark.cpp OPENZL
)

# MlIdDecodeGatherBenchmark: scattered multi-range gather, the TableScan skip
# pattern.
nimble_add_ml_id_benchmark(
  nimble_ml_id_decode_gather_benchmark MlIdDecodeGatherBenchmark.cpp OPENZL
)

# MlIdCostModelOracleBenchmark: validates SubIntSplit's DP cost model against an
# oracle that actually encodes each candidate bit range.
nimble_add_ml_id_benchmark(
  nimble_ml_id_cost_model_oracle_benchmark MlIdCostModelOracleBenchmark.cpp
)

# MlIdIndexOracleBenchmark: sweeps FPE index types, Pareto frontier, scalarised
# oracle.
nimble_add_ml_id_benchmark(
  nimble_ml_id_index_oracle_benchmark MlIdIndexOracleBenchmark.cpp
)

# MlIdAblationBenchmark: progressive encoding-set restriction for SIS sections.
nimble_add_ml_id_benchmark(
  nimble_ml_id_ablation_benchmark MlIdAblationBenchmark.cpp
)

# MlIdSmokeBenchmark: encode + decode correctness check (no timing). Sweeps
# every supported element type, so it is also the compile-coverage gate.
nimble_add_ml_id_benchmark(
  nimble_ml_id_smoke_benchmark MlIdSmokeBenchmark.cpp)

# MlIdSelectionPolicyBenchmark: measures whether the bit-flip-probability
# top-level policies would correctly predict "SubIntSplit wins" against real
# candidate encodings -- evidence-gathering only, does not touch production
# selection.
nimble_add_ml_id_benchmark(
  nimble_ml_id_selection_policy_benchmark MlIdSelectionPolicyBenchmark.cpp)
