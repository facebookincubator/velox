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

# --- TPC-H benchmarks ---

add_executable(velox_cudf_tpch_benchmark CudfTpchBenchmark.cpp)

target_link_libraries(
  velox_cudf_tpch_benchmark
  velox_cudf_exec
  velox_cudf_exec_test_lib
  velox_tpch_benchmark_lib
)

# --- TPC-DS benchmarks ---

# Base TPC-DS benchmark library (Hive-only, no CuDF dependency in the lib
# itself, but placed here since it is consumed by the CuDF benchmark binary).
add_library(velox_tpcds_benchmark_lib TpcdsBenchmark.cpp)

target_link_libraries(
  velox_tpcds_benchmark_lib
  velox_query_benchmark
  velox_exec
  velox_exec_test_lib
  velox_dwio_common
  velox_dwio_parquet_reader
  velox_dwio_common_test_utils
  velox_hive_connector
  velox_exception
  velox_memory
  velox_process
  velox_serialization
  velox_encode
  velox_type
  velox_caching
  velox_vector_test_lib
  velox_aggregates
  Folly::follybenchmark
  Folly::folly
  fmt::fmt
)

# CuDF TPC-DS benchmark (single binary, --cudf_enabled flag selects mode).
add_executable(velox_cudf_tpcds_benchmark CudfTpcdsBenchmark.cpp)

target_link_libraries(
  velox_cudf_tpcds_benchmark
  velox_tpcds_benchmark_lib
  velox_cudf_exec
  velox_cudf_exec_test_lib
)
