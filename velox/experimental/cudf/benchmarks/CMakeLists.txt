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

add_library(velox_cudf_benchmark_lib
  CudfBenchmarkHelpers.cpp
  CudfTpchBenchmark.cpp
  TpchPlanRewriter.cpp
)

target_link_libraries(
  velox_cudf_benchmark_lib
  velox_cudf_exec
  velox_cudf_exec_test_lib
  velox_tpch_benchmark_lib
  velox_tpch_gen
)

add_executable(velox_cudf_tpch_benchmark CudfTpchBenchmarkMain.cpp)

target_link_libraries(
  velox_cudf_tpch_benchmark
  velox_cudf_benchmark_lib
)

add_executable(
  velox_cudf_hashjoin_spill_benchmark CudfHashJoinSpillBenchmark.cpp)

target_link_libraries(
  velox_cudf_hashjoin_spill_benchmark
  velox_cudf_benchmark_lib
  velox_tpch_gen
)
