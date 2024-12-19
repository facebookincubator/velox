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

add_executable(velox_plan_test PlanTest.cpp Tpch.cpp ParquetTpchTest.cpp)

add_test(velox_plan_test velox_plan_test)

target_link_libraries(
  velox_plan_test
  velox_verax
  velox_tpch_gen
  velox_exec_test_lib
  velox_dwio_parquet_reader
  velox_dwio_native_parquet_reader
  gtest
  gtest_main)

add_executable(velox_sql VeloxSql.cpp)

target_link_libraries(
  velox_sql
  velox_local_runner
  velox_verax
  velox_connector_split_source
  velox_hive_connector_metadata
  velox_exec_test_lib
  velox_dwio_common
  velox_dwio_parquet_reader
  velox_dwio_native_parquet_reader
  velox_parse_parser
  velox_parse_expression
  velox_parse_utils)
