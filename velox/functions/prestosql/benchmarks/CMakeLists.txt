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

add_library(velox_benchmark_deps_no_func INTERFACE)
target_link_libraries(
  velox_benchmark_deps_no_func
  INTERFACE velox_functions_test_lib velox_exec velox_exec_test_lib
            velox_benchmark_builder)

add_library(velox_presto_benchmark_deps INTERFACE)
target_link_libraries(
  velox_presto_benchmark_deps
  INTERFACE velox_functions_prestosql velox_functions_lib velox_vector_fuzzer
            velox_benchmark_deps_no_func)

add_executable(velox_functions_prestosql_benchmarks_array_contains
               ArrayContainsBenchmark.cpp)
target_link_libraries(
  velox_functions_prestosql_benchmarks_array_contains
  PRIVATE velox_presto_benchmark_deps)

add_executable(velox_functions_prestosql_benchmarks_array_min_max
               ArrayMinMaxBenchmark.cpp)
target_link_libraries(
  velox_functions_prestosql_benchmarks_array_min_max
  PRIVATE velox_presto_benchmark_deps)

add_executable(velox_functions_prestosql_benchmarks_array_position
               ArrayPositionBenchmark.cpp)
target_link_libraries(
  velox_functions_prestosql_benchmarks_array_position
  PRIVATE velox_presto_benchmark_deps)

add_executable(velox_functions_prestosql_benchmarks_array_sum
               ArraySumBenchmark.cpp)

target_link_libraries(
  velox_functions_prestosql_benchmarks_array_sum
  PRIVATE velox_presto_benchmark_deps)

add_executable(velox_functions_prestosql_benchmarks_field_reference
               FieldReferenceBenchmark.cpp)
target_link_libraries(
  velox_functions_prestosql_benchmarks_field_reference
  PRIVATE velox_presto_benchmark_deps)

add_executable(velox_functions_prestosql_benchmarks_width_bucket
               WidthBucketBenchmark.cpp)
target_link_libraries(
  velox_functions_prestosql_benchmarks_width_bucket
  PRIVATE velox_presto_benchmark_deps)

add_executable(velox_functions_prestosql_benchmarks_string_ascii_utf_functions
               StringAsciiUTFFunctionBenchmarks.cpp)
target_link_libraries(
  velox_functions_prestosql_benchmarks_string_ascii_utf_functions
  PRIVATE velox_presto_benchmark_deps velox_common_fuzzer_util)

add_executable(velox_functions_prestosql_benchmarks_not NotBenchmark.cpp)
target_link_libraries(
  velox_functions_prestosql_benchmarks_not
  PRIVATE velox_presto_benchmark_deps)

add_executable(velox_functions_prestosql_benchmarks_date_time
               DateTimeBenchmark.cpp)
target_link_libraries(
  velox_functions_prestosql_benchmarks_date_time
  PRIVATE velox_type_tz velox_presto_benchmark_deps)

add_executable(velox_functions_prestosql_benchmarks_bitwise
               BitwiseBenchmark.cpp)
target_link_libraries(
  velox_functions_prestosql_benchmarks_bitwise
  PRIVATE velox_presto_benchmark_deps)

add_executable(velox_functions_prestosql_benchmarks_in InBenchmark.cpp)
target_link_libraries(
  velox_functions_prestosql_benchmarks_in
  PRIVATE velox_presto_benchmark_deps)

add_executable(velox_functions_prestosql_benchmarks_map_input
               MapInputBenchmark.cpp)
target_link_libraries(
  velox_functions_prestosql_benchmarks_map_input
  PRIVATE velox_presto_benchmark_deps)

add_executable(velox_functions_benchmarks_url URLBenchmark.cpp)
target_link_libraries(
  velox_functions_benchmarks_url
  PRIVATE velox_presto_benchmark_deps)

add_executable(velox_functions_benchmarks_compare CompareBenchmark.cpp)
target_link_libraries(
  velox_functions_benchmarks_compare
  PRIVATE velox_presto_benchmark_deps)

add_executable(velox_benchmark_array_writer_with_nulls ArrayWriterBenchmark.cpp)
target_compile_definitions(velox_benchmark_array_writer_with_nulls
                           PUBLIC WITH_NULLS=true)
target_link_libraries(
  velox_benchmark_array_writer_with_nulls
  PRIVATE velox_benchmark_deps_no_func)

add_executable(velox_benchmark_array_writer_no_nulls ArrayWriterBenchmark.cpp)
target_link_libraries(
  velox_benchmark_array_writer_no_nulls
  PRIVATE velox_benchmark_deps_no_func)
target_compile_definitions(velox_benchmark_array_writer_no_nulls
                           PUBLIC WITH_NULLS=false)

add_executable(velox_benchmark_nested_array_writer_no_nulls
               NestedArrayWriterBenchmark.cpp)
target_link_libraries(
  velox_benchmark_nested_array_writer_no_nulls
  PRIVATE velox_benchmark_deps_no_func)
target_compile_definitions(velox_benchmark_nested_array_writer_no_nulls
                           PUBLIC WITH_NULLS=false)

add_executable(velox_benchmark_nested_array_writer_with_nulls
               NestedArrayWriterBenchmark.cpp)
target_compile_definitions(velox_benchmark_nested_array_writer_with_nulls
                           PUBLIC WITH_NULLS=true)
target_link_libraries(
  velox_benchmark_nested_array_writer_with_nulls
  PRIVATE velox_benchmark_deps_no_func)

add_executable(velox_benchmark_map_writer_with_nulls MapWriterBenchmarks.cpp)
target_compile_definitions(velox_benchmark_map_writer_with_nulls
                           PUBLIC WITH_NULLS=true)
target_link_libraries(
  velox_benchmark_map_writer_with_nulls
  PRIVATE velox_benchmark_deps_no_func)

add_executable(velox_benchmark_map_writer_no_nulls MapWriterBenchmarks.cpp)
target_link_libraries(
  velox_benchmark_map_writer_no_nulls
  PRIVATE velox_benchmark_deps_no_func)
target_compile_definitions(velox_benchmark_map_writer_no_nulls
                           PUBLIC WITH_NULLS=false)

add_executable(velox_functions_benchmarks_row_writer_no_nulls
               RowWriterBenchmark.cpp)
target_link_libraries(
  velox_functions_benchmarks_row_writer_no_nulls
  PRIVATE velox_benchmark_deps_no_func)

add_executable(velox_functions_benchmarks_string_writer_no_nulls
               StringWriterBenchmark.cpp)
target_link_libraries(
  velox_functions_benchmarks_string_writer_no_nulls
  PRIVATE velox_benchmark_deps_no_func)

add_executable(velox_functions_prestosql_benchmarks_zip ZipBenchmark.cpp)
target_link_libraries(
  velox_functions_prestosql_benchmarks_zip
  PRIVATE velox_presto_benchmark_deps)

add_executable(velox_functions_prestosql_benchmarks_row Row.cpp)
target_link_libraries(
  velox_functions_prestosql_benchmarks_row
  PRIVATE velox_presto_benchmark_deps)

add_executable(velox_functions_prestosql_benchmarks_comparisons
               ComparisonsBenchmark.cpp)
target_link_libraries(
  velox_functions_prestosql_benchmarks_comparisons
  PRIVATE velox_presto_benchmark_deps)

add_executable(velox_functions_prestosql_benchmarks_concat ConcatBenchmark.cpp)
target_link_libraries(
  velox_functions_prestosql_benchmarks_concat
  PRIVATE velox_presto_benchmark_deps)

add_executable(velox_functions_prestosql_benchmarks_zip_with
               ZipWithBenchmark.cpp)
target_link_libraries(
  velox_functions_prestosql_benchmarks_zip_with
  PRIVATE velox_presto_benchmark_deps)

add_executable(velox_functions_prestosql_benchmarks_map_zip_with
               MapZipWithBenchmark.cpp)
target_link_libraries(
  velox_functions_prestosql_benchmarks_map_zip_with
  PRIVATE velox_presto_benchmark_deps)

add_executable(velox_functions_prestosql_benchmarks_cardinality
               CardinalityBenchmark.cpp)
target_link_libraries(
  velox_functions_prestosql_benchmarks_cardinality
  PRIVATE velox_presto_benchmark_deps)

add_executable(velox_functions_benchmarks_simdjson_function_with_expr
               JsonExprBenchmark.cpp)
target_link_libraries(
  velox_functions_benchmarks_simdjson_function_with_expr
  PRIVATE velox_presto_benchmark_deps)

add_executable(velox_functions_prestosql_benchmarks_map_subscript
               MapSubscriptCachingBenchmark.cpp)
target_link_libraries(
  velox_functions_prestosql_benchmarks_map_subscript
  PRIVATE velox_presto_benchmark_deps)

add_executable(velox_functions_prestosql_benchmarks_regexp_replace
               RegexpReplaceBenchmark.cpp)
target_link_libraries(
  velox_functions_prestosql_benchmarks_regexp_replace
  PRIVATE velox_presto_benchmark_deps)

add_executable(velox_functions_prestosql_benchmarks_generic
               GenericBenchmark.cpp)
target_link_libraries(
  velox_functions_prestosql_benchmarks_generic
  PRIVATE velox_presto_benchmark_deps)

add_executable(velox_functions_prestosql_benchmarks_uuid_cast
               UuidCastBenchmark.cpp)
target_link_libraries(
  velox_functions_prestosql_benchmarks_uuid_cast
  PRIVATE velox_presto_benchmark_deps)
