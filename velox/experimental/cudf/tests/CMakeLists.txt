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

# Common libraries shared by most cudf tests.
set(CUDF_TEST_DEFAULT_LIBS velox_cudf_exec velox_exec velox_exec_test_lib velox_test_util)

# Adds a cudf test target with standard boilerplate.
#
# Usage:
#   velox_add_cudf_test(
#     NAME <test_name>
#     SOURCES <source_files...>
#     LIBS <libraries...>
#     [TIMEOUT <seconds>]
#   )
#
# gtest and gtest_main are always appended to LIBS.
# Default timeout: 300 seconds
function(velox_add_cudf_test)
  set(options "")
  set(one_value_args NAME TIMEOUT)
  set(multi_value_args SOURCES LIBS)
  cmake_parse_arguments(ARG "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})

  if(NOT ARG_TIMEOUT)
    set(ARG_TIMEOUT 300)
  endif()

  add_executable(${ARG_NAME} ${ARG_SOURCES})
  add_test(NAME ${ARG_NAME} COMMAND ${ARG_NAME} WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
  set_tests_properties(${ARG_NAME} PROPERTIES LABELS cuda_driver TIMEOUT ${ARG_TIMEOUT})
  target_link_libraries(${ARG_NAME} ${ARG_LIBS} gtest gtest_main)
endfunction()

velox_add_cudf_test(
  NAME velox_cudf_aggregation_test
  SOURCES Main.cpp AggregationTest.cpp
  LIBS ${CUDF_TEST_DEFAULT_LIBS} velox_vector_fuzzer
)

velox_add_cudf_test(
  NAME velox_cudf_aggregation_selection_test
  SOURCES Main.cpp AggregationSelectionTest.cpp
  LIBS ${CUDF_TEST_DEFAULT_LIBS} velox_functions_spark
)

velox_add_cudf_test(
  NAME velox_cudf_assign_unique_id_test
  SOURCES Main.cpp AssignUniqueIdTest.cpp
  LIBS ${CUDF_TEST_DEFAULT_LIBS}
)

velox_add_cudf_test(
  NAME velox_cudf_batch_concat_test
  SOURCES Main.cpp BatchConcatTest.cpp
  LIBS ${CUDF_TEST_DEFAULT_LIBS}
)

velox_add_cudf_test(
  NAME velox_cudf_config_test
  SOURCES Main.cpp ConfigTest.cpp
  LIBS velox_cudf_exec Folly::folly
)

velox_add_cudf_test(
  NAME velox_cudf_decimal_expression_test
  SOURCES Main.cpp DecimalExpressionTest.cpp
  LIBS ${CUDF_TEST_DEFAULT_LIBS} velox_functions_test_lib
  TIMEOUT 3000
)

velox_add_cudf_test(
  NAME velox_cudf_enforce_single_row_test
  SOURCES Main.cpp EnforceSingleRowTest.cpp
  LIBS ${CUDF_TEST_DEFAULT_LIBS}
)

velox_add_cudf_test(
  NAME velox_cudf_expression_selection_test
  SOURCES Main.cpp ExpressionEvaluatorSelectionTest.cpp
  LIBS ${CUDF_TEST_DEFAULT_LIBS} velox_functions_spark
)

velox_add_cudf_test(
  NAME velox_cudf_filter_project_test
  SOURCES Main.cpp FilterProjectTest.cpp
  LIBS ${CUDF_TEST_DEFAULT_LIBS} velox_functions_test_lib
)

velox_add_cudf_test(
  NAME velox_cudf_function_registry_test
  SOURCES Main.cpp FunctionRegistryTest.cpp
  LIBS velox_cudf_expression velox_cudf_exec velox_exec velox_exec_test_lib velox_test_util
)

velox_add_cudf_test(
  NAME velox_cudf_group_id_test
  SOURCES Main.cpp GroupIdTest.cpp
  LIBS ${CUDF_TEST_DEFAULT_LIBS}
)

velox_add_cudf_test(
  NAME velox_cudf_hash_join_test
  SOURCES HashJoinTest.cpp Main.cpp
  LIBS ${CUDF_TEST_DEFAULT_LIBS} velox_vector_fuzzer Folly::folly fmt::fmt
)

velox_add_cudf_test(
  NAME velox_cudf_interop_test
  SOURCES Main.cpp InteropTest.cpp
  LIBS ${CUDF_TEST_DEFAULT_LIBS}
)

velox_add_cudf_test(
  NAME velox_cudf_limit_test
  SOURCES Main.cpp LimitTest.cpp
  LIBS ${CUDF_TEST_DEFAULT_LIBS}
)

velox_add_cudf_test(
  NAME velox_cudf_local_partition_test
  SOURCES Main.cpp LocalPartitionTest.cpp
  LIBS ${CUDF_TEST_DEFAULT_LIBS} fmt::fmt
)

velox_add_cudf_test(
  NAME velox_cudf_mark_distinct_test
  SOURCES Main.cpp MarkDistinctTest.cpp
  LIBS ${CUDF_TEST_DEFAULT_LIBS}
)

velox_add_cudf_test(
  NAME velox_cudf_nested_loop_join_test
  SOURCES Main.cpp NestedLoopJoinTest.cpp
  LIBS ${CUDF_TEST_DEFAULT_LIBS}
)

velox_add_cudf_test(
  NAME velox_cudf_order_by_test
  SOURCES Main.cpp OrderByTest.cpp
  LIBS ${CUDF_TEST_DEFAULT_LIBS} fmt::fmt
)

if(VELOX_ENABLE_S3)
  velox_add_cudf_test(
    NAME velox_cudf_s3_read_test
    SOURCES Main.cpp S3ReadTest.cpp
    LIBS
      velox_cudf_exec_test_lib
      velox_cudf_hive_connector
      velox_exec
      velox_exec_test_lib
      velox_file
      velox_s3fs
      velox_test_util
  )
endif()

velox_add_cudf_test(
  NAME velox_cudf_step_aware_aggregation_registry_test
  SOURCES Main.cpp StepAwareAggregationRegistryTest.cpp
  LIBS ${CUDF_TEST_DEFAULT_LIBS} velox_cudf_exec_test_lib
)

velox_add_cudf_test(
  NAME velox_cudf_string_function_test
  SOURCES Main.cpp StringFunctionTest.cpp
  LIBS ${CUDF_TEST_DEFAULT_LIBS} velox_functions_test_lib
)

velox_add_cudf_test(
  NAME velox_cudf_subfield_filter_ast_test
  SOURCES Main.cpp SubfieldFilterAstTest.cpp
  LIBS ${CUDF_TEST_DEFAULT_LIBS}
)

velox_add_cudf_test(
  NAME velox_cudf_table_scan_test
  SOURCES Main.cpp TableScanTest.cpp
  LIBS
    velox_cudf_exec_test_lib
    velox_cudf_hive_connector
    velox_exec
    velox_exec_test_lib
    velox_test_util
    fmt::fmt
)

# Disabling writer tests until we re-add writing ability to CudfHiveConnector
# velox_add_cudf_test(
#   NAME velox_cudf_table_write_test
#   SOURCES Main.cpp TableWriteTest.cpp
#   LIBS
#     velox_cudf_exec_test_lib
#     velox_cudf_hive_connector
#     velox_exec
#     velox_exec_test_lib
#     velox_test_util
#     fmt::fmt
# )

velox_add_cudf_test(
  NAME velox_cudf_tocudf_selection_test
  SOURCES Main.cpp ToCudfSelectionTest.cpp
  LIBS ${CUDF_TEST_DEFAULT_LIBS} velox_vector_fuzzer
)

velox_add_cudf_test(
  NAME velox_cudf_topn_test
  SOURCES Main.cpp TopNTest.cpp
  LIBS ${CUDF_TEST_DEFAULT_LIBS} fmt::fmt
)

add_subdirectory(utils)

if(${VELOX_ENABLE_SPARK_FUNCTIONS})
  add_subdirectory(sparksql)
endif()
