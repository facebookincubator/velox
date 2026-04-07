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
add_subdirectory(utils)

add_executable(
  aggregate_companion_functions_test
  AggregateCompanionAdapterTest.cpp
  AggregateCompanionSignaturesTest.cpp
  DummyAggregateFunction.cpp
)
velox_add_test_headers(
  aggregate_companion_functions_test
  DummyAggregateFunction.h
  PrestoQueryRunnerIntermediateTypeTransformTestBase.h
)

add_test(
  NAME aggregate_companion_functions_test
  COMMAND aggregate_companion_functions_test
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(
  aggregate_companion_functions_test
  velox_exec
  velox_function_registry
  velox_exec_test_lib
  velox_presto_types
  GTest::gtest
  GTest::gtest_main
)

# Sources are ordered via greedy bin-packing by measured EC2 execution time.
# With VELOX_TESTS_PER_GROUP=10, CMake takes files positionally in batches of
# 10 (files 1-10 = group0, 11-20 = group1, etc.).
#
# Measured per-file EC2 timings (sequential, 8-core, 30GB):
#   IndexLookupJoinTest 783s, MultiFragmentTest 599s,
#   HashJoinTestExtra ~585s, SpillerTest 584s, HashJoinTest ~575s,
#   TableWriterTest 225s, TableScanTest 189s, IndexLookupJoinTestExtra 188s,
#   MergeJoinTest 111s, AggregationTest 105s, OutputBufferManagerTest 98s,
#   ScaleWriterLocalPartitionTest 96s, OrderByTest 71s, TopNRowNumberTest 64s,
#   HashTableTest 60s, StreamingAggregationTest 58s, ExchangeClientTest 54s,
#   RowNumberTest 42s.
#
# Max group ~784s (bounded by IndexLookupJoinTest.cpp).
set(
  VELOX_EXEC_TEST_SOURCES
  # group0 (~784s): IndexLookupJoinTest 783 + 9 lightweight
  IndexLookupJoinTest.cpp
  ConcatFilesSpillMergeStreamTest.cpp
  MixedUnionWithTableScanTest.cpp
  MemoryReclaimerTest.cpp
  EnforceDistinctTest.cpp
  TraceUtilTest.cpp
  HashPartitionFunctionTest.cpp
  SpatialIndexTest.cpp
  ValuesTest.cpp
  ParallelProjectTest.cpp
  # group1 (~599s): MultiFragmentTest 599 + 9 lightweight
  MultiFragmentTest.cpp
  EnforceSingleRowTest.cpp
  FilterToExpressionTest.cpp
  ScaledScanControllerTest.cpp
  HilbertIndexTest.cpp
  OperatorTraceTest.cpp
  LimitTest.cpp
  SplitListenerTest.cpp
  AddressableNonNullValueListTest.cpp
  ArrowStreamTest.cpp
  # group2 (~585s): HashJoinTestExtra ~585 + 9 lightweight
  HashJoinTestExtra.cpp
  AggregateFunctionRegistryTest.cpp
  RoundRobinPartitionFunctionTest.cpp
  ColumnStatsCollectorTest.cpp
  MixedUnionTest.cpp
  ExpandTest.cpp
  FunctionResolutionTest.cpp
  SpillStatsTest.cpp
  MarkSortedTest.cpp
  PartitionedOutputTest.cpp
  # group3 (~584s): SpillerTest 584 + 9 lightweight
  SpillerTest.cpp
  PlanNodeToSummaryStringTest.cpp
  CountingJoinTest.cpp
  CustomJoinTest.cpp
  TaskListenerTest.cpp
  SplitTest.cpp
  SqlTest.cpp
  WindowFunctionRegistryTest.cpp
  StreamingEnforceDistinctTest.cpp
  HashBitRangeTest.cpp
  # group4 (~575s): HashJoinTest ~575 + 9 lightweight
  HashJoinTest.cpp
  SpillTest.cpp
  WindowTest.cpp
  PrefixSortTest.cpp
  MergerTest.cpp
  LocalPartitionTest.cpp
  PrintPlanWithStatsTest.cpp
  ProbeOperatorStateTest.cpp
  MarkDistinctTest.cpp
  MergeTest.cpp
  # group5 (~445s): TableWriterTest 225 + OutputBufMgr 98
  # + TopNRowNumber 64 + StreamingAgg 58
  TableWriterTest.cpp
  OutputBufferManagerTest.cpp
  TopNRowNumberTest.cpp
  StreamingAggregationTest.cpp
  ContainerRowSerdeTest.cpp
  RowContainerTest.cpp
  TopNTest.cpp
  WriterFuzzerUtilTest.cpp
  PlanNodeStatsTest.cpp
  ThreadDebugInfoTest.cpp
  # group6 (~467s): TableScanTest 189 + AggregationTest 105
  # + OrderBy 71 + HashTable 60 + RowNumber 42
  TableScanTest.cpp
  AggregationTest.cpp
  OrderByTest.cpp
  HashTableTest.cpp
  RowNumberTest.cpp
  UnnestTest.cpp
  PlanNodeSerdeTest.cpp
  HashJoinBridgeTest.cpp
  SortBufferTest.cpp
  VectorHasherTest.cpp
  # group7 (~449s): IndexLookupJoinTestExtra 188
  # + MergeJoin 111 + ScaleWriter 96 + Exchange 54
  IndexLookupJoinTestExtra.cpp
  MergeJoinTest.cpp
  ScaleWriterLocalPartitionTest.cpp
  ExchangeClientTest.cpp
  NestedLoopJoinTest.cpp
  UnorderedStreamReaderTest.cpp
  PlanNodeToStringTest.cpp
  AssignUniqueIdTest.cpp
  FilterProjectTest.cpp
  AsyncConnectorTest.cpp
)

set(
  VELOX_EXEC_TEST_DEPS
  velox_aggregates
  velox_dwio_common
  velox_dwio_common_exception
  velox_dwio_common_test_utils
  velox_dwio_orc_reader
  velox_dwio_parquet_reader
  velox_dwio_parquet_writer
  velox_exec
  velox_exec_test_lib
  velox_file_test_utils
  velox_functions_json
  velox_functions_lib
  velox_functions_prestosql
  velox_functions_test_lib
  velox_hive_connector
  velox_memory
  velox_serialization
  velox_test_util
  velox_type
  velox_type_test_lib
  velox_trace
  velox_vector
  velox_vector_fuzzer
  velox_writer_fuzzer
  velox_window
  Boost::atomic
  Boost::context
  Boost::date_time
  Boost::filesystem
  Boost::program_options
  Boost::thread
  Boost::system
  GTest::gtest
  GTest::gtest_main
  GTest::gmock
  Folly::folly
  gflags::gflags
  glog::glog
  fmt::fmt
)

velox_add_grouped_tests(
  PREFIX velox_exec_test
  SOURCES ${VELOX_EXEC_TEST_SOURCES}
  DEPS ${VELOX_EXEC_TEST_DEPS}
  EXTRA_SOURCES Main.cpp
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

if(VELOX_ENABLE_GEO)
  add_executable(velox_exec_SpatialJoinTest SpatialJoinTest.cpp Main.cpp)
  add_test(
    NAME velox_exec_SpatialJoinTest
    COMMAND velox_exec_SpatialJoinTest
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  )
  target_link_libraries(velox_exec_SpatialJoinTest ${VELOX_EXEC_TEST_DEPS})
endif()

add_executable(
  velox_exec_infra_test
  AssertQueryBuilderTest.cpp
  DriverTest.cpp
  FunctionSignatureBuilderTest.cpp
  GroupedExecutionTest.cpp
  Main.cpp
  OperatorUtilsTest.cpp
  PlanBuilderTest.cpp
  PrestoQueryRunnerTest.cpp
  QueryAssertionsTest.cpp
  TaskTest.cpp
  TreeOfLosersTest.cpp
)

add_test(
  NAME velox_exec_infra_test
  COMMAND velox_exec_infra_test
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(
  velox_exec_infra_test
  velox_dwio_common
  velox_dwio_common_exception
  velox_dwio_common_test_utils
  velox_exec
  velox_exec_test_lib
  velox_fuzzer_util
  velox_functions_lib
  velox_functions_prestosql
  velox_functions_test_lib
  velox_window
  velox_hive_connector
  velox_memory
  velox_serialization
  velox_test_util
  velox_type
  velox_vector
  velox_vector_fuzzer
  velox_presto_serializer
  velox_dwio_dwrf_writer
  Boost::atomic
  Boost::context
  Boost::date_time
  Boost::filesystem
  Boost::program_options
  Boost::thread
  Boost::system
  GTest::gtest
  GTest::gtest_main
  GTest::gmock
  Folly::folly
  gflags::gflags
  glog::glog
  fmt::fmt
)

# Split velox_exec_util_test into individual test binaries for parallel execution.
# Each test depends on the shared base class PrestoQueryRunnerIntermediateTypeTransformTestBase.cpp.
set(
  VELOX_EXEC_UTIL_TEST_SOURCES
  PrestoQueryRunnerHyperLogLogTransformTest.cpp
  PrestoQueryRunnerTDigestTransformTest.cpp
  PrestoQueryRunnerQDigestTransformTest.cpp
  PrestoQueryRunnerKHyperLogLogTransformTest.cpp
  PrestoQueryRunnerSetDigestTransformTest.cpp
  PrestoQueryRunnerJsonTransformTest.cpp
  PrestoQueryRunnerIntervalTransformTest.cpp
  PrestoQueryRunnerTimeTransformTest.cpp
  PrestoQueryRunnerTimestampWithTimeZoneTransformTest.cpp
)

set(
  VELOX_EXEC_UTIL_TEST_DEPS
  velox_fuzzer_util
  velox_exec_test_lib
  velox_functions_test_lib
  velox_presto_types
  velox_presto_types_fuzzer_utils
  GTest::gtest
  GTest::gtest_main
)

velox_add_grouped_tests(
  PREFIX velox_exec_util_test
  SOURCES ${VELOX_EXEC_UTIL_TEST_SOURCES}
  DEPS ${VELOX_EXEC_UTIL_TEST_DEPS}
  EXTRA_SOURCES PrestoQueryRunnerIntermediateTypeTransformTestBase.cpp Main.cpp
)

add_executable(velox_in_10_min_demo VeloxIn10MinDemo.cpp)

target_link_libraries(
  velox_in_10_min_demo
  velox_aggregates
  velox_type
  velox_vector
  velox_vector_test_lib
  velox_exec
  velox_exec_test_lib
  velox_tpch_connector
  velox_memory
)

add_executable(
  velox_table_evolution_fuzzer_test
  TableEvolutionFuzzerTest.cpp
  TableEvolutionFuzzer.cpp
)
velox_add_test_headers(velox_table_evolution_fuzzer_test TableEvolutionFuzzer.h)

target_link_libraries(
  velox_table_evolution_fuzzer_test
  velox_expression_fuzzer
  velox_expression_test_utility
  velox_constrained_input_generators
  velox_fuzzer_util
  velox_functions_prestosql
  velox_exec_test_lib
  velox_test_util
  velox_vector_fuzzer
  GTest::gtest
  GTest::gtest_main
)

add_test(
  NAME velox_table_evolution_fuzzer_test
  COMMAND velox_table_evolution_fuzzer_test
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

add_executable(velox_aggregation_runner_test AggregationRunnerTest.cpp)

target_link_libraries(
  velox_aggregation_runner_test
  velox_aggregation_fuzzer
  velox_fuzzer_util
  velox_aggregates
  velox_vector_test_lib
  GTest::gtest
  GTest::gtest_main
)

add_library(
  velox_simple_aggregate
  SimpleAverageAggregate.cpp
  SimpleArrayAggAggregate.cpp
  SimpleVariadicArrayAggAggregate.cpp
  SimpleVariadicSumAggregate.cpp
)
velox_add_test_headers(velox_simple_aggregate SimpleAggregateFunctionsRegistration.h)

target_link_libraries(
  velox_simple_aggregate
  velox_exec
  velox_expression
  velox_expression_functions
  velox_aggregates
)

add_executable(velox_simple_aggregate_test SimpleAggregateAdapterTest.cpp Main.cpp)

target_link_libraries(
  velox_simple_aggregate_test
  velox_simple_aggregate
  velox_exec
  velox_functions_aggregates_test_lib
  GTest::gtest
)

add_test(velox_simple_aggregate_test velox_simple_aggregate_test)

if(VELOX_ENABLE_BENCHMARKS)
  add_library(
    velox_spiller_join_benchmark_base
    JoinSpillInputBenchmarkBase.cpp
    SpillerBenchmarkBase.cpp
  )
  velox_add_test_headers(
    velox_spiller_join_benchmark_base
    AggregateSpillBenchmarkBase.h
    JoinSpillInputBenchmarkBase.h
    SpillerBenchmarkBase.h
  )
  target_link_libraries(
    velox_spiller_join_benchmark_base
    velox_exec
    velox_exec_test_lib
    velox_memory
    velox_vector_fuzzer
    glog::glog
    gflags::gflags
    Folly::folly
    pthread
  )

  add_executable(velox_spiller_join_benchmark SpillerJoinInputBenchmarkTest.cpp)
  target_link_libraries(
    velox_spiller_join_benchmark
    velox_exec
    velox_exec_test_lib
    velox_spiller_join_benchmark_base
  )

  add_library(
    velox_spiller_aggregate_benchmark_base
    AggregateSpillBenchmarkBase.cpp
    SpillerBenchmarkBase.cpp
  )
  target_link_libraries(
    velox_spiller_aggregate_benchmark_base
    velox_exec
    velox_exec_test_lib
    velox_memory
    velox_vector_fuzzer
    glog::glog
    gflags::gflags
    Folly::folly
    pthread
  )

  add_executable(velox_spiller_aggregate_benchmark SpillerAggregateBenchmarkTest.cpp)
  target_link_libraries(
    velox_spiller_aggregate_benchmark
    velox_exec
    velox_exec_test_lib
    velox_spiller_aggregate_benchmark_base
  )
endif()

add_executable(velox_driver_test OperatorReplacementTest.cpp Main.cpp)
add_test(
  NAME velox_driver_test
  COMMAND velox_driver_test
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(velox_driver_test velox_exec velox_exec_test_lib GTest::gtest)

velox_add_library(velox_aggregate_registry_util INTERFACE HEADERS AggregateRegistryTestUtil.h)
