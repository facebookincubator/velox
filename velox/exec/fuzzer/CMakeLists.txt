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

# Generate Thrift library for LocalRunnerService early since velox_fuzzer_util
# depends on it when VELOX_ENABLE_REMOTE_FUNCTIONS is enabled.
if(VELOX_ENABLE_REMOTE_FUNCTIONS)
  include(FBThriftCppLibrary)
  add_fbthrift_cpp_library(
    local_runner_service_thrift
    if/LocalRunnerService.thrift
    SERVICES
    LocalRunnerService
  )
  target_compile_options(local_runner_service_thrift PRIVATE -Wno-error=deprecated-declarations)
endif()

add_library(
  velox_fuzzer_util
  ReferenceQueryRunner.cpp
  DuckQueryRunner.cpp
  DuckQueryRunnerToSqlPlanNodeVisitor.cpp
  PrestoQueryRunner.cpp
  PrestoQueryRunnerIntermediateTypeTransforms.cpp
  PrestoQueryRunnerTimestampWithTimeZoneTransform.cpp
  PrestoQueryRunnerJsonTransform.cpp
  PrestoQueryRunnerIntervalTransform.cpp
  PrestoQueryRunnerToSqlPlanNodeVisitor.cpp
  FuzzerUtil.cpp
  PrestoSql.cpp
)

# TODO Add VeloxQueryRunner to velox_fuzzer_util to support in
# ExpressionFuzzerTest. More information can be found here:
# https://github.com/facebookincubator/velox/issues/15414
if(VELOX_ENABLE_REMOTE_FUNCTIONS)
  target_sources(velox_fuzzer_util PRIVATE VeloxQueryRunner.cpp)
  target_link_libraries(velox_fuzzer_util local_runner_service_thrift)
  target_include_directories(velox_fuzzer_util PUBLIC ${CMAKE_CURRENT_BINARY_DIR})
endif()

target_link_libraries(
  velox_fuzzer_util
  velox_common_fuzzer_util
  velox_vector_fuzzer
  velox_core
  velox_exec_test_lib
  velox_expression_functions
  velox_presto_types
  CURL::libcurl
  velox_presto_type_parser
  Folly::folly
  velox_hive_connector
  velox_dwio_dwrf_writer
  velox_dwio_catalog_fbhive
  velox_dwio_faulty_file_sink
)

add_library(velox_aggregation_fuzzer_base AggregationFuzzerBase.cpp)

target_link_libraries(
  velox_aggregation_fuzzer_base
  velox_exec_test_lib
  velox_temp_path
  velox_common_base
  velox_file
  velox_hive_connector
  velox_dwio_dwrf_reader
  velox_dwio_dwrf_writer
  velox_type
  velox_vector_fuzzer
  velox_fuzzer_util
  velox_expression_test_utility
  velox_vector
  velox_core
)

add_library(velox_aggregation_fuzzer AggregationFuzzer.cpp)

target_link_libraries(
  velox_aggregation_fuzzer
  velox_type
  velox_vector_fuzzer
  velox_exec_test_lib
  velox_expression_test_utility
  velox_aggregation_fuzzer_base
  velox_fuzzer_util
)

add_library(velox_window_fuzzer WindowFuzzer.cpp)

target_link_libraries(
  velox_window_fuzzer
  velox_fuzzer_util
  velox_type
  velox_vector_fuzzer
  velox_exec_test_lib
  velox_expression_test_utility
  velox_aggregation_fuzzer_base
  velox_temp_path
)

add_library(velox_row_number_fuzzer_base_lib RowNumberFuzzerBase.cpp)

target_link_libraries(
  velox_row_number_fuzzer_base_lib
  velox_dwio_dwrf_reader
  velox_fuzzer_util
  velox_vector_fuzzer
  velox_exec_test_lib
)

add_library(velox_row_number_fuzzer_lib RowNumberFuzzer.cpp)

target_link_libraries(
  velox_row_number_fuzzer_lib
  velox_row_number_fuzzer_base_lib
  velox_type
  velox_expression_test_utility
)

# RowNumber Fuzzer.
add_executable(velox_row_number_fuzzer RowNumberFuzzerRunner.cpp)

target_link_libraries(velox_row_number_fuzzer velox_row_number_fuzzer_lib)

add_library(velox_topn_row_number_fuzzer_lib TopNRowNumberFuzzer.cpp)

target_link_libraries(
  velox_topn_row_number_fuzzer_lib
  velox_row_number_fuzzer_base_lib
  velox_type
  velox_expression_test_utility
)

# TopNRowNumber Fuzzer.
add_executable(velox_topn_row_number_fuzzer TopNRowNumberFuzzerRunner.cpp)

target_link_libraries(velox_topn_row_number_fuzzer velox_topn_row_number_fuzzer_lib)

# Join Fuzzer Base Library.
add_library(velox_join_fuzzer_base JoinFuzzerBase.cpp)

target_link_libraries(
  velox_join_fuzzer_base
  velox_type
  velox_vector_fuzzer
  velox_fuzzer_util
  velox_exec_test_lib
)

# Join Fuzzer.
add_executable(velox_join_fuzzer JoinFuzzerRunner.cpp JoinFuzzer.cpp JoinMaker.cpp)

target_link_libraries(
  velox_join_fuzzer
  velox_join_fuzzer_base
  velox_type
  velox_vector_fuzzer
  velox_fuzzer_util
  velox_exec_test_lib
  velox_expression_test_utility
)

# Spatial Join Fuzzer.
add_executable(velox_spatial_join_fuzzer SpatialJoinFuzzerRunner.cpp SpatialJoinFuzzer.cpp)

target_link_libraries(
  velox_spatial_join_fuzzer
  velox_type
  velox_vector_fuzzer
  velox_fuzzer_util
  velox_exec_test_lib
  velox_expression_test_utility
  velox_vector_test_lib
)

add_library(velox_writer_fuzzer WriterFuzzer.cpp)

target_link_libraries(
  velox_writer_fuzzer
  velox_fuzzer_util
  velox_type
  velox_vector_fuzzer
  velox_exec_test_lib
  velox_expression_test_utility
  velox_temp_path
  velox_vector_test_lib
  velox_dwio_faulty_file_sink
  velox_file_test_utils
)

# Arbitration Fuzzer.
add_executable(
  velox_memory_arbitration_fuzzer
  MemoryArbitrationFuzzerRunner.cpp
  MemoryArbitrationFuzzer.cpp
)

target_link_libraries(
  velox_memory_arbitration_fuzzer
  velox_common_fuzzer_util
  velox_fuzzer_util
  velox_type
  velox_vector_fuzzer
  velox_exec_test_lib
  velox_expression_test_utility
  velox_functions_prestosql
  velox_aggregates
)

add_library(velox_cache_fuzzer_lib CacheFuzzer.cpp)

# Cache Fuzzer
add_executable(velox_cache_fuzzer CacheFuzzerRunner.cpp)

target_link_libraries(velox_cache_fuzzer velox_cache_fuzzer_lib velox_fuzzer_util)

target_link_libraries(
  velox_cache_fuzzer_lib
  velox_dwio_common
  velox_temp_path
  velox_vector_test_lib
)

# Exchange Fuzzer
add_executable(velox_exchange_fuzzer ExchangeFuzzer.cpp)

target_link_libraries(
  velox_exchange_fuzzer
  velox_exec_test_lib
  velox_aggregates
  velox_vector_test_lib
  velox_vector_fuzzer
)

# LocalRunnerService Library (requires FBThrift support)
if(VELOX_ENABLE_REMOTE_FUNCTIONS)
  add_library(velox_local_runner_service_lib LocalRunnerService.cpp)

  target_link_libraries(
    velox_local_runner_service_lib
    local_runner_service_thrift
    velox_core
    velox_exec
    velox_exec_test_lib
    velox_expression
    velox_functions_prestosql
    velox_common_base
    velox_memory
    Folly::folly
    FBThrift::thriftcpp2
    gflags
    glog::glog
  )

  target_include_directories(velox_local_runner_service_lib PUBLIC ${CMAKE_CURRENT_BINARY_DIR})

  # LocalRunnerService Executable
  add_executable(velox_local_runner_service_runner LocalRunnerServiceRunner.cpp)

  target_link_libraries(
    velox_local_runner_service_runner
    velox_local_runner_service_lib
    velox_functions_prestosql
    gtest
    gflags
    Folly::folly
  )
endif()

if(${VELOX_BUILD_TESTING})
  add_subdirectory(tests)
endif()
