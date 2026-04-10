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

add_library(velox_aggregation_result_verifier MinMaxByResultVerifier.cpp)
velox_add_test_headers(
  velox_aggregation_result_verifier
  ApproxDistinctResultVerifier.h
  ApproxPercentileResultVerifier.h
  ArbitraryResultVerifier.h
  AverageResultVerifier.h
  KHyperLogLogResultVerifier.h
  MinMaxByResultVerifier.h
  NoisyAvgResultVerifier.h
  NoisyCountIfResultVerifier.h
  NoisyCountResultVerifier.h
  NoisySumResultVerifier.h
  QDigestAggResultVerifier.h
  SetDigestResultVerifier.h
  TDigestAggregateResultVerifier.h
)

target_link_libraries(
  velox_aggregation_result_verifier
  velox_common_hyperloglog
  velox_core
  velox_aggregation_fuzzer_base
  velox_exec_test_lib
  velox_functions_lib
  velox_vector
  velox_vector_test_lib
)

add_executable(velox_aggregation_fuzzer_test AggregationFuzzerTest.cpp)

target_link_libraries(
  velox_aggregation_fuzzer_test
  velox_aggregation_fuzzer
  velox_aggregation_fuzzer_base
  velox_aggregation_result_verifier
  velox_aggregates
  velox_window
  velox_vector_fuzzer
  velox_vector_test_lib
  GTest::gtest
  GTest::gtest_main
)

# Window Fuzzer
add_executable(velox_window_fuzzer_test WindowFuzzerTest.cpp)

target_link_libraries(
  velox_window_fuzzer_test
  velox_window_fuzzer
  velox_fuzzer_util
  velox_aggregates
  velox_window
  velox_expression_test_utility
  velox_functions_prestosql
  velox_vector_test_lib
  GTest::gtest
  GTest::gtest_main
)

# Writer Fuzzer
add_executable(velox_writer_fuzzer_test WriterFuzzerTest.cpp)

target_link_libraries(
  velox_writer_fuzzer_test
  velox_writer_fuzzer
  velox_fuzzer_util
  GTest::gtest
  GTest::gtest_main
)

velox_add_library(
  velox_aggregation_input_generator
  INTERFACE
  HEADERS
  ApproxDistinctInputGenerator.h
  ApproxPercentileInputGenerator.h
  ClassificationAggregationInputGenerator.h
  DivideArgTypesGenerator.h
  FbDatelistStringMergeAggregationInputGenerator.h
  FloorCeilRoundArgTypesGenerator.h
  MapUnionSumInputGenerator.h
  MinMaxInputGenerator.h
  ModulusArgTypesGenerator.h
  MultiplyArgTypesGenerator.h
  NoisyAvgInputGenerator.h
  NoisyCountIfInputGenerator.h
  NoisyCountInputGenerator.h
  NoisySumInputGenerator.h
  NumericHistogramInputGenerator.h
  PlusMinusArgTypesGenerator.h
  QDigestAggInputGenerator.h
  SkipIPAddressArgTypesGenerator.h
  TDigestAggregateInputGenerator.h
  TruncateArgTypesGenerator.h
  WindowOffsetInputGenerator.h
)

velox_add_library(velox_custome_expression_transformer INTERFACE HEADERS SortArrayTransformer.h)
