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
add_executable(velox_key_encoder_test KeyEncoderTest.cpp)

add_test(
  NAME velox_key_encoder_test
  COMMAND velox_key_encoder_test
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(
  velox_key_encoder_test
  velox_exec
  velox_exec_test_lib
  velox_key_encoder
  velox_functions_prestosql
  velox_vector_fuzzer
  velox_vector_test_lib
  GTest::gtest
  GTest::gtest_main
  GTest::gmock
)

add_executable(
  velox_serializer_test
  CompactRowSerializerTest.cpp
  PrestoOutputStreamListenerTest.cpp
  PrestoSerializerTest.cpp
  SerializedPageFileTest.cpp
  UnsafeRowSerializerTest.cpp
)

add_test(velox_serializer_test velox_serializer_test)

target_link_libraries(
  velox_serializer_test
  velox_expression_test_utility
  velox_presto_serializer
  velox_vector_test_lib
  velox_vector_fuzzer
  velox_row_fast
  GTest::gtest
  GTest::gtest_main
  glog::glog
)

if(VELOX_ENABLE_BENCHMARKS)
  add_executable(velox_serializer_benchmark SerializerBenchmark.cpp)

  target_link_libraries(
    velox_serializer_benchmark
    velox_presto_serializer
    velox_vector_test_lib
    velox_vector_fuzzer
    GTest::gtest
    GTest::gtest_main
    glog::glog
  )
endif()
