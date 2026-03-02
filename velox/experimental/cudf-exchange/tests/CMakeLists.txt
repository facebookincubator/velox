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

add_executable(
  cudf_exchange_test
  CudfOutputQueueManagerTest.cpp
  CudfExchangeTest.cpp
  Main.cpp
  CudfTestHelpers.cpp
  CudfTestData.cpp
  CudfPartitionedOutputMock.cpp
  SinkDriverMock.cpp
  SourceDriverMock.cpp
)

add_test(
  NAME cudf_exchange_test
  COMMAND cudf_exchange_test
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

set_tests_properties(cudf_exchange_test PROPERTIES LABELS cuda_driver TIMEOUT 3000)

target_link_libraries(
  cudf_exchange_test
  velox_cudf_exchange
  velox_cudf_exec
  velox_exec
  velox_exec_test_lib
  velox_test_util
  velox_vector_fuzzer
  gtest
  gtest_main
  Folly::folly
  fmt::fmt
  ucxx::ucxx
)

add_executable(1brc_server 1brc_server.cpp)
target_compile_options(1brc_server PRIVATE -Wno-missing-field-initializers)

add_executable(1brc_client 1brc_client.cpp common.cpp TableWithNames.cpp)
target_compile_options(1brc_client PRIVATE -Wno-missing-field-initializers)

target_link_libraries(
  1brc_server
  ucxx::ucxx
  velox_cudf_exchange
  velox_aggregates
  velox_type
  velox_vector
  velox_vector_test_lib
  velox_exec
  velox_exec_test_lib
  velox_tpch_connector
  velox_memory
  velox_cudf_exec
)

target_link_libraries(
  1brc_client
  ucxx::ucxx
  velox_cudf_exchange
  velox_aggregates
  velox_type
  velox_vector
  velox_vector_test_lib
  velox_exec
  velox_exec_test_lib
  velox_tpch_connector
  velox_memory
  velox_cudf_exec
)

add_custom_target(exchange_tests)
add_dependencies(exchange_tests 1brc_server 1brc_client)

# Wide table exchange server and client for profiling
add_executable(
  wide_table_exchange_server
  WideTableExchangeServer.cpp
  SourceDriverMock.cpp
  CudfTestHelpers.cpp
  CudfTestData.cpp
)
target_compile_options(wide_table_exchange_server PRIVATE -Wno-missing-field-initializers)

add_executable(
  wide_table_exchange_client
  WideTableExchangeClient.cpp
  SinkDriverMock.cpp
  CudfTestHelpers.cpp
  CudfTestData.cpp
)
target_compile_options(wide_table_exchange_client PRIVATE -Wno-missing-field-initializers)

target_link_libraries(
  wide_table_exchange_server
  velox_cudf_exchange
  velox_cudf_exec
  velox_exec
  velox_exec_test_lib
  velox_test_util
  Folly::folly
  fmt::fmt
  ucxx::ucxx
  gflags
  nvtx3-cpp
)

target_link_libraries(
  wide_table_exchange_client
  velox_cudf_exchange
  velox_cudf_exec
  velox_exec
  velox_exec_test_lib
  velox_test_util
  Folly::folly
  fmt::fmt
  ucxx::ucxx
  gflags
  nvtx3-cpp
)

add_dependencies(exchange_tests wide_table_exchange_server wide_table_exchange_client)
