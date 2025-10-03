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
add_executable(velox_cudf_queue_mgr_test CudfOutputQueueManagerTest.cpp
                                         Main.cpp)

add_test(
  NAME velox_cudf_queue_mgr_test
  COMMAND velox_cudf_queue_mgr_test
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

set_tests_properties(velox_cudf_queue_mgr_test PROPERTIES LABELS cuda_driver
                                                          TIMEOUT 3000)

target_link_libraries(
  velox_cudf_queue_mgr_test
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
  ucxx::ucxx)

add_executable(exchange_srv_tst ExchangeServerMain.cpp)
target_compile_options(exchange_srv_tst PRIVATE -Wno-missing-field-initializers)
add_executable(exchange_client_tst ExchangeClientMain.cpp common.cpp
                                   TableWithNames.cpp)
target_compile_options(exchange_client_tst
                       PRIVATE -Wno-missing-field-initializers)

target_link_libraries(
  exchange_srv_tst
  velox_cudf_exchange
  velox_cudf_exec
  velox_exec
  velox_exec_test_lib
  velox_test_util
  Folly::folly
  fmt::fmt
  ucxx::ucxx)

target_link_libraries(
  exchange_client_tst
  velox_cudf_exchange
  velox_cudf_exec
  velox_exec
  velox_exec_test_lib
  velox_test_util
  Folly::folly
  fmt::fmt
  ucxx::ucxx
  gflags)

add_custom_target(exchange_tests)
add_dependencies(
  exchange_tests
  exchange_srv_tst
  exchange_client_tst
  1brc_server
  1brc_client)
