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

add_executable(velox_mock_rpc_client_test MockRPCClientTest.cpp)

target_link_libraries(
  velox_mock_rpc_client_test
  velox_mock_rpc_client
  GTest::gtest
  GTest::gtest_main
  Folly::folly
)

add_test(NAME velox_mock_rpc_client_test COMMAND velox_mock_rpc_client_test)

add_executable(velox_rpc_state_test RPCStateTest.cpp)

target_link_libraries(
  velox_rpc_state_test
  velox_rpc_state
  velox_rpc_types
  GTest::gtest
  GTest::gtest_main
  Folly::folly
)

add_test(NAME velox_rpc_state_test COMMAND velox_rpc_state_test)

add_executable(velox_rpc_node_test RPCNodeTest.cpp)

target_link_libraries(
  velox_rpc_node_test
  velox_mock_rpc_client
  velox_core
  velox_memory
  velox_vector
  GTest::gtest
  GTest::gtest_main
  Folly::folly
)

add_test(NAME velox_rpc_node_test COMMAND velox_rpc_node_test)

add_executable(velox_demo_rpc_function_test DemoRPCFunctionTest.cpp)
velox_add_test_headers(velox_demo_rpc_function_test DemoRPCFunction.h)

target_link_libraries(
  velox_demo_rpc_function_test
  velox_demo_rpc_function
  velox_core
  velox_memory
  velox_vector
  GTest::gtest
  GTest::gtest_main
  Folly::folly
)

add_test(NAME velox_demo_rpc_function_test COMMAND velox_demo_rpc_function_test)

add_executable(velox_rpc_rate_limiter_test RPCRateLimiterTest.cpp)

target_link_libraries(
  velox_rpc_rate_limiter_test
  velox_rpc_rate_limiter
  GTest::gtest
  GTest::gtest_main
)

add_test(NAME velox_rpc_rate_limiter_test COMMAND velox_rpc_rate_limiter_test)

add_executable(velox_rpc_operator_test RPCOperatorTest.cpp Main.cpp)

target_link_libraries(
  velox_rpc_operator_test
  velox_demo_rpc_function
  velox_rpc_plan_node_translator
  velox_async_rpc_function_registry
  velox_exec_test_lib
  GTest::gtest
  Folly::folly
)

add_test(NAME velox_rpc_operator_test COMMAND velox_rpc_operator_test)
