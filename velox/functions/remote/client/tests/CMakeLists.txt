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

add_executable(velox_functions_remote_client_thrift_test
               RemoteFunctionThriftTest.cpp)

add_test(velox_functions_remote_client_thrift_test
         velox_functions_remote_client_thrift_test)

target_link_libraries(
  velox_functions_remote_client_thrift_test
  velox_remote_thrift_libs
  velox_function_registry
  velox_functions_test_lib
  velox_exec_test_lib
  velox_functions_remote_utils
  GTest::gmock
  GTest::gtest
  GTest::gtest_main)

add_executable(velox_functions_remote_client_rest_test
               RemoteFunctionRestTest.cpp)

add_test(velox_functions_remote_client_rest_test
         velox_functions_remote_client_rest_test)

target_link_libraries(
  velox_functions_remote_client_rest_test
  velox_functions_remote_rest_client
  velox_functions_remote_rest_server
  velox_functions_remote
  velox_functions_test_lib
  velox_exec_test_lib
  GTest::gmock
  GTest::gtest
  GTest::gtest_main)
