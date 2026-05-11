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
  ucx_exchange_test
  UcxOutputQueueManagerTest.cpp
  UcxExchangeTest.cpp
  Main.cpp
  UcxTestHelpers.cpp
  UcxTestData.cpp
  UcxPartitionedOutputMock.cpp
  SinkDriverMock.cpp
  SourceDriverMock.cpp
)

add_test(
  NAME ucx_exchange_test
  COMMAND ucx_exchange_test
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

set_tests_properties(ucx_exchange_test PROPERTIES LABELS cuda_driver TIMEOUT 3000)

target_link_libraries(
  ucx_exchange_test
  velox_ucx_exchange
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
