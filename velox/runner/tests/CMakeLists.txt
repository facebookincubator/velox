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

add_executable(velox_local_runner_test LocalRunnerTest.cpp Main.cpp)

add_test(velox_local_runner_test velox_local_runner_test)

target_link_libraries(
  velox_local_runner_test
  velox_exec_runner_test_util
  velox_exec_test_lib
  velox_parse_parser
  velox_parse_expression
  GTest::gtest)

velox_add_library(velox_query_replay_runner QueryReplayRunner.cpp)

velox_link_libraries(
  velox_query_replay_runner
  velox_exec
  velox_exec_test_lib
  velox_local_runner
)

add_executable(velox_query_replay_runner_test QueryReplayRunnerTest.cpp)

add_test(velox_query_replay_runner_test velox_query_replay_runner_test)

target_link_libraries(velox_query_replay_runner_test velox_query_replay_runner
                      velox_exec_test_lib
)
