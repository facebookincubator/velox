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

add_executable(presto_sql_test PrestoSqlTest.cpp)
add_test(presto_sql_test presto_sql_test)

target_link_libraries(presto_sql_test velox_fuzzer_util velox_presto_types)

# LocalRunnerService Test (requires FBThrift support)
if(VELOX_ENABLE_REMOTE_FUNCTIONS)
  add_executable(local_runner_service_test LocalRunnerServiceTest.cpp)
  add_test(local_runner_service_test local_runner_service_test)

  target_link_libraries(
    local_runner_service_test
    velox_local_runner_service_lib
    local_runner_service_thrift
    velox_core
    velox_type
    velox_functions_prestosql
    velox_functions_test_lib
    velox_vector_test_lib
    velox_common_base
    Folly::folly
    gtest
    gtest_main
  )
endif()
