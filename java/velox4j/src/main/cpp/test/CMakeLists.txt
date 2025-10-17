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
set(BUILD_TESTING ON)
include(CTest)

add_library(velox4j_test INTERFACE)
target_include_directories(velox4j_test INTERFACE ${CMAKE_CURRENT_LIST_DIR})
target_link_libraries(
  velox4j_test
  INTERFACE
    velox4j
    velox_vector_test_lib
    velox_dwio_common_test_utils
    velox_file_test_utils
    velox_temp_path
    velox_exec_test_lib)
set(TEST_OBJECT_FILES $<TARGET_OBJECTS:velox_fuzzer_connector>)

add_executable(velox4j_query_serde_test ${TEST_OBJECT_FILES}
                                        velox4j/query/QuerySerdeTest.cpp)
target_link_libraries(
  velox4j_query_serde_test velox4j_test)
add_test(velox4j_query_serde_test velox4j_query_serde_test)

add_executable(velox4j_query_test ${TEST_OBJECT_FILES}
                                  velox4j/query/QueryTest.cpp)
target_link_libraries(
  velox4j_query_test velox4j_test)
add_test(velox4j_query_test velox4j_query_test)

add_executable(velox4j_blocking_queue_test
               ${TEST_OBJECT_FILES} velox4j/iterator/BlockingQueueTest.cpp)
target_link_libraries(
  velox4j_blocking_queue_test velox4j_test)
add_test(velox4j_blocking_queue_test velox4j_blocking_queue_test)
