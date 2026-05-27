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

find_package(Python3 REQUIRED COMPONENTS Interpreter)

set(TORCHWAVE_TEST_DATA_DIR ${CMAKE_CURRENT_SOURCE_DIR}/data)

# Generate element_test.pt2 and element_test_results.pt at build time.
add_custom_command(
  OUTPUT
    ${TORCHWAVE_TEST_DATA_DIR}/element_test.pt2
    ${TORCHWAVE_TEST_DATA_DIR}/element_test_results.pt
  COMMAND
    ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/element_test_run_pt2.py --output_dir
    ${TORCHWAVE_TEST_DATA_DIR}
  DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/element_test_run_pt2.py
    ${CMAKE_CURRENT_SOURCE_DIR}/element_test.py
  COMMENT "Generating element_test.pt2 and reference results"
)

add_custom_target(
  element_test_data
  DEPENDS
    ${TORCHWAVE_TEST_DATA_DIR}/element_test.pt2
    ${TORCHWAVE_TEST_DATA_DIR}/element_test_results.pt
)

add_executable(torchwave_executor_test ExecutorTest.cpp)

add_dependencies(torchwave_executor_test element_test_data)

target_link_libraries(
  torchwave_executor_test
  torch_wave
  GTest::gtest
  GTest::gtest_main
  Folly::folly
  CUDA::cudart
)

add_test(
  NAME torchwave_executor_test
  COMMAND torchwave_executor_test
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
set_tests_properties(torchwave_executor_test PROPERTIES LABELS cuda_driver)

add_executable(torchwave_graph_tool GraphTool.cpp)

target_link_libraries(torchwave_graph_tool torch_wave torch_wave_graph_view Folly::folly)
