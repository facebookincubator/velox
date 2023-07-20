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

add_executable(velox_example_simple_functions SimpleFunctions.cpp)

target_link_libraries(
  velox_example_simple_functions PUBLIC velox_core velox_expression
                                        velox_functions_lib re2::re2)

add_executable(velox_example_expression_eval ExpressionEval.cpp)

target_link_libraries(
  velox_example_expression_eval PUBLIC velox_expression velox_memory velox_type
                                       velox_vector)

add_executable(velox_example_opaque_type OpaqueType.cpp)

target_link_libraries(
  velox_example_opaque_type PUBLIC velox_expression velox_memory velox_type
                                   velox_vector)

# This is disabled temporarily until we figure out why g++ is crashing linking
# it on linux builds.

# add_executable(velox_example_scan_and_sort ScanAndSort.cpp)

# target_link_libraries( velox_example_scan_and_sort velox_type velox_vector
# velox_exec velox_exec_test_lib velox_hive_connector velox_memory
# velox_vector_test_lib)

add_executable(velox_example_scan_orc ScanOrc.cpp)

target_link_libraries(
  velox_example_scan_orc
  PUBLIC velox_exec velox_exec_test_lib velox_hive_connector velox_memory
         velox_type velox_vector)

add_executable(velox_example_vector_reader_writer VectorReaderWriter.cpp)

target_link_libraries(velox_example_vector_reader_writer
                      PUBLIC velox_expression velox_type velox_vector)

add_executable(velox_example_operator_extensibility OperatorExtensibility.cpp)

target_link_libraries(
  velox_example_operator_extensibility
  PUBLIC velox_exec
         velox_exec_test_lib
         velox_memory
         velox_type
         velox_vector
         velox_vector_test_lib
         velox_vector_fuzzer)
