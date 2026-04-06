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

add_library(velox_cudf_iceberg_test_lib CudfIcebergTestBase.cpp)

set_target_properties(velox_cudf_iceberg_test_lib PROPERTIES CUDA_ARCHITECTURES native)

target_link_libraries(
  velox_cudf_iceberg_test_lib
  velox_cudf_exec_test_lib
  velox_cudf_iceberg_connector
  velox_hive_iceberg_splitreader
  velox_exec_test_lib
  velox_test_util
)

# End-to-end read tests (connector-level tests with full Velox pipeline).
add_executable(
  velox_cudf_iceberg_read_test
  Main.cpp
  CudfIcebergReadTest.cpp
  CudfEqualityDeleteFileReaderTest.cpp
)

target_link_libraries(
  velox_cudf_iceberg_read_test
  velox_cudf_iceberg_test_lib
  velox_cudf_iceberg_connector
  velox_cudf_exec_test_lib
  velox_exec
  velox_exec_test_lib
  velox_test_util
  gtest
  gtest_main
  fmt::fmt
)

add_test(
  NAME velox_cudf_iceberg_read_test
  COMMAND velox_cudf_iceberg_read_test
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

set_tests_properties(velox_cudf_iceberg_read_test PROPERTIES LABELS cuda_driver TIMEOUT 3000)

# Unit tests for CudfDeletionVectorReader (standalone, no connector pipeline).
add_executable(velox_cudf_iceberg_deletion_vector_test CudfDeletionVectorReaderTest.cpp)

target_link_libraries(
  velox_cudf_iceberg_deletion_vector_test
  velox_cudf_iceberg_connector
  velox_cudf_exec_test_lib
  velox_exec
  velox_exec_test_lib
  velox_test_util
  gtest
  gtest_main
  fmt::fmt
)

add_test(
  NAME velox_cudf_iceberg_deletion_vector_test
  COMMAND velox_cudf_iceberg_deletion_vector_test
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

set_tests_properties(
  velox_cudf_iceberg_deletion_vector_test
  PROPERTIES LABELS cuda_driver TIMEOUT 3000
)
