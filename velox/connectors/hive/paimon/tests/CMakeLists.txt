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

if(NOT VELOX_DISABLE_GOOGLETEST)
  add_executable(velox_hive_paimon_connector_test Main.cpp PaimonConnectorTest.cpp)
  add_test(velox_hive_paimon_connector_test velox_hive_paimon_connector_test)
  target_link_libraries(
    velox_hive_paimon_connector_test
    velox_hive_paimon_connector
    velox_exec_test_lib
    velox_vector_fuzzer
    velox_vector_test_lib
    velox_dwio_orc_reader
    GTest::gtest
    GTest::gmock
  )
  if(VELOX_ENABLE_PARQUET)
    target_compile_definitions(
      velox_hive_paimon_connector_test
      PRIVATE
        VELOX_ENABLE_PARQUET
        PAIMON_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/fixtures/append"
    )
    target_link_libraries(velox_hive_paimon_connector_test velox_dwio_parquet_reader)
  endif()

  add_executable(velox_hive_paimon_split_test PaimonConnectorSplitTest.cpp)
  add_test(velox_hive_paimon_split_test velox_hive_paimon_split_test)

  target_link_libraries(
    velox_hive_paimon_split_test
    velox_hive_paimon_split
    velox_dwio_common
    GTest::gtest
    GTest::gtest_main
    GTest::gmock
  )

  add_executable(velox_hive_paimon_data_file_meta_test PaimonDataFileMetaTest.cpp)
  add_test(velox_hive_paimon_data_file_meta_test velox_hive_paimon_data_file_meta_test)

  target_link_libraries(
    velox_hive_paimon_data_file_meta_test
    velox_hive_paimon_split
    GTest::gtest
    GTest::gtest_main
    GTest::gmock
  )

  add_executable(velox_hive_paimon_deletion_file_test PaimonDeletionFileTest.cpp)
  add_test(velox_hive_paimon_deletion_file_test velox_hive_paimon_deletion_file_test)

  target_link_libraries(
    velox_hive_paimon_deletion_file_test
    velox_hive_paimon_split
    GTest::gtest
    GTest::gtest_main
    GTest::gmock
  )

  add_executable(velox_hive_paimon_row_kind_test PaimonRowKindTest.cpp)
  add_test(velox_hive_paimon_row_kind_test velox_hive_paimon_row_kind_test)

  target_link_libraries(
    velox_hive_paimon_row_kind_test
    velox_hive_paimon_split
    GTest::gtest
    GTest::gtest_main
    fmt::fmt
  )
endif()
