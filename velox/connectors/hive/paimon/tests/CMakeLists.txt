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
  add_executable(velox_hive_paimon_split_reader_test PaimonSplitReaderTest.cpp)
  add_test(velox_hive_paimon_split_reader_test velox_hive_paimon_split_reader_test)

  target_link_libraries(
    velox_hive_paimon_split_reader_test
    velox_hive_paimon_connector
    velox_dwio_orc_reader
    velox_exec_test_lib
    velox_vector_test_lib
    GTest::gtest
    GTest::gtest_main
  )

  if(VELOX_ENABLE_NIMBLE)
    target_include_directories(
      velox_hive_paimon_split_reader_test
      BEFORE
      PRIVATE "${PROJECT_SOURCE_DIR}/velox/external/fsst"
    )
    target_compile_definitions(
      velox_hive_paimon_split_reader_test
      PRIVATE VELOX_ENABLE_NIMBLE DISABLE_META_INTERNAL_COMPRESSOR=1
    )
    if(NIMBLE_ENABLE_EXPERIMENTAL_ENCODINGS)
      target_compile_definitions(
        velox_hive_paimon_split_reader_test
        PRIVATE NIMBLE_ENABLE_EXPERIMENTAL_ENCODINGS
      )
    endif()
    target_link_libraries(velox_hive_paimon_split_reader_test nimble_writer nimble_velox_selective)
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
