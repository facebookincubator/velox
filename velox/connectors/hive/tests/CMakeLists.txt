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
  velox_hive_connector_test
  FileColumnHandleTest.cpp
  FileConfigTest.cpp
  FileDataSourceTest.cpp
  HiveIndexSourceUtilsTest.cpp
  FileConnectorSplitTest.cpp
  FileConnectorUtilTest.cpp
  FileHandleTest.cpp
  HiveConfigTest.cpp
  HiveDataSinkTest.cpp
  HiveConnectorTest.cpp
  HiveConnectorUtilTest.cpp
  HiveConnectorSerDeTest.cpp
  HivePartitionFunctionTest.cpp
  HivePartitionNameTest.cpp
  HiveSplitTest.cpp
  PartitionIdGeneratorTest.cpp
  PartitionValueTest.cpp
  TableHandleTest.cpp
)
add_test(velox_hive_connector_test velox_hive_connector_test)

target_link_libraries(
  velox_hive_connector_test
  velox_hive_connector
  velox_hive_partition_function
  velox_dwio_common_exception
  velox_dwio_orc_reader
  velox_vector_fuzzer
  velox_vector_test_lib
  velox_exec
  velox_exec_test_lib
  GTest::gmock
  GTest::gtest
  GTest::gtest_main
)

if(VELOX_ENABLE_PARQUET)
  target_include_directories(velox_hive_connector_test PUBLIC ${ARROW_PREFIX}/install/include)
  target_link_libraries(
    velox_hive_connector_test
    velox_dwio_parquet_writer
    velox_dwio_parquet_reader
  )
endif()

if(VELOX_ENABLE_NIMBLE)
  # These tests include Nimble writer headers outside the Nimble subtree.
  # Use the same OSS header configuration and prefer its FSST over DuckDB's.
  target_include_directories(
    velox_hive_connector_test
    BEFORE
    PRIVATE "${PROJECT_SOURCE_DIR}/velox/external/fsst"
  )
  target_compile_definitions(
    velox_hive_connector_test
    PRIVATE VELOX_ENABLE_NIMBLE DISABLE_META_INTERNAL_COMPRESSOR=1
  )
  if(NIMBLE_ENABLE_EXPERIMENTAL_ENCODINGS)
    target_compile_definitions(
      velox_hive_connector_test
      PRIVATE NIMBLE_ENABLE_EXPERIMENTAL_ENCODINGS
    )
  endif()
  target_link_libraries(velox_hive_connector_test nimble_writer nimble_velox_selective)
endif()
