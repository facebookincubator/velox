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
add_executable(velox_file_connector_test FileConnectorTest.cpp)
add_test(velox_file_connector_test velox_file_connector_test)
target_link_libraries(
  velox_file_connector_test
  velox_file_connector
  velox_test_util
  GTest::gtest
  GTest::gtest_main
)

add_executable(velox_file_split_reader_test FileSplitReaderTest.cpp)
add_test(velox_file_split_reader_test velox_file_split_reader_test)
target_link_libraries(
  velox_file_split_reader_test
  velox_file_connector
  velox_dwio_dwrf_reader
  velox_dwio_orc_reader
  velox_vector_test_lib
  GTest::gtest
  GTest::gtest_main
)
if(VELOX_ENABLE_PARQUET)
  target_include_directories(velox_file_split_reader_test PUBLIC ${ARROW_PREFIX}/install/include)
  target_link_libraries(velox_file_split_reader_test velox_dwio_parquet_reader)
endif()
