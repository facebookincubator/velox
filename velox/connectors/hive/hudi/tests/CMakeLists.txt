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
  add_executable(velox_hive_hudi_log_reader_test HudiLogFileReaderTest.cpp)
  add_test(velox_hive_hudi_log_reader_test velox_hive_hudi_log_reader_test)

  target_link_libraries(
    velox_hive_hudi_log_reader_test
    velox_hive_connector
    velox_hive_hudi_splitreader
    velox_exec_test_lib
    velox_dwio_common_test_utils
    GTest::gtest
    GTest::gtest_main
  )

  add_executable(velox_hive_hudi_delete_block_decoder_test HudiDeleteBlockDecoderTest.cpp)
  add_test(velox_hive_hudi_delete_block_decoder_test velox_hive_hudi_delete_block_decoder_test)

  target_link_libraries(
    velox_hive_hudi_delete_block_decoder_test
    velox_hive_connector
    velox_hive_hudi_splitreader
    velox_exec_test_lib
    velox_dwio_common_test_utils
    GTest::gmock
    GTest::gtest
    GTest::gtest_main
  )
endif()
