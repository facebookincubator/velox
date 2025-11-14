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

add_executable(velox_dwio_orc_reader_test ReaderTest.cpp)
add_test(
  NAME velox_dwio_orc_reader_test
  COMMAND velox_dwio_orc_reader_test
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(
  velox_dwio_orc_reader_test
  velox_dwrf_test_utils
  velox_dwio_common_test_utils
  GTest::gtest
  GTest::gtest_main
  GTest::gmock
)

add_executable(velox_dwio_orc_reader_filter_test ReaderFilterTest.cpp)
add_test(
  NAME velox_dwio_orc_reader_filter_test
  COMMAND velox_dwio_orc_reader_filter_test
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(
  velox_dwio_orc_reader_filter_test
  velox_dwrf_test_utils
  velox_dwio_common_test_utils
  GTest::gtest
  GTest::gtest_main
  GTest::gmock
)

add_executable(velox_orc_e2e_writer_test E2EWriterTest.cpp)
add_test(velox_orc_e2e_writer_test velox_orc_e2e_writer_test)

target_link_libraries(
  velox_orc_e2e_writer_test
  velox_dwrf_test_utils
  velox_vector_fuzzer
  velox_vector_test_lib
  velox_link_libs
  Folly::folly
  fmt::fmt
  lz4::lz4
  zstd::zstd
  ZLIB::ZLIB
  GTest::gtest
  GTest::gtest_main
  GTest::gmock
)

add_executable(velox_orc_e2e_filter_test E2EFilterTest.cpp)
add_test(velox_orc_e2e_filter_test velox_orc_e2e_filter_test)

target_link_libraries(
  velox_orc_e2e_filter_test
  velox_dwio_common_test_utils
  velox_dwrf_test_utils
  velox_link_libs
  velox_test_util
  Folly::folly
  fmt::fmt
  lz4::lz4
  zstd::zstd
  ZLIB::ZLIB
  GTest::gtest
  GTest::gtest_main
  GTest::gmock
)

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/examples DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
