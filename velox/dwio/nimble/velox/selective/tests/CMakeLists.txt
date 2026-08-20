# Copyright (c) Meta Platforms, Inc. and affiliates.
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
  nimble_velox_selective_tests
  ChunkedDecoderTest.cpp
  # This requires VELOX_BUILD_TEST_UTILS=ON but it also requires
  # VELOX_ENABLE_EXEC=ON. VELOX_ENABLE_EXEC is OFF by
  # VELOX_BUILD_MINIMAL_WITH_DWIO=ON.
  #
  # E2EFilterTest.cpp
  # E2EIndexTest.cpp
  FlatMapColumnReaderTest.cpp
  NimbleDataTest.cpp
  NullColumnReaderTest.cpp
  ScalarColumnReaderTest.cpp
  SelectiveNimbleReaderTest.cpp
  StringColumnReaderTest.cpp
  TimestampColumnReaderTest.cpp
  VariableLengthColumnReaderTest.cpp
)
add_test(nimble_velox_selective_tests nimble_velox_selective_tests)

target_link_libraries(
  nimble_velox_selective_tests
  nimble_velox_selective
  nimble_common_file_writer
  nimble_encoding_layout_test_helper
  nimble_index_test_base
  velox_vector_test_lib
  velox_vector_fuzzer
  # See also the above VELOX_BUILD_TEST_UTILS=ON comment.
  #
  # velox_dwio_common_test_utils
  gmock
  gtest
  gtest_main
  Folly::folly
)
