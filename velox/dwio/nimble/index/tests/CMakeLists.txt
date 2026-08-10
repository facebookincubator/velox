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
add_library(nimble_index_test_utils ClusterIndexTestUtils.cpp)
target_link_libraries(
  nimble_index_test_utils
  nimble_index
  nimble_writer
  nimble_common
  velox_memory
  velox_vector_fuzzer
  velox_file
  Folly::folly
)

add_library(nimble_index_test_base ClusterIndexTestBase.cpp)
target_link_libraries(
  nimble_index_test_base
  nimble_index
  nimble_writer
  velox_core
  velox_vector
  velox_memory
  velox_dwio_common
  gtest
  Folly::folly
)

add_executable(
  nimble_index_tests
  BloomFilterTest.cpp
  IndexConstantsTest.cpp
  IndexConfigTest.cpp
  IndexFilterTest.cpp
  IndexLookupTest.cpp
  ClusterIndexTest.cpp
  KeyChunkDecoderTest.cpp
  ClusterIndexTestBase.cpp
)

add_test(nimble_index_tests nimble_index_tests)

target_link_libraries(
  nimble_index_tests
  nimble_index
  nimble_index_test_utils
  nimble_tablet_reader
  nimble_tablet_writer
  nimble_velox_reader
  nimble_writer
  velox_core
  velox_file
  velox_vector
  velox_vector_test_lib
  velox_memory
  velox_dwio_common
  gmock
  gtest
  gtest_main
  Folly::folly
)
