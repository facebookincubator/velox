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
  nimble_tablet_tests
  CompressionTest.cpp
  FileFeaturesTest.cpp
  FileLayoutTest.cpp
  MetadataBufferTest.cpp
  TabletTest.cpp
)

add_test(nimble_tablet_tests nimble_tablet_tests)

target_link_libraries(
  nimble_tablet_tests
  nimble_tablet_reader
  nimble_tablet_writer
  nimble_index
  nimble_index_test_utils
  nimble_common
  velox_dwio_common
  velox_memory
  velox_file
  gtest
  gtest_main
  glog::glog
  Folly::folly
)
