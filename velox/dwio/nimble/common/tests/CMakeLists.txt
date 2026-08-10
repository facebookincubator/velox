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
add_library(nimble_common_file_writer NimbleFileWriter.cpp)

target_link_libraries(
  nimble_common_file_writer
  nimble_common
  nimble_writer
  velox_vector
)

add_executable(
  nimble_common_tests
  BitEncoderTests.cpp
  BitsTests.cpp
  ChunkHeaderTest.cpp
  ConstantsTest.cpp
  DataTypeDispatchTest.cpp
  ExceptionHelperTest.cpp
  ExceptionTests.cpp
  FeatureGateTest.cpp
  FixedBitArrayTests.cpp
  VarintTests.cpp
  VectorTest.cpp
)

add_test(nimble_common_tests nimble_common_tests)

target_link_libraries(
  nimble_common_tests
  nimble_common
  velox_common_base
  gtest
  gtest_main
  glog::glog
  Folly::folly
)
