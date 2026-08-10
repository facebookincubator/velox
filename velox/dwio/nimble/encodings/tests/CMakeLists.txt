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
add_library(
  nimble_encodings_tests_utils
  SharedDictionaryEncodingTestUtils.cpp
  TestUtils.cpp
  ../selection/tests/RandomEncodingSelectionPolicy.cpp
)
target_link_libraries(nimble_encodings_tests_utils nimble_encodings fsst)

add_library(nimble_encoding_layout_test_helper EncodingLayoutTestHelper.cpp)
target_link_libraries(nimble_encoding_layout_test_helper nimble_encodings)

add_executable(
  nimble_encodings_tests
  ALPEncodingViewTest.cpp
  BufferedEncodingTest.cpp
  BlockBitPackingEncodingViewTest.cpp
  ConstantEncodingViewTest.cpp
  ConstantEncodingTest.cpp
  DictionaryEncodingTest.cpp
  DictionaryEncodingViewTest.cpp
  EncodingLayoutTest.cpp
  EncodingSelectionTest.cpp
  EncodingViewFactoryTest.cpp
  CompressionTest.cpp
  EncodingTest.cpp
  EncodingLegacyTest.cpp
  EncodingPrefixTest.cpp
  EncodingTypeDispatchTest.cpp
  FixedBitWidthEncodingViewTest.cpp
  FsstEncodingTest.cpp
  HuffmanEncodingTest.cpp
  HuffmanEncodingViewTest.cpp
  FOREncodingViewTest.cpp
  Lz4CompressorTest.cpp
  ForEncodingTest.cpp
  FrequencyPartitionEncodingTest.cpp
  MainlyConstantEncodingTest.cpp
  MainlyConstantEncodingViewTest.cpp
  NullableEncodingTest.cpp
  OpenZLCompressorTest.cpp
  PFOREncodingViewTest.cpp
  PFOREncodingTest.cpp
  PrefixEncodingTest.cpp
  RLEEncodingViewTest.cpp
  RLEEncodingTest.cpp
  SentinelEncodingTest.cpp
  SharedDictionaryBuilderTest.cpp
  SharedDictionaryEncodingTest.cpp
  SharedDictionaryTypesTest.cpp
  SimdForBitpackEncodingViewTest.cpp
  SliceEncodingTest.cpp
  SparseBoolEncodingViewTest.cpp
  StatisticsTest.cpp
  TrivialEncodingViewTest.cpp
  TrivialEncodingTest.cpp
)

if(NIMBLE_ENABLE_EXPERIMENTAL_ENCODINGS)
  target_sources(nimble_encodings_tests PRIVATE SubIntSplitEncodingTest.cpp)
endif()

add_test(nimble_encodings_tests nimble_encodings_tests)

target_link_libraries(
  nimble_encodings_tests
  nimble_encodings_tests_utils
  nimble_encodings
  nimble_encodings_legacy
  nimble_common
  nimble_tools_common
  velox_common_base
  gmock
  gtest
  gtest_main
  fmt::fmt
  Folly::folly
)
