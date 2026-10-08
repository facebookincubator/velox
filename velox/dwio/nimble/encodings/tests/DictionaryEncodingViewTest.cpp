/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "velox/dwio/nimble/encodings/tests/EncodingViewTestUtils.h"

#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "velox/dwio/nimble/encodings/DictionaryEncoding.h"

using namespace facebook;

using EncodingViewTest = nimble::test::EncodingViewTest;
using DictionaryEncodingViewTest = nimble::test::EncodingViewTest;

TEST_F(EncodingViewTest, readsDictionaryEncoding) {
  expectReads<nimble::DictionaryEncoding<int32_t>>(
      makeVector({5, 7, 5, 9, 7, 5, 11}), {6, 0, 3, 2, 5});
}

TEST_F(EncodingViewTest, readsDictionaryEncodingForSupportedTypeFamilies) {
  expectReads<nimble::DictionaryEncoding<uint32_t>>(
      makeVector<uint32_t>({5, 7, 5, 9, 7, 5, 11}), {6, 0, 3, 2, 5});
  expectReads<nimble::DictionaryEncoding<float>>(
      makeVector<float>({1.25F, 2.5F, 1.25F, -3.75F, 2.5F}), {4, 0, 3, 2, 1});
  expectReads<nimble::DictionaryEncoding<double>>(
      makeVector<double>({1.25, 2.5, 1.25, -3.75, 2.5}), {4, 0, 3, 2, 1});
  expectReads<nimble::DictionaryEncoding<std::string_view>>(
      makeVector<std::string_view>({"alpha", "beta", "alpha", "gamma", "beta"}),
      {4, 0, 3, 2, 1});
}

TEST_F(DictionaryEncodingViewTest, concurrent) {
  const auto positions = randomizedPositions(/*seed=*/8);
  std::vector<std::string> backing;

  expectConcurrentReads<nimble::DictionaryEncoding<uint32_t>>(
      randomDictionaryUint32(/*seed=*/9), positions);
  expectConcurrentReads<nimble::DictionaryEncoding<std::string_view>>(
      randomStringViews(backing, /*seed=*/10), positions);
}
