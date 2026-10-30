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
#include <tuple>
#include <vector>

#include <gtest/gtest.h>

#include "velox/dwio/nimble/common/tests/GTestUtils.h"
#include "velox/dwio/nimble/encodings/TrivialEncoding.h"

using namespace facebook;

using TrivialEncodingViewTest = nimble::test::EncodingViewTest;

TEST_P(TrivialEncodingViewTest, readsTrivialEncoding) {
  expectReads<nimble::TrivialEncoding<int32_t>>(
      makeVector({10, 20, 30, 40, 50}), {4, 1, 3, 0, 1});
}

TEST_P(TrivialEncodingViewTest, readsTrivialEncodingForAllLayoutFamilies) {
  expectReads<nimble::TrivialEncoding<double>>(
      makeVector<double>({1.25, -0.5, 3.0, 1024.125}), {3, 0, 2, 1});
  expectReads<nimble::TrivialEncoding<bool>>(
      makeVector<bool>({true, false, true, true, false}), {4, 0, 1, 3, 2});
  expectReads<nimble::TrivialEncoding<std::string_view>>(
      makeVector<std::string_view>({"alpha", "", "beta", "longer-value"}),
      {3, 0, 1, 2});
}

TEST_P(TrivialEncodingViewTest, lowerBoundSearchesRange) {
  const auto values = makeVector<int32_t>({-7, -2, -2, 0, 5, 1, 3});
  nimble::Encoding::Options options{.cacheEncodingView = cacheEncodingView()};
  const auto serialized =
      nimble::test::Encoder<nimble::TrivialEncoding<int32_t>>::encode(
          *buffer_, values, nimble::CompressionType::Uncompressed, options);
  auto view = nimble::createEncodingView(serialized, pool_.get(), options);
  const auto& typedView =
      dynamic_cast<const nimble::TypedEncodingView<int32_t>&>(*view);

  for (const auto& [begin, end, value, expected] :
       std::vector<std::tuple<uint32_t, uint32_t, int32_t, uint32_t>>{
           {0, 5, -8, 0},
           {0, 5, -2, 1},
           {0, 5, -1, 3},
           {1, 3, -2, 1},
           {0, 5, 6, 5},
           {5, 7, 2, 6},
       }) {
    SCOPED_TRACE(
        ::testing::Message()
        << "begin=" << begin << ", end=" << end << ", value=" << value);
    int32_t found{0};
    const auto row = typedView.lowerBound(begin, end, value, found);
    EXPECT_EQ(row, expected);
    if (row < end) {
      EXPECT_EQ(found, values[row]);
    }
  }
  int32_t found{0};
  EXPECT_EQ(typedView.lowerBound(7, 7, 0, found), 7);
  NIMBLE_ASSERT_THROW(
      typedView.lowerBound(3, 2, 0, found), "Invalid lowerBound range");
  NIMBLE_ASSERT_THROW(
      typedView.lowerBound(0, 8, 0, found), "Invalid lowerBound range");
}

DEBUG_ONLY_TEST_P(TrivialEncodingViewTest, lowerBoundRejectsUnsortedRange) {
  // The search probes rows 2 and 3, both below 4, and returns 4. Row 0 is
  // above 4, so the range cannot be sorted.
  const auto values = makeVector<int32_t>({5, 1, 2, 3});
  const auto serialized =
      nimble::test::Encoder<nimble::TrivialEncoding<int32_t>>::encode(
          *buffer_, values);
  auto view = nimble::createEncodingView(serialized, pool_.get());
  const auto& typedView =
      dynamic_cast<const nimble::TypedEncodingView<int32_t>&>(*view);
  int32_t found{0};
  NIMBLE_ASSERT_THROW(
      typedView.lowerBound(0, 4, 4, found),
      "lowerBound range must be non-decreasing");
  EXPECT_EQ(typedView.lowerBound(1, 4, 2, found), 2);
}

TEST_P(TrivialEncodingViewTest, readsInternallyCompressedPayloads) {
  const auto positions = randomizedPositions(/*seed=*/33);
  const auto integers = constantInt32(42);
  const auto booleans = randomBool(/*seed=*/34);
  std::vector<std::string> backing;
  const auto strings = randomStringViews(backing, /*seed=*/35);

  for (const auto compressionType :
       {nimble::CompressionType::Zstd, nimble::CompressionType::MetaInternal}) {
    SCOPED_TRACE(nimble::toString(compressionType));
    expectReads<nimble::TrivialEncoding<int32_t>>(
        integers, positions, {}, compressionType);
    expectReads<nimble::TrivialEncoding<bool>>(
        booleans, positions, {}, compressionType);
    expectReads<nimble::TrivialEncoding<std::string_view>>(
        strings, positions, {}, compressionType);
  }
}

TEST_P(TrivialEncodingViewTest, concurrent) {
  const auto positions = randomizedPositions(/*seed=*/2);
  std::vector<std::string> backing;

  expectConcurrentReads<nimble::TrivialEncoding<int32_t>>(
      randomInt32(/*seed=*/3), positions);
  expectConcurrentReads<nimble::TrivialEncoding<bool>>(
      randomBool(/*seed=*/4), positions);
  expectConcurrentReads<nimble::TrivialEncoding<std::string_view>>(
      randomStringViews(backing, /*seed=*/5), positions);
}

INSTANTIATE_ENCODING_VIEW_TEST_SUITE(TrivialEncodingViewTest);
