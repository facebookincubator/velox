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

#include <gtest/gtest.h>

#include <limits>
#include <span>
#include <vector>

#include "fmt/core.h"
#include "velox/dwio/nimble/encodings/BlockBitPackingEncoding.h"

using namespace facebook;

using EncodingViewTest = nimble::test::EncodingViewTest;
using BlockBitPackingEncodingViewTest = nimble::test::EncodingViewTest;

TEST_F(EncodingViewTest, readsBlockBitPackingEncoding) {
  nimble::Vector<int32_t> values{pool_.get()};
  for (auto i = 0; i < 20; ++i) {
    values.push_back(i < 10 ? 500 + i : 9000 + (i % 3));
  }

  nimble::Encoding::Options options;
  options.blockBitPackingBlockSize = 8;
  expectReads<nimble::BlockBitPackingEncoding<int32_t>>(
      values, {17, 0, 8, 9, 19, 3}, options);
}

TEST_F(BlockBitPackingEncodingViewTest, readsIndexedBlockRuns) {
  using Encoding = nimble::BlockBitPackingEncoding<uint32_t>;
  auto values = makeVector<uint32_t>({
      42,  42,
      42,  42,
      42,  42,
      42,  42,
      100, 101,
      103, 104,
      108, 110,
      111, 112,
      0,   std::numeric_limits<uint32_t>::max(),
      1,   std::numeric_limits<uint32_t>::max() - 1,
      2,   std::numeric_limits<uint32_t>::max() - 2,
      3,   std::numeric_limits<uint32_t>::max() - 3,
      200, 205,
      206, 210,
      211, 212,
      220, 221,
  });

  const std::vector<uint32_t> indices{
      1, 7, 0, 8, 10, 15, 9, 16, 18, 23, 17, 24, 26, 29, 25, 31, 31, 30, 2};
  std::vector<uint32_t> expected;
  expected.reserve(indices.size());
  for (const auto index : indices) {
    expected.push_back(values[index]);
  }

  nimble::Encoding::Options baseOptions;
  baseOptions.blockBitPackingBlockSize = 8;
  for (const auto useVarintRowCount : {false, true}) {
    SCOPED_TRACE(fmt::format("useVarintRowCount={}", useVarintRowCount));
    auto options = baseOptions;
    options.useVarintRowCount = useVarintRowCount;
    auto serialized = nimble::test::Encoder<Encoding>::encode(
        *buffer_, values, nimble::CompressionType::Uncompressed, options);
    auto view = nimble::createEncodingView(serialized, pool_.get(), options);
    ASSERT_NE(view, nullptr);

    std::vector<uint32_t> actual(indices.size());
    view->readAt(
        std::span<const uint32_t>{indices.data(), indices.size()},
        actual.data());
    EXPECT_EQ(actual, expected);
  }
}

TEST_F(BlockBitPackingEncodingViewTest, concurrent) {
  nimble::Encoding::Options options;
  options.blockBitPackingBlockSize = 64;

  expectConcurrentReads<nimble::BlockBitPackingEncoding<uint32_t>>(
      randomNarrowUnsigned<uint32_t>(/*seed=*/23),
      randomizedPositions(/*seed=*/24),
      options);
}
