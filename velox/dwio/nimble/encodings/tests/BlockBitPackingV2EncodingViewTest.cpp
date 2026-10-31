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
#include <string>
#include <vector>

#include "fmt/core.h"
#include "velox/dwio/nimble/common/tests/GTestUtils.h"
#include "velox/dwio/nimble/encodings/BlockBitPackingV2Encoding.h"

using namespace facebook;

using BlockBitPackingV2EncodingViewTest = nimble::test::EncodingViewTest;

TEST_P(BlockBitPackingV2EncodingViewTest, readsBlockBitPackingV2Encoding) {
  // 20 rows with blockSize 8 give blocks of 8, 8 and 4 rows.
  nimble::Vector<int32_t> values{pool_.get()};
  for (auto i = 0; i < 20; ++i) {
    values.push_back(i < 10 ? 500 + i : 9000 + (i % 3));
  }

  nimble::Encoding::Options options;
  options.blockBitPackingBlockSize = 8;
  expectReads<nimble::BlockBitPackingV2Encoding<int32_t>>(
      values, {17, 0, 8, 9, 19, 3, 16}, options);
}

TEST_P(BlockBitPackingV2EncodingViewTest, readsInternallyCompressedPayload) {
  const auto values = randomNarrowUnsigned<uint32_t>(/*seed=*/40);
  const auto positions = randomizedPositions(/*seed=*/41);
  nimble::Encoding::Options options;
  options.blockBitPackingBlockSize = 64;
  for (const auto compressionType :
       {nimble::CompressionType::Zstd, nimble::CompressionType::MetaInternal}) {
    SCOPED_TRACE(nimble::toString(compressionType));
    expectReads<nimble::BlockBitPackingV2Encoding<uint32_t>>(
        values, positions, options, compressionType);
  }
}

DEBUG_ONLY_TEST_P(
    BlockBitPackingV2EncodingViewTest,
    rejectsTruncatedPackedPayload) {
  using Encoding = nimble::BlockBitPackingV2Encoding<uint32_t>;
  const auto values = randomNarrowUnsigned<uint32_t>(/*seed=*/42);
  nimble::Encoding::Options options;
  options.blockBitPackingBlockSize = 64;
  const auto serialized = nimble::test::Encoder<Encoding>::encode(
      *buffer_, values, nimble::CompressionType::Uncompressed, options);
  const std::string truncated{serialized.substr(0, serialized.size() - 1)};

  options.cacheEncodingView = cacheEncodingView();
  if (cacheEncodingView()) {
    // A cached view checks every block against the payload when it is created.
    NIMBLE_ASSERT_FILE_THROW(
        nimble::createEncodingView(truncated, pool_.get(), options),
        "BlockBitPackingV2 block payload exceeds packed payload");
    return;
  }

  // Without caching, a read checks the block it reaches.
  const auto view = nimble::createEncodingView(truncated, pool_.get(), options);
  uint32_t value;
  view->readAt(0, &value);
  NIMBLE_ASSERT_FILE_THROW(
      view->readAt(view->rowCount() - 1, &value),
      "BlockBitPackingV2 block payload exceeds packed payload");
}

TEST_P(BlockBitPackingV2EncodingViewTest, readsIndexedBlockRuns) {
  using Encoding = nimble::BlockBitPackingV2Encoding<uint32_t>;
  // With blockSize 8: a constant block, a packed block, a raw block and a
  // packed block.
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

TEST_P(BlockBitPackingV2EncodingViewTest, concurrent) {
  nimble::Encoding::Options options;
  options.blockBitPackingBlockSize = 64;

  expectConcurrentReads<nimble::BlockBitPackingV2Encoding<uint32_t>>(
      randomNarrowUnsigned<uint32_t>(/*seed=*/23),
      randomizedPositions(/*seed=*/24),
      options);
}

INSTANTIATE_ENCODING_VIEW_TEST_SUITE(BlockBitPackingV2EncodingViewTest);
