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

#include <cstdint>
#include <limits>
#include <random>
#include <tuple>

#include <gtest/gtest.h>

#include "velox/dwio/nimble/encodings/PartitionedEliasFanoEncoding.h"

using namespace facebook;

using PartitionedEliasFanoEncodingViewTest = nimble::test::EncodingViewTest;

TEST_P(PartitionedEliasFanoEncodingViewTest, readsSignedAndUnsignedValues) {
  expectReads<nimble::PartitionedEliasFanoEncoding<int64_t>>(
      makeVector<int64_t>({
          0,
          17,
          std::numeric_limits<int64_t>::max(),
          std::numeric_limits<int64_t>::min(),
          std::numeric_limits<int64_t>::min() + 1,
          -1,
          -1,
      }),
      {6, 0, 3, 1, 3, 4, 2});
  expectReads<nimble::PartitionedEliasFanoEncoding<uint32_t>>(
      makeVector<uint32_t>({7, 8, 9, 1, 1, 2, 1'000, 3}), {7, 0, 2, 3, 6, 1});
}

TEST_P(PartitionedEliasFanoEncodingViewTest, lowerBound) {
  const auto values = makeVector<uint64_t>({3, 7, 7, 20, 1, 9});
  nimble::Encoding::Options options{.cacheEncodingView = cacheEncodingView()};
  const auto serialized = nimble::test::
      Encoder<nimble::PartitionedEliasFanoEncoding<uint64_t>>::encode(
          *buffer_, values, nimble::CompressionType::Uncompressed, options);
  auto view = nimble::createEncodingView(serialized, pool_.get(), options);
  ASSERT_EQ(view->encodingType(), nimble::EncodingType::PartitionedEliasFano);
  const auto& typedView =
      dynamic_cast<const nimble::TypedEncodingView<uint64_t>&>(*view);

  for (const auto& [begin, end, value, expected] :
       std::vector<std::tuple<uint32_t, uint32_t, uint64_t, uint32_t>>{
           {0, 4, 7, 1},
           {0, 4, 21, 4},
           {4, 6, 0, 4},
           {4, 6, 2, 5},
           {4, 6, 10, 6},
       }) {
    SCOPED_TRACE(
        ::testing::Message()
        << "begin=" << begin << ", end=" << end << ", value=" << value);
    uint64_t found{0};
    const auto row = typedView.lowerBound(begin, end, value, found);
    EXPECT_EQ(row, expected);
    if (row < end) {
      EXPECT_EQ(found, values[row]);
    }
  }
}

TEST_P(PartitionedEliasFanoEncodingViewTest, supportsConcurrentReads) {
  // Ascending runs of 128 rows give the view many partitions.
  nimble::Vector<uint64_t> values{pool_.get()};
  values.reserve(kConcurrentRows);
  for (uint32_t row{0}; row < kConcurrentRows; ++row) {
    values.push_back(1'000 + row % 128);
  }

  expectConcurrentReads<nimble::PartitionedEliasFanoEncoding<uint64_t>>(
      values, randomizedPositions(/*seed=*/43));
}

TEST_P(PartitionedEliasFanoEncodingViewTest, randomizedReads) {
  std::mt19937_64 random{44};
  std::uniform_int_distribution<uint32_t> rowCountDistribution{1, 512};
  std::uniform_int_distribution<uint64_t> valueDistribution{0, 1'000};

  for (uint32_t iteration{0}; iteration < 32; ++iteration) {
    SCOPED_TRACE(::testing::Message() << "iteration=" << iteration);
    const auto rowCount = rowCountDistribution(random);
    nimble::Vector<uint64_t> values{pool_.get()};
    values.reserve(rowCount);
    uint64_t value{valueDistribution(random)};
    for (uint32_t row{0}; row < rowCount; ++row) {
      // Starts a new run about once every 16 rows.
      value = random() % 16 == 0 ? valueDistribution(random)
                                 : value + valueDistribution(random) % 50;
      values.push_back(value);
    }

    std::uniform_int_distribution<uint32_t> positionDistribution{
        0, rowCount - 1};
    std::vector<uint32_t> positions;
    positions.reserve(64);
    for (uint32_t read{0}; read < 64; ++read) {
      positions.push_back(positionDistribution(random));
    }
    expectReads<nimble::PartitionedEliasFanoEncoding<uint64_t>>(
        values, positions);
  }
}

INSTANTIATE_ENCODING_VIEW_TEST_SUITE(PartitionedEliasFanoEncodingViewTest);
