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

#include <algorithm>
#include <array>
#include <optional>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelectionPolicy.h"

namespace facebook::nimble::test {
namespace {

// Builds ordered, disjoint ranges with randomized gaps and lengths.
std::vector<RowRange> makeRandomRanges(uint32_t rowCount, std::mt19937& rng) {
  std::vector<RowRange> ranges;
  uint32_t row = static_cast<uint32_t>(rng() % std::min<uint32_t>(rowCount, 8));
  while (row < rowCount) {
    const auto maxLength = std::min<uint32_t>(rowCount - row, 16);
    const uint32_t length{1 + static_cast<uint32_t>(rng() % maxLength)};
    const auto end = row + length;
    ranges.emplace_back(row, end);
    row = std::min<uint32_t>(rowCount, end + static_cast<uint32_t>(rng() % 8));
  }
  return ranges;
}

class NullableEncodingViewTest : public EncodingViewTest {
 protected:
  std::unique_ptr<EncodingView> makeView(
      const std::vector<std::optional<int64_t>>& rows) {
    Vector<int64_t> nonNullValues{pool_.get()};
    Vector<bool> isNonNull{pool_.get(), rows.size()};
    for (size_t i{0}; i < rows.size(); ++i) {
      isNonNull[i] = rows[i].has_value();
      if (rows[i].has_value()) {
        nonNullValues.push_back(*rows[i]);
      }
    }

    auto policy = std::make_unique<ManualEncodingSelectionPolicy<int64_t>>(
        std::vector<std::pair<EncodingType, float>>{
            {EncodingType::Trivial, 1.0}},
        CompressionOptions{},
        std::nullopt);
    const auto encoded = EncodingFactory::encodeNullable<int64_t>(
        std::move(policy), nonNullValues, isNonNull, *buffer_);
    return createEncodingView(encoded, pool_.get());
  }
};

TEST_F(NullableEncodingViewTest, readsMixedRows) {
  auto view = makeView({10, std::nullopt, 12, 13, std::nullopt, 15});
  const std::array<uint32_t, 4> indices{1, 2, 4, 5};
  std::array<int64_t, 4> output{};
  std::vector<uint32_t> nullIndices;

  const auto numNonNulls = view->read(
      indices,
      [&](uint32_t outputIndex) { nullIndices.push_back(outputIndex); },
      output.data());

  EXPECT_EQ(numNonNulls, 2);
  EXPECT_EQ(output, (std::array<int64_t, 4>{0, 12, 0, 15}));
  EXPECT_EQ(nullIndices, (std::vector<uint32_t>{0, 2}));
}

TEST_F(NullableEncodingViewTest, readsNonNullRows) {
  auto view = makeView({10, std::nullopt, 12, 13, std::nullopt, 15});
  const std::array<uint32_t, 4> indices{5, 0, 2, 2};
  std::array<int64_t, 4> output{};
  std::vector<uint32_t> nullIndices;

  const auto numNonNulls = view->read(
      indices,
      [&](uint32_t outputIndex) { nullIndices.push_back(outputIndex); },
      output.data());

  EXPECT_EQ(numNonNulls, indices.size());
  EXPECT_EQ(output, (std::array<int64_t, 4>{15, 10, 12, 12}));
  EXPECT_TRUE(nullIndices.empty());
}

TEST_F(NullableEncodingViewTest, readsNullRows) {
  auto view = makeView({10, std::nullopt, 12, 13, std::nullopt, 15});
  const std::array<uint32_t, 2> indices{1, 4};
  std::array<int64_t, 2> output{};
  std::vector<uint32_t> nullIndices;

  const auto numNonNulls = view->read(
      indices,
      [&](uint32_t outputIndex) { nullIndices.push_back(outputIndex); },
      output.data());

  EXPECT_EQ(numNonNulls, 0);
  EXPECT_EQ(output, (std::array<int64_t, 2>{0, 0}));
  EXPECT_EQ(nullIndices, (std::vector<uint32_t>{0, 1}));
}

TEST_F(NullableEncodingViewTest, readsRanges) {
  auto view = makeView({10, std::nullopt, 12, 13, std::nullopt, 15});
  const std::array<RowRange, 2> ranges{{{0, 2}, {4, 6}}};
  std::array<int64_t, 4> output{};
  std::vector<uint32_t> nullIndices;

  const auto numNonNulls = view->read(
      ranges,
      [&](uint32_t outputIndex) { nullIndices.push_back(outputIndex); },
      output.data());

  EXPECT_EQ(numNonNulls, 2);
  EXPECT_EQ(output, (std::array<int64_t, 4>{10, 0, 0, 15}));
  EXPECT_EQ(nullIndices, (std::vector<uint32_t>{1, 2}));
}

TEST_F(NullableEncodingViewTest, readsNonNullRangesAfterNulls) {
  auto view = makeView({10, std::nullopt, 12, 13, std::nullopt, 15});
  const std::array<RowRange, 2> ranges{{{2, 4}, {5, 6}}};
  std::array<int64_t, 3> output{};
  std::vector<uint32_t> nullIndices;

  const auto numNonNulls = view->read(
      ranges,
      [&](uint32_t outputIndex) { nullIndices.push_back(outputIndex); },
      output.data());

  EXPECT_EQ(numNonNulls, ranges[0].numRows() + ranges[1].numRows());
  EXPECT_EQ(output, (std::array<int64_t, 3>{12, 13, 15}));
  EXPECT_TRUE(nullIndices.empty());
}

TEST_F(NullableEncodingViewTest, readsRandomRanges) {
  constexpr uint32_t kSeed{1'384'921};
  std::mt19937 rng{kSeed};
  for (uint32_t iteration{0}; iteration < 100; ++iteration) {
    SCOPED_TRACE(
        ::testing::Message()
        << "seed=" << kSeed << ", iteration=" << iteration);
    const uint32_t rowCount{1 + static_cast<uint32_t>(rng() % 256)};
    const uint32_t nullModulo{1 + static_cast<uint32_t>(rng() % 8)};
    std::vector<std::optional<int64_t>> rows;
    rows.reserve(rowCount);
    for (uint32_t row{0}; row < rowCount; ++row) {
      rows.push_back(
          rng() % nullModulo == 0
              ? std::nullopt
              : std::optional<int64_t>{static_cast<int64_t>(rng())});
    }
    const auto ranges = makeRandomRanges(rowCount, rng);
    std::vector<std::optional<int64_t>> expected;
    for (const auto& range : ranges) {
      expected.insert(
          expected.end(),
          rows.begin() + range.startRow,
          rows.begin() + range.endRow);
    }
    std::vector<int64_t> output(expected.size(), -1);
    std::vector<bool> isNull(expected.size(), false);
    auto view = makeView(rows);

    const auto numNonNulls = view->read(
        ranges,
        [&](uint32_t outputIndex) { isNull[outputIndex] = true; },
        output.data());

    std::vector<std::optional<int64_t>> actual;
    actual.reserve(output.size());
    for (size_t i{0}; i < output.size(); ++i) {
      actual.push_back(
          isNull[i] ? std::nullopt : std::optional<int64_t>{output[i]});
    }
    EXPECT_EQ(actual, expected);
    EXPECT_EQ(
        numNonNulls,
        static_cast<uint32_t>(std::count_if(
            expected.begin(), expected.end(), [](const auto& value) {
              return value.has_value();
            })));
  }
}

} // namespace
} // namespace facebook::nimble::test
