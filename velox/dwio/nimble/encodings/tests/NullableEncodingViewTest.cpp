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

#include "velox/dwio/nimble/common/tests/GTestUtils.h"
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

// Builds ordered, disjoint ranges long enough to span many null runs.
std::vector<RowRange> makeWideRandomRanges(
    uint32_t rowCount,
    std::mt19937& rng) {
  std::vector<RowRange> ranges;
  uint32_t row =
      static_cast<uint32_t>(rng() % std::min<uint32_t>(rowCount, 64));
  while (row < rowCount) {
    const auto maxLength = std::min<uint32_t>(rowCount - row, 700);
    const uint32_t length{1 + static_cast<uint32_t>(rng() % maxLength)};
    const auto end = row + length;
    ranges.emplace_back(row, end);
    row = std::min<uint32_t>(rowCount, end + static_cast<uint32_t>(rng() % 64));
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
    Encoding::Options options;
    options.cacheEncodingView = cacheEncodingView();
    return createEncodingView(encoded, pool_.get(), options);
  }
};

TEST_P(NullableEncodingViewTest, requiresNonNullRowsForSingleValueReads) {
  auto view = makeView({10, std::nullopt, 12, std::nullopt});
  int64_t value{0};
  view->readAt(2, &value);
  EXPECT_EQ(value, 12);
  for (const uint32_t index : {1, 3}) {
    SCOPED_TRACE(index);
    NIMBLE_ASSERT_THROW(
        view->readAt(index, &value),
        "Nullable EncodingView physical reads require a non-null row");
  }
}

TEST_P(NullableEncodingViewTest, readsMixedRows) {
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

TEST_P(NullableEncodingViewTest, readsNonNullRows) {
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

TEST_P(NullableEncodingViewTest, readsNullRows) {
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

// Rows far apart make a read span many more rows than it selects, which a view
// without caching reads through a window for each row or range.
TEST_P(NullableEncodingViewTest, readsScatteredIndicesAndRanges) {
  constexpr uint32_t kNumRows{5'000};
  std::vector<std::optional<int64_t>> rows(kNumRows);
  for (uint32_t row{0}; row < kNumRows; ++row) {
    if (row % 3 != 0) {
      rows[row] = static_cast<int64_t>(row) * 7;
    }
  }
  auto view = makeView(rows);

  const std::vector<uint32_t> indices{4'999, 0, 2'500, 17, 17, 3'001};
  std::vector<int64_t> indexOutput(indices.size());
  std::vector<uint32_t> indexNulls;
  std::vector<int64_t> expectedIndexOutput;
  std::vector<uint32_t> expectedIndexNulls;
  for (uint32_t i{0}; i < indices.size(); ++i) {
    expectedIndexOutput.push_back(rows[indices[i]].value_or(0));
    if (!rows[indices[i]].has_value()) {
      expectedIndexNulls.push_back(i);
    }
  }
  const auto numNonNullIndices = view->read(
      indices,
      [&](uint32_t outputIndex) { indexNulls.push_back(outputIndex); },
      indexOutput.data());
  EXPECT_EQ(
      numNonNullIndices,
      indices.size() - static_cast<uint32_t>(expectedIndexNulls.size()));
  EXPECT_EQ(indexOutput, expectedIndexOutput);
  EXPECT_EQ(indexNulls, expectedIndexNulls);

  const std::vector<RowRange> ranges{{10, 14}, {2'000, 2'003}, {4'990, 4'996}};
  std::vector<int64_t> rangeOutput(13);
  std::vector<uint32_t> rangeNulls;
  std::vector<int64_t> expectedRangeOutput;
  std::vector<uint32_t> expectedRangeNulls;
  uint32_t outputIndex{0};
  for (const auto& range : ranges) {
    for (uint32_t row{range.startRow}; row < range.endRow; ++row) {
      expectedRangeOutput.push_back(rows[row].value_or(0));
      if (!rows[row].has_value()) {
        expectedRangeNulls.push_back(outputIndex);
      }
      ++outputIndex;
    }
  }
  ASSERT_EQ(expectedRangeOutput.size(), rangeOutput.size());
  const auto numNonNullRanges = view->read(
      ranges,
      [&](uint32_t index) { rangeNulls.push_back(index); },
      rangeOutput.data());
  EXPECT_EQ(
      numNonNullRanges,
      rangeOutput.size() - static_cast<uint32_t>(expectedRangeNulls.size()));
  EXPECT_EQ(rangeOutput, expectedRangeOutput);
  EXPECT_EQ(rangeNulls, expectedRangeNulls);
}

TEST_P(NullableEncodingViewTest, readsRanges) {
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

TEST_P(NullableEncodingViewTest, readsNonNullRangesAfterNulls) {
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

TEST_P(NullableEncodingViewTest, readsRangesWithoutNulls) {
  auto view = makeView({10, 11, 12, 13, 14, 15});
  const std::array<RowRange, 3> ranges{{{0, 2}, {2, 3}, {4, 6}}};
  std::array<int64_t, 5> output{};
  std::vector<uint32_t> nullIndices;

  const auto numNonNulls = view->read(
      ranges,
      [&](uint32_t outputIndex) { nullIndices.push_back(outputIndex); },
      output.data());

  EXPECT_EQ(numNonNulls, 5);
  EXPECT_EQ(output, (std::array<int64_t, 5>{10, 11, 12, 14, 15}));
  EXPECT_TRUE(nullIndices.empty());

  int64_t value{0};
  view->readAt(3, &value);
  EXPECT_EQ(value, 13);

  const std::array<uint32_t, 3> indices{5, 0, 3};
  std::array<int64_t, 3> indexedOutput{};
  EXPECT_EQ(
      view->read(
          indices,
          [&](uint32_t outputIndex) { nullIndices.push_back(outputIndex); },
          indexedOutput.data()),
      indices.size());
  EXPECT_EQ(indexedOutput, (std::array<int64_t, 3>{15, 10, 13}));
  EXPECT_TRUE(nullIndices.empty());
}

TEST_P(NullableEncodingViewTest, readsRangesWithOnlyNulls) {
  auto view =
      makeView({std::nullopt, std::nullopt, std::nullopt, std::nullopt});
  const std::array<RowRange, 2> ranges{{{0, 1}, {2, 4}}};
  std::array<int64_t, 3> output{-1, -1, -1};
  std::vector<uint32_t> nullIndices;

  const auto numNonNulls = view->read(
      ranges,
      [&](uint32_t outputIndex) { nullIndices.push_back(outputIndex); },
      output.data());

  EXPECT_EQ(numNonNulls, 0);
  EXPECT_EQ(output, (std::array<int64_t, 3>{0, 0, 0}));
  EXPECT_EQ(nullIndices, (std::vector<uint32_t>{0, 1, 2}));
}

TEST_P(NullableEncodingViewTest, readsRangesAroundSelectedNulls) {
  auto view =
      makeView({10, std::nullopt, 12, std::nullopt, std::nullopt, 15, 16, 17});
  // Rows 2 and 5 start where the previous run ends in the compact child, but a
  // selected null separates them in the output.
  const std::array<RowRange, 4> ranges{{{0, 2}, {2, 3}, {3, 5}, {5, 8}}};
  std::array<int64_t, 8> output{-1, -1, -1, -1, -1, -1, -1, -1};
  std::vector<uint32_t> nullIndices;

  const auto numNonNulls = view->read(
      ranges,
      [&](uint32_t outputIndex) { nullIndices.push_back(outputIndex); },
      output.data());

  EXPECT_EQ(numNonNulls, 5);
  EXPECT_EQ(output, (std::array<int64_t, 8>{10, 0, 12, 0, 0, 15, 16, 17}));
  EXPECT_EQ(nullIndices, (std::vector<uint32_t>{1, 3, 4}));
}

TEST_P(NullableEncodingViewTest, readsLongRunsAroundSelectedNulls) {
  // Few nulls leave long runs of non-null rows.
  const auto isNullRow = [](uint32_t row) {
    return row == 5 || row == 20 || row == 21 || row == 40;
  };
  std::vector<std::optional<int64_t>> rows;
  for (uint32_t row{0}; row < 64; ++row) {
    rows.push_back(
        isNullRow(row)
            ? std::nullopt
            : std::optional<int64_t>{static_cast<int64_t>(row) * 10});
  }
  // Row 6 starts where the run ending at row 4 ends in the compact child, but
  // the selected null at row 5 separates them in the output. Rows 10, 38 and 44
  // continue the previous run in both. Rows 20 and 21 form a range of nulls,
  // and row 40 starts a range with both nulls and non-nulls.
  const std::vector<RowRange> ranges{
      {0, 6},
      {6, 10},
      {10, 12},
      {20, 22},
      {22, 38},
      {38, 40},
      {40, 44},
      {44, 46},
      {60, 64},
  };
  std::vector<std::optional<int64_t>> expected;
  for (const auto& range : ranges) {
    expected.insert(
        expected.end(),
        rows.begin() + range.startRow,
        rows.begin() + range.endRow);
  }
  std::vector<int64_t> output(expected.size(), -1);
  std::vector<bool> isNull(expected.size(), false);

  const auto numNonNulls = makeView(rows)->read(
      ranges,
      [&](uint32_t outputIndex) { isNull[outputIndex] = true; },
      output.data());

  std::vector<std::optional<int64_t>> actual;
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

TEST_P(NullableEncodingViewTest, readsRandomRanges) {
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

TEST_P(NullableEncodingViewTest, readsWideRandomRanges) {
  constexpr uint32_t kSeed{7'204'113};
  std::mt19937 rng{kSeed};
  // Null rates in thousandths: none, sparse, mixed, dense and all.
  for (const uint32_t nullsPerThousand : {0, 2, 40, 120, 500, 950, 1'000}) {
    for (uint32_t iteration{0}; iteration < 8; ++iteration) {
      SCOPED_TRACE(
          ::testing::Message()
          << "seed=" << kSeed << ", nullsPerThousand=" << nullsPerThousand
          << ", iteration=" << iteration);
      const uint32_t rowCount{1 + static_cast<uint32_t>(rng() % 4'096)};
      std::vector<std::optional<int64_t>> rows;
      rows.reserve(rowCount);
      for (uint32_t row{0}; row < rowCount; ++row) {
        rows.push_back(
            rng() % 1'000 < nullsPerThousand
                ? std::nullopt
                : std::optional<int64_t>{static_cast<int64_t>(rng())});
      }
      const auto ranges = makeWideRandomRanges(rowCount, rng);
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
}

TEST_P(NullableEncodingViewTest, concurrentReads) {
  constexpr uint32_t kNumRows{2'048};
  std::vector<std::optional<int64_t>> rows(kNumRows);
  for (uint32_t row{0}; row < kNumRows; ++row) {
    if (row % 5 != 0 && row % 7 != 3) {
      rows[row] = static_cast<int64_t>(row) * 3;
    }
  }
  const auto view = makeView(rows);

  // Checks that 'output' holds the rows in 'selected', in order, with zeros and
  // reported nulls at the null rows.
  const auto matches = [&](const std::vector<uint32_t>& selected,
                           const std::vector<int64_t>& output,
                           const std::vector<uint32_t>& nullIndices) {
    size_t numNulls{0};
    for (uint32_t i{0}; i < selected.size(); ++i) {
      const auto& row = rows[selected[i]];
      if (row.has_value()) {
        if (output[i] != *row) {
          return false;
        }
        continue;
      }
      if (output[i] != 0 || numNulls >= nullIndices.size() ||
          nullIndices[numNulls++] != i) {
        return false;
      }
    }
    return numNulls == nullIndices.size();
  };

  expectConcurrent([&](int threadIndex) {
    std::mt19937 rng{static_cast<uint32_t>(threadIndex) + 11};
    for (auto iteration = 0; iteration < 64; ++iteration) {
      const auto point = static_cast<uint32_t>(rng() % kNumRows);
      if (rows[point].has_value()) {
        int64_t value{0};
        view->readAt(point, &value);
        if (value != *rows[point]) {
          return false;
        }
      }

      std::vector<uint32_t> indices(16);
      for (auto& index : indices) {
        index = static_cast<uint32_t>(rng() % kNumRows);
      }
      std::vector<int64_t> output(indices.size());
      std::vector<uint32_t> nullIndices;
      view->read(
          indices,
          [&](uint32_t outputIndex) { nullIndices.push_back(outputIndex); },
          output.data());
      if (!matches(indices, output, nullIndices)) {
        return false;
      }

      // Two ordered, disjoint ranges.
      const auto firstStart = static_cast<uint32_t>(rng() % 1'000);
      const auto firstLength = 1 + static_cast<uint32_t>(rng() % 64);
      const auto secondStart =
          firstStart + firstLength + static_cast<uint32_t>(rng() % 64);
      const auto secondLength = 1 + static_cast<uint32_t>(rng() % 64);
      const std::array<RowRange, 2> ranges{
          RowRange{firstStart, firstStart + firstLength},
          RowRange{secondStart, secondStart + secondLength}};
      std::vector<uint32_t> selected;
      for (const auto& range : ranges) {
        for (auto row = range.startRow; row < range.endRow; ++row) {
          selected.push_back(row);
        }
      }
      output.assign(selected.size(), 0);
      nullIndices.clear();
      view->read(
          ranges,
          [&](uint32_t outputIndex) { nullIndices.push_back(outputIndex); },
          output.data());
      if (!matches(selected, output, nullIndices)) {
        return false;
      }
    }
    return true;
  });
}

INSTANTIATE_ENCODING_VIEW_TEST_SUITE(NullableEncodingViewTest);

} // namespace
} // namespace facebook::nimble::test
