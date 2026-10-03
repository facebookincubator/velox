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
#pragma once

#include <algorithm>
#include <cstdint>
#include <memory>
#include <random>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "fmt/core.h"
#include "velox/dwio/nimble/common/Types.h"
#include "velox/dwio/nimble/common/Vector.h"
#include "velox/dwio/nimble/encodings/views/EncodingView.h"

namespace facebook::nimble::test {

/// Checks the range-list reads of a SubIntSplit view against the values it was
/// encoded from, over lists shaped to hit the view's gap bridging, its chunk
/// edges and its transformed-stream planning.
class SubIntSplitRangeListReads {
 public:
  /// Range lists that exercise a view's range-list read: none at all, the whole
  /// column in one and in several ranges, single rows at strides either side of
  /// where a planner might stop bridging gaps, random lists at several run
  /// lengths and densities, ranges straddling 1024-row chunk edges and the
  /// ragged tail. Every list is ascending, disjoint and free of empty ranges,
  /// as EncodingView::read requires, and every range is clipped to `rowCount`,
  /// so any column length is valid.
  static std::vector<std::vector<std::pair<uint32_t, uint32_t>>> makeLists(
      uint32_t rowCount) {
    using RangeList = std::vector<std::pair<uint32_t, uint32_t>>;
    std::vector<RangeList> lists;
    const auto clipped = [rowCount](RangeList ranges) {
      RangeList kept;
      for (auto [offset, length] : ranges) {
        if (offset > rowCount) {
          continue;
        }
        kept.emplace_back(offset, std::min(length, rowCount - offset));
      }
      return kept;
    };

    lists.push_back({});
    lists.push_back(clipped({{0, rowCount}}));
    const uint32_t quarter = rowCount / 4;
    lists.push_back(clipped(
        {{0, quarter},
         {quarter, quarter},
         {2 * quarter, quarter},
         {3 * quarter, rowCount - 3 * quarter}}));

    for (uint32_t stride : {1u, 2u, 7u, 64u, 65u, 66u, 1000u, 5000u}) {
      RangeList ranges;
      for (uint32_t row = 0; row < rowCount; row += stride) {
        ranges.emplace_back(row, 1);
      }
      lists.push_back(std::move(ranges));
    }

    std::mt19937 rng{17};
    for (uint32_t runLength : {1u, 3u, 111u, 1500u}) {
      for (double density : {0.001, 0.05, 0.33, 0.9}) {
        const double meanGap = runLength * (1.0 - density) / density;
        std::uniform_real_distribution<double> gapDraw{0.0, 2.0 * meanGap};
        RangeList ranges;
        uint64_t position = 0;
        while (true) {
          const uint64_t start = position + static_cast<uint64_t>(gapDraw(rng));
          if (start >= rowCount) {
            break;
          }
          const auto length = static_cast<uint32_t>(
              std::min<uint64_t>(runLength, rowCount - start));
          ranges.emplace_back(static_cast<uint32_t>(start), length);
          position = start + length;
        }
        lists.push_back(std::move(ranges));
      }
    }

    lists.push_back(clipped(
        {{1023, 1},
         {1024, 1},
         {1025, 2},
         {2047, 2},
         {3000, 1100},
         {4100, 1},
         {rowCount > 3 ? rowCount - 3 : 0, 3}}));
    lists.push_back(clipped(
        {{1000, 48}, {1100, 1}, {1101, 900}, {2010, 1}, {rowCount, 0}}));
    lists.push_back(clipped({{5, 0}, {6, 1}, {6, 0}, {7, 2}, {rowCount, 0}}));
    // Clipping a short column can leave a list out of order or with empty
    // ranges, which the read does not accept, so both are dropped here.
    for (auto& list : lists) {
      RangeList kept;
      uint32_t end = 0;
      for (const auto& [offset, length] : list) {
        if (length == 0 || offset < end) {
          continue;
        }
        kept.emplace_back(offset, length);
        end = offset + length;
      }
      list = std::move(kept);
    }
    return lists;
  }

  /// Reads `ranges` through the view's range-list read and compares every row
  /// against the rows it was encoded from.
  template <typename PhysicalType>
  static void expectRead(
      const nimble::EncodingView& view,
      std::span<const PhysicalType> values,
      const std::vector<std::pair<uint32_t, uint32_t>>& ranges) {
    size_t numRows{0};
    for (const auto& [offset, length] : ranges) {
      numRows += length;
    }
    SCOPED_TRACE(
        fmt::format("numRanges={}, numRows={}", ranges.size(), numRows));
    // One past the end holds a sentinel, so a read that writes more rows than
    // the ranges add up to is caught rather than landing in unowned memory. A
    // plain array rather than std::vector, which has no data() for bool.
    PhysicalType sentinel{};
    if constexpr (
        std::is_integral_v<PhysicalType> &&
        !std::is_same_v<PhysicalType, bool>) {
      sentinel = static_cast<PhysicalType>(0x5A5A5A5A);
    }
    const auto buffer = std::make_unique<PhysicalType[]>(numRows + 1);
    std::fill_n(buffer.get(), numRows + 1, sentinel);
    std::vector<nimble::RowRange> rowRanges;
    for (const auto& [offset, length] : ranges) {
      if (length > 0) {
        rowRanges.emplace_back(offset, offset + length);
      }
    }
    view.read(rowRanges, {}, buffer.get());
    const std::vector<PhysicalType> actual(
        buffer.get(), buffer.get() + numRows + 1);
    std::vector<PhysicalType> expected;
    expected.reserve(numRows + 1);
    for (const auto& [offset, length] : ranges) {
      expected.insert(
          expected.end(),
          values.begin() + offset,
          values.begin() + offset + length);
    }
    expected.push_back(sentinel);
    ASSERT_EQ(actual, expected);
  }

  /// Runs expectRead over every list makeLists builds for `values`.
  template <typename T>
  static void expectReads(
      const nimble::EncodingView& view,
      const nimble::Vector<T>& values) {
    using PhysicalType = typename nimble::TypeTraits<T>::physicalType;
    const std::span<const PhysicalType> physical{
        reinterpret_cast<const PhysicalType*>(values.data()), values.size()};
    const auto lists = makeLists(static_cast<uint32_t>(values.size()));
    for (size_t i = 0; i < lists.size(); ++i) {
      SCOPED_TRACE(fmt::format("rangeList={}", i));
      expectRead(view, physical, lists[i]);
    }
  }
};

} // namespace facebook::nimble::test
