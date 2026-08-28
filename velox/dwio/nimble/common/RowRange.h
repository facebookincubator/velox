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

#include <cstdint>

#include <fmt/format.h>
#include <folly/hash/Hash.h>

namespace facebook::nimble {

/// Row range within a stripe [startRow, endRow).
struct RowRange {
  uint32_t startRow{0};
  uint32_t endRow{0};

  RowRange() = default;
  RowRange(uint32_t start, uint32_t end) : startRow(start), endRow(end) {}

  inline uint32_t numRows() const {
    return endRow - startRow;
  }

  inline bool empty() const {
    return startRow >= endRow;
  }

  inline bool contains(uint32_t row) const {
    return row >= startRow && row < endRow;
  }

  inline bool contains(const RowRange& other) const {
    return startRow <= other.startRow && other.endRow <= endRow;
  }

  /// Returns the overlap between this range and 'other', or an empty range if
  /// they don't overlap.
  inline RowRange intersect(const RowRange& other) const {
    const auto start = std::max(startRow, other.startRow);
    const auto end = std::min(endRow, other.endRow);
    return start < end ? RowRange(start, end) : RowRange();
  }

  inline bool operator==(const RowRange& other) const {
    return startRow == other.startRow && endRow == other.endRow;
  }

  inline std::string toString() const {
    return fmt::format("[{}, {})", startRow, endRow);
  }
};

struct RowRangeHash {
  inline size_t operator()(const RowRange& range) const {
    return folly::hash::hash_combine(range.startRow, range.endRow);
  }
};

} // namespace facebook::nimble
