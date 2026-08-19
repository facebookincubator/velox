/*
 * Copyright (c) Facebook, Inc. and its affiliates.
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
#include <string>
#include <utility>
#include <vector>

namespace facebook::velox::dwio::common {

/// Represents a non-empty half-open interval of row numbers.
struct RowInterval {
  uint64_t begin{0};
  uint64_t end{0};

  bool empty() const {
    return begin >= end;
  }

  uint64_t size() const {
    return empty() ? 0 : end - begin;
  }
};

/// Stores normalized, non-adjacent half-open row intervals.
class RowIntervalSet {
 public:
  RowIntervalSet() = default;

  /// Creates an interval set containing [0, numRows).
  static RowIntervalSet full(uint64_t numRows);

  /// Adds an interval and merges overlapping or adjacent intervals.
  void add(RowInterval interval);

  /// Returns the union of two normalized interval sets.
  static RowIntervalSet setUnion(
      const RowIntervalSet& left,
      const RowIntervalSet& right);

  /// Returns the intersection of two normalized interval sets.
  static RowIntervalSet intersection(
      const RowIntervalSet& left,
      const RowIntervalSet& right);

  /// Returns left minus right.
  static RowIntervalSet difference(
      const RowIntervalSet& left,
      const RowIntervalSet& right);

  /// Returns whether any stored interval overlaps the given interval.
  bool overlaps(RowInterval interval) const;

  /// Returns the first contiguous prefix of 'interval' and whether it is
  /// retained by this set. The cursor identifies the first candidate interval
  /// for the next call and is advanced past intervals that end before it.
  std::pair<RowInterval, bool> firstSplit(RowInterval interval, size_t& cursor)
      const;

  /// Returns the normalized intervals.
  const std::vector<RowInterval>& intervals() const {
    return intervals_;
  }

  /// Returns a human-readable representation of the intervals.
  std::string toString() const;

 private:
  std::vector<RowInterval> intervals_;
};

/// Describes the next contiguous source-row chunk and whether it is retained.
struct RowReadChunk {
  uint64_t numRows{0};
  bool retained{false};
};

} // namespace facebook::velox::dwio::common
