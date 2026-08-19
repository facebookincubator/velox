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

#include "velox/dwio/common/RowIntervalSet.h"

#include <algorithm>
#include <sstream>

#include "velox/common/base/Exceptions.h"

namespace facebook::velox::dwio::common {

RowIntervalSet RowIntervalSet::full(uint64_t numRows) {
  RowIntervalSet result;
  if (numRows != 0) {
    result.intervals_.push_back({0, numRows});
  }
  return result;
}

void RowIntervalSet::add(RowInterval interval) {
  VELOX_CHECK_LT(
      interval.begin,
      interval.end,
      "Row interval must be non-empty: [{}, {})",
      interval.begin,
      interval.end);

  auto it = std::lower_bound(
      intervals_.begin(),
      intervals_.end(),
      interval.begin,
      [](const RowInterval& current, uint64_t begin) {
        return current.end < begin;
      });

  auto mergedBegin = interval.begin;
  auto mergedEnd = interval.end;
  while (it != intervals_.end() && it->begin <= mergedEnd) {
    mergedBegin = std::min(mergedBegin, it->begin);
    mergedEnd = std::max(mergedEnd, it->end);
    it = intervals_.erase(it);
  }
  intervals_.insert(it, {mergedBegin, mergedEnd});
}

RowIntervalSet RowIntervalSet::setUnion(
    const RowIntervalSet& left,
    const RowIntervalSet& right) {
  RowIntervalSet result;
  size_t leftIndex = 0;
  size_t rightIndex = 0;
  while (leftIndex < left.intervals_.size() ||
         rightIndex < right.intervals_.size()) {
    if (rightIndex == right.intervals_.size() ||
        (leftIndex < left.intervals_.size() &&
         left.intervals_[leftIndex].begin <=
             right.intervals_[rightIndex].begin)) {
      result.add(left.intervals_[leftIndex++]);
    } else {
      result.add(right.intervals_[rightIndex++]);
    }
  }
  return result;
}

RowIntervalSet RowIntervalSet::intersection(
    const RowIntervalSet& left,
    const RowIntervalSet& right) {
  RowIntervalSet result;
  size_t leftIndex = 0;
  size_t rightIndex = 0;
  while (leftIndex < left.intervals_.size() &&
         rightIndex < right.intervals_.size()) {
    const auto& leftInterval = left.intervals_[leftIndex];
    const auto& rightInterval = right.intervals_[rightIndex];
    const auto begin = std::max(leftInterval.begin, rightInterval.begin);
    const auto end = std::min(leftInterval.end, rightInterval.end);
    if (begin < end) {
      result.intervals_.push_back({begin, end});
    }
    if (leftInterval.end < rightInterval.end) {
      ++leftIndex;
    } else {
      ++rightIndex;
    }
  }
  return result;
}

RowIntervalSet RowIntervalSet::difference(
    const RowIntervalSet& left,
    const RowIntervalSet& right) {
  RowIntervalSet result;
  size_t rightIndex = 0;
  for (const auto& leftInterval : left.intervals_) {
    auto cursor = leftInterval.begin;
    while (rightIndex < right.intervals_.size() &&
           right.intervals_[rightIndex].end <= cursor) {
      ++rightIndex;
    }
    auto candidate = rightIndex;
    while (candidate < right.intervals_.size() &&
           right.intervals_[candidate].begin < leftInterval.end) {
      const auto& rightInterval = right.intervals_[candidate];
      if (cursor < rightInterval.begin) {
        result.intervals_.push_back(
            {cursor, std::min(rightInterval.begin, leftInterval.end)});
      }
      if (rightInterval.end > cursor) {
        cursor = std::max(cursor, rightInterval.end);
      }
      if (cursor >= leftInterval.end) {
        break;
      }
      ++candidate;
    }
    if (cursor < leftInterval.end) {
      result.intervals_.push_back({cursor, leftInterval.end});
    }
  }
  return result;
}

bool RowIntervalSet::overlaps(RowInterval interval) const {
  if (interval.empty()) {
    return false;
  }
  auto it = std::lower_bound(
      intervals_.begin(),
      intervals_.end(),
      interval.begin,
      [](const RowInterval& current, uint64_t begin) {
        return current.end <= begin;
      });
  return it != intervals_.end() && it->begin < interval.end;
}

std::pair<RowInterval, bool> RowIntervalSet::firstSplit(
    RowInterval interval,
    size_t& cursor) const {
  VELOX_CHECK_LT(interval.begin, interval.end);
  cursor = std::min(cursor, intervals_.size());
  while (cursor < intervals_.size() &&
         intervals_[cursor].end <= interval.begin) {
    ++cursor;
  }
  if (cursor == intervals_.size() || intervals_[cursor].begin >= interval.end) {
    return {interval, false};
  }
  const auto& retained = intervals_[cursor];
  if (interval.begin < retained.begin) {
    return {{interval.begin, std::min(interval.end, retained.begin)}, false};
  }
  return {{interval.begin, std::min(interval.end, retained.end)}, true};
}

std::string RowIntervalSet::toString() const {
  std::ostringstream out;
  out << "[";
  for (size_t i = 0; i < intervals_.size(); ++i) {
    if (i != 0) {
      out << ", ";
    }
    out << "[" << intervals_[i].begin << ", " << intervals_[i].end << ")";
  }
  out << "]";
  return out.str();
}

} // namespace facebook::velox::dwio::common
