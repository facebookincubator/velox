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
#include <tuple>
#include <vector>

namespace facebook::nimble::range_helper {

// Range helper used in cases that order need to be maintained. It doesn't
// handle overlaps.
// Suppose we have:
//  offsets: [0, 1, 100, 50]
//  sizes: [1, 2, 2, 50]
// The result will be:
//  [0, 3), [100, 102), [50, 100)
// Time complexity of `add()` is O(1).
template <typename T = int32_t>
class OrderedRanges {
 public:
  template <typename F>
  inline void apply(F f) const {
    for (auto& range : ranges_) {
      f(std::get<0>(range), std::get<1>(range));
    }
  }

  template <typename F>
  inline void applyEach(F f) const {
    for (auto& range : ranges_) {
      for (auto offset = std::get<0>(range), end = offset + std::get<1>(range);
           offset < end;
           ++offset) {
        f(offset);
      }
    }
  }

  inline void add(T offset, T size) {
    size_ += size;
    if (ranges_.size() > 0) {
      auto& last = ranges_.back();
      auto& end = std::get<1>(last);
      if (std::get<0>(last) + end == offset) {
        end += size;
        return;
      }
    }
    ranges_.emplace_back(offset, size);
  }

  inline T size() const {
    return size_;
  }

  inline void clear() {
    ranges_.clear();
    size_ = 0;
  }

  static OrderedRanges of(T offset, T size) {
    OrderedRanges r;
    r.add(offset, size);
    return r;
  }

  const std::vector<std::tuple<T, T>>& ranges() const {
    return ranges_;
  }

 private:
  std::vector<std::tuple<T, T>> ranges_;
  T size_{0};
};

} // namespace facebook::nimble::range_helper
