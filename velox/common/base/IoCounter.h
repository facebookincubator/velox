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

#include <atomic>
#include <cstdint>
#include <functional>
#include <limits>

namespace facebook::velox::io {

class IoCounter {
 public:
  IoCounter() = default;

  IoCounter(const IoCounter& other) noexcept
      : count_{other.count_.load(std::memory_order_relaxed)},
        sum_{other.sum_.load(std::memory_order_relaxed)},
        min_{other.min_.load(std::memory_order_relaxed)},
        max_{other.max_.load(std::memory_order_relaxed)} {}

  IoCounter& operator=(const IoCounter& other) noexcept {
    if (this != &other) {
      count_.store(
          other.count_.load(std::memory_order_relaxed),
          std::memory_order_relaxed);
      sum_.store(
          other.sum_.load(std::memory_order_relaxed),
          std::memory_order_relaxed);
      min_.store(
          other.min_.load(std::memory_order_relaxed),
          std::memory_order_relaxed);
      max_.store(
          other.max_.load(std::memory_order_relaxed),
          std::memory_order_relaxed);
    }
    return *this;
  }

  uint64_t count() const {
    return count_;
  }

  uint64_t sum() const {
    return sum_;
  }

  uint64_t min() const {
    return min_;
  }

  uint64_t max() const {
    return max_;
  }

  void increment(uint64_t amount) {
    ++count_;
    sum_ += amount;
    casLoop(min_, amount, std::greater());
    casLoop(max_, amount, std::less());
  }

  void merge(const IoCounter& other) {
    sum_ += other.sum_;
    count_ += other.count_;
    casLoop(min_, other.min_, std::greater());
    casLoop(max_, other.max_, std::less());
  }

 private:
  template <typename Compare>
  static void
  casLoop(std::atomic<uint64_t>& value, uint64_t newValue, Compare compare) {
    uint64_t old = value;
    while (compare(old, newValue) &&
           !value.compare_exchange_weak(old, newValue)) {
    }
  }

  std::atomic<uint64_t> count_{0};
  std::atomic<uint64_t> sum_{0};
  std::atomic<uint64_t> min_{std::numeric_limits<uint64_t>::max()};
  std::atomic<uint64_t> max_{0};
};

} // namespace facebook::velox::io
