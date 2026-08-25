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

#include <limits>
#include <optional>
#include <span>
#include <type_traits>
#include <vector>
#include "velox/dwio/nimble/common/Constants.h"
#include "velox/dwio/nimble/common/Types.h"

#include "absl/container/flat_hash_map.h" // @manual=fbsource//third-party/abseil-cpp:container__flat_hash_map

namespace facebook::nimble {

template <typename T, typename InputType = T>
class UniqueValueCounts {
 public:
  using MapType = absl::flat_hash_map<T, uint64_t>;

  struct Iterator {
    using iterator_category = std::forward_iterator_tag;
    using value_type = typename MapType::value_type;
    using difference_type = typename MapType::difference_type;
    using const_reference = typename MapType::const_reference;
    using const_iterator = typename MapType::const_iterator;

    const_reference operator*() const {
      return *iterator_;
    }
    const_iterator operator->() const {
      return iterator_;
    }

    // Prefix increment
    Iterator& operator++() {
      ++iterator_;
      return *this;
    }

    // Postfix increment
    Iterator operator++(int) {
      Iterator tmp = *this;
      ++(*this);
      return tmp;
    }

    friend bool operator==(const Iterator& a, const Iterator& b) {
      return a.iterator_ == b.iterator_;
    }
    friend bool operator!=(const Iterator& a, const Iterator& b) {
      return a.iterator_ != b.iterator_;
    }

   private:
    explicit Iterator(typename MapType::const_iterator iterator)
        : iterator_{iterator} {}

    typename MapType::const_iterator iterator_;

    friend class UniqueValueCounts<T, InputType>;
  };

  using const_iterator = Iterator;

  uint64_t at(T key) const noexcept {
    auto it = uniqueCounts_.find(key);
    if (it == uniqueCounts_.end()) {
      return 0;
    }
    return it->second;
  }

  size_t size() const noexcept {
    return uniqueCounts_.size();
  }

  uint64_t uniqueStringBytes() const noexcept {
    static_assert(nimble::isStringType<T>());
    uint64_t totalBytes = 0;
    for (const auto& unique : uniqueCounts_) {
      totalBytes += unique.first.size();
    }
    return totalBytes;
  }

  const_iterator begin() const noexcept {
    return Iterator{uniqueCounts_.cbegin()};
  }
  const_iterator cbegin() const noexcept {
    return Iterator{uniqueCounts_.cbegin()};
  }

  const_iterator end() const noexcept {
    return Iterator{uniqueCounts_.cend()};
  }
  const_iterator cend() const noexcept {
    return Iterator{uniqueCounts_.cend()};
  }

  UniqueValueCounts() = default;
  explicit UniqueValueCounts(MapType&& uniqueCounts)
      : uniqueCounts_{std::move(uniqueCounts)} {}

 private:
  MapType uniqueCounts_;
};

template <typename T, typename InputType = T>
class Statistics {
 public:
  using valueType = T;

  static Statistics<T, InputType> create(std::span<const InputType> data);

  uint64_t consecutiveRepeatCount() const noexcept {
    if (!consecutiveRepeatCount_.has_value()) {
      populateRepeats();
    }
    return consecutiveRepeatCount_.value();
  }

  uint64_t minRepeat() const noexcept {
    if (!minRepeat_.has_value()) {
      populateRepeats();
    }
    return minRepeat_.value();
  }

  uint64_t maxRepeat() const noexcept {
    if (!maxRepeat_.has_value()) {
      populateRepeats();
    }
    return maxRepeat_.value();
  }

  uint64_t totalStringsLength() const noexcept {
    static_assert(nimble::isStringType<T>());
    if (!totalStringsLength_.has_value()) {
      populateStringLength();
    }
    return totalStringsLength_.value();
  }

  uint64_t totalStringsRepeatLength() const noexcept {
    static_assert(nimble::isStringType<T>());
    if (!totalStringsRepeatLength_.has_value()) {
      populateRepeats();
    }
    return totalStringsRepeatLength_.value();
  }

  T min() const noexcept {
    static_assert(!nimble::isBoolType<T>());
    if (!min_.has_value()) {
      populateMinMax();
    }
    return min_.value();
  }

  T max() const noexcept {
    static_assert(!nimble::isBoolType<T>());
    if (!max_.has_value()) {
      populateMinMax();
    }
    return max_.value();
  }

  const std::vector<uint64_t>& bucketCounts() const noexcept {
    static_assert(nimble::isIntegralType<T>());
    if (!bucketCounts_.has_value()) {
      populateBucketCounts();
    }
    return bucketCounts_.value();
  }

  const std::optional<UniqueValueCounts<T, InputType>>& uniqueCounts()
      const noexcept {
    if (!uniqueCounts_.has_value()) {
      populateUniques();
    }
    return uniqueCounts_.value();
  }

  /// Returns one value per consecutive run in input order. The sequence is
  /// computed lazily and cached independently from aggregate repeat metrics.
  const std::vector<T>& runValues() const {
    if (runValues_.has_value()) {
      return runValues_.value();
    }
    if (!consecutiveRepeatCount_.has_value()) {
      populateRepeats(/*collectRunValues=*/true);
      return runValues_.value();
    }

    std::vector<T> values;
    if (!data_.empty()) {
      values.reserve(consecutiveRepeatCount());
      T last = data_.front();
      values.push_back(last);
      for (size_t i = 1; i < data_.size(); ++i) {
        if (data_[i] != last) {
          last = data_[i];
          values.push_back(last);
        }
      }
    }
    return runValues_.emplace(std::move(values));
  }

  struct BlockStats {
    uint64_t count;
    uint64_t min;
    uint64_t max;
  };

  const std::vector<BlockStats>& minMaxBlocks(
      uint16_t blockSize = kBlockBitPackingBlockSize) const noexcept {
    static_assert(nimble::isNumericType<T>());
    if (!minMaxBlocks_.has_value() || minMaxBlockSize_ != blockSize) {
      populateMinMaxBlocks(blockSize);
      minMaxBlockSize_ = blockSize;
    }
    return minMaxBlocks_.value();
  }

 private:
  Statistics() = default;
  std::span<const InputType> data_;

  class BlockStatsAccumulator {
   public:
    explicit BlockStatsAccumulator(uint16_t blockSize)
        : blockSize_{blockSize} {}

    void add(uint64_t val) {
      if (val < blockMin_) {
        blockMin_ = val;
      }
      if (val > blockMax_) {
        blockMax_ = val;
      }
      if (++blockCount_ == blockSize_) {
        flush();
      }
    }

    std::vector<BlockStats> finish() {
      flush();
      return std::move(result_);
    }

   private:
    void flush() {
      if (blockCount_ > 0) {
        result_.push_back({blockCount_, blockMin_, blockMax_});
      }
      blockCount_ = 0;
      blockMin_ = std::numeric_limits<uint64_t>::max();
      blockMax_ = 0;
    }

    const uint16_t blockSize_;
    uint64_t blockCount_ = 0;
    uint64_t blockMin_ = std::numeric_limits<uint64_t>::max();
    uint64_t blockMax_ = 0;
    std::vector<BlockStats> result_;
  };

  void populateRepeats(bool collectRunValues = false) const;
  void populateUniques() const;
  void populateMinMax() const;
  void populateBucketCounts() const;
  void populateMinMaxBlocks(uint16_t blockSize) const;
  void populateStringLength() const;

  mutable std::optional<uint64_t> consecutiveRepeatCount_;
  mutable std::optional<uint64_t> minRepeat_;
  mutable std::optional<uint64_t> maxRepeat_;
  mutable std::optional<uint64_t> totalStringsLength_;
  mutable std::optional<uint64_t> totalStringsRepeatLength_;
  mutable std::optional<T> min_;
  mutable std::optional<T> max_;
  mutable std::optional<std::vector<uint64_t>> bucketCounts_;
  mutable std::optional<std::vector<BlockStats>> minMaxBlocks_;
  mutable uint16_t minMaxBlockSize_{0};
  mutable std::optional<std::optional<UniqueValueCounts<T, InputType>>>
      uniqueCounts_;
  mutable std::optional<std::vector<T>> runValues_;
};

} // namespace facebook::nimble
