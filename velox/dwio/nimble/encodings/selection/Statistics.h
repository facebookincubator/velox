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
#include <limits>
#include <optional>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>
#include "velox/dwio/nimble/common/Constants.h"
#include "velox/dwio/nimble/common/Types.h"
#include "velox/dwio/nimble/encodings/selection/BitFlipProfile.h"

#include "absl/container/flat_hash_map.h" // @manual=fbsource//third-party/abseil-cpp:container__flat_hash_map

namespace facebook::nimble {

/// Each distinct value of a stream with the number of rows holding it.
///
/// Held either as a hash map or as a vector sorted by value, a cost decision
/// made in Statistics; consumers are indifferent to the iteration order.
template <typename T, typename InputType = T>
class UniqueValueCounts {
 public:
  using MapType = absl::flat_hash_map<T, uint64_t>;
  using value_type = typename MapType::value_type;

  /// Entries in ascending value order, each value appearing once.
  using SortedType = std::vector<value_type>;

  struct Iterator {
    using iterator_category = std::forward_iterator_tag;
    using value_type = typename MapType::value_type;
    using difference_type = typename MapType::difference_type;
    using const_reference = typename MapType::const_reference;

    const_reference operator*() const {
      return sortedEntry_ != nullptr ? *sortedEntry_ : *mapIterator_;
    }
    const value_type* operator->() const {
      return &**this;
    }

    // Prefix increment
    Iterator& operator++() {
      if (sortedEntry_ != nullptr) {
        ++sortedEntry_;
      } else {
        ++mapIterator_;
      }
      return *this;
    }

    // Postfix increment
    Iterator operator++(int) {
      Iterator tmp = *this;
      ++(*this);
      return tmp;
    }

    friend bool operator==(const Iterator& a, const Iterator& b) {
      return a.sortedEntry_ == b.sortedEntry_ &&
          (a.sortedEntry_ != nullptr || a.mapIterator_ == b.mapIterator_);
    }
    friend bool operator!=(const Iterator& a, const Iterator& b) {
      return !(a == b);
    }

   private:
    explicit Iterator(typename MapType::const_iterator iterator)
        : mapIterator_{iterator} {}
    explicit Iterator(const value_type* sortedEntry)
        : sortedEntry_{sortedEntry} {}

    typename MapType::const_iterator mapIterator_{};
    // Null when iterating the map. Statistics never builds an empty sorted
    // vector, whose data pointer may itself be null.
    const value_type* sortedEntry_{nullptr};

    friend class UniqueValueCounts<T, InputType>;
  };

  using const_iterator = Iterator;

  uint64_t at(T key) const noexcept {
    if (sorted_) {
      const auto it = std::lower_bound(
          sortedCounts_.begin(),
          sortedCounts_.end(),
          key,
          [](const value_type& entry, const T& value) {
            return entry.first < value;
          });
      return it != sortedCounts_.end() && it->first == key ? it->second : 0;
    }
    auto it = uniqueCounts_.find(key);
    if (it == uniqueCounts_.end()) {
      return 0;
    }
    return it->second;
  }

  size_t size() const noexcept {
    return sorted_ ? sortedCounts_.size() : uniqueCounts_.size();
  }

  std::optional<std::pair<T, uint64_t>> mostFrequent() const noexcept {
    if (size() == 0) {
      return std::nullopt;
    }
    if (!mostFrequent_.has_value()) {
      const auto it = std::max_element(
          cbegin(), cend(), [](const auto& left, const auto& right) {
            if (left.second != right.second) {
              return left.second < right.second;
            }
            return left.first > right.first;
          });
      mostFrequent_.emplace(it->first, it->second);
    }
    return mostFrequent_;
  }

  uint64_t uniqueStringBytes() const noexcept {
    static_assert(nimble::isStringType<T>());
    if (!uniqueStringBytes_.has_value()) {
      uint64_t totalBytes = 0;
      for (const auto& unique : *this) {
        totalBytes += unique.first.size();
      }
      uniqueStringBytes_ = totalBytes;
    }
    return uniqueStringBytes_.value();
  }

  const_iterator begin() const noexcept {
    return sorted_ ? Iterator{sortedCounts_.data()}
                   : Iterator{uniqueCounts_.cbegin()};
  }
  const_iterator cbegin() const noexcept {
    return begin();
  }

  const_iterator end() const noexcept {
    return sorted_ ? Iterator{sortedCounts_.data() + sortedCounts_.size()}
                   : Iterator{uniqueCounts_.cend()};
  }
  const_iterator cend() const noexcept {
    return end();
  }

  UniqueValueCounts() = default;
  explicit UniqueValueCounts(MapType&& uniqueCounts)
      : uniqueCounts_{std::move(uniqueCounts)} {}
  explicit UniqueValueCounts(SortedType&& sortedCounts)
      : sortedCounts_{std::move(sortedCounts)}, sorted_{true} {}

 private:
  MapType uniqueCounts_;
  SortedType sortedCounts_;
  bool sorted_{false};
  mutable std::optional<std::pair<T, uint64_t>> mostFrequent_;
  mutable std::optional<uint64_t> uniqueStringBytes_;
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

  /// Whether every value in the stream is identical.
  ///
  /// Answering constancy from the data costs a comparison per value and stops
  /// at the first one that differs, so a stream that is not constant is
  /// usually settled within a handful of reads. uniqueCounts() reaches the
  /// same verdict only after inserting one hash entry per value and sizing a
  /// table to match -- on a wide column that is the dominant cost of encoding
  /// selection, paid to learn a fact the second element already gave away.
  bool isConstant() const noexcept {
    if (!isConstant_.has_value()) {
      populateIsConstant();
    }
    return isConstant_.value();
  }

  /// Returns whether integral input values are non-decreasing when interpreted
  /// as LogicalType. LogicalType must be explicit because signed values use
  /// unsigned physical storage.
  template <typename LogicalType>
  bool isNonDecreasing() const noexcept {
    static_assert(nimble::isIntegralType<LogicalType>());
    static_assert(nimble::isIntegralType<InputType>());
    static_assert(sizeof(LogicalType) == sizeof(InputType));
    if constexpr (
        std::is_signed_v<LogicalType> == std::is_signed_v<InputType>) {
      if (!physicalOrderNonDecreasing_.has_value()) {
        populatePhysicalOrderNonDecreasing();
      }
      return physicalOrderNonDecreasing_.value();
    } else {
      static_assert(
          std::is_signed_v<LogicalType> && std::is_unsigned_v<InputType>);
      if (!signedOrderNonDecreasing_.has_value()) {
        populateSignedOrderNonDecreasing();
      }
      return signedOrderNonDecreasing_.value();
    }
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

  /// A lower bound on the number of distinct values: the number of distinct
  /// offsets from min in their low kDistinctBoundBits bits. Exact where the
  /// unique counts are already built or every offset fits in those bits;
  /// otherwise costs one pass and an L2-sized bitmap, far cheaper than
  /// counting the distinct values outright.
  uint64_t distinctLowerBound() const {
    static_assert(nimble::isIntegralType<T>());
    if (uniqueCounts_.has_value() && uniqueCounts_->has_value()) {
      return uniqueCounts_->value().size();
    }
    if (!distinctLowerBound_.has_value()) {
      populateDistinctLowerBound();
    }
    return distinctLowerBound_.value();
  }

  /// Low bits of the offsets from min that distinctLowerBound() tells apart.
  static constexpr int kDistinctBoundBits{20};

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

  /// Returns the length of each consecutive run in input order, aligned with
  /// runValues(). Computed lazily and cached.
  const std::vector<uint32_t>& runLengths() const {
    if (runLengths_.has_value()) {
      return runLengths_.value();
    }
    if constexpr (!nimble::isStringType<T>() && !nimble::isBoolType<T>()) {
      if (!data_.empty()) {
        populateRepeats();
        return runLengths_.value();
      }
    }
    std::vector<uint32_t> lengths;
    if (!data_.empty()) {
      lengths.reserve(consecutiveRepeatCount());
      uint32_t length{1};
      for (size_t i = 1; i < data_.size(); ++i) {
        if (data_[i] == data_[i - 1]) {
          ++length;
        } else {
          lengths.push_back(length);
          length = 1;
        }
      }
      lengths.push_back(length);
    }
    return runLengths_.emplace(std::move(lengths));
  }

  struct BlockStats {
    uint64_t count;
    uint64_t min;
    uint64_t max;
  };

  /// Aggregates over adjacent value pairs, in input order. Grouped because
  /// they all come from one pass over consecutive pairs.
  struct AdjacentPairStats {
    /// Pairs where the later value is not below the earlier one, i.e. steps an
    /// encoding storing non-negative deltas can represent without restating.
    uint64_t nonDecreasingCount{0};
    /// Largest step over a non-decreasing pair; sizes a fixed-width delta
    /// array, which must cover the widest delta it stores.
    uint64_t maxIncrease{0};
    /// Sum of |v[i] - v[i-1]| over every pair.
    uint64_t sumAbsoluteDelta{0};
  };

  /// See AdjacentPairStats. Empty for a stream of fewer than two values.
  const AdjacentPairStats& adjacentPairStats() const noexcept {
    static_assert(nimble::isIntegralType<T>());
    if (!adjacentPairStats_.has_value()) {
      populateAdjacentPairStats();
    }
    return adjacentPairStats_.value();
  }

  const std::vector<BlockStats>& minMaxBlocks(
      uint16_t blockSize = kBlockBitPackingBlockSize) const noexcept {
    static_assert(nimble::isNumericType<T>());
    if (!minMaxBlocks_.has_value() || minMaxBlockSize_ != blockSize) {
      populateMinMaxBlocks(blockSize);
      minMaxBlockSize_ = blockSize;
    }
    return minMaxBlocks_.value();
  }

  /// Per-bit-position bit-flip-probability profile between consecutive
  /// values, plus its variance and discrete gradient. Used to cheaply predict
  /// whether a stream is likely to contain multiple concatenated bit-field
  /// distributions (see subintsplit/TopLevelPolicy.h).
  const BitFlipProfile& bitFlipProfile() const noexcept {
    static_assert(nimble::isIntegralType<T>());
    if (!bitFlipProfile_.has_value()) {
      populateBitFlipProfile();
    }
    return bitFlipProfile_.value();
  }

 private:
  Statistics() = default;
  std::span<const InputType> data_;

  void populateRepeats(bool collectRunValues = false) const;

  // Compares against the first value and stops at the first mismatch.
  void populateIsConstant() const noexcept;
  void populateUniques() const;
  void populateMinMax() const;
  void populateBucketCounts() const;
  void populateMinMaxBlocks(uint16_t blockSize) const;
  // Checks the order of the physical input values.
  void populatePhysicalOrderNonDecreasing() const noexcept;

  // Checks signed logical order over unsigned physical input values.
  void populateSignedOrderNonDecreasing() const noexcept;
  void populateStringLength() const;
  void populateBitFlipProfile() const;
  void populateAdjacentPairStats() const;
  void populateDistinctLowerBound() const;

  mutable std::optional<uint64_t> consecutiveRepeatCount_;
  mutable std::optional<uint64_t> minRepeat_;
  mutable std::optional<uint64_t> maxRepeat_;
  mutable std::optional<uint64_t> totalStringsLength_;
  mutable std::optional<uint64_t> totalStringsRepeatLength_;
  mutable std::optional<bool> isConstant_;
  mutable std::optional<T> min_;
  mutable std::optional<T> max_;

  // Caches the physical input order independently from signed logical order.
  mutable std::optional<bool> physicalOrderNonDecreasing_;

  // Caches signed logical order for unsigned physical input.
  mutable std::optional<bool> signedOrderNonDecreasing_;
  mutable std::optional<std::vector<uint64_t>> bucketCounts_;
  mutable std::optional<std::vector<BlockStats>> minMaxBlocks_;
  mutable uint16_t minMaxBlockSize_{0};
  mutable std::optional<std::optional<UniqueValueCounts<T, InputType>>>
      uniqueCounts_;
  mutable std::optional<std::vector<T>> runValues_;
  mutable std::optional<std::vector<uint32_t>> runLengths_;
  mutable std::optional<BitFlipProfile> bitFlipProfile_;
  mutable std::optional<AdjacentPairStats> adjacentPairStats_;
  mutable std::optional<uint64_t> distinctLowerBound_;
};

/// Copies `numBlocks` contiguous blocks of `blockRows` rows, spread evenly
/// over `values`, so runs, frames and local ranges survive in the sample.
/// `values` must hold at least `numBlocks * blockRows` rows.
template <typename T>
std::vector<T> sampleSpreadBlocks(
    std::span<const T> values,
    size_t numBlocks,
    size_t blockRows) {
  std::vector<T> sample;
  sample.reserve(numBlocks * blockRows);
  const size_t stride = values.size() / numBlocks;
  for (size_t block = 0; block < numBlocks; ++block) {
    const auto first = values.begin() + block * stride;
    sample.insert(sample.end(), first, first + blockRows);
  }
  return sample;
}

} // namespace facebook::nimble
