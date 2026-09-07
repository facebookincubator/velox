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
#include "velox/dwio/nimble/encodings/selection/Statistics.h"
#include "velox/dwio/nimble/common/StatsUtil.h"
#include "velox/dwio/nimble/common/Types.h"

#include <algorithm>
#include <bit>
#include <cstdint>
#include <limits>
#include <type_traits>
#include "velox/common/base/SimdUtil.h"

namespace facebook::nimble {

namespace {

constexpr uint32_t kMaxDenseRangeSize{4096};

template <typename T, typename InputType>
using MapType = typename UniqueValueCounts<T, InputType>::MapType;

template <typename T>
uint64_t integralRangeDistance(T value, T rangeBase) {
  return static_cast<uint64_t>(value) - static_cast<uint64_t>(rangeBase);
}

template <typename T>
uint32_t denseRangeOffset(T value, T rangeBase) {
  return static_cast<uint32_t>(integralRangeDistance(value, rangeBase));
}

template <typename T>
T integralValueAtOffset(T rangeBase, uint32_t offset) {
  if constexpr (std::is_signed_v<T>) {
    return static_cast<T>(
        static_cast<int64_t>(rangeBase) + static_cast<int64_t>(offset));
  } else {
    return static_cast<T>(static_cast<uint64_t>(rangeBase) + offset);
  }
}

template <typename T>
std::optional<uint32_t>
denseRangeSize(T minValue, T maxValue, size_t valueCount) {
  const auto rangeDistance = integralRangeDistance(maxValue, minValue);
  const auto maxRangeSize = std::min<uint64_t>(kMaxDenseRangeSize, valueCount);
  if (rangeDistance >= maxRangeSize) {
    return std::nullopt;
  }
  return static_cast<uint32_t>(rangeDistance) + 1;
}

template <typename T, typename InputType>
MapType<T, InputType> populateHashUniqueCounts(
    std::span<const InputType> values) {
  MapType<T, InputType> uniqueCounts;
  // NOTE: There is no science behind the reservation size. Just trying to
  // minimize internal allocations...
  uniqueCounts.reserve(values.size() / 3);
  for (auto i = 0; i < values.size(); ++i) {
    ++uniqueCounts[values[i]];
  }
  return uniqueCounts;
}

template <typename T>
MapType<T, T> populateRangeUniqueCounts(
    std::span<const T> values,
    T minValue,
    uint32_t rangeSize) {
  if (rangeSize == 1) {
    MapType<T, T> uniqueCounts;
    uniqueCounts.reserve(1);
    uniqueCounts.emplace(minValue, static_cast<uint64_t>(values.size()));
    return uniqueCounts;
  }

  std::vector<uint64_t> counts(rangeSize);
  for (const auto value : values) {
    ++counts[denseRangeOffset(value, minValue)];
  }

  MapType<T, T> uniqueCounts;
  uniqueCounts.reserve(std::min<size_t>(rangeSize, values.size()));
  for (uint32_t offset = 0; offset < rangeSize; ++offset) {
    if (counts[offset] > 0) {
      uniqueCounts.emplace(
          integralValueAtOffset(minValue, offset), counts[offset]);
    }
  }
  return uniqueCounts;
}

uint64_t countTrueValues(std::span<const bool> values) {
  static_assert(sizeof(bool) == sizeof(uint8_t));
  constexpr auto kBatchSize = xsimd::batch<uint8_t>::size;
  const auto* rawValues = reinterpret_cast<const uint8_t*>(values.data());
  const auto zero = xsimd::batch<uint8_t>::broadcast(0);

  uint64_t trueCount{0};
  size_t offset{0};
  for (; offset + kBatchSize <= values.size(); offset += kBatchSize) {
    const auto batch =
        xsimd::batch<uint8_t>::load_unaligned(rawValues + offset);
    trueCount += std::popcount(
        static_cast<uint32_t>(facebook::velox::simd::toBitMask(batch != zero)));
  }
  for (; offset < values.size(); ++offset) {
    trueCount += static_cast<uint8_t>(values[offset]);
  }
  return trueCount;
}

} // namespace

template <typename T, typename InputType>
void Statistics<T, InputType>::populateRepeats(bool collectRunValues) const {
  uint64_t consecutiveRepeatCount = 0;
  uint64_t minRepeat = std::numeric_limits<uint64_t>::max();
  uint64_t maxRepeat = 0;

  uint64_t totalRepeatLength = 0; // only needed for strings
  if constexpr (nimble::isStringType<T>()) {
    totalRepeatLength = data_[0].size();
  }

  T currentValue = data_[0];
  uint64_t currentRepeat = 0;
  std::vector<T> runValues;
  if (collectRunValues) {
    runValues.push_back(currentValue);
  }

  for (auto i = 0; i < data_.size(); ++i) {
    const auto& value = data_[i];

    if (value == currentValue) {
      ++currentRepeat;
    } else {
      if constexpr (nimble::isStringType<T>()) {
        totalRepeatLength += value.size();
      }
      if (currentRepeat > maxRepeat) {
        maxRepeat = currentRepeat;
      }

      if (currentRepeat < minRepeat) {
        minRepeat = currentRepeat;
      }

      currentRepeat = 1;
      currentValue = value;
      if (collectRunValues) {
        runValues.push_back(currentValue);
      }
      ++consecutiveRepeatCount;
    }
  }

  if (currentRepeat > maxRepeat) {
    maxRepeat = currentRepeat;
  }

  if (currentRepeat < minRepeat) {
    minRepeat = currentRepeat;
  }

  ++consecutiveRepeatCount;
  minRepeat_ = minRepeat;
  maxRepeat_ = maxRepeat;
  consecutiveRepeatCount_ = consecutiveRepeatCount;
  totalStringsRepeatLength_ = totalRepeatLength;
  if (collectRunValues) {
    runValues_ = std::move(runValues);
  }
}

template <typename T, typename InputType>
void Statistics<T, InputType>::populateMinMax() const {
  if constexpr (nimble::isNumericType<InputType>()) {
    if constexpr (kIntegralMinMaxType<InputType>) {
      const auto minMax = findMinMax(data_);
      min_ = minMax.min;
      max_ = minMax.max;
    } else {
      // Floating point uses std::minmax_element to preserve NaN ordering.
      const auto [min, max] = std::minmax_element(data_.begin(), data_.end());
      min_ = *min;
      max_ = *max;
    }
  } else if constexpr (nimble::isStringType<InputType>()) {
    populateStringLength();
  }
}

template <typename T, typename InputType>
void Statistics<T, InputType>::populateUniques() const {
  MapType<T, InputType> uniqueCounts;
  if constexpr (nimble::isBoolType<T>()) {
    const uint64_t trueCount = countTrueValues(data_);
    const uint64_t falseCount = static_cast<uint64_t>(data_.size()) - trueCount;
    uniqueCounts.reserve(2);
    if (falseCount > 0) {
      uniqueCounts.emplace(false, falseCount);
    }
    if (trueCount > 0) {
      uniqueCounts.emplace(true, trueCount);
    }
  } else if constexpr (
      nimble::isIntegralType<T>() && std::is_same_v<T, InputType>) {
    const T minValue = min();
    const T maxValue = max();
    if (const auto rangeSize =
            denseRangeSize(minValue, maxValue, data_.size())) {
      uniqueCounts =
          populateRangeUniqueCounts<T>(data_, minValue, rangeSize.value());
    } else {
      uniqueCounts = populateHashUniqueCounts<T, InputType>(data_);
    }
  } else {
    uniqueCounts = populateHashUniqueCounts<T, InputType>(data_);
  }
  uniqueCounts_.emplace(std::make_optional(std::move(uniqueCounts)));
}

template <typename T, typename InputType>
void Statistics<T, InputType>::populateMinMaxBlocks(uint16_t blockSize) const {
  static_assert(std::is_unsigned_v<T>);
  BlockStatsAccumulator acc(blockSize);
  for (const auto& v : data_) {
    acc.add(static_cast<uint64_t>(static_cast<T>(v)));
  }
  minMaxBlocks_ = acc.finish();
}

template <typename T, typename InputType>
void Statistics<T, InputType>::populateBucketCounts() const {
  using UnsignedT = typename std::make_unsigned<T>::type;
  // Bucket counts are calculated in two phases. In phase one, we iterate on all
  // entries, and (efficiently) count the occurrences based on the MSB (most
  // significant bit) of the entry. In phase two, we merge the results of phase
  // one, for each consecutive 7 bits.
  // See benchmarks in
  // velox/dwio/nimble/encodings/tests:bucket_benchmark for why this method is
  // used.
  std::array<uint64_t, std::numeric_limits<UnsignedT>::digits + 1> bitCounts{};
  for (auto i = 0; i < data_.size(); ++i) {
    ++(bitCounts
           [std::numeric_limits<UnsignedT>::digits -
            std::countl_zero(
                static_cast<UnsignedT>(
                    static_cast<UnsignedT>(data_[i]) -
                    static_cast<UnsignedT>(min())))]);
  }

  std::vector<uint64_t> bucketCounts(sizeof(T) * 8 / 7 + 1, 0);
  uint8_t start = 0;
  uint8_t end = 8;
  uint8_t iteration = 0;
  while (start < bitCounts.size()) {
    for (auto i = start; i < end; ++i) {
      bucketCounts[iteration] += bitCounts[i];
    }
    ++iteration;
    start = end;
    end += 7;
    if (bitCounts.size() < end) {
      end = bitCounts.size();
    }
  }

  bucketCounts_ = std::move(bucketCounts);
}

template <typename T, typename InputType>
void Statistics<T, InputType>::populateStringLength() const {
  uint64_t totalBytes = 0;
  std::string_view minString = data_[0];
  std::string_view maxString = data_[0];
  for (int i = 0; i < data_.size(); ++i) {
    const auto& value = data_[i];
    totalBytes += value.size();
    if (value.size() > maxString.size()) {
      maxString = value;
    }
    if (value.size() < minString.size()) {
      minString = value;
    }
  }
  totalStringsLength_ = totalBytes;
  min_ = minString;
  max_ = maxString;
}

template <typename T, typename InputType>
Statistics<T, InputType> Statistics<T, InputType>::create(
    std::span<const InputType> data) {
  Statistics<T, InputType> statistics;
  if (data.size() == 0) {
    statistics.consecutiveRepeatCount_ = 0;
    statistics.minRepeat_ = 0;
    statistics.maxRepeat_ = 0;
    statistics.totalStringsLength_ = 0;
    statistics.totalStringsRepeatLength_ = 0;
    statistics.min_ = T();
    statistics.max_ = T();

    statistics.bucketCounts_ = {};
    statistics.uniqueCounts_ = std::make_optional(
        std::make_optional(UniqueValueCounts<T, InputType>()));
    return statistics;
  }

  statistics.data_ = data;
  return statistics;
}

template Statistics<int8_t> Statistics<int8_t>::create(
    std::span<const int8_t> data);
template Statistics<uint8_t> Statistics<uint8_t>::create(
    std::span<const uint8_t> data);
template Statistics<int16_t> Statistics<int16_t>::create(
    std::span<const int16_t> data);
template Statistics<uint16_t> Statistics<uint16_t>::create(
    std::span<const uint16_t> data);
template Statistics<int32_t> Statistics<int32_t>::create(
    std::span<const int32_t> data);
template Statistics<uint32_t> Statistics<uint32_t>::create(
    std::span<const uint32_t> data);
template Statistics<int64_t> Statistics<int64_t>::create(
    std::span<const int64_t> data);
template Statistics<uint64_t> Statistics<uint64_t>::create(
    std::span<const uint64_t> data);
template Statistics<float> Statistics<float>::create(
    std::span<const float> data);
template Statistics<double> Statistics<double>::create(
    std::span<const double> data);
template Statistics<bool> Statistics<bool>::create(std::span<const bool> data);
template Statistics<std::string_view> Statistics<std::string_view>::create(
    std::span<const std::string_view> data);
template Statistics<std::string_view, std::string>
Statistics<std::string_view, std::string>::create(
    std::span<const std::string> data);

// populateRepeats works on all types
template void Statistics<int8_t>::populateRepeats(bool) const;
template void Statistics<uint8_t>::populateRepeats(bool) const;
template void Statistics<int16_t>::populateRepeats(bool) const;
template void Statistics<uint16_t>::populateRepeats(bool) const;
template void Statistics<int32_t>::populateRepeats(bool) const;
template void Statistics<uint32_t>::populateRepeats(bool) const;
template void Statistics<int64_t>::populateRepeats(bool) const;
template void Statistics<uint64_t>::populateRepeats(bool) const;
template void Statistics<float>::populateRepeats(bool) const;
template void Statistics<double>::populateRepeats(bool) const;
template void Statistics<bool>::populateRepeats(bool) const;
template void Statistics<std::string_view>::populateRepeats(bool) const;
template void Statistics<std::string_view, std::string>::populateRepeats(
    bool) const;

// populateUniques works on all types
template void Statistics<int8_t>::populateUniques() const;
template void Statistics<uint8_t>::populateUniques() const;
template void Statistics<int16_t>::populateUniques() const;
template void Statistics<uint16_t>::populateUniques() const;
template void Statistics<int32_t>::populateUniques() const;
template void Statistics<uint32_t>::populateUniques() const;
template void Statistics<int64_t>::populateUniques() const;
template void Statistics<uint64_t>::populateUniques() const;
template void Statistics<float>::populateUniques() const;
template void Statistics<double>::populateUniques() const;
template void Statistics<bool>::populateUniques() const;
template void Statistics<std::string_view>::populateUniques() const;
template void Statistics<std::string_view, std::string>::populateUniques()
    const;

// populateMinMax works on numeric types only
template void Statistics<int8_t>::populateMinMax() const;
template void Statistics<uint8_t>::populateMinMax() const;
template void Statistics<int16_t>::populateMinMax() const;
template void Statistics<uint16_t>::populateMinMax() const;
template void Statistics<int32_t>::populateMinMax() const;
template void Statistics<uint32_t>::populateMinMax() const;
template void Statistics<int64_t>::populateMinMax() const;
template void Statistics<uint64_t>::populateMinMax() const;
template void Statistics<float>::populateMinMax() const;
template void Statistics<double>::populateMinMax() const;
template void Statistics<std::string_view>::populateMinMax() const;
template void Statistics<std::string_view, std::string>::populateMinMax() const;

// populateMinMaxBlocks is used through the estimation path where T is always
// the unsigned physicalType.
template void Statistics<uint8_t>::populateMinMaxBlocks(uint16_t) const;
template void Statistics<uint16_t>::populateMinMaxBlocks(uint16_t) const;
template void Statistics<uint32_t>::populateMinMaxBlocks(uint16_t) const;
template void Statistics<uint64_t>::populateMinMaxBlocks(uint16_t) const;

// populateBucketCounts works on integral types only
template void Statistics<int8_t>::populateBucketCounts() const;
template void Statistics<uint8_t>::populateBucketCounts() const;
template void Statistics<int16_t>::populateBucketCounts() const;
template void Statistics<uint16_t>::populateBucketCounts() const;
template void Statistics<int32_t>::populateBucketCounts() const;
template void Statistics<uint32_t>::populateBucketCounts() const;
template void Statistics<int64_t>::populateBucketCounts() const;
template void Statistics<uint64_t>::populateBucketCounts() const;

// String functions
template void Statistics<std::string_view>::populateStringLength() const;
template void Statistics<std::string_view, std::string>::populateStringLength()
    const;

} // namespace facebook::nimble
