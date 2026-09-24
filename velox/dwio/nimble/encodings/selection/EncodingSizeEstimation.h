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

#include <glog/logging.h>
#include <optional>
#include <span>
#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/common/Types.h"
#include "velox/dwio/nimble/encodings/ALPEncoding.h"
#include "velox/dwio/nimble/encodings/BlockBitPackingEncoding.h"
#include "velox/dwio/nimble/encodings/ConstantEncoding.h"
#include "velox/dwio/nimble/encodings/DeltaBlockEncoding.h"
#include "velox/dwio/nimble/encodings/DeltaEncoding.h"
#include "velox/dwio/nimble/encodings/DictionaryEncoding.h"
#include "velox/dwio/nimble/encodings/EliasFanoEncoding.h"
#include "velox/dwio/nimble/encodings/FixedBitWidthEncoding.h"
#include "velox/dwio/nimble/encodings/ForEncoding.h"
#include "velox/dwio/nimble/encodings/FrequencyPartitionEncoding.h"
#include "velox/dwio/nimble/encodings/FsstEncoding.h"
#include "velox/dwio/nimble/encodings/HuffmanEncoding.h"
#include "velox/dwio/nimble/encodings/MainlyConstantEncoding.h"
#include "velox/dwio/nimble/encodings/PFOREncoding.h"
#include "velox/dwio/nimble/encodings/RLEEncoding.h"
#include "velox/dwio/nimble/encodings/SimdForBitpackEncoding.h"
#include "velox/dwio/nimble/encodings/SparseBoolEncoding.h"
#include "velox/dwio/nimble/encodings/SubIntSplitEncoding.h"
#include "velox/dwio/nimble/encodings/TrivialEncoding.h"
#include "velox/dwio/nimble/encodings/VarintEncoding.h"

namespace facebook::nimble {

namespace detail {

/// Estimates encoded data size for encoding selection. Estimates are heuristic
/// and are not expected to match the serialized size exactly.
template <typename T>
struct EncodingSizeEstimation {
  using physicalType = typename TypeTraits<T>::physicalType;

  static std::optional<uint64_t> estimateSize(
      const EncodingType encodingType,
      const size_t entryCount,
      const Statistics<physicalType>& statistics,
      const Encoding::Options& options) {
    if constexpr (isNumericType<physicalType>()) {
      return estimateNumericSize(encodingType, entryCount, statistics, options);
    } else if constexpr (isBoolType<physicalType>()) {
      return estimateBoolSize(encodingType, entryCount, statistics, options);
    } else if constexpr (isStringType<physicalType>()) {
      return estimateStringSize(encodingType, entryCount, statistics, options);
    }

    NIMBLE_UNREACHABLE(
        "Unable to estimate size for type {}.", folly::demangle(typeid(T)));
  }

  static std::optional<uint64_t> estimateSize(
      const EncodingType encodingType,
      std::span<const physicalType> values,
      const Statistics<physicalType>& statistics,
      const Encoding::Options& options) {
    if constexpr (isNumericType<physicalType>()) {
      return estimateNumericSize(encodingType, values, statistics, options);
    } else if constexpr (isBoolType<physicalType>()) {
      return estimateBoolSize(encodingType, values, statistics, options);
    } else if constexpr (isStringType<physicalType>()) {
      return estimateStringSize(encodingType, values, statistics, options);
    }

    NIMBLE_UNREACHABLE(
        "Unable to estimate size for type {}.", folly::demangle(typeid(T)));
  }

  /// A size estimateSize never quotes below for `encodingType`, for encodings
  /// whose estimate counts the stream's distinct values but where a bound can
  /// be had without counting them (MainlyConstant, Dictionary,
  /// FrequencyPartition on a too-wide-to-table integer stream). nullopt
  /// otherwise.
  static std::optional<uint64_t> estimateSizeLowerBound(
      const EncodingType encodingType,
      std::span<const physicalType> values,
      const Statistics<physicalType>& statistics,
      const Encoding::Options& options) {
    if constexpr (
        !isIntegralType<T>() || !isIntegralType<physicalType>() ||
        isBoolType<physicalType>()) {
      return std::nullopt;
    } else {
#ifdef NIMBLE_ENABLE_EXPERIMENTAL_ENCODINGS
      // Bounded from the bit-flip profile rather than from a distinct count,
      // so it is answered before the preamble below.
      if (encodingType == EncodingType::SubIntSplit) {
        if constexpr (sizeof(physicalType) == 4 || sizeof(physicalType) == 8) {
          return SubIntSplitEncoding<T>::estimateSizeLowerBound(
              values, statistics, options);
        } else {
          return std::nullopt;
        }
      }
#endif
      if (encodingType != EncodingType::MainlyConstant &&
          encodingType != EncodingType::Dictionary
#ifdef NIMBLE_ENABLE_EXPERIMENTAL_ENCODINGS
          && encodingType != EncodingType::FrequencyPartition
#endif
      ) {
        return std::nullopt;
      }
      // Below this range Statistics counts distinct values in a table, which
      // costs less than the pass the bound takes.
      constexpr uint64_t kMinBoundedRange{65'536};
      const uint64_t rowCount = values.size();
      if (rowCount == 0 ||
          static_cast<uint64_t>(statistics.max() - statistics.min()) <
              kMinBoundedRange) {
        return std::nullopt;
      }
      const uint64_t distinctLowerBound = statistics.distinctLowerBound();
      switch (encodingType) {
        case EncodingType::Dictionary: {
          // Grows with the distinct count and reads nothing else of the counts.
          return DictionaryEncoding<T>::estimateIntegralSize(
              rowCount,
              distinctLowerBound,
              statistics.min(),
              statistics.max(),
              options);
        }
#ifdef NIMBLE_ENABLE_EXPERIMENTAL_ENCODINGS
        case EncodingType::FrequencyPartition: {
          return FrequencyPartitionEncoding<physicalType>::
              estimateSizeLowerBound(
                  rowCount, distinctLowerBound, statistics, options);
        }
#endif
        default: {
          return mainlyConstantSizeLowerBound(
              values, statistics, distinctLowerBound, options);
        }
      }
    }
  }

 private:
  // Lower bound for MainlyConstantEncodingBase::estimateSize with the common
  // value unknown: with D distinct values, at least D - 1 rows are uncommon,
  // and they span at least the smaller of max - (second smallest) and
  // (second largest) - min.
  static std::optional<uint64_t> mainlyConstantSizeLowerBound(
      std::span<const physicalType> values,
      const Statistics<physicalType>& statistics,
      uint64_t distinctLowerBound,
      const Encoding::Options& options) {
    // Three distinct values guarantee a second smallest and a second largest.
    if (distinctLowerBound < 3) {
      return std::nullopt;
    }
    const physicalType minValue = statistics.min();
    const physicalType maxValue = statistics.max();
    physicalType secondSmallest = maxValue;
    physicalType secondLargest = minValue;
    for (const physicalType value : values) {
      secondSmallest =
          std::min(secondSmallest, value == minValue ? maxValue : value);
      secondLargest =
          std::max(secondLargest, value == maxValue ? minValue : value);
    }
    const uint64_t uncommonRange = std::min<uint64_t>(
        static_cast<uint64_t>(maxValue - secondSmallest),
        static_cast<uint64_t>(secondLargest - minValue));
    const uint64_t uncommonCount = distinctLowerBound - 1;
    const uint64_t otherValuesSize = std::min(
        TrivialEncoding<physicalType>::estimateSize(uncommonCount),
        FixedBitWidthEncoding<physicalType>::estimateSize(
            uncommonCount, 0, uncommonRange, options));
    const uint64_t isCommonEncodingSize =
        SparseBoolEncoding::estimateSize(values.size(), uncommonCount, options);
    const uint64_t outerEncodingSize = EncodingPrefix::kFixedPrefixSize +
        2 * sizeof(uint32_t) + sizeof(physicalType);
    return outerEncodingSize + otherValuesSize + isCommonEncodingSize;
  }

  // Prices RLE's run-lengths stream with the encodings nested selection would
  // pick for it, rather than only the flat FixedBitWidth RLE's own estimate
  // uses; that flat price stays a candidate, so this never quotes above it.
  static uint64_t estimateRunLengthsSize(
      const Statistics<physicalType>& statistics,
      const Encoding::Options& options) {
    const uint64_t runCount = statistics.consecutiveRepeatCount();
    uint64_t bestSize = FixedBitWidthEncoding<uint32_t>::estimateSize(
        runCount, statistics.minRepeat(), statistics.maxRepeat(), options);
    if (runCount < 2) {
      return bestSize;
    }
    const auto& runLengths = statistics.runLengths();
    const auto runLengthsStatistics = Statistics<uint32_t>::create(
        std::span<const uint32_t>{runLengths.data(), runLengths.size()});
    bestSize = std::min(
        bestSize,
        MainlyConstantEncoding<uint32_t>::estimateSize(
            runCount, runLengthsStatistics, options));
#ifdef NIMBLE_ENABLE_EXPERIMENTAL_ENCODINGS
    bestSize = std::min(
        bestSize,
        FrequencyPartitionEncoding<uint32_t>::estimateSize(
            runCount, runLengthsStatistics, options));
#endif
    return bestSize;
  }

  static std::optional<uint64_t> estimateNumericSize(
      const EncodingType encodingType,
      const uint64_t entryCount,
      const Statistics<physicalType>& statistics,
      const Encoding::Options& options) {
    switch (encodingType) {
      case EncodingType::Constant: {
        return std::nullopt;
      }
      case EncodingType::MainlyConstant: {
        // TODO: Wire per-block tightening when BlockBitPacking is a
        // candidate. MainlyConstantEncoding::estimateSize already supports
        // usePerBlockStats + blockSize params.
        return MainlyConstantEncoding<T>::estimateSize(
            entryCount, statistics, options);
      }
      case EncodingType::Trivial: {
        return TrivialEncoding<physicalType>::estimateSize(entryCount);
      }
      case EncodingType::FixedBitWidth: {
        return FixedBitWidthEncoding<physicalType>::estimateSize(
            entryCount, statistics, options);
      }
      case EncodingType::Dictionary: {
        // TODO: Wire per-block tightening when BlockBitPacking is a
        // candidate. DictionaryEncoding::estimateSize already supports
        // usePerBlockStats + blockSize params.
        return DictionaryEncoding<T>::estimateSize(
            entryCount, statistics, options);
      }
      case EncodingType::RLE: {
        return RLEEncoding<T>::estimateSize(
            entryCount,
            statistics,
            estimateRunLengthsSize(statistics, options),
            options);
      }
      case EncodingType::Varint: {
        // Note: the condition below actually support floating point numbers as
        // well, as we use physicalType, which, for floating point numbers, is
        // an integer.
        if constexpr (isIntegralType<physicalType>() && sizeof(T) >= 4) {
          return VarintEncoding<physicalType>::estimateSize(statistics);
        } else {
          return std::nullopt;
        }
      }
      case EncodingType::PFOR: {
        if constexpr (isIntegralType<physicalType>()) {
          return PFOREncoding<physicalType>::estimateSize(
              entryCount, statistics, options);
        } else {
          return std::nullopt;
        }
      }
      case EncodingType::SimdForBitpack: {
        if constexpr (isIntegralType<physicalType>()) {
          return SimdForBitpackEncoding<physicalType>::estimateSize(
              entryCount, statistics.min(), statistics.max());
        } else {
          return std::nullopt;
        }
      }
      case EncodingType::BlockBitPacking: {
        // estimateSize() returns nullopt when the stream has more blocks than
        // the uint16_t block index can address, so selection skips this
        // encoding instead of picking it and hard-failing in encode().
        return BlockBitPackingEncoding<physicalType>::estimateSize(
            statistics, options.blockBitPackingBlockSize);
      }
#ifdef NIMBLE_ENABLE_EXPERIMENTAL_ENCODINGS
      case EncodingType::SubIntSplit: {
        if constexpr (
            isNumericType<physicalType>() &&
            (sizeof(physicalType) == 4 || sizeof(physicalType) == 8)) {
          // No values to plan a split over, so this falls back to the
          // FixedBitWidth bound the encoder's whole-value floor guarantees.
          return SubIntSplitEncoding<T>::estimateSize(
              entryCount, {}, statistics, options);
        } else {
          return std::nullopt;
        }
      }
#endif
#ifdef NIMBLE_ENABLE_EXPERIMENTAL_ENCODINGS
      case EncodingType::Delta: {
        if constexpr (isIntegralType<physicalType>()) {
          return DeltaEncoding<physicalType>::estimateSize(
              entryCount, statistics, options);
        } else {
          return std::nullopt;
        }
      }
      case EncodingType::FOR: {
        if constexpr (isIntegralType<physicalType>()) {
          return ForEncoding<physicalType>::estimateSize(
              entryCount, statistics);
        } else {
          return std::nullopt;
        }
      }
      case EncodingType::FrequencyPartition: {
        if constexpr (isIntegralType<physicalType>()) {
          // Options carry the index type, which is a large part of what a
          // FrequencyPartition encode costs and is not visible from statistics.
          return FrequencyPartitionEncoding<physicalType>::estimateSize(
              entryCount, statistics, options);
        } else {
          return std::nullopt;
        }
      }
#endif
      default: {
        return std::nullopt;
      }
    }
  }

  static std::optional<uint64_t> estimateNumericSize(
      const EncodingType encodingType,
      std::span<const physicalType> values,
      const Statistics<physicalType>& statistics,
      const Encoding::Options& options) {
    switch (encodingType) {
      case EncodingType::Constant: {
        return ConstantEncoding<T>::estimateSize(values, statistics, options);
      }
      case EncodingType::Huffman: {
        // Gates on T, not physicalType: EncodingFactory reads Huffman back
        // through RETURN_ENCODING_BY_INTEGER_TYPE, which rejects floating-point
        // data types. Neighbouring cases may gate on physicalType because their
        // read dispatch accepts it.
        if constexpr (isIntegralType<T>()) {
          return HuffmanEncoding<T>::estimateSize(values, statistics, options);
        } else {
          return std::nullopt;
        }
      }
      case EncodingType::ALP: {
        if constexpr (isFloatingPointType<T>()) {
          return ALPEncoding<T>::estimateSize(values, options);
        } else {
          return std::nullopt;
        }
      }
      case EncodingType::DeltaBlock: {
        if constexpr (isIntegralType<T>()) {
          return DeltaBlockEncoding<T>::estimateSize(values, options);
        } else {
          return std::nullopt;
        }
      }
      case EncodingType::EliasFano: {
        if constexpr (isIntegralType<T>()) {
          return EliasFanoEncoding<T>::estimateSize(
              values, statistics, options);
        } else {
          return std::nullopt;
        }
      }
#ifdef NIMBLE_ENABLE_EXPERIMENTAL_ENCODINGS
      case EncodingType::FOR: {
        // Measured over the values rather than inferred from statistics: a
        // frame's local range is a property of the values in position, not
        // of any summary of them.
        if constexpr (isIntegralType<physicalType>()) {
          return ForEncoding<physicalType>::estimateSize(values, options);
        } else {
          return std::nullopt;
        }
      }
      case EncodingType::SubIntSplit: {
        // Planned over the values, for the same reason FOR is: where the bit
        // fields of a value sit, and how each behaves down the stream, is not
        // in any summary of the values.
        if constexpr (
            isNumericType<physicalType>() &&
            (sizeof(physicalType) == 4 || sizeof(physicalType) == 8)) {
          return SubIntSplitEncoding<T>::estimateSize(
              values.size(), values, statistics, options);
        } else {
          return std::nullopt;
        }
      }
#endif
      default: {
        return estimateNumericSize(
            encodingType, values.size(), statistics, options);
      }
    }
  }

  static std::optional<uint64_t> estimateBoolSize(
      const EncodingType encodingType,
      const size_t entryCount,
      const Statistics<physicalType>& statistics,
      const Encoding::Options& options) {
    switch (encodingType) {
      case EncodingType::Constant: {
        return std::nullopt;
      }
      case EncodingType::SparseBool: {
        return SparseBoolEncoding::estimateSize(
            entryCount, statistics, options);
      }
      case EncodingType::Trivial: {
        return TrivialEncoding<bool>::estimateSize(entryCount);
      }
      case EncodingType::RLE: {
        return RLEEncoding<bool>::estimateSize(entryCount, statistics, options);
      }
      default: {
        return std::nullopt;
      }
    }
  }

  static std::optional<uint64_t> estimateBoolSize(
      const EncodingType encodingType,
      std::span<const physicalType> values,
      const Statistics<physicalType>& statistics,
      const Encoding::Options& options) {
    switch (encodingType) {
      case EncodingType::Constant: {
        return ConstantEncoding<bool>::estimateSize(
            values, statistics, options);
      }
      default: {
        return estimateBoolSize(
            encodingType, values.size(), statistics, options);
      }
    }
  }

  static std::optional<uint64_t> estimateStringSize(
      const EncodingType encodingType,
      const size_t entryCount,
      const Statistics<std::string_view>& statistics,
      const Encoding::Options& options) {
    switch (encodingType) {
      case EncodingType::Constant: {
        return std::nullopt;
      }
      case EncodingType::MainlyConstant: {
        return MainlyConstantEncoding<std::string_view>::estimateSize(
            entryCount, statistics, options);
      }
      case EncodingType::Trivial: {
        return TrivialEncoding<std::string_view>::estimateSize(
            entryCount, statistics, options);
      }
      case EncodingType::Dictionary: {
        return DictionaryEncoding<std::string_view>::estimateSize(
            entryCount, statistics, options);
      }
      case EncodingType::RLE: {
        return RLEEncoding<std::string_view>::estimateSize(
            entryCount, statistics, options);
      }
      case EncodingType::Fsst: {
        return FsstEncoding::estimateSize(entryCount, statistics, options);
      }
      default: {
        return std::nullopt;
      }
    }
  }

  static std::optional<uint64_t> estimateStringSize(
      const EncodingType encodingType,
      std::span<const physicalType> values,
      const Statistics<std::string_view>& statistics,
      const Encoding::Options& options) {
    switch (encodingType) {
      case EncodingType::Constant: {
        return ConstantEncoding<std::string_view>::estimateSize(
            values, statistics, options);
      }
      default: {
        return estimateStringSize(
            encodingType, values.size(), statistics, options);
      }
    }
  }
};
} // namespace detail
} // namespace facebook::nimble
