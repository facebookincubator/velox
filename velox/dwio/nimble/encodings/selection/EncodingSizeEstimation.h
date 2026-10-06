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
#include "velox/dwio/nimble/encodings/DictionaryEncoding.h"
#include "velox/dwio/nimble/encodings/EliasFanoEncoding.h"
#include "velox/dwio/nimble/encodings/FixedBitWidthEncoding.h"
#include "velox/dwio/nimble/encodings/FsstEncoding.h"
#include "velox/dwio/nimble/encodings/HuffmanEncoding.h"
#include "velox/dwio/nimble/encodings/MainlyConstantEncoding.h"
#include "velox/dwio/nimble/encodings/PFOREncoding.h"
#include "velox/dwio/nimble/encodings/RLEEncoding.h"
#include "velox/dwio/nimble/encodings/SimdForBitpackEncoding.h"
#include "velox/dwio/nimble/encodings/SparseBoolEncoding.h"
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
  /// be had without counting them (MainlyConstant and Dictionary on a
  /// too-wide-to-table integer stream). nullopt otherwise.
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
      if (encodingType != EncodingType::MainlyConstant &&
          encodingType != EncodingType::Dictionary) {
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
      if (encodingType == EncodingType::Dictionary) {
        // Grows with the distinct count and reads nothing else of the counts.
        return DictionaryEncoding<T>::estimateIntegralSize(
            rowCount,
            distinctLowerBound,
            statistics.min(),
            statistics.max(),
            options);
      }
      return mainlyConstantSizeLowerBound(
          values, statistics, distinctLowerBound, options);
    }
  }

 private:
  // Lower bound for MainlyConstantEncodingBase::estimateSize with the common
  // value unknown: with D distinct values, at least D - 1 rows are uncommon,
  // and they span at least the smaller of max - (second smallest) and
  // (second largest) - min. The other-values stream is priced at the cheaper
  // of Trivial and FixedBitWidth over that span, which bounds any pricing of
  // it over the uncommon values or over the whole stream's range.
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
        return RLEEncoding<T>::estimateSize(entryCount, statistics, options);
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
