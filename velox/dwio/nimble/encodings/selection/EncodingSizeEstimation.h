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

 private:
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
      // SubIntSplit integration (re-enabled for
      // NIMBLE_ENABLE_EXPERIMENTAL_ENCODINGS; was commented out by #636):
#ifdef NIMBLE_ENABLE_EXPERIMENTAL_ENCODINGS
      case EncodingType::SubIntSplit: {
        if constexpr (
            isNumericType<physicalType>() &&
            (sizeof(physicalType) == 4 || sizeof(physicalType) == 8)) {
          constexpr uint64_t kTypeWidthBits =
              static_cast<uint64_t>(sizeof(physicalType)) * 8u;
          const uint64_t rangeBits =
              velox::bits::bitsRequired(statistics.max() - statistics.min());
          if (rangeBits > (kTypeWidthBits * 3) / 4) {
            return std::nullopt;
          }
          const auto fbwEst = estimateNumericSize(
              EncodingType::FixedBitWidth, entryCount, statistics, options);
          if (!fbwEst.has_value()) {
            return std::nullopt;
          }
          constexpr uint64_t kOverheadBytes = 6u + 2u + 4u * 6u + 4u * 8u;
          const uint64_t estimate =
              static_cast<uint64_t>(
                  static_cast<double>(fbwEst.value()) * 0.90) +
              kOverheadBytes;
          return estimate;
        } else {
          return std::nullopt;
        }
      }
#endif
      // Delta/FOR integration (re-enabled for
      // NIMBLE_ENABLE_EXPERIMENTAL_ENCODINGS; was commented out by #636):
#ifdef NIMBLE_ENABLE_EXPERIMENTAL_ENCODINGS
      case EncodingType::Delta: {
        if constexpr (isIntegralType<physicalType>()) {
          return DeltaEncoding<physicalType>::estimateSize(
              entryCount, statistics);
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
      // FrequencyPartition integration (re-enabled for
      // NIMBLE_ENABLE_EXPERIMENTAL_ENCODINGS; was commented out by #636):
      case EncodingType::FrequencyPartition: {
        if constexpr (isIntegralType<physicalType>()) {
          return FrequencyPartitionEncoding<physicalType>::estimateSize(
              entryCount, statistics);
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
        if constexpr (isIntegralType<physicalType>()) {
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
