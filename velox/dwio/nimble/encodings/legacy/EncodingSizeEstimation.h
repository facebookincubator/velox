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
#include <algorithm>
#include <optional>
#include "velox/common/base/BitUtil.h"
#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/common/FixedBitArray.h"
#include "velox/dwio/nimble/common/Types.h"

namespace facebook::nimble::legacy {
namespace detail {

// This class is meant to quickly estimate the size of encoded data using a
// given encoding type. It does a lot of assumptions, and it is not meant to be
// 100% accurate.
template <typename T, bool FixedByteWidth>
struct EncodingSizeEstimation {
  using physicalType = typename TypeTraits<T>::physicalType;

  static std::optional<uint64_t> estimateNumericSize(
      const EncodingType encodingType,
      const uint64_t entryCount,
      const nimble::Statistics<physicalType>& statistics) {
    switch (encodingType) {
      case EncodingType::Constant: {
        return statistics.uniqueCounts().value().size() == 1
            ? std::optional<uint64_t>{getEncodingOverhead<
                  EncodingType::Constant,
                  physicalType>()}
            : std::nullopt;
      }
      case EncodingType::MainlyConstant: {
        // Assumptions:
        // We store one entry for the common value.
        // Number of uncommon values is total item count minus the max unique
        // count (the common value count).
        // For each uncommon value we store the its value. We assume they will
        // be stored bit-packed.
        // We also store a bitmap for all rows (is-common bitmap). This bitmap
        // will most likely be stored as SparseBool. Therefore, for each
        // uncommon value there will be an index. These indices will most likely
        // be stored bit-packed, with bit width of max(rowCount).

        // Find most common item count
        const auto maxUniqueCount = std::max_element(
            statistics.uniqueCounts().value().cbegin(),
            statistics.uniqueCounts().value().cend(),
            [](const auto& a, const auto& b) { return a.second < b.second; });
        // Deduce uncommon values count
        const auto uncommonCount = entryCount - maxUniqueCount->second;
        // Assuming the uncommon values will be stored bit packed.
        const auto uncommonValueSize =
            bitPackedBytes(statistics.min(), statistics.max(), uncommonCount);
        // Uncommon (sparse bool) bitmap will have index per uncommon value,
        // stored bit packed.
        const auto uncommonIndicesSize =
            bitPackedBytes(0, entryCount, uncommonCount);
        uint32_t overhead =
            getEncodingOverhead<EncodingType::MainlyConstant, physicalType>() +
            // Overhead for storing uncommon values
            getEncodingOverhead<EncodingType::FixedBitWidth, physicalType>() +
            // Overhead for storing uncommon bitmap
            getEncodingOverhead<EncodingType::SparseBool, bool>() +
            getEncodingOverhead<EncodingType::FixedBitWidth, uint32_t>();
        return overhead + sizeof(physicalType) + uncommonValueSize +
            uncommonIndicesSize;
      }
      case EncodingType::Trivial: {
        return getEncodingOverhead<EncodingType::Trivial, physicalType>() +
            (entryCount * sizeof(physicalType));
      }
      case EncodingType::FixedBitWidth: {
        return getEncodingOverhead<
                   EncodingType::FixedBitWidth,
                   physicalType>() +
            bitPackedBytes(statistics.min(), statistics.max(), entryCount);
      }
      case EncodingType::Dictionary: {
        // Assumptions:
        // Alphabet stored trivially.
        // Indices are stored bit-packed, with bit width needed to store max
        // dictionary size (which is the unique value count).
        const uint64_t indicesSize = bitPackedBytes(
            0, statistics.uniqueCounts().value().size(), entryCount);
        const uint64_t alphabetSize =
            statistics.uniqueCounts().value().size() * sizeof(physicalType);
        uint32_t overhead =
            getEncodingOverhead<EncodingType::Dictionary, physicalType>() +
            // Alphabet overhead
            getEncodingOverhead<EncodingType::Trivial, physicalType>() +
            // Indices overhead
            getEncodingOverhead<EncodingType::FixedBitWidth, uint32_t>();
        return overhead + alphabetSize + indicesSize;
      }
      case EncodingType::RLE: {
        // Assumptions:
        // Run values are stored bit-packed (with bit width needed to store max
        // value). Run lengths are stored using bit-packing (with bit width
        // needed to store max repetition count).

        const auto runValuesSize = bitPackedBytes(
            statistics.min(),
            statistics.max(),
            statistics.consecutiveRepeatCount());
        const auto runLengthsSize = bitPackedBytes(
            statistics.minRepeat(),
            statistics.maxRepeat(),
            statistics.consecutiveRepeatCount());
        uint32_t overhead =
            getEncodingOverhead<EncodingType::RLE, physicalType>() +
            // Overhead of run values
            getEncodingOverhead<EncodingType::FixedBitWidth, physicalType>() +
            // Overhead of run lengths
            getEncodingOverhead<EncodingType::FixedBitWidth, uint32_t>();
        return overhead + runValuesSize + runLengthsSize;
      }
      case EncodingType::Varint: {
        // Note: the condition below actually support floating point numbers as
        // well, as we use physicalType, which, for floating point numbers, is
        // an integer.
        if constexpr (isIntegralType<physicalType>() && sizeof(T) >= 4) {
          // First (7 bit) bucket produces 1 byte number. Second bucket produce
          // 2 byte number and so forth.
          size_t i = 0;
          const uint64_t dataSize = std::accumulate(
              statistics.bucketCounts().cbegin(),
              statistics.bucketCounts().cend(),
              0,
              [&i](const uint64_t sum, const uint64_t bucketSize) {
                return sum + (bucketSize * (++i));
              });
          return getEncodingOverhead<EncodingType::Varint, physicalType>() +
              dataSize;
        } else {
          return std::nullopt;
        }
      }
      default: {
        return std::nullopt;
      }
    }
  }

  static std::optional<uint64_t> estimateBoolSize(
      const EncodingType encodingType,
      const size_t entryCount,
      const Statistics<physicalType>& statistics) {
    switch (encodingType) {
      case EncodingType::Constant: {
        return statistics.uniqueCounts().value().size() == 1
            ? std::optional<uint64_t>{getEncodingOverhead<
                  EncodingType::Constant,
                  physicalType>()}
            : std::nullopt;
      }
      case EncodingType::SparseBool: {
        // Assumptions:
        // Uncommon indices are stored bit-packed (with bit width capable of
        // representing max entry count).

        const auto exceptionCount = std::min(
            statistics.uniqueCounts().value().at(true),
            statistics.uniqueCounts().value().at(false));
        uint32_t overhead =
            getEncodingOverhead<EncodingType::SparseBool, physicalType>() +
            // Overhead for storing exception indices
            getEncodingOverhead<EncodingType::FixedBitWidth, uint32_t>();
        return overhead + sizeof(bool) +
            bitPackedBytes(0, entryCount, exceptionCount);
      }
      case EncodingType::Trivial: {
        return getEncodingOverhead<EncodingType::Trivial, physicalType>() +
            FixedBitArray::bufferSize(entryCount, 1);
      }
      case EncodingType::RLE: {
        // Assumptions:
        // Run lengths are stored using bit-packing (with bit width
        // needed to store max repetition count).

        const auto runLengthsSize = bitPackedBytes(
            statistics.minRepeat(),
            statistics.maxRepeat(),
            statistics.consecutiveRepeatCount());
        uint32_t overhead =
            getEncodingOverhead<EncodingType::RLE, physicalType>() +
            // Overhead of run lengths
            getEncodingOverhead<EncodingType::FixedBitWidth, uint32_t>();
        return overhead + sizeof(bool) + runLengthsSize;
      }
      default: {
        return std::nullopt;
      }
    }
  }

  static std::optional<uint64_t> estimateStringSize(
      const EncodingType encodingType,
      const size_t entryCount,
      const Statistics<std::string_view>& statistics) {
    const uint32_t maxStringSize = statistics.max().size();
    switch (encodingType) {
      case EncodingType::Constant: {
        return statistics.uniqueCounts().value().size() == 1
            ? std::optional<uint64_t>{getEncodingOverhead<
                  EncodingType::Constant,
                  physicalType>(maxStringSize)}
            : std::nullopt;
      }
      case EncodingType::MainlyConstant: {
        // Assumptions:
        // We store one entry for the common value.
        // For each uncommon value we store the its value.
        // Uncommon values will be stored trivially, or if there are
        // repetitions, we assume they will be nested as a dictionary. Either
        // way, it means each (unique) string value will be stored exactly once.
        // (Note: for strings we store the blob size and the lengths, assuming
        // bit-packed).
        // We also store a bitmap for all rows (is-common bitmap). This bitmap
        // will most likely be stored as SparseBool. Therefore, for each
        // uncommon value there will be an index. These indices will most likely
        // be stored bit-packed, with bit width of max(rowCount).

        // Find the most common item count
        const auto maxUniqueCount = std::max_element(
            statistics.uniqueCounts().value().cbegin(),
            statistics.uniqueCounts().value().cend(),
            [](const auto& a, const auto& b) { return a.second < b.second; });
        // Get the total blob size for all (unique) strings
        const uint64_t alphabetByteSize = std::accumulate(
            statistics.uniqueCounts().value().cbegin(),
            statistics.uniqueCounts().value().cend(),
            0,
            [](const uint32_t sum, const auto& unique) {
              return sum + unique.first.size();
            });
        // Deduce uncommon values count
        const auto uncommonCount = entryCount - maxUniqueCount->second;
        // Remove common string from blob, and add the (bit-packed) string
        // lengths to the size.
        // Note: we remove the common string, as it is later added back when
        // calculating the header overhead.
        const uint64_t alphabetSize = alphabetByteSize -
            maxUniqueCount->first.size() +
            bitPackedBytes(statistics.min().size(),
                           statistics.max().size(),
                           statistics.uniqueCounts().value().size());
        // Uncommon values (sparse bool) bitmap will have index per value,
        // stored bit packed.
        const auto uncommonIndicesSize =
            bitPackedBytes(0, entryCount, uncommonCount);
        uint32_t overhead =
            getEncodingOverhead<EncodingType::MainlyConstant, physicalType>(
                maxUniqueCount->first.size()) +
            // Overhead for storing uncommon values
            getEncodingOverhead<EncodingType::Trivial, physicalType>(
                maxStringSize) +
            // Overhead for storing uncommon bitmap
            getEncodingOverhead<EncodingType::SparseBool, bool>();

        return overhead + alphabetSize + uncommonIndicesSize;
      }
      case EncodingType::Trivial: {
        // We assume string lengths will be stored bit packed.
        return getEncodingOverhead<EncodingType::Trivial, physicalType>(
                   maxStringSize) +
            statistics.totalStringsLength() +
            bitPackedBytes(
                   statistics.min().size(),
                   statistics.max().size(),
                   entryCount);
      }
      case EncodingType::Dictionary: {
        // Assumptions:
        // Alphabet stored trivially.
        // Indices are stored bit-packed, with bit width needed to store max
        // dictionary size (which is the unique value count).
        const uint64_t indicesSize = bitPackedBytes(
            0, statistics.uniqueCounts().value().size(), entryCount);
        // Get the total blob size for all (unique) strings
        const uint64_t alphabetByteSize = std::accumulate(
            statistics.uniqueCounts().value().cbegin(),
            statistics.uniqueCounts().value().cend(),
            0,
            [](const uint32_t sum, const auto& unique) {
              return sum + unique.first.size();
            });
        // Add (bit-packed) string lengths
        const uint64_t alphabetSize = alphabetByteSize +
            bitPackedBytes(statistics.min().size(),
                           statistics.max().size(),
                           statistics.uniqueCounts().value().size());
        uint32_t overhead =
            getEncodingOverhead<EncodingType::Dictionary, physicalType>(
                maxStringSize) +
            // Alphabet overhead
            getEncodingOverhead<EncodingType::Trivial, physicalType>(
                maxStringSize) +
            // Indices overhead
            getEncodingOverhead<EncodingType::FixedBitWidth, uint32_t>();
        return overhead + alphabetSize + indicesSize;
      }
      case EncodingType::RLE: {
        // Assumptions:
        // Run values are stored using dictionary (and inside, trivial +
        // bit-packing). Run lengths are stored using bit-packing (with bit
        // width needed to store max repetition count).

        uint64_t runValuesSize =
            // (unique) strings blob size
            std::accumulate(
                statistics.uniqueCounts().value().cbegin(),
                statistics.uniqueCounts().value().cend(),
                0,
                [](const uint32_t sum, const auto& unique) {
                  return sum + unique.first.size();
                }) +
            // (bit-packed) string lengths
            bitPackedBytes(
                statistics.min().size(),
                statistics.max().size(),
                statistics.consecutiveRepeatCount()) +
            // dictionary indices
            bitPackedBytes(
                0,
                statistics.uniqueCounts().value().size(),
                statistics.consecutiveRepeatCount());
        const auto runLengthsSize = bitPackedBytes(
            statistics.minRepeat(),
            statistics.maxRepeat(),
            statistics.consecutiveRepeatCount());
        uint32_t overhead =
            getEncodingOverhead<EncodingType::RLE, physicalType>() +
            // Overhead of run values
            getEncodingOverhead<EncodingType::Dictionary, physicalType>() +
            getEncodingOverhead<EncodingType::Trivial, physicalType>() +
            getEncodingOverhead<EncodingType::FixedBitWidth, uint32_t>() +
            // Overhead of run lengths
            getEncodingOverhead<EncodingType::FixedBitWidth, uint32_t>();
        return overhead + runValuesSize + runLengthsSize;
      }
      default: {
        return std::nullopt;
      }
    }
  }

  static std::optional<uint64_t> estimateSize(
      const EncodingType encodingType,
      const size_t entryCount,
      const Statistics<physicalType>& statistics) {
    if constexpr (isNumericType<physicalType>()) {
      return estimateNumericSize(encodingType, entryCount, statistics);
    } else if constexpr (isBoolType<physicalType>()) {
      return estimateBoolSize(encodingType, entryCount, statistics);
    } else if constexpr (isStringType<physicalType>()) {
      return estimateStringSize(encodingType, entryCount, statistics);
    }

    NIMBLE_UNREACHABLE(
        fmt::format(
            "Unable to estimate size for type {}.",
            folly::demangle(typeid(T))));
  }

  template <EncodingType encodingType, typename dataType>
  static uint32_t getEncodingOverhead(uint32_t size = sizeof(dataType)) {
    constexpr uint32_t commonPrefixSize = 6;
    if constexpr (encodingType == EncodingType::Trivial) {
      if constexpr (isStringType<dataType>()) {
        return commonPrefixSize +
            // Trivial strings also have nested encoding for lengths.
            getEncodingOverhead<EncodingType::FixedBitWidth, uint32_t>() +
            5; // CompressionType (1 byte), BlobOffset (4 bytes)
      } else {
        return commonPrefixSize + 1; // CompressionType (1 byte)
      }
    } else if constexpr (encodingType == EncodingType::Constant) {
      if constexpr (isStringType<dataType>()) {
        return commonPrefixSize + size +
            sizeof(uint32_t); // CommonValue (string bytes and string length)
      } else {
        return commonPrefixSize + sizeof(dataType); // CompressionType (1 byte)
      }
    } else if constexpr (encodingType == EncodingType::FixedBitWidth) {
      return commonPrefixSize + 2 +
          sizeof(dataType); // CompressionType (1 byte), Baseline (size of
                            // dataType), Bit Width (1 byte)
    } else if constexpr (encodingType == EncodingType::MainlyConstant) {
      if constexpr (isStringType<dataType>()) {
        return commonPrefixSize + (2 * sizeof(uint32_t)) + size +
            sizeof(uint32_t); // IsCommon size (4 bytes), OtherValues size (4
                              // bytes), Common value (string bytes and string
                              // length)
      } else {
        return commonPrefixSize + (2 * sizeof(uint32_t)) +
            sizeof(dataType); // IsCommon size (4 bytes), OtherValues size (4
                              // bytes), Common value (size of dataType)
      }
    } else if constexpr (encodingType == EncodingType::SparseBool) {
      return commonPrefixSize + 1; // IsSet flag (1 byte)
    } else if constexpr (encodingType == EncodingType::Dictionary) {
      return commonPrefixSize + sizeof(uint32_t); // Alphabet size (4 byte)
    } else if constexpr (encodingType == EncodingType::RLE) {
      return commonPrefixSize + sizeof(uint32_t); // Run Lengths size (4 byte)
    } else if constexpr (encodingType == EncodingType::Varint) {
      return commonPrefixSize + sizeof(dataType); // Baseline (size of dataType)
    } else if constexpr (encodingType == EncodingType::Nullable) {
      return commonPrefixSize + sizeof(uint32_t); // Data size (4 bytes)
    }
  }

  static inline uint64_t bitPackedBytes(
      const uint64_t minValue,
      const uint64_t maxValue,
      const uint64_t count) {
    if constexpr (FixedByteWidth) {
      // NOTE: We round up the bits required to the next byte boundary.
      // This is to match a (temporary) hack we added to FixedBitWidthEncoding,
      // to mitigare compression issue on non-rounded bit widths. See
      // velox/dwio/nimble/encodings/FixedBitWidthEncoding.h for full details.
      return velox::bits::nbytes(
          ((velox::bits::bitsRequired(maxValue - minValue) + 7) & ~7) * count);
    } else {
      return velox::bits::nbytes(
          velox::bits::bitsRequired(maxValue - minValue) * count);
    }
  }
};
} // namespace detail
} // namespace facebook::nimble::legacy
