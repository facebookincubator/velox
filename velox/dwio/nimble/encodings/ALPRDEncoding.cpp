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
#include "velox/dwio/nimble/encodings/ALPRDEncoding.h"

#include "velox/dwio/nimble/encodings/selection/NestedAlpSizeEstimation.h"

namespace facebook::nimble {

ALPRDEncodingBase::Metadata ALPRDEncodingBase::readMetadata(
    std::string_view data,
    const Encoding::Options& options) {
  const auto prefix = EncodingPrefix::consume(data, options.useVarintRowCount);
  NIMBLE_CHECK_EQ(EncodingPrefix::encodingType(prefix), EncodingType::ALPRD);
  Metadata metadata{};
  metadata.dataType = EncodingPrefix::dataType(prefix);
  NIMBLE_CHECK(
      metadata.dataType == DataType::Float ||
          metadata.dataType == DataType::Double,
      "ALPRD requires a floating-point type.");
  metadata.rowCount =
      EncodingPrefix::readRowCount(prefix, options.useVarintRowCount);
  NIMBLE_CHECK_GT(metadata.rowCount, 0, "Empty ALPRD encoding.");
  auto& parameters = metadata.parameters;
  parameters.rightBitWidth = encoding::readByte(data);
  parameters.dictionarySize = encoding::readByte(data);
  const auto valueBitWidth = metadata.valueBitWidth();
  NIMBLE_CHECK_GE(parameters.rightBitWidth, valueBitWidth - kMaxHighBitWidth);
  NIMBLE_CHECK_LT(parameters.rightBitWidth, valueBitWidth);
  NIMBLE_CHECK_GT(parameters.dictionarySize, 0);
  NIMBLE_CHECK_LE(parameters.dictionarySize, kMaxDictionarySize);
  metadata.exceptionCount = encoding::readVarint32(data);
  NIMBLE_CHECK_LE(metadata.exceptionCount, metadata.rowCount);

  const auto highLimit = metadata.highLimit();
  for (uint8_t i = 0; i < parameters.dictionarySize; ++i) {
    const auto high = encoding::readUint16(data);
    NIMBLE_CHECK_LT(high, highLimit, "Invalid ALPRD dictionary value.");
    for (uint8_t j = 0; j < i; ++j) {
      NIMBLE_CHECK_NE(high, parameters.dictionary[j], "Duplicate ALPRD key.");
    }
    parameters.dictionary[i] = high;
  }

  const std::array<DataType, 4> childTypes{
      DataType::Uint16,
      metadata.dataType == DataType::Float ? DataType::Uint32
                                           : DataType::Uint64,
      DataType::Uint32,
      DataType::Uint16,
  };
  for (uint8_t i = 0; i < metadata.childrenCount(); ++i) {
    const auto child = encoding::readLengthPrefixedBytes(data);
    auto cursor = child;
    const auto childPrefix =
        EncodingPrefix::consume(cursor, options.useVarintRowCount);
    NIMBLE_CHECK_EQ(
        EncodingPrefix::dataType(childPrefix),
        childTypes[i],
        "Invalid ALPRD child type.");
    NIMBLE_CHECK_EQ(
        EncodingPrefix::readRowCount(childPrefix, options.useVarintRowCount),
        i < 2 ? metadata.rowCount : metadata.exceptionCount,
        "Invalid ALPRD child row count.");
    metadata.children[i] = child;
  }
  NIMBLE_CHECK(data.empty(), "Unexpected bytes after ALPRD children.");
  return metadata;
}

std::unique_ptr<Encoding> ALPRDEncodingBase::createChild(
    velox::memory::MemoryPool& pool,
    std::string_view data,
    const Encoding::Options& options) {
  auto child = EncodingFactory(options).create(
      pool, data, [](uint32_t) -> void* { return nullptr; });
  NIMBLE_CHECK(!child->isNullable(), "ALPRD children must be non-nullable.");
  return child;
}

void ALPRDEncodingBase::loadExceptions(
    velox::memory::MemoryPool& pool,
    const Metadata& metadata,
    const Encoding::Options& options,
    Vector<uint32_t>& positions,
    Vector<uint16_t>& highParts) {
  NIMBLE_DCHECK(positions.empty());
  NIMBLE_DCHECK(highParts.empty());
  if (metadata.exceptionCount == 0) {
    return;
  }
  auto positionDecoder = createChild(pool, metadata.children[2], options);
  auto highPartDecoder = createChild(pool, metadata.children[3], options);
  positions.resize(metadata.exceptionCount);
  highParts.resize(metadata.exceptionCount);
  positionDecoder->materialize(metadata.exceptionCount, positions.data());
  highPartDecoder->materialize(metadata.exceptionCount, highParts.data());
  const auto highLimit = metadata.highLimit();
  for (uint32_t i = 0; i < metadata.exceptionCount; ++i) {
    NIMBLE_CHECK_LT(
        positions[i], metadata.rowCount, "Invalid ALPRD exception position.");
    if (i != 0) {
      NIMBLE_CHECK_GT(
          positions[i],
          positions[i - 1],
          "ALPRD exception positions must increase.");
    }
    NIMBLE_CHECK_LT(
        highParts[i], highLimit, "Invalid ALPRD exception high part.");
  }
}

namespace {

// Returns the parameters and heuristic size from the same bounded sample.
struct SplitSelection {
  // Defines the split and dictionary associated with the estimated size.
  ALPRDEncodingBase::Parameters parameters;
  // Includes ALPRD metadata and its four child estimates for all target rows.
  uint64_t size{std::numeric_limits<uint64_t>::max()};
};

// Adds the outer metadata and one length field for each present child.
uint64_t splitSize(
    uint8_t dictionarySize,
    uint32_t numRows,
    uint32_t numExceptions,
    const std::array<uint64_t, 4>& childSizes,
    const Encoding::Options& options) {
  uint64_t size =
      EncodingPrefix::serializedSize(numRows, options.useVarintRowCount) + 2 +
      2 * dictionarySize + varint::varintSize(numExceptions);
  for (auto childSize : childSizes) {
    if (childSize != 0) {
      size += varint::varintSize(childSize) + childSize;
    }
  }
  return size;
}

// Owns the bounded sample and frequency storage for ALPRD parameter selection.
template <typename PhysicalType>
class SplitSelector {
 public:
  // Samples the input while retaining the target row count for cost estimates.
  SplitSelector(
      std::span<const PhysicalType> values,
      uint32_t numRows,
      const Encoding::Options& options);

  // Scores splits using Constant, FixedBitWidth and Trivial child estimates.
  SplitSelection select();

 private:
  using Base = ALPRDEncodingBase;

  // Number of values represented by the sample in the target payload.
  const uint32_t numRows_;
  // Limits the amount of data inspected for every split width.
  const uint32_t numSamples_;
  // Applies the caller's byte-rounded or exact-bit packing option.
  const Encoding::Options& options_;
  // Retains the original sampled bits across split widths.
  std::array<PhysicalType, Base::kSampleSize> sample_{};
  // Sorts sampled prefixes for frequency counting.
  std::array<uint16_t, Base::kSampleSize> sortedHighParts_{};
  // Ranks distinct prefixes by frequency for dictionary construction.
  std::array<std::pair<uint16_t, uint32_t>, Base::kSampleSize> frequencies_{};
};

template <typename PhysicalType>
SplitSelector<PhysicalType>::SplitSelector(
    std::span<const PhysicalType> values,
    uint32_t numRows,
    const Encoding::Options& options)
    : numRows_{numRows},
      numSamples_{std::min<uint32_t>(values.size(), Base::kSampleSize)},
      options_{options} {
  NIMBLE_CHECK(
      !values.empty(), "ALPRD split selection requires non-empty input.");
  NIMBLE_CHECK_LE(values.size(), numRows_);
  for (uint32_t i = 0; i < numSamples_; ++i) {
    sample_[i] = values[detail::NestedAlpSizeEstimation::sampledRowIndex(
        i, numSamples_, values.size())];
  }
}

template <typename PhysicalType>
SplitSelection SplitSelector<PhysicalType>::select() {
  using Cost = detail::NestedAlpSizeEstimation;
  SplitSelection best;
  for (uint8_t highBitWidth = 1; highBitWidth <= Base::kMaxHighBitWidth;
       ++highBitWidth) {
    const uint8_t rightBitWidth = sizeof(PhysicalType) * 8 - highBitWidth;
    const auto mask = (PhysicalType{1} << rightBitWidth) - 1;
    PhysicalType minRight{std::numeric_limits<PhysicalType>::max()};
    PhysicalType maxRight{0};
    for (uint32_t i = 0; i < numSamples_; ++i) {
      const auto right = sample_[i] & mask;
      minRight = std::min(minRight, right);
      maxRight = std::max(maxRight, right);
      sortedHighParts_[i] = sample_[i] >> rightBitWidth;
    }
    const auto rightSize = Cost::estimateChildSize<PhysicalType>(
        numRows_, minRight, maxRight, options_);

    std::sort(sortedHighParts_.begin(), sortedHighParts_.begin() + numSamples_);
    uint32_t numPrefixes{0};
    for (uint32_t i = 0; i < numSamples_; ++i) {
      if (i == 0 || sortedHighParts_[i] != sortedHighParts_[i - 1]) {
        frequencies_[numPrefixes++] = {sortedHighParts_[i], 1};
      } else {
        ++frequencies_[numPrefixes - 1].second;
      }
    }
    const auto maxDictionarySize =
        std::min<uint32_t>(numPrefixes, Base::kMaxDictionarySize);
    std::partial_sort(
        frequencies_.begin(),
        frequencies_.begin() + maxDictionarySize,
        frequencies_.begin() + numPrefixes,
        [](const auto& lhs, const auto& rhs) {
          return lhs.second != rhs.second ? lhs.second > rhs.second
                                          : lhs.first < rhs.first;
        });

    Base::Parameters parameters{.rightBitWidth = rightBitWidth};
    uint32_t numSampleExceptions{numSamples_};
    for (uint8_t dictionarySize = 1; dictionarySize <= maxDictionarySize;
         ++dictionarySize) {
      parameters.dictionarySize = dictionarySize;
      parameters.dictionary[dictionarySize - 1] =
          frequencies_[dictionarySize - 1].first;
      numSampleExceptions -= frequencies_[dictionarySize - 1].second;
      const uint32_t numExceptions =
          (uint64_t{numSampleExceptions} * numRows_ + numSamples_ - 1) /
          numSamples_;
      uint64_t positionsSize{0};
      uint64_t highPartsSize{0};
      if (numExceptions != 0) {
        // Positions are distinct absolute row numbers. Even a single sampled
        // exception can represent many positions in the full stream.
        positionsSize = Cost::estimateChildSize<uint32_t>(
            numExceptions, 0, numExceptions == 1 ? 0 : numRows_ - 1, options_);
        uint16_t minHigh{std::numeric_limits<uint16_t>::max()};
        uint16_t maxHigh{0};
        for (uint32_t i = dictionarySize; i < numPrefixes; ++i) {
          minHigh = std::min(minHigh, frequencies_[i].first);
          maxHigh = std::max(maxHigh, frequencies_[i].first);
        }
        highPartsSize = Cost::estimateChildSize<uint16_t>(
            numExceptions, minHigh, maxHigh, options_);
      }
      const auto size = splitSize(
          dictionarySize,
          numRows_,
          numExceptions,
          {
              Cost::estimateChildSize<uint16_t>(
                  numRows_, 0, dictionarySize - 1, options_),
              rightSize,
              positionsSize,
              highPartsSize,
          },
          options_);
      // Prefer narrower low parts on ties. Dictionary sizes increase, so
      // equal costs at one width retain the smaller dictionary.
      if (size < best.size ||
          (size == best.size &&
           rightBitWidth < best.parameters.rightBitWidth)) {
        best = {parameters, size};
      }
    }
  }
  return best;
}

} // namespace

uint64_t ALPRDEncodingBase::estimateContainerSize(
    const Parameters& parameters,
    uint32_t numRows,
    uint32_t numExceptions,
    const std::array<uint64_t, 4>& childSizes,
    const Encoding::Options& options) {
  return splitSize(
      parameters.dictionarySize, numRows, numExceptions, childSizes, options);
}

template <typename PhysicalType>
ALPRDEncodingBase::Parameters ALPRDEncodingBase::selectParameters(
    std::span<const PhysicalType> values,
    const Encoding::Options& options) {
  NIMBLE_CHECK_LE(values.size(), std::numeric_limits<uint32_t>::max());
  return SplitSelector<PhysicalType>{
      values, static_cast<uint32_t>(values.size()), options}
      .select()
      .parameters;
}

template <typename PhysicalType>
ALPRDEncodingBase::Children<PhysicalType> ALPRDEncodingBase::decomposeChildren(
    std::span<const PhysicalType> values,
    const Encoding::Options& options) {
  NIMBLE_CHECK_LE(values.size(), std::numeric_limits<uint32_t>::max());
  NIMBLE_CHECK(!values.empty(), "Cannot decompose empty ALPRD input.");
  Children<PhysicalType> result;
  result.parameters = selectParameters(values, options);
  result.rowCount = static_cast<uint32_t>(values.size());
  const auto numSamples =
      std::min(static_cast<uint32_t>(values.size()), kSampleSize);
  NIMBLE_CHECK_GT(numSamples, 0);
  std::vector<uint32_t> sampledRows(numSamples);
  for (uint32_t i = 0; i < numSamples; ++i) {
    sampledRows[i] = detail::NestedAlpSizeEstimation::sampledRowIndex(
        i, numSamples, result.rowCount);
  }
  decompose(
      values,
      result.parameters,
      sampledRows,
      result.codes,
      result.rightParts,
      result.exceptionPositions,
      result.exceptionHighParts);
  result.exceptionCount = static_cast<uint32_t>(
      (uint64_t{result.exceptionPositions.size()} * result.rowCount +
       numSamples - 1) /
      numSamples);
  return result;
}

template <typename PhysicalType>
std::optional<uint64_t> ALPRDEncodingBase::estimateSize(
    std::span<const PhysicalType> sampleValues,
    uint32_t numRows,
    const Encoding::Options& options) {
  if (sampleValues.empty()) {
    return std::nullopt;
  }
  return SplitSelector<PhysicalType>{sampleValues, numRows, options}
      .select()
      .size;
}

template ALPRDEncodingBase::Parameters ALPRDEncodingBase::selectParameters<
    uint32_t>(std::span<const uint32_t>, const Encoding::Options&);
template ALPRDEncodingBase::Parameters ALPRDEncodingBase::selectParameters<
    uint64_t>(std::span<const uint64_t>, const Encoding::Options&);
template ALPRDEncodingBase::Children<uint32_t>
ALPRDEncodingBase::decomposeChildren<uint32_t>(
    std::span<const uint32_t>,
    const Encoding::Options&);
template ALPRDEncodingBase::Children<uint64_t>
ALPRDEncodingBase::decomposeChildren<uint64_t>(
    std::span<const uint64_t>,
    const Encoding::Options&);
template std::optional<uint64_t> ALPRDEncodingBase::estimateSize<uint32_t>(
    std::span<const uint32_t>,
    uint32_t,
    const Encoding::Options&);
template std::optional<uint64_t> ALPRDEncodingBase::estimateSize<uint64_t>(
    std::span<const uint64_t>,
    uint32_t,
    const Encoding::Options&);

} // namespace facebook::nimble
