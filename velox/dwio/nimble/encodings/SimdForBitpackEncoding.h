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
#include <array>
#include <cstring>
#include <span>
#include <type_traits>

#include "folly/Likely.h"
#include "velox/common/base/BitUtil.h"
#include "velox/dwio/common/Lemire/BitPacking/bitpackinghelpers.h"
#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/common/Types.h"
#include "velox/dwio/nimble/common/Varint.h"
#include "velox/dwio/nimble/encodings/common/Encoding.h"
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrefix.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrimitives.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelection.h"

// SIMD Frame-of-Reference bitpacking: value = baseline + residual.
// Packs residuals in groups of 32 via Lemire FastPFor (fastpack/fastunpack).
// Ported from AusList SIMD_FOR_BITPACKED (D97646777).

namespace facebook::nimble {

template <typename T>
class SimdForBitpackEncodingView;

/// SIMD Frame-of-Reference bitpacking for integer streams.
///
/// Wire format (after Encoding prefix):
///   [baseline: sizeof(T)] [bitWidth: 1B] [firstGroupRows: varint]
///   [packed groups]
///   Each group = bitWidth * 4 bytes (32 values). Omitted when bitWidth == 0.
///   firstGroupRows is only less than 32 for sliced streams whose first group
///   starts from a partial source group.
template <typename T>
class SimdForBitpackEncoding final
    : public TypedEncoding<T, typename TypeTraits<T>::physicalType> {
  static_assert(
      isIntegralType<typename TypeTraits<T>::physicalType>(),
      "SimdForBitpack only supports integral data types.");

 public:
  using cppDataType = T;
  using physicalType = typename TypeTraits<T>::physicalType;

  static constexpr uint32_t kGroupSize =
      facebook::velox::fastpforlib::BITPACKING_ALGORITHM_GROUP_SIZE;

  SimdForBitpackEncoding(
      velox::memory::MemoryPool& pool,
      std::string_view data,
      const std::function<void*(uint32_t)>& stringBufferFactory,
      const Encoding::Options& options = {});

  void reset() final;
  void skip(uint32_t rowCount) final;
  void materialize(uint32_t rowCount, void* buffer) final;

  template <typename DecoderVisitor>
  void readWithVisitor(DecoderVisitor& visitor, ReadWithVisitorParams& params);

  static std::string_view encode(
      EncodingSelection<physicalType>& selection,
      std::span<const physicalType> values,
      Buffer& buffer,
      const Encoding::Options& options = {});

  static std::string_view slice(
      std::string_view encoded,
      uint32_t offset,
      uint32_t length,
      Buffer& buffer,
      const Encoding::Options& options = {});

  std::string debugString(int offset) const final;

  /// Return the fixed-prefix encoded size estimate used by encoding selection.
  static uint64_t
  estimateSize(uint64_t rowCount, physicalType min, physicalType max) {
    const auto firstGroupRows = firstGroupRowCount(rowCount);
    return EncodingPrefix::kFixedPrefixSize + fixedHeaderSize() +
        varint::varintSize(firstGroupRows) +
        packedDataSize(numGroups(rowCount), bitWidth(min, max));
  }

 private:
  friend class SimdForBitpackEncodingView<T>;

  // Parsed stream metadata shared by normal decode and native slice. The
  // packed data pointer references the first byte of the group payload.
  struct Header {
    uint32_t rowCount{0};
    physicalType baseline{};
    uint8_t bitWidth{0};
    uint32_t firstGroupRows{0};
    uint32_t lastGroupRows{0};
    uint32_t numGroups{0};
    const char* packedData{nullptr};
  };

  // Describes where a row falls in a packed group. During normal reads this is
  // relative to the current encoding. During slice construction this is
  // relative to the source encoding group that feeds an output slice group.
  struct GroupInfo {
    // Packed group containing the row.
    uint32_t groupIndex{0};
    // Row offset inside that packed group.
    uint32_t rowOffset{0};
    // Number of rows represented by that packed group.
    uint32_t rowCount{0};
  };

  static constexpr uint32_t fixedHeaderSize() {
    return sizeof(physicalType) + sizeof(uint8_t);
  }

  static constexpr uint32_t firstGroupRowCount(uint64_t rowCount) {
    return static_cast<uint32_t>(std::min<uint64_t>(rowCount, kGroupSize));
  }

  static constexpr uint32_t numGroups(uint64_t rowCount) {
    return static_cast<uint32_t>(velox::bits::divRoundUp(rowCount, kGroupSize));
  }

  static constexpr uint32_t numGroups(const Header& header) {
    if (header.rowCount == 0) {
      return 0;
    }
    return static_cast<uint32_t>(
        1 +
        velox::bits::divRoundUp(
            header.rowCount - header.firstGroupRows, kGroupSize));
  }

  // Return the total packed data size in bytes.
  // FastPFor emits `bitWidth` uint32_t words per 32-value group.
  static constexpr uint64_t packedDataSize(
      uint32_t groupCount,
      uint8_t bitWidth) {
    return static_cast<uint64_t>(groupCount) * bitWidth * sizeof(uint32_t);
  }

  static uint8_t bitWidth(physicalType min, physicalType max) {
    const physicalType fullRange = static_cast<physicalType>(max - min);
    return fullRange == 0
        ? uint8_t{0}
        : static_cast<uint8_t>(velox::bits::bitsRequired(fullRange));
  }

  static void populateGroupCounts(Header& header) {
    header.numGroups = numGroups(header);
    header.lastGroupRows =
        header.rowCount - groupRowOffset(header, header.numGroups - 1);
  }

  static uint32_t groupRowOffset(const Header& header, uint32_t groupIndex) {
    if (groupIndex == 0) {
      return 0;
    }
    return header.firstGroupRows + (groupIndex - 1) * kGroupSize;
  }

  static uint32_t groupRowCount(const Header& header, uint32_t groupIndex) {
    if (groupIndex == 0) {
      return header.firstGroupRows;
    }
    if (groupIndex == header.numGroups - 1) {
      return header.lastGroupRows;
    }
    return kGroupSize;
  }

  static GroupInfo groupInfo(const Header& header, uint32_t row) {
    const auto groupIndex = row < header.firstGroupRows
        ? uint32_t{0}
        : 1 + (row - header.firstGroupRows) / kGroupSize;
    NIMBLE_CHECK_LT(groupIndex, header.numGroups);
    return {
        .groupIndex = groupIndex,
        .rowOffset = row - groupRowOffset(header, groupIndex),
        .rowCount = groupRowCount(header, groupIndex),
    };
  }

  static Header parseHeader(
      std::string_view encoded,
      const Encoding::Options& options);

  static void unpackGroup(
      const char* packedData,
      uint8_t bitWidth,
      uint32_t groupIndex,
      physicalType* output);

  static void
  packGroup(const physicalType* residuals, uint8_t bitWidth, uint32_t* output);

  static void writeSlicedGroupsPayload(
      const Header& sourceHeader,
      const Header& sliceHeader,
      uint32_t offset,
      uint64_t packedSize,
      char*& pos);

  // Unpack a single group at the given group index into `output`.
  // `output` must have room for kGroupSize elements.
  void unpackGroup(uint32_t groupIndex, physicalType* output) const;

  // Materialize a slice from one packed group into `output`.
  inline void materializePartialGroup(
      uint32_t groupIndex,
      uint32_t offsetInGroup,
      uint32_t count,
      physicalType* output) const;

  Header header_;
  uint32_t row_{0};

  // Avoid re-unpacking when consecutive rows hit the same group.
  mutable uint32_t cachedGroupIndex_{~0u};
  mutable std::array<physicalType, kGroupSize> cachedGroup_{};
};

//
// End of class declaration. Implementations follow.
//

template <typename T>
SimdForBitpackEncoding<T>::SimdForBitpackEncoding(
    velox::memory::MemoryPool& pool,
    std::string_view data,
    const std::function<void*(uint32_t)>& /* stringBufferFactory */,
    const Encoding::Options& options)
    : TypedEncoding<T, physicalType>{pool, data, options} {
  if constexpr (!isIntegralType<physicalType>()) {
    NIMBLE_INCOMPATIBLE_ENCODING(
        "SimdForBitpack encoding only supports integral data types.");
  }
  header_ = parseHeader(data, options);
  NIMBLE_CHECK_EQ(header_.rowCount, this->rowCount_);
  reset();
}

template <typename T>
void SimdForBitpackEncoding<T>::reset() {
  row_ = 0;
  cachedGroupIndex_ = ~0u;
}

template <typename T>
void SimdForBitpackEncoding<T>::skip(uint32_t rowCount) {
  row_ += rowCount;
}

template <typename T>
typename SimdForBitpackEncoding<T>::Header
SimdForBitpackEncoding<T>::parseHeader(
    std::string_view encoded,
    const Encoding::Options& options) {
  Header header;
  header.rowCount =
      EncodingPrefix::readRowCount(encoded, options.useVarintRowCount);
  NIMBLE_CHECK_GT(
      header.rowCount, 0, "SimdForBitpack stream must contain rows.");
  const char* pos = encoded.data() +
      EncodingPrefix::prefixSize(encoded, options.useVarintRowCount);
  header.baseline = encoding::read<physicalType>(pos);
  header.bitWidth = static_cast<uint8_t>(encoding::readChar(pos));
  header.firstGroupRows = varint::readVarint32(&pos);
  NIMBLE_CHECK_GT(
      header.firstGroupRows, 0, "First group row count must be set.");
  NIMBLE_CHECK_LE(
      header.firstGroupRows,
      kGroupSize,
      "Invalid SimdForBitpack first group row count.");
  populateGroupCounts(header);
  constexpr uint8_t kMaxBits = static_cast<uint8_t>(sizeof(physicalType) * 8);
  NIMBLE_CHECK_LE(
      header.bitWidth,
      kMaxBits,
      "SimdForBitpack bit width exceeds physical type size.");
  header.packedData = pos;
  return header;
}

template <typename T>
void SimdForBitpackEncoding<T>::unpackGroup(
    const char* packedData,
    uint8_t bitWidth,
    uint32_t groupIndex,
    physicalType* output) {
  static_assert(
      isOneByteIntegralType<physicalType>() ||
          isTwoByteIntegralType<physicalType>() ||
          isFourByteIntegralType<physicalType>() ||
          isEightByteIntegralType<physicalType>(),
      "Unexpected SimdForBitpack physical type width.");
  const uint32_t* groupStart = reinterpret_cast<const uint32_t*>(packedData) +
      static_cast<uint64_t>(groupIndex) * bitWidth;
  if constexpr (isEightByteIntegralType<physicalType>()) {
    facebook::velox::fastpforlib::fastunpack(
        groupStart, reinterpret_cast<uint64_t*>(output), bitWidth);
  } else if constexpr (isFourByteIntegralType<physicalType>()) {
    facebook::velox::fastpforlib::fastunpack(
        groupStart, reinterpret_cast<uint32_t*>(output), bitWidth);
  } else {
    std::array<uint32_t, kGroupSize> temp{};
    facebook::velox::fastpforlib::fastunpack(groupStart, temp.data(), bitWidth);
    for (uint32_t i = 0; i < kGroupSize; ++i) {
      output[i] = static_cast<physicalType>(temp[i]);
    }
  }
}

template <typename T>
void SimdForBitpackEncoding<T>::packGroup(
    const physicalType* residuals,
    uint8_t bitWidth,
    uint32_t* output) {
  if constexpr (isEightByteIntegralType<physicalType>()) {
    facebook::velox::fastpforlib::fastpack(
        reinterpret_cast<const uint64_t*>(residuals), output, bitWidth);
  } else if constexpr (isFourByteIntegralType<physicalType>()) {
    facebook::velox::fastpforlib::fastpack(
        reinterpret_cast<const uint32_t*>(residuals), output, bitWidth);
  } else {
    std::array<uint32_t, kGroupSize> widenedResiduals{};
    for (uint32_t i = 0; i < kGroupSize; ++i) {
      widenedResiduals[i] = static_cast<uint32_t>(residuals[i]);
    }
    facebook::velox::fastpforlib::fastpack(
        widenedResiduals.data(), output, bitWidth);
  }
}

template <typename T>
void SimdForBitpackEncoding<T>::writeSlicedGroupsPayload(
    const Header& sourceHeader,
    const Header& sliceHeader,
    uint32_t offset,
    uint64_t packedSize,
    char*& pos) {
  std::memset(pos, 0, packedSize);
  char* outputData = pos;
  std::array<physicalType, kGroupSize> sourceResiduals{};
  std::array<physicalType, kGroupSize> sliceResiduals{};

  const auto groupBytes =
      static_cast<uint64_t>(sourceHeader.bitWidth) * sizeof(uint32_t);
  const auto getSliceGroupSource = [&](uint32_t group) {
    return groupInfo(sourceHeader, offset + groupRowOffset(sliceHeader, group));
  };
  const auto isPartialGroup = [&](uint32_t group,
                                  const GroupInfo& sourceSlice) {
    return sourceSlice.rowOffset != 0 ||
        groupRowCount(sliceHeader, group) != sourceSlice.rowCount;
  };
  const auto copyPartialGroup = [&](uint32_t group,
                                    const GroupInfo& sourceSlice) {
    const uint32_t rowCount = groupRowCount(sliceHeader, group);
    unpackGroup(
        sourceHeader.packedData,
        sourceHeader.bitWidth,
        sourceSlice.groupIndex,
        sourceResiduals.data());
    std::fill(sliceResiduals.begin(), sliceResiduals.end(), physicalType{0});
    for (uint32_t i = 0; i < rowCount; ++i) {
      sliceResiduals[i] = sourceResiduals[sourceSlice.rowOffset + i];
    }
    packGroup(
        sliceResiduals.data(),
        sourceHeader.bitWidth,
        reinterpret_cast<uint32_t*>(outputData + group * groupBytes));
  };

  const auto firstSourceSlice = getSliceGroupSource(0);
  const bool firstPartialGroup = isPartialGroup(0, firstSourceSlice);
  if (firstPartialGroup) {
    copyPartialGroup(0, firstSourceSlice);
  }

  const uint32_t copyBegin = firstPartialGroup ? 1 : 0;
  const uint32_t lastGroup = sliceHeader.numGroups - 1;
  const auto lastSourceSlice = getSliceGroupSource(lastGroup);
  const bool lastPartialGroup =
      lastGroup != 0 && isPartialGroup(lastGroup, lastSourceSlice);
  const uint32_t copyEnd = lastPartialGroup ? lastGroup : sliceHeader.numGroups;
  if (copyBegin < copyEnd) {
    const auto beginSourceSlice = getSliceGroupSource(copyBegin);
    NIMBLE_CHECK_EQ(beginSourceSlice.rowOffset, 0);
    NIMBLE_CHECK_EQ(
        groupRowCount(sliceHeader, copyBegin), beginSourceSlice.rowCount);
    std::memcpy(
        outputData + static_cast<uint64_t>(copyBegin) * groupBytes,
        sourceHeader.packedData +
            static_cast<uint64_t>(beginSourceSlice.groupIndex) * groupBytes,
        static_cast<uint64_t>(copyEnd - copyBegin) * groupBytes);
  }

  if (lastPartialGroup) {
    copyPartialGroup(lastGroup, lastSourceSlice);
  }
  pos += packedSize;
}

template <typename T>
void SimdForBitpackEncoding<T>::unpackGroup(
    uint32_t groupIndex,
    physicalType* output) const {
  unpackGroup(header_.packedData, header_.bitWidth, groupIndex, output);
}

template <typename T>
inline void SimdForBitpackEncoding<T>::materializePartialGroup(
    uint32_t groupIndex,
    uint32_t offsetInGroup,
    uint32_t count,
    physicalType* output) const {
  std::array<physicalType, kGroupSize> temp{};
  unpackGroup(groupIndex, temp.data());
  for (uint32_t i = 0; i < count; ++i) {
    output[i] =
        static_cast<physicalType>(temp[offsetInGroup + i] + header_.baseline);
  }
}

template <typename T>
void SimdForBitpackEncoding<T>::materialize(uint32_t rowCount, void* buffer) {
  if (FOLLY_UNLIKELY(rowCount == 0)) {
    return;
  }
  auto* output = static_cast<physicalType*>(buffer);

  if (header_.bitWidth == 0) {
    // All values match the baseline, so there are no residual bits to unpack.
    std::fill(output, output + rowCount, header_.baseline);
    row_ += rowCount;
    return;
  }

  uint32_t remaining = rowCount;
  uint32_t outputIndex = 0;

  // Partial first group.
  const auto firstGroup = groupInfo(header_, row_);
  if (firstGroup.rowOffset != 0) {
    const uint32_t available = firstGroup.rowCount - firstGroup.rowOffset;
    const uint32_t toCopy = std::min(available, remaining);
    materializePartialGroup(
        firstGroup.groupIndex,
        firstGroup.rowOffset,
        toCopy,
        output + outputIndex);
    outputIndex += toCopy;
    remaining -= toCopy;
  }

  // Full groups can unpack directly into the caller's output. Short groups
  // need the bounded path because FastPFor always writes 32 values.
  while (remaining > 0) {
    const auto group = groupInfo(header_, row_ + outputIndex);
    const uint32_t toCopy = std::min(remaining, group.rowCount);
    if (toCopy == kGroupSize) {
      unpackGroup(group.groupIndex, output + outputIndex);
      for (uint32_t i = 0; i < kGroupSize; ++i) {
        output[outputIndex + i] += header_.baseline;
      }
    } else {
      materializePartialGroup(
          group.groupIndex, 0, toCopy, output + outputIndex);
    }
    outputIndex += toCopy;
    remaining -= toCopy;
  }

  row_ += rowCount;
}

template <typename T>
template <typename V>
void SimdForBitpackEncoding<T>::readWithVisitor(
    V& visitor,
    ReadWithVisitorParams& params) {
  detail::readWithVisitorSlow(
      visitor,
      params,
      [&](auto toSkip) { skip(toSkip); },
      [&]() {
        physicalType value;
        if (header_.bitWidth == 0) {
          value = header_.baseline;
        } else {
          const auto group = groupInfo(header_, row_);
          if (group.groupIndex != cachedGroupIndex_) {
            unpackGroup(group.groupIndex, cachedGroup_.data());
            cachedGroupIndex_ = group.groupIndex;
          }
          value = static_cast<physicalType>(
              cachedGroup_[group.rowOffset] + header_.baseline);
        }
        ++row_;
        return value;
      });
}

template <typename T>
std::string_view SimdForBitpackEncoding<T>::encode(
    EncodingSelection<typename TypeTraits<T>::physicalType>& selection,
    std::span<const typename TypeTraits<T>::physicalType> values,
    Buffer& buffer,
    const Encoding::Options& options) {
  if constexpr (!isIntegralType<physicalType>()) {
    NIMBLE_INCOMPATIBLE_ENCODING(
        "SimdForBitpack encoding only supports integral data types.");
  }

  static_assert(
      std::is_same_v<
          typename std::make_unsigned<physicalType>::type,
          physicalType>,
      "SimdForBitpack physical type must be unsigned.");
  static_assert(
      isOneByteIntegralType<physicalType>() ||
          isTwoByteIntegralType<physicalType>() ||
          isFourByteIntegralType<physicalType>() ||
          isEightByteIntegralType<physicalType>(),
      "Unexpected SimdForBitpack physical type width.");
  const bool useVarint = options.useVarintRowCount;
  NIMBLE_CHECK(
      !values.empty(), "SimdForBitpack encoding cannot be used with 0 rows.");
  Header header{
      .rowCount = static_cast<uint32_t>(values.size()),
      .baseline = selection.statistics().min(),
      .bitWidth =
          bitWidth(selection.statistics().min(), selection.statistics().max()),
      .firstGroupRows =
          firstGroupRowCount(static_cast<uint32_t>(values.size())),
  };
  populateGroupCounts(header);
  const uint64_t packedSize = packedDataSize(header.numGroups, header.bitWidth);

  const uint32_t encodingSize =
      Encoding::serializePrefixSize(header.rowCount, useVarint) +
      fixedHeaderSize() + varint::varintSize(header.firstGroupRows) +
      static_cast<uint32_t>(packedSize);

  char* reserved = buffer.reserve(encodingSize);
  char* pos = reserved;
  Encoding::serializePrefix(
      EncodingType::SimdForBitpack,
      TypeTraits<T>::dataType,
      header.rowCount,
      useVarint,
      pos);
  encoding::write(header.baseline, pos);
  encoding::writeChar(static_cast<char>(header.bitWidth), pos);
  varint::writeVarint(header.firstGroupRows, &pos);

  if (header.bitWidth > 0) {
    auto* packedOut = reinterpret_cast<uint32_t*>(pos);
    std::array<physicalType, kGroupSize> residuals{};

    // FastPFor packs one group of 32 residuals at a time:
    //
    //   32 values * bitWidth bits/value = bitWidth uint32_t words
    //
    // The last group is zero-padded before packing when rowCount is not a
    // multiple of 32.
    for (uint32_t group = 0; group < header.numGroups; ++group) {
      const uint32_t groupStart = groupRowOffset(header, group);
      const uint32_t groupLen = groupRowCount(header, group);

      // Only zero-fill the last partial group; full groups overwrite all 32.
      if (groupLen < kGroupSize) {
        std::fill(residuals.begin(), residuals.end(), physicalType{0});
      }
      for (uint32_t i = 0; i < groupLen; ++i) {
        residuals[i] =
            static_cast<physicalType>(values[groupStart + i] - header.baseline);
      }

      packGroup(residuals.data(), header.bitWidth, packedOut);
      // `packedOut` is a uint32_t pointer. One packed group occupies
      // `bitWidth` uint32_t words.
      packedOut += header.bitWidth;
    }
    pos += packedSize;
  }

  NIMBLE_CHECK_EQ(
      pos - reserved, encodingSize, "SimdForBitpack encoding size mismatch.");
  return {reserved, encodingSize};
}

template <typename T>
std::string_view SimdForBitpackEncoding<T>::slice(
    std::string_view encoded,
    uint32_t offset,
    uint32_t length,
    Buffer& buffer,
    const Encoding::Options& options) {
  const auto sourceRowCount =
      EncodingPrefix::readRowCount(encoded, options.useVarintRowCount);
  NIMBLE_CHECK_GT(length, 0, "Cannot slice zero rows.");
  NIMBLE_CHECK_LE(offset, sourceRowCount);
  NIMBLE_CHECK_LE(length, sourceRowCount - offset);

  const auto sourceHeader = parseHeader(encoded, options);
  const auto firstGroup = groupInfo(sourceHeader, offset);
  Header sliceHeader;
  sliceHeader.rowCount = length;
  sliceHeader.baseline = sourceHeader.baseline;
  sliceHeader.bitWidth = sourceHeader.bitWidth;
  sliceHeader.firstGroupRows =
      std::min(length, firstGroup.rowCount - firstGroup.rowOffset);
  populateGroupCounts(sliceHeader);
  const uint64_t packedSize =
      packedDataSize(sliceHeader.numGroups, sliceHeader.bitWidth);

  const bool useVarint = options.useVarintRowCount;
  const uint32_t encodingSize =
      Encoding::serializePrefixSize(length, useVarint) + fixedHeaderSize() +
      varint::varintSize(sliceHeader.firstGroupRows) +
      static_cast<uint32_t>(packedSize);
  char* reserved = buffer.reserve(encodingSize);
  char* pos = reserved;
  Encoding::serializePrefix(
      EncodingType::SimdForBitpack,
      TypeTraits<T>::dataType,
      length,
      useVarint,
      pos);
  encoding::write(sliceHeader.baseline, pos);
  encoding::writeChar(static_cast<char>(sliceHeader.bitWidth), pos);
  varint::writeVarint(sliceHeader.firstGroupRows, &pos);

  if (sliceHeader.bitWidth > 0) {
    writeSlicedGroupsPayload(
        sourceHeader, sliceHeader, offset, packedSize, pos);
  }

  NIMBLE_CHECK_EQ(
      pos - reserved, encodingSize, "SimdForBitpack encoding size mismatch.");
  return {reserved, encodingSize};
}

template <typename T>
std::string SimdForBitpackEncoding<T>::debugString(int offset) const {
  return fmt::format(
      "{}{}<{}> rowCount={} bitWidth={}",
      std::string(offset, ' '),
      Encoding::encodingType(),
      Encoding::dataType(),
      Encoding::rowCount(),
      static_cast<int>(header_.bitWidth));
}

} // namespace facebook::nimble
