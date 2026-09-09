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
#include <cstdint>
#include <exception>
#include <limits>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include <folly/Conv.h>

#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/common/DataTypeDispatch.h"
#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/common/Vector.h"
#include "velox/dwio/nimble/encodings/EncodingSliceFactory.h"
#include "velox/dwio/nimble/encodings/common/Encoding.h"
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/common/EncodingLayout.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrefix.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrimitives.h"
#include "velox/dwio/nimble/encodings/selection/EncodingIdentifier.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelection.h"

namespace facebook::nimble {

namespace detail {

// Defines the serialized BitRangeSplit layout shared by decoders and tooling.
class BitRangeSplitEncodingBase {
 public:
  // EncodingLayout config key for semicolon-separated inclusive ranges.
  static constexpr std::string_view kRangesConfigKey = "bit-range-split.ranges";

  // Number of bytes in a serialized section descriptor.
  static constexpr size_t kSectionDescriptorBytes = 2 + sizeof(uint32_t);

  // Describes one contiguous bit range and its nested encoded payload.
  struct Section {
    // First bit in the range, inclusive.
    uint8_t bitStart{0};
    // Last bit in the range, inclusive.
    uint8_t bitEnd{0};
    // Byte width of the unsigned integer type used by the child encoding.
    uint8_t storageBytes{0};
    // Start of the child payload, or null for a range parsed from config.
    const char* data{nullptr};
    // Child payload size, or zero for a range parsed from config.
    uint32_t dataBytes{0};
  };

  // Describes all sections in one validated encoded chunk.
  struct Header {
    // Number of values encoded in every child stream.
    uint32_t rowCount{0};
    // Ordered, contiguous ranges covering the physical integer type.
    std::vector<Section> sections{};
  };

  // Returns the narrowest integer storage width containing a bit range.
  static constexpr uint8_t sectionStorageBytes(uint8_t bitWidth) {
    if (bitWidth <= 8) {
      return 1;
    }
    if (bitWidth <= 16) {
      return 2;
    }
    if (bitWidth <= 32) {
      return 4;
    }
    return 8;
  }

  // Returns the unsigned data type matching a section storage width.
  static DataType sectionDataType(uint8_t storageBytes) {
    switch (storageBytes) {
      case 1:
        return DataType::Uint8;
      case 2:
        return DataType::Uint16;
      case 4:
        return DataType::Uint32;
      case 8:
        return DataType::Uint64;
      default:
        NIMBLE_UNREACHABLE(
            "Invalid BitRangeSplit section storage width: {}.",
            static_cast<uint32_t>(storageBytes));
    }
  }

  // Returns a mask restricted to the section's serialized bit range.
  static constexpr uint64_t sectionMask(const Section& section) {
    const auto width =
        static_cast<uint8_t>(section.bitEnd - section.bitStart + 1);
    return width == 64 ? std::numeric_limits<uint64_t>::max()
                       : (uint64_t{1} << width) - 1;
  }

  // Returns whether an encoding participates in default child selection.
  // Explicitly replayed layouts are not restricted by this preference.
  static constexpr bool isValidSectionEncodingCandidate(
      EncodingType encodingType) {
    return encodingType != EncodingType::BitRangeSplit &&
        encodingType != EncodingType::RLE &&
        encodingType != EncodingType::MainlyConstant;
  }

  // Parses configured contiguous ranges or throws when they are invalid.
  static std::vector<Section> parseSections(
      std::string_view config,
      uint8_t physicalBits);

  // Serializes ranges for EncodingLayout replay.
  static std::string serializeSections(std::span<const Section> sections);

  // Parses and validates section metadata and payload boundaries.
  static Header parseHeader(
      std::string_view encoded,
      const Encoding::Options& options);

  // Visits each nested section and returns the validated outer header.
  template <typename Visitor>
  static Header visitSections(
      std::string_view encoded,
      const Encoding::Options& options,
      Visitor&& visitor) {
    auto header = parseHeader(encoded, options);
    for (NestedEncodingIdentifier sectionIndex{0};
         sectionIndex < header.sections.size();
         ++sectionIndex) {
      visitor(sectionIndex, header.sections[sectionIndex]);
    }
    return header;
  }

  // Captures child layouts and configuration required for replay.
  template <typename Visitor>
  static void captureLayout(
      std::string_view encoded,
      const Encoding::Options& options,
      Visitor&& visitor,
      EncodingLayout::Config& encodingConfig) {
    const auto header =
        visitSections(encoded, options, std::forward<Visitor>(visitor));
    encodingConfig = EncodingLayout::Config{{{
        std::string(kRangesConfigKey),
        serializeSections(header.sections),
    }}};
  }
};

} // namespace detail

/// Stores configured integer bit ranges as independently encoded child streams
/// inside one self-describing encoding chunk.
template <typename T>
class BitRangeSplitEncoding final
    : public TypedEncoding<T, typename TypeTraits<T>::physicalType> {
  static_assert(
      isIntegralType<T>() && (sizeof(T) == 4 || sizeof(T) == 8),
      "BitRangeSplit only supports 32- and 64-bit integers.");

 public:
  using physicalType = typename TypeTraits<T>::physicalType;

  /// EncodingLayout config key for semicolon-separated inclusive ranges.
  static constexpr std::string_view kRangesConfigKey =
      detail::BitRangeSplitEncodingBase::kRangesConfigKey;

  /// Constructs a decoder from a self-describing encoded chunk.
  BitRangeSplitEncoding(
      velox::memory::MemoryPool& pool,
      std::string_view data,
      const std::function<void*(uint32_t)>& stringBufferFactory,
      const Encoding::Options& options = {});

  void reset() final;
  void skip(uint32_t rowCount) final;
  void materialize(uint32_t rowCount, void* buffer) final;

  template <typename DecoderVisitor>
  void readWithVisitor(DecoderVisitor& visitor, ReadWithVisitorParams& params);

  /// Encodes values using ranges supplied by the per-stream encoding config.
  static std::string_view encode(
      EncodingSelection<physicalType>& selection,
      std::span<const physicalType> values,
      Buffer& buffer,
      const Encoding::Options& options = {});

  /// Encodes rows [offset, offset + length) by slicing each child stream.
  static std::string_view slice(
      std::string_view encoded,
      uint32_t offset,
      uint32_t length,
      Buffer& buffer,
      const Encoding::Options& options = {});

  std::string debugString(int offset) const final;

 private:
  using Base = detail::BitRangeSplitEncodingBase;
  using Section = Base::Section;
  using Header = Base::Header;

  // Encodes one range using the narrowest integer type that contains it.
  template <typename SectionType>
  static std::string_view encodeSection(
      EncodingSelection<physicalType>& selection,
      std::span<const physicalType> values,
      const Section& section,
      NestedEncodingIdentifier identifier,
      Buffer& buffer,
      const Encoding::Options& options);

  // Selects and encodes all range child streams.
  static std::vector<std::string_view> encodeSections(
      EncodingSelection<physicalType>& selection,
      std::span<const physicalType> values,
      std::span<const Section> sections,
      Buffer& buffer,
      const Encoding::Options& options);

  // Serializes the outer chunk around already encoded child streams.
  static std::string_view serializeSections(
      uint32_t rowCount,
      std::span<const Section> sections,
      std::span<const std::string_view> sectionData,
      Buffer& buffer,
      const Encoding::Options& options);

  // Materializes child streams and merges their disjoint ranges.
  void readValues(uint32_t length, physicalType* output);

  // Validated metadata and child-stream boundaries for this chunk.
  Header header_{};
  // Decoders ordered by the corresponding range in header_.sections.
  std::vector<std::unique_ptr<Encoding>> nestedEncodings_{};
  // Reused materialization storage sized for the widest child in a read.
  Vector<uint8_t> scratch_;
  // Next row consumed by the sequential interface.
  uint32_t row_{0};
};

template <typename T>
BitRangeSplitEncoding<T>::BitRangeSplitEncoding(
    velox::memory::MemoryPool& pool,
    std::string_view data,
    const std::function<void*(uint32_t)>& stringBufferFactory,
    const Encoding::Options& options)
    : TypedEncoding<T, physicalType>{pool, data, options},
      header_{Base::parseHeader(data, options)},
      scratch_{&pool} {
  NIMBLE_CHECK_EQ(this->encodingType_, EncodingType::BitRangeSplit);
  NIMBLE_CHECK_EQ(header_.rowCount, this->rowCount_);
  nestedEncodings_.reserve(header_.sections.size());
  for (const auto& section : header_.sections) {
    auto nested = EncodingFactory().create(
        pool, {section.data, section.dataBytes}, stringBufferFactory, options);
    NIMBLE_CHECK_EQ(nested->rowCount(), header_.rowCount);
    NIMBLE_CHECK_EQ(
        nested->dataType(), Base::sectionDataType(section.storageBytes));
    nestedEncodings_.push_back(std::move(nested));
  }
}

inline std::vector<detail::BitRangeSplitEncodingBase::Section>
detail::BitRangeSplitEncodingBase::parseSections(
    std::string_view config,
    uint8_t physicalBits) {
  NIMBLE_CHECK(!config.empty(), "BitRangeSplit ranges cannot be empty.");
  NIMBLE_CHECK(
      physicalBits == 32 || physicalBits == 64,
      "BitRangeSplit requires a 32- or 64-bit physical type: {}.",
      physicalBits);

  std::vector<Section> sections;
  sections.reserve(8);
  uint16_t expectedStart{0};
  size_t cursor{0};
  while (cursor < config.size()) {
    const auto dash = config.find('-', cursor);
    NIMBLE_CHECK_NE(
        dash,
        std::string_view::npos,
        "Invalid BitRangeSplit range config: '{}'.",
        config);
    const auto separator = config.find(';', dash + 1);
    const auto startText = config.substr(cursor, dash - cursor);
    const auto endText = config.substr(
        dash + 1,
        separator == std::string_view::npos ? config.size() - dash - 1
                                            : separator - dash - 1);

    uint32_t bitStart{0};
    uint32_t bitEnd{0};
    try {
      bitStart = folly::to<uint32_t>(startText);
      bitEnd = folly::to<uint32_t>(endText);
    } catch (const std::exception&) {
      NIMBLE_FAIL("Invalid BitRangeSplit range config: '{}'.", config);
    }
    NIMBLE_CHECK_EQ(
        bitStart,
        expectedStart,
        "BitRangeSplit ranges are not contiguous: '{}'.",
        config);
    NIMBLE_CHECK_GE(
        bitEnd, bitStart, "BitRangeSplit range is reversed: '{}'.", config);
    NIMBLE_CHECK_LT(
        bitEnd,
        physicalBits,
        "BitRangeSplit range exceeds the physical type: '{}'.",
        config);

    const auto width = static_cast<uint8_t>(bitEnd - bitStart + 1);
    sections.push_back({
        .bitStart = static_cast<uint8_t>(bitStart),
        .bitEnd = static_cast<uint8_t>(bitEnd),
        .storageBytes = sectionStorageBytes(width),
    });
    expectedStart = bitEnd + 1;
    if (separator == std::string_view::npos) {
      break;
    }
    cursor = separator + 1;
    NIMBLE_CHECK_LT(
        cursor,
        config.size(),
        "Invalid trailing separator in BitRangeSplit range config: '{}'.",
        config);
  }

  NIMBLE_CHECK_EQ(
      expectedStart,
      physicalBits,
      "BitRangeSplit ranges do not cover the physical type: '{}'.",
      config);
  return sections;
}

inline std::string detail::BitRangeSplitEncodingBase::serializeSections(
    std::span<const Section> sections) {
  std::string config;
  for (size_t i{0}; i < sections.size(); ++i) {
    if (i > 0) {
      config.push_back(';');
    }
    config += std::to_string(sections[i].bitStart);
    config.push_back('-');
    config += std::to_string(sections[i].bitEnd);
  }
  return config;
}

inline detail::BitRangeSplitEncodingBase::Header
detail::BitRangeSplitEncodingBase::parseHeader(
    std::string_view encoded,
    const Encoding::Options& options) {
  const auto prefixSize =
      EncodingPrefix::prefixSize(encoded, options.useVarintRowCount);
  NIMBLE_CHECK_FILE_GE(
      encoded.size(), prefixSize + 1, "Truncated BitRangeSplit header.");
  Header header{
      .rowCount =
          EncodingPrefix::readRowCount(encoded, options.useVarintRowCount),
  };
  NIMBLE_CHECK_FILE_GT(
      header.rowCount, 0, "BitRangeSplit stream must contain rows.");

  const auto dataType = EncodingPrefix::dataType(encoded);
  const bool is32Bit =
      dataType == DataType::Int32 || dataType == DataType::Uint32;
  const bool is64Bit =
      dataType == DataType::Int64 || dataType == DataType::Uint64;
  NIMBLE_CHECK_FILE(
      is32Bit || is64Bit,
      "BitRangeSplit requires a 32- or 64-bit integer type: {}.",
      dataType);
  const uint8_t physicalBits = is32Bit ? 32 : 64;

  const char* cursor = encoded.data() + prefixSize;
  const char* const encodedEnd = encoded.end();
  const auto numSections = encoding::read<uint8_t>(cursor);
  NIMBLE_CHECK_FILE_GT(numSections, 0, "BitRangeSplit must contain sections.");
  const auto descriptorBytes =
      static_cast<size_t>(numSections) * kSectionDescriptorBytes;
  NIMBLE_CHECK_FILE_GE(
      static_cast<size_t>(encodedEnd - cursor),
      descriptorBytes,
      "Truncated BitRangeSplit section metadata.");

  header.sections.reserve(numSections);
  uint16_t expectedStart{0};
  for (uint8_t sectionIndex{0}; sectionIndex < numSections; ++sectionIndex) {
    const auto bitStart = encoding::read<uint8_t>(cursor);
    const auto bitEnd = encoding::read<uint8_t>(cursor);
    NIMBLE_CHECK_FILE_EQ(
        bitStart, expectedStart, "BitRangeSplit ranges are not contiguous.");
    NIMBLE_CHECK_FILE_GE(
        bitEnd, bitStart, "Invalid BitRangeSplit range start.");
    NIMBLE_CHECK_FILE_LT(
        bitEnd, physicalBits, "Invalid BitRangeSplit range end.");
    const auto rangeWidth = static_cast<uint8_t>(bitEnd - bitStart + 1);
    header.sections.push_back({
        .bitStart = bitStart,
        .bitEnd = bitEnd,
        .storageBytes = sectionStorageBytes(rangeWidth),
        .dataBytes = encoding::readUint32(cursor),
    });
    expectedStart = bitEnd + 1;
  }
  NIMBLE_CHECK_FILE_EQ(
      expectedStart,
      physicalBits,
      "BitRangeSplit ranges do not cover every physical bit.");

  for (auto& section : header.sections) {
    NIMBLE_CHECK_FILE_LE(
        section.dataBytes,
        static_cast<uint64_t>(encodedEnd - cursor),
        "Truncated BitRangeSplit section payload.");
    section.data = cursor;
    cursor += section.dataBytes;
  }
  NIMBLE_CHECK_FILE_EQ(
      cursor, encodedEnd, "Unexpected BitRangeSplit trailing bytes.");
  return header;
}

template <typename T>
void BitRangeSplitEncoding<T>::readValues(
    uint32_t length,
    physicalType* output) {
  std::fill(output, output + length, physicalType{0});
  const auto mergeSection = [&]<typename SectionType>(size_t sectionIndex) {
    scratch_.resize(length * sizeof(SectionType));
    auto* sectionValues = reinterpret_cast<SectionType*>(scratch_.data());
    nestedEncodings_[sectionIndex]->materialize(length, sectionValues);
    const auto& section = header_.sections[sectionIndex];
    const auto mask = Base::sectionMask(section);
    for (uint32_t row{0}; row < length; ++row) {
      output[row] |= static_cast<physicalType>(
          (static_cast<uint64_t>(sectionValues[row]) & mask)
          << section.bitStart);
    }
  };
  for (size_t sectionIndex{0}; sectionIndex < header_.sections.size();
       ++sectionIndex) {
    const auto storageBytes = header_.sections[sectionIndex].storageBytes;
    [&] {
      NIMBLE_RETURN_BY_UNSIGNED_INTEGER_DATA_TYPE(
          Base::sectionDataType(storageBytes),
          SectionType,
          mergeSection.template operator()<SectionType>(sectionIndex));
    }();
  }
}

template <typename T>
template <typename SectionType>
std::string_view BitRangeSplitEncoding<T>::encodeSection(
    EncodingSelection<physicalType>& selection,
    std::span<const physicalType> values,
    const Section& section,
    NestedEncodingIdentifier identifier,
    Buffer& buffer,
    const Encoding::Options& options) {
  const auto rowCount = static_cast<uint32_t>(values.size());
  const auto mask = Base::sectionMask(section);
  ScopedVector<SectionType> sectionValues{
      rowCount, &buffer.getMemoryPool(), options.bufferPool};
  for (uint32_t row{0}; row < rowCount; ++row) {
    const uint64_t rawValue = values[row];
    sectionValues[row] =
        static_cast<SectionType>((rawValue >> section.bitStart) & mask);
  }
  Encoding::Options sectionOptions = options;
  sectionOptions.fixedBitWidthUseExactBits = true;
  return selection.template encodeNested<SectionType>(
      identifier,
      std::span<const SectionType>(sectionValues.data(), sectionValues.size()),
      buffer,
      sectionOptions);
}

template <typename T>
std::vector<std::string_view> BitRangeSplitEncoding<T>::encodeSections(
    EncodingSelection<physicalType>& selection,
    std::span<const physicalType> values,
    std::span<const Section> sections,
    Buffer& buffer,
    const Encoding::Options& options) {
  std::vector<std::string_view> sectionData;
  sectionData.reserve(sections.size());
  for (size_t sectionIndex{0}; sectionIndex < sections.size(); ++sectionIndex) {
    const auto& section = sections[sectionIndex];
    const auto encoded = [&]() -> std::string_view {
      NIMBLE_RETURN_BY_UNSIGNED_INTEGER_DATA_TYPE(
          Base::sectionDataType(section.storageBytes),
          SectionType,
          encodeSection<SectionType>(
              selection,
              values,
              section,
              static_cast<NestedEncodingIdentifier>(sectionIndex),
              buffer,
              options));
    }();
    NIMBLE_CHECK_LE(
        encoded.size(),
        std::numeric_limits<uint32_t>::max(),
        "BitRangeSplit child encoding exceeds the maximum chunk size.");
    sectionData.push_back(encoded);
  }
  return sectionData;
}

template <typename T>
std::string_view BitRangeSplitEncoding<T>::serializeSections(
    uint32_t rowCount,
    std::span<const Section> sections,
    std::span<const std::string_view> sectionData,
    Buffer& buffer,
    const Encoding::Options& options) {
  NIMBLE_CHECK_EQ(sections.size(), sectionData.size());
  uint64_t totalPayloadSize{0};
  for (const auto encoded : sectionData) {
    totalPayloadSize += encoded.size();
  }
  const auto prefixSize =
      Encoding::serializePrefixSize(rowCount, options.useVarintRowCount);
  const auto metadataSize =
      uint64_t{1} + sections.size() * Base::kSectionDescriptorBytes;
  const auto encodingSize = prefixSize + metadataSize + totalPayloadSize;
  NIMBLE_CHECK_LE(
      encodingSize,
      std::numeric_limits<uint32_t>::max(),
      "BitRangeSplit encoding exceeds the maximum chunk size.");

  char* const reserved = buffer.reserve(static_cast<uint32_t>(encodingSize));
  char* cursor = reserved;
  Encoding::serializePrefix(
      EncodingType::BitRangeSplit,
      TypeTraits<T>::dataType,
      rowCount,
      options.useVarintRowCount,
      cursor);
  encoding::write<uint8_t>(static_cast<uint8_t>(sections.size()), cursor);
  for (size_t sectionIndex{0}; sectionIndex < sections.size(); ++sectionIndex) {
    encoding::write<uint8_t>(sections[sectionIndex].bitStart, cursor);
    encoding::write<uint8_t>(sections[sectionIndex].bitEnd, cursor);
    encoding::writeUint32(
        static_cast<uint32_t>(sectionData[sectionIndex].size()), cursor);
  }
  for (const auto encoded : sectionData) {
    encoding::writeBytes(encoded, cursor);
  }
  NIMBLE_CHECK_EQ(
      cursor - reserved, encodingSize, "BitRangeSplit encoding size mismatch.");
  return {reserved, static_cast<size_t>(encodingSize)};
}

template <typename T>
std::string_view BitRangeSplitEncoding<T>::encode(
    EncodingSelection<physicalType>& selection,
    std::span<const physicalType> values,
    Buffer& buffer,
    const Encoding::Options& options) {
  NIMBLE_CHECK(
      !values.empty(), "BitRangeSplit encoding cannot be used with 0 rows.");
  const auto rangesConfig =
      selection.getConfig(std::string(Base::kRangesConfigKey));
  NIMBLE_CHECK(
      rangesConfig.has_value(),
      "BitRangeSplit requires the '{}' encoding config.",
      Base::kRangesConfigKey);
  const auto sections = Base::parseSections(
      *rangesConfig, static_cast<uint8_t>(sizeof(physicalType) * 8));

  ScopedEncodingBuffer scopedBuffer{
      &buffer.getMemoryPool(), options.encodingBufferPool};
  auto& sectionBuffer = scopedBuffer.get();
  const auto sectionData =
      encodeSections(selection, values, sections, sectionBuffer, options);
  const auto rowCount = static_cast<uint32_t>(values.size());
  return serializeSections(rowCount, sections, sectionData, buffer, options);
}

template <typename T>
std::string_view BitRangeSplitEncoding<T>::slice(
    std::string_view encoded,
    uint32_t offset,
    uint32_t length,
    Buffer& buffer,
    const Encoding::Options& options) {
  const auto header = Base::parseHeader(encoded, options);
  NIMBLE_CHECK_LE(offset, header.rowCount);
  NIMBLE_CHECK_LE(length, header.rowCount - offset);
  NIMBLE_CHECK_GT(length, 0, "Cannot slice zero rows.");
  if (offset == 0 && length == header.rowCount) {
    return buffer.writeString(encoded);
  }

  ScopedEncodingBuffer scopedBuffer{
      &buffer.getMemoryPool(), options.encodingBufferPool};
  auto& sectionBuffer = scopedBuffer.get();
  std::vector<std::string_view> sectionData;
  sectionData.reserve(header.sections.size());
  for (const auto& section : header.sections) {
    sectionData.push_back(
        EncodingSliceFactory::slice(
            {section.data, section.dataBytes},
            offset,
            length,
            sectionBuffer,
            options));
  }
  return serializeSections(
      length, header.sections, sectionData, buffer, options);
}

template <typename T>
void BitRangeSplitEncoding<T>::reset() {
  for (auto& nested : nestedEncodings_) {
    nested->reset();
  }
  row_ = 0;
}

template <typename T>
void BitRangeSplitEncoding<T>::skip(uint32_t rowCount) {
  NIMBLE_CHECK_LE(rowCount, this->rowCount_ - row_);
  for (auto& nested : nestedEncodings_) {
    nested->skip(rowCount);
  }
  row_ += rowCount;
}

template <typename T>
void BitRangeSplitEncoding<T>::materialize(uint32_t rowCount, void* buffer) {
  NIMBLE_CHECK_LE(rowCount, this->rowCount_ - row_);
  readValues(rowCount, static_cast<physicalType*>(buffer));
  row_ += rowCount;
}

template <typename T>
template <typename DecoderVisitor>
void BitRangeSplitEncoding<T>::readWithVisitor(
    DecoderVisitor& visitor,
    ReadWithVisitorParams& params) {
  detail::readWithVisitorSlow(
      visitor,
      params,
      [&](auto toSkip) { skip(toSkip); },
      [&]() {
        physicalType value{0};
        materialize(1, &value);
        return value;
      });
}

template <typename T>
std::string BitRangeSplitEncoding<T>::debugString(int offset) const {
  return fmt::format(
      "{}{}<{}> rowCount={} sections={}",
      std::string(offset, ' '),
      Encoding::encodingType(),
      Encoding::dataType(),
      Encoding::rowCount(),
      header_.sections.size());
}

} // namespace facebook::nimble
