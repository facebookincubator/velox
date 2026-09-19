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
#include <bit>
#include <limits>
#include <span>
#include <unordered_map>
#include <vector>

#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/common/Varint.h"
#include "velox/dwio/nimble/common/Vector.h"
#include "velox/dwio/nimble/encodings/common/Encoding.h"
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrefix.h"
#include "velox/dwio/nimble/encodings/selection/EncodingIdentifier.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelection.h"

namespace facebook::nimble {

namespace detail::alprd {

inline constexpr uint8_t kMaxDictionarySize = 8;
inline constexpr uint8_t kMaxHighBitWidth = 16;

struct Parameters {
  uint8_t rightBitWidth{0};
  uint8_t dictionarySize{0};
  std::array<uint16_t, kMaxDictionarySize> dictionary{};
};

// Bound all metadata reads before passing child payloads to their decoders.
class MetadataReader {
 public:
  explicit MetadataReader(std::string_view data) : remaining_(data) {}

  uint8_t readByte() {
    NIMBLE_CHECK(!remaining_.empty(), "Truncated ALPRD metadata.");
    const auto value = static_cast<uint8_t>(remaining_.front());
    remaining_.remove_prefix(1);
    return value;
  }

  uint32_t readInteger(uint8_t bytes) {
    uint32_t value = 0;
    for (uint8_t i = 0; i < bytes; ++i) {
      value |= static_cast<uint32_t>(readByte()) << (8 * i);
    }
    return value;
  }

  uint32_t readVarint() {
    uint32_t value = 0;
    for (uint8_t i = 0; i < 5; ++i) {
      const auto byte = readByte();
      NIMBLE_CHECK(i != 4 || byte < 16, "Invalid ALPRD varint.");
      value |= static_cast<uint32_t>(byte & 0x7f) << (7 * i);
      if ((byte & 0x80) == 0) {
        return value;
      }
    }
    NIMBLE_UNREACHABLE("Invalid ALPRD varint.");
  }

  uint32_t readRowCount(bool useVarint) {
    return useVarint ? readVarint() : readInteger(4);
  }

  std::string_view readChild() {
    const auto size = readVarint();
    NIMBLE_CHECK_LE(size, remaining_.size(), "Truncated ALPRD child.");
    const auto child = remaining_.substr(0, size);
    remaining_.remove_prefix(size);
    return child;
  }

  bool atEnd() const {
    return remaining_.empty();
  }

 private:
  std::string_view remaining_;
};

inline std::string_view checkedPrefix(
    std::string_view data,
    const Encoding::Options& options) {
  MetadataReader reader(data);
  NIMBLE_CHECK_EQ(reader.readByte(), static_cast<uint8_t>(EncodingType::ALPRD));
  const auto type = static_cast<DataType>(reader.readByte());
  NIMBLE_CHECK(
      type == DataType::Float || type == DataType::Double,
      "ALPRD requires a floating-point type.");
  reader.readRowCount(options.useVarintRowCount);
  return data;
}

struct Metadata {
  Parameters parameters;
  DataType dataType;
  uint32_t rowCount;
  uint32_t exceptionCount;
  std::array<std::string_view, 4> children;

  uint8_t childrenCount() const {
    return exceptionCount == 0 ? 2 : 4;
  }
};

inline Metadata readMetadata(
    std::string_view data,
    const Encoding::Options& options) {
  MetadataReader reader(checkedPrefix(data, options));
  reader.readByte();
  Metadata metadata{};
  metadata.dataType = static_cast<DataType>(reader.readByte());
  metadata.rowCount = reader.readRowCount(options.useVarintRowCount);
  NIMBLE_CHECK_GT(metadata.rowCount, 0, "Empty ALPRD encoding.");
  auto& parameters = metadata.parameters;
  parameters.rightBitWidth = reader.readByte();
  parameters.dictionarySize = reader.readByte();
  const uint8_t valueBitWidth = metadata.dataType == DataType::Float ? 32 : 64;
  NIMBLE_CHECK_GE(parameters.rightBitWidth, valueBitWidth - kMaxHighBitWidth);
  NIMBLE_CHECK_LT(parameters.rightBitWidth, valueBitWidth);
  NIMBLE_CHECK_GT(parameters.dictionarySize, 0);
  NIMBLE_CHECK_LE(parameters.dictionarySize, kMaxDictionarySize);
  metadata.exceptionCount = reader.readVarint();
  NIMBLE_CHECK_LE(metadata.exceptionCount, metadata.rowCount);

  const auto highLimit = uint32_t{1}
      << (valueBitWidth - parameters.rightBitWidth);
  for (uint8_t i = 0; i < parameters.dictionarySize; ++i) {
    const auto high = reader.readInteger(2);
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
      DataType::Uint16};
  for (uint8_t i = 0; i < metadata.childrenCount(); ++i) {
    const auto child = reader.readChild();
    MetadataReader childReader(child);
    childReader.readByte();
    NIMBLE_CHECK_EQ(
        static_cast<DataType>(childReader.readByte()),
        childTypes[i],
        "Invalid ALPRD child type.");
    NIMBLE_CHECK_EQ(
        childReader.readRowCount(options.useVarintRowCount),
        i < 2 ? metadata.rowCount : metadata.exceptionCount,
        "Invalid ALPRD child row count.");
    metadata.children[i] = child;
  }
  NIMBLE_CHECK(reader.atEnd(), "Unexpected bytes after ALPRD children.");
  return metadata;
}

template <typename PhysicalType>
Parameters selectParameters(std::span<const PhysicalType> values) {
  constexpr uint32_t kSampleSize = 1024;
  const auto sampleSize = std::min<size_t>(values.size(), kSampleSize);
  Parameters best;
  uint64_t bestCost = std::numeric_limits<uint64_t>::max();
  for (uint8_t highBitWidth = 1; highBitWidth <= kMaxHighBitWidth;
       ++highBitWidth) {
    const uint8_t rightBitWidth = sizeof(PhysicalType) * 8 - highBitWidth;
    std::unordered_map<uint16_t, uint32_t> frequencies;
    for (uint32_t i = 0; i < sampleSize; ++i) {
      const auto index = uint64_t{i} * values.size() / sampleSize;
      ++frequencies[values[index] >> rightBitWidth];
    }
    std::vector<std::pair<uint16_t, uint32_t>> entries(
        frequencies.begin(), frequencies.end());
    std::sort(entries.begin(), entries.end(), [](const auto& a, const auto& b) {
      return a.second != b.second ? a.second > b.second : a.first < b.first;
    });
    const auto dictionarySize =
        std::min<size_t>(entries.size(), kMaxDictionarySize);
    uint32_t exceptions = sampleSize;
    for (uint8_t i = 0; i < dictionarySize; ++i) {
      exceptions -= entries[i].second;
    }
    // Estimate the split using packed integers. Exceptions carry a 32-bit
    // position and a 16-bit high part; child codecs choose their own layout.
    const auto codeBitWidth = std::bit_width(dictionarySize - 1);
    const uint64_t cost = sampleSize * (rightBitWidth + codeBitWidth) +
        uint64_t{exceptions} * 48 + dictionarySize * 16;
    if (cost <= bestCost) {
      bestCost = cost;
      best.rightBitWidth = rightBitWidth;
      best.dictionarySize = dictionarySize;
      for (uint8_t i = 0; i < dictionarySize; ++i) {
        best.dictionary[i] = entries[i].first;
      }
    }
  }
  return best;
}

} // namespace detail::alprd

/// Encodes floating-point bit patterns using a small high-part dictionary.
/// Stores codes and low parts as integer child encodings, with separate
/// uint32 positions and uint16 high parts for dictionary misses.
template <typename T>
class ALPRDEncoding final
    : public TypedEncoding<T, typename TypeTraits<T>::physicalType> {
  static_assert(isFloatingPointType<T>());

 public:
  using cppDataType = T;
  using physicalType = typename TypeTraits<T>::physicalType;

  ALPRDEncoding(
      velox::memory::MemoryPool& pool,
      std::string_view data,
      std::function<void*(uint32_t)> /* stringBufferFactory */,
      const Encoding::Options& options = {})
      : TypedEncoding<T, physicalType>(
            pool,
            detail::alprd::checkedPrefix(data, options),
            options),
        metadata_(detail::alprd::readMetadata(data, options)),
        exceptionPositions_(&pool),
        exceptionHighParts_(&pool) {
    NIMBLE_CHECK_EQ(metadata_.dataType, TypeTraits<T>::dataType);
    const EncodingFactory factory(options);
    auto createChild = [&](uint8_t index) {
      auto child = factory.create(
          pool, metadata_.children[index], [](uint32_t) -> void* {
            return nullptr;
          });
      NIMBLE_CHECK(
          !child->isNullable(), "ALPRD children must be non-nullable.");
      return child;
    };
    codes_ = createChild(0);
    rightParts_ = createChild(1);
    if (metadata_.exceptionCount != 0) {
      exceptionPositions_.resize(metadata_.exceptionCount);
      exceptionHighParts_.resize(metadata_.exceptionCount);
      createChild(2)->materialize(
          metadata_.exceptionCount, exceptionPositions_.data());
      createChild(3)->materialize(
          metadata_.exceptionCount, exceptionHighParts_.data());
      const auto highLimit = uint32_t{1}
          << (sizeof(T) * 8 - metadata_.parameters.rightBitWidth);
      for (uint32_t i = 0; i < metadata_.exceptionCount; ++i) {
        NIMBLE_CHECK_LT(
            exceptionPositions_[i],
            this->rowCount(),
            "Invalid ALPRD exception position.");
        if (i != 0) {
          NIMBLE_CHECK_GT(
              exceptionPositions_[i],
              exceptionPositions_[i - 1],
              "ALPRD exception positions must increase.");
        }
        NIMBLE_CHECK_LT(
            exceptionHighParts_[i],
            highLimit,
            "Invalid ALPRD exception high part.");
      }
    }
  }

  void reset() final {
    codes_->reset();
    rightParts_->reset();
    position_ = 0;
    exceptionIndex_ = 0;
  }

  void skip(uint32_t rowCount) final {
    checkRead(rowCount);
    if (rowCount == 0) {
      return;
    }
    codes_->skip(rowCount);
    rightParts_->skip(rowCount);
    position_ += rowCount;
    if (metadata_.exceptionCount != 0) {
      exceptionIndex_ =
          std::lower_bound(
              exceptionPositions_.data() + exceptionIndex_,
              exceptionPositions_.data() + metadata_.exceptionCount,
              position_) -
          exceptionPositions_.data();
    }
  }

  void materialize(uint32_t rowCount, void* buffer) final {
    checkRead(rowCount);
    auto* output = static_cast<physicalType*>(buffer);
    std::array<uint16_t, 256> codes;
    const auto& parameters = metadata_.parameters;
    const auto mask = (physicalType{1} << parameters.rightBitWidth) - 1;
    while (rowCount != 0) {
      const auto count = std::min<uint32_t>(rowCount, codes.size());
      codes_->materialize(count, codes.data());
      rightParts_->materialize(count, output);
      for (uint32_t i = 0; i < count; ++i) {
        NIMBLE_CHECK_LT(
            codes[i],
            parameters.dictionarySize,
            "Invalid ALPRD dictionary code.");
        NIMBLE_CHECK_LE(output[i], mask, "Invalid ALPRD low part.");
        output[i] |= static_cast<physicalType>(parameters.dictionary[codes[i]])
            << parameters.rightBitWidth;
      }
      while (exceptionIndex_ < metadata_.exceptionCount &&
             exceptionPositions_[exceptionIndex_] < position_ + count) {
        const auto index = exceptionPositions_[exceptionIndex_] - position_;
        output[index] =
            (static_cast<physicalType>(exceptionHighParts_[exceptionIndex_])
             << parameters.rightBitWidth) |
            (output[index] & mask);
        ++exceptionIndex_;
      }
      output += count;
      rowCount -= count;
      position_ += count;
    }
  }

  void materializeBoolsAsBits(uint32_t, uint64_t*, int) final {
    NIMBLE_UNREACHABLE("ALPRD does not support bool values.");
  }

  template <typename Visitor>
  void readWithVisitor(Visitor& visitor, ReadWithVisitorParams& params) {
    auto skipValues = [&](auto count) { skip(count); };
    auto decodeOne = [&] {
      physicalType value;
      materialize(1, &value);
      return value;
    };
    detail::readWithVisitorSlow(visitor, params, skipValues, decodeOne);
  }

  std::string debugString(int offset) const final {
    return fmt::format(
        "{}{}<{}> rowCount={} rightBitWidth={} dictionarySize={} exceptions={}",
        std::string(offset, ' '),
        this->encodingType(),
        this->dataType(),
        this->rowCount(),
        metadata_.parameters.rightBitWidth,
        metadata_.parameters.dictionarySize,
        metadata_.exceptionCount);
  }

  static std::string_view encode(
      EncodingSelection<physicalType>& selection,
      std::span<const physicalType> values,
      Buffer& buffer,
      const Encoding::Options& options = {}) {
    NIMBLE_CHECK_LE(values.size(), std::numeric_limits<uint32_t>::max());
    if (values.empty()) {
      NIMBLE_INCOMPATIBLE_ENCODING("ALPRD cannot encode empty data.");
    }
    const auto parameters = detail::alprd::selectParameters(values);
    const uint32_t rowCount = values.size();
    auto* pool = &buffer.getMemoryPool();
    ScopedVector<uint16_t> codes(rowCount, pool, options.bufferPool);
    ScopedVector<physicalType> rightParts(rowCount, pool, options.bufferPool);
    ScopedVector<uint32_t> exceptionPositions(0, pool, options.bufferPool);
    ScopedVector<uint16_t> exceptionHighParts(0, pool, options.bufferPool);
    const auto mask = (physicalType{1} << parameters.rightBitWidth) - 1;
    for (uint32_t i = 0; i < rowCount; ++i) {
      rightParts[i] = values[i] & mask;
      const auto high =
          static_cast<uint16_t>(values[i] >> parameters.rightBitWidth);
      uint16_t code = 0;
      while (code < parameters.dictionarySize &&
             parameters.dictionary[code] != high) {
        ++code;
      }
      if (code == parameters.dictionarySize) {
        exceptionPositions.push_back(i);
        exceptionHighParts.push_back(high);
        code = 0;
      }
      codes[i] = code;
    }
    ScopedEncodingBuffer scratch(pool, options.encodingBufferPool);
    std::array<std::string_view, 4> children;
    children[0] = selection.template encodeNested<uint16_t>(
        EncodingIdentifiers::ALPRD::Codes,
        {codes.data(), codes.size()},
        scratch.get(),
        options);
    children[1] = selection.template encodeNested<physicalType>(
        EncodingIdentifiers::ALPRD::RightParts,
        {rightParts.data(), rightParts.size()},
        scratch.get(),
        options);
    if (!exceptionPositions.empty()) {
      children[2] = selection.template encodeNested<uint32_t>(
          EncodingIdentifiers::ALPRD::ExceptionPositions,
          {exceptionPositions.data(), exceptionPositions.size()},
          scratch.get(),
          options);
      children[3] = selection.template encodeNested<uint16_t>(
          EncodingIdentifiers::ALPRD::ExceptionHighParts,
          {exceptionHighParts.data(), exceptionHighParts.size()},
          scratch.get(),
          options);
    }
    return serialize(
        parameters,
        rowCount,
        exceptionPositions.size(),
        children,
        buffer,
        options);
  }

  static std::string_view slice(
      std::string_view encoded,
      uint32_t offset,
      uint32_t length,
      Buffer& buffer,
      const Encoding::Options& options = {}) {
    const auto metadata = detail::alprd::readMetadata(encoded, options);
    NIMBLE_CHECK_EQ(metadata.dataType, TypeTraits<T>::dataType);
    NIMBLE_CHECK_LE(offset, metadata.rowCount);
    NIMBLE_CHECK_LE(length, metadata.rowCount - offset);
    NIMBLE_CHECK_GT(length, 0, "Cannot slice zero rows.");
    auto* pool = &buffer.getMemoryPool();
    ScopedEncodingBuffer scratch(pool, options.encodingBufferPool);
    std::array<std::string_view, 4> children;
    children[0] = EncodingFactory::slice(
        metadata.children[0], offset, length, scratch.get(), options);
    children[1] = EncodingFactory::slice(
        metadata.children[1], offset, length, scratch.get(), options);
    uint32_t exceptionCount = 0;
    if (metadata.exceptionCount != 0) {
      // Validate exception ordering and bounds before using lower_bound.
      ALPRDEncoding source(*pool, encoded, nullptr, options);
      const auto* begin = source.exceptionPositions_.data();
      const auto* end = begin + metadata.exceptionCount;
      const auto* first = std::lower_bound(begin, end, offset);
      const auto* last = std::lower_bound(first, end, offset + length);
      exceptionCount = last - first;
      if (exceptionCount != 0) {
        ScopedVector<uint32_t> positions(
            exceptionCount, pool, options.bufferPool);
        for (uint32_t i = 0; i < exceptionCount; ++i) {
          positions[i] = first[i] - offset;
        }
        children[2] = EncodingFactory::encodeWithCapturedLayout<uint32_t>(
            metadata.children[2],
            {positions.data(), positions.size()},
            scratch.get(),
            options);
        children[3] = EncodingFactory::slice(
            metadata.children[3],
            first - begin,
            exceptionCount,
            scratch.get(),
            options);
      }
    }
    return serialize(
        metadata.parameters, length, exceptionCount, children, buffer, options);
  }

 private:
  void checkRead(uint32_t count) const {
    NIMBLE_CHECK_LE(
        count, this->rowCount() - position_, "ALPRD read exceeds row count.");
  }

  static std::string_view serialize(
      const detail::alprd::Parameters& parameters,
      uint32_t rowCount,
      uint32_t exceptionCount,
      const std::array<std::string_view, 4>& children,
      Buffer& buffer,
      const Encoding::Options& options) {
    const auto childrenCount = exceptionCount == 0 ? 2 : 4;
    uint64_t size =
        EncodingPrefix::serializedSize(rowCount, options.useVarintRowCount) +
        2 + varint::varintSize(exceptionCount) + 2 * parameters.dictionarySize;
    for (int i = 0; i < childrenCount; ++i) {
      NIMBLE_CHECK_LE(children[i].size(), std::numeric_limits<uint32_t>::max());
      size += varint::varintSize(children[i].size()) + children[i].size();
    }
    NIMBLE_CHECK_LE(
        size,
        std::numeric_limits<uint32_t>::max(),
        "ALPRD encoding is too large.");
    char* start = buffer.reserve(size);
    auto* pos = start;
    EncodingPrefix::serialize(
        EncodingType::ALPRD,
        TypeTraits<T>::dataType,
        rowCount,
        options.useVarintRowCount,
        pos);
    *pos++ = parameters.rightBitWidth;
    *pos++ = parameters.dictionarySize;
    varint::writeVarint(exceptionCount, &pos);
    for (uint8_t i = 0; i < parameters.dictionarySize; ++i) {
      *pos++ = static_cast<char>(parameters.dictionary[i] & 0xff);
      *pos++ = static_cast<char>(parameters.dictionary[i] >> 8);
    }
    for (int i = 0; i < childrenCount; ++i) {
      varint::writeVarint(children[i].size(), &pos);
      encoding::writeBytes(children[i], pos);
    }
    NIMBLE_CHECK_EQ(pos - start, size);
    return {start, size};
  }

  detail::alprd::Metadata metadata_;
  std::unique_ptr<Encoding> codes_;
  std::unique_ptr<Encoding> rightParts_;
  Vector<uint32_t> exceptionPositions_;
  Vector<uint16_t> exceptionHighParts_;
  uint32_t position_{0};
  uint32_t exceptionIndex_{0};
};

} // namespace facebook::nimble
