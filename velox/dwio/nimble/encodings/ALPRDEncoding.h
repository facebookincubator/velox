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

/// Shares split metadata, validation and exception loading across ALPRD types.
class ALPRDEncodingBase {
 public:
  /// Maximum number of high-part prefixes represented by dictionary codes.
  static constexpr uint8_t kMaxDictionarySize = 8;
  /// Maximum number of high bits stored in a dictionary or exception entry.
  static constexpr uint8_t kMaxHighBitWidth = 16;

  /// Describes a split and the high-bit prefixes represented by dictionary
  /// codes.
  struct Parameters {
    /// Number of low bits stored in the right-parts child.
    uint8_t rightBitWidth{0};
    /// Number of valid entries in dictionary; never exceeds kMaxDictionarySize.
    uint8_t dictionarySize{0};
    /// Maps a code to its high-bit prefix; unused entries are zero.
    std::array<uint16_t, kMaxDictionarySize> dictionary{};
  };

  /// Holds validated split metadata and views of the serialized child payloads.
  struct Metadata {
    /// Defines the dictionary and split shared by every row in the payload.
    Parameters parameters;
    /// Distinguishes the 32-bit FLOAT and 64-bit DOUBLE representations.
    DataType dataType;
    /// Number of values in each main child stream.
    uint32_t rowCount;
    /// Number of values in each exception child stream.
    uint32_t exceptionCount;
    /// Stores codes, low parts, exception positions and exception high parts.
    std::array<std::string_view, 4> children;

    /// Omits the exception children when every high part is in the dictionary.
    uint8_t childrenCount() const {
      return exceptionCount == 0 ? 2 : 4;
    }

    /// Returns the width of the logical type validated by readMetadata().
    uint8_t valueBitWidth() const {
      return dataType == DataType::Float ? 32 : 64;
    }

    /// Returns the exclusive upper bound of a high part after width validation.
    uint32_t highLimit() const {
      return uint32_t{1} << (valueBitWidth() - parameters.rightBitWidth);
    }
  };

  /// Validates metadata and child prefixes without materializing child values.
  static Metadata readMetadata(
      std::string_view data,
      const Encoding::Options& options);

  /// Trains a bounded sample using the initial packed-bit cost approximation.
  template <typename PhysicalType>
  static Parameters selectParameters(std::span<const PhysicalType> values) {
    constexpr uint32_t kSampleSize = 1'024;
    const auto sampleSize = std::min<size_t>(values.size(), kSampleSize);
    Parameters best;
    uint64_t bestCost = std::numeric_limits<uint64_t>::max();
    for (uint8_t highBitWidth = 1; highBitWidth <= kMaxHighBitWidth;
         ++highBitWidth) {
      const uint8_t rightBitWidth = sizeof(PhysicalType) * 8 - highBitWidth;
      std::unordered_map<uint16_t, uint32_t> frequencies;
      for (uint32_t i = 0; i < sampleSize; ++i) {
        const auto sampleIndex = uint64_t{i} * values.size() / sampleSize;
        ++frequencies[values[sampleIndex] >> rightBitWidth];
      }
      std::vector<std::pair<uint16_t, uint32_t>> entries(
          frequencies.begin(), frequencies.end());
      std::sort(
          entries.begin(), entries.end(), [](const auto& lhs, const auto& rhs) {
            return lhs.second != rhs.second ? lhs.second > rhs.second
                                            : lhs.first < rhs.first;
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
      // Prefer the narrower right part when minimum estimated costs tie.
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

 protected:
  /// Creates a child decoder and rejects NULL wrappers within ALPRD streams.
  static std::unique_ptr<Encoding> createChild(
      velox::memory::MemoryPool& pool,
      std::string_view data,
      const Encoding::Options& options);

  /// Materializes both exception streams in full, then validates their values.
  /// Output vectors must be empty and use the caller's memory pool.
  static void loadExceptions(
      velox::memory::MemoryPool& pool,
      const Metadata& metadata,
      const Encoding::Options& options,
      Vector<uint32_t>& positions,
      Vector<uint16_t>& highParts);
};

/// Encodes floating-point bit patterns using a small high-part dictionary.
/// Stores codes and low parts as integer child encodings, with separate
/// uint32 positions and uint16 high parts for dictionary misses.
template <typename T>
class ALPRDEncoding final
    : public TypedEncoding<T, typename TypeTraits<T>::physicalType>,
      private ALPRDEncodingBase {
  static_assert(isFloatingPointType<T>());

 public:
  using cppDataType = T;
  using physicalType = typename TypeTraits<T>::physicalType;

  /// Validates metadata before the base constructor reads the encoding prefix.
  ALPRDEncoding(
      velox::memory::MemoryPool& pool,
      std::string_view data,
      std::function<void*(uint32_t)> /* stringBufferFactory */,
      const Encoding::Options& options = {})
      : ALPRDEncoding(pool, data, options, readMetadata(data, options)) {}

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
    // Each batch overwrites the used prefix before reading it. Avoid clearing
    // the whole scratch array when the generic visitor requests a single row.
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
        NIMBLE_DCHECK_GE(exceptionPositions_[exceptionIndex_], position_);
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
    // TODO: Add a bulk visitor path while preserving this generic fallback.
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
    const auto parameters = selectParameters(values);
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
    const auto metadata = readMetadata(encoded, options);
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
      Vector<uint32_t> exceptionPositions(pool);
      Vector<uint16_t> exceptionHighParts(pool);
      loadExceptions(
          *pool, metadata, options, exceptionPositions, exceptionHighParts);
      const auto* begin = exceptionPositions.data();
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
  // Accepts validated metadata so the base can safely read the raw prefix.
  ALPRDEncoding(
      velox::memory::MemoryPool& pool,
      std::string_view data,
      const Encoding::Options& options,
      const Metadata& metadata)
      : TypedEncoding<T, physicalType>(pool, data, options),
        metadata_(metadata),
        exceptionPositions_(&pool),
        exceptionHighParts_(&pool) {
    NIMBLE_CHECK_EQ(metadata_.dataType, TypeTraits<T>::dataType);
    codes_ = createChild(pool, metadata_.children[0], options);
    rightParts_ = createChild(pool, metadata_.children[1], options);
    loadExceptions(
        pool, metadata_, options, exceptionPositions_, exceptionHighParts_);
  }

  void checkRead(uint32_t count) const {
    NIMBLE_CHECK_LE(
        count, this->rowCount() - position_, "ALPRD read exceeds row count.");
  }

  static std::string_view serialize(
      const Parameters& parameters,
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
    NIMBLE_CHECK_EQ(static_cast<uint64_t>(pos - start), size);
    return {start, size};
  }

  // Keeps views into the caller-owned encoded payload.
  Metadata metadata_;
  // Advances in lockstep with rightParts_, one code per non-null value.
  std::unique_ptr<Encoding> codes_;
  // Stores the low bits to combine with dictionary or exception high parts.
  std::unique_ptr<Encoding> rightParts_;
  // Holds strictly increasing positions in the non-null value stream.
  Vector<uint32_t> exceptionPositions_;
  // Holds one replacement high part for each exception position.
  Vector<uint16_t> exceptionHighParts_;
  // Points to the next non-null value to decode.
  uint32_t position_{0};
  // Points to the first exception at or after position_.
  uint32_t exceptionIndex_{0};
};

} // namespace facebook::nimble
