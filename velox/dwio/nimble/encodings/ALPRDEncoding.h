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
#include <utility>
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

  /// Bounded child samples and projected counts used to score an ALPRD tree.
  template <typename PhysicalType>
  struct Children {
    /// Defines the dictionary and split used to produce the child samples.
    Parameters parameters;
    /// Number of rows represented by the sampled main child streams.
    uint32_t rowCount{0};
    /// Projected number of rows in each full exception child stream.
    uint32_t exceptionCount{0};
    /// Sampled dictionary codes for the main child stream.
    std::vector<uint16_t> codes;
    /// Sampled low-bit parts for the main child stream.
    std::vector<PhysicalType> rightParts;
    /// Sampled absolute row positions for exception values.
    std::vector<uint32_t> exceptionPositions;
    /// Sampled high-bit parts for exception values.
    std::vector<uint16_t> exceptionHighParts;
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

  /// Maximum number of input values inspected by split encoding selection.
  static constexpr uint32_t kSampleSize = 1'024;

  /// Selects a split and dictionary using bounded sampling and scalar child
  /// estimates. Each child writer applies its own encoding selection policy.
  template <typename PhysicalType>
  static Parameters selectParameters(
      std::span<const PhysicalType> values,
      const Encoding::Options& options);

  /// Splits a bounded representative sample using the same parameters and
  /// decomposition as encode(). Projected counts let selection policies score
  /// the full child streams with their normal encoding models.
  template <typename PhysicalType>
  static Children<PhysicalType> decomposeChildren(
      std::span<const PhysicalType> values,
      const Encoding::Options& options);

  /// Adds ALPRD metadata and length prefixes to child serialized sizes.
  static uint64_t estimateContainerSize(
      const Parameters& parameters,
      uint32_t numRows,
      uint32_t numExceptions,
      const std::array<uint64_t, 4>& childSizes,
      const Encoding::Options& options);

  /// Estimates numRows values from a representative sample, which may contain
  /// the full input. Uses the same split selection as encode(), adding ALPRD
  /// metadata once to the heuristic child sizes.
  template <typename PhysicalType>
  static std::optional<uint64_t> estimateSize(
      std::span<const PhysicalType> sampleValues,
      uint32_t numRows,
      const Encoding::Options& options);

 protected:
  // Splits either every input row (when sampledRows is empty) or the specified
  // representative rows into the four ALPRD child streams.
  template <
      typename PhysicalType,
      typename Codes,
      typename RightParts,
      typename ExceptionPositions,
      typename ExceptionHighParts>
  static void decompose(
      std::span<const PhysicalType> values,
      const Parameters& parameters,
      std::span<const uint32_t> sampledRows,
      Codes& codes,
      RightParts& rightParts,
      ExceptionPositions& exceptionPositions,
      ExceptionHighParts& exceptionHighParts) {
    NIMBLE_CHECK_LE(values.size(), std::numeric_limits<uint32_t>::max());
    NIMBLE_CHECK_LE(sampledRows.size(), std::numeric_limits<uint32_t>::max());
    const uint32_t rowCount = sampledRows.empty()
        ? static_cast<uint32_t>(values.size())
        : static_cast<uint32_t>(sampledRows.size());
    codes.resize(rowCount);
    rightParts.resize(rowCount);
    exceptionPositions.resize(0);
    exceptionHighParts.resize(0);
    const auto mask = (PhysicalType{1} << parameters.rightBitWidth) - 1;
    for (uint32_t i = 0; i < rowCount; ++i) {
      const auto row = sampledRows.empty() ? i : sampledRows[i];
      NIMBLE_CHECK_LT(row, values.size());
      const auto value = values[row];
      rightParts[i] = value & mask;
      const auto high =
          static_cast<uint16_t>(value >> parameters.rightBitWidth);
      uint16_t code = 0;
      while (code < parameters.dictionarySize &&
             parameters.dictionary[code] != high) {
        ++code;
      }
      if (code == parameters.dictionarySize) {
        exceptionPositions.push_back(row);
        exceptionHighParts.push_back(high);
        code = 0;
      }
      codes[i] = code;
    }
  }

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

  /// Estimates the full input with the bounded scalar child-cost model.
  static std::optional<uint64_t> estimateSize(
      std::span<const physicalType> values,
      const Encoding::Options& options) {
    NIMBLE_CHECK_LE(values.size(), std::numeric_limits<uint32_t>::max());
    return ALPRDEncodingBase::estimateSize(values, values.size(), options);
  }

  /// Scores ALPRD using the policy-selected encoding for each child.
  /// The root factor applies only to container overhead.
  template <typename NestedChildScorer>
  static std::optional<EncodingCandidateScore> estimateSelectionScore(
      std::span<const physicalType> values,
      const Encoding::Options& options,
      const double containerCostFactor,
      NestedChildScorer&& scoreNestedChild) {
    if (values.empty()) {
      return std::nullopt;
    }
    const auto children = decomposeChildren<physicalType>(values, options);
    const auto codes = scoreNestedChild(
        EncodingType::ALPRD,
        EncodingIdentifiers::ALPRD::Codes,
        std::span<const uint16_t>{children.codes},
        children.rowCount,
        options);
    const auto rightParts = scoreNestedChild(
        EncodingType::ALPRD,
        EncodingIdentifiers::ALPRD::RightParts,
        std::span<const physicalType>{children.rightParts},
        children.rowCount,
        options);
    if (!codes.has_value() || !rightParts.has_value()) {
      return std::nullopt;
    }

    EncodingCandidateScore exceptionPositions{0, 0};
    EncodingCandidateScore exceptionHighParts{0, 0};
    if (!children.exceptionPositions.empty()) {
      std::array<uint32_t, 2> projectedExceptionPositions{};
      std::span<const uint32_t> exceptionPositionValues =
          children.exceptionPositions;
      if (children.exceptionPositions.size() < children.exceptionCount) {
        // Projected exception positions may span the full input.
        projectedExceptionPositions = {0, children.rowCount - 1};
        exceptionPositionValues = projectedExceptionPositions;
      }
      const auto selectedExceptionPositions = scoreNestedChild(
          EncodingType::ALPRD,
          EncodingIdentifiers::ALPRD::ExceptionPositions,
          exceptionPositionValues,
          children.exceptionCount,
          options);
      // High parts may repeat, so retain their sampled distribution.
      const auto selectedExceptionHighParts = scoreNestedChild(
          EncodingType::ALPRD,
          EncodingIdentifiers::ALPRD::ExceptionHighParts,
          std::span<const uint16_t>{children.exceptionHighParts},
          children.exceptionCount,
          options);
      if (!selectedExceptionPositions.has_value() ||
          !selectedExceptionHighParts.has_value()) {
        return std::nullopt;
      }
      exceptionPositions = selectedExceptionPositions.value();
      exceptionHighParts = selectedExceptionHighParts.value();
    }

    const std::array<uint64_t, 4> childSizes{
        codes->estimatedSize,
        rightParts->estimatedSize,
        exceptionPositions.estimatedSize,
        exceptionHighParts.estimatedSize,
    };
    const auto estimatedSize = estimateContainerSize(
        children.parameters,
        children.rowCount,
        children.exceptionCount,
        childSizes,
        options);
    const std::array<EncodingCandidateScore, 4> childScores{
        codes.value(),
        rightParts.value(),
        exceptionPositions,
        exceptionHighParts,
    };
    uint64_t childSize{0};
    double childCost{0};
    for (const auto& child : childScores) {
      childSize += child.estimatedSize;
      childCost += child.cost;
    }
    NIMBLE_CHECK_LE(childSize, estimatedSize);
    return EncodingCandidateScore{
        .estimatedSize = estimatedSize,
        .cost = static_cast<double>(estimatedSize - childSize) *
                containerCostFactor +
            childCost};
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
    const auto parameters = selectParameters(values, options);
    const uint32_t rowCount = values.size();
    auto* pool = &buffer.getMemoryPool();
    ScopedVector<uint16_t> codes(rowCount, pool, options.bufferPool);
    ScopedVector<physicalType> rightParts(rowCount, pool, options.bufferPool);
    ScopedVector<uint32_t> exceptionPositions(0, pool, options.bufferPool);
    ScopedVector<uint16_t> exceptionHighParts(0, pool, options.bufferPool);
    decompose(
        values,
        parameters,
        {},
        codes,
        rightParts,
        exceptionPositions,
        exceptionHighParts);
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
      ScopedVector<uint32_t> exceptionPositions{0, pool, options.bufferPool};
      ScopedVector<uint16_t> exceptionHighParts{0, pool, options.bufferPool};
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
