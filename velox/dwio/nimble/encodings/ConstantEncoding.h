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
#include <optional>
#include <span>
#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/encodings/common/Encoding.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrefix.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrimitives.h"
#include "velox/dwio/nimble/encodings/common/EncodingType.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelection.h"

// Encodes data that is constant, i.e. there is a single unique value.

namespace facebook::nimble {

// Data layout is:
// EncodingPrefix::kFixedPrefixSize bytes: standard Encoding prefix
// X bytes: the constant value via encoding primitive.

template <typename T>
class ConstantEncodingBase
    : public TypedEncoding<T, typename TypeTraits<T>::physicalType> {
 public:
  using cppDataType = T;
  using physicalType = typename TypeTraits<T>::physicalType;

  ConstantEncodingBase(
      velox::memory::MemoryPool& pool,
      std::string_view data,
      const Encoding::Options& options = {})
      : TypedEncoding<T, physicalType>(pool, data, options) {}

  static std::optional<uint64_t> estimateSize(
      std::span<const physicalType> values,
      const Statistics<physicalType>& statistics,
      const Encoding::Options& options) {
    if (!isConstant(values, statistics, options)) {
      return std::nullopt;
    }
    const uint64_t outerEncodingSize =
        Encoding::serializePrefixSize(values.size(), options.useVarintRowCount);
    if constexpr (isStringType<physicalType>()) {
      const uint64_t valueSize = statistics.max().size() + sizeof(uint32_t);
      return outerEncodingSize + valueSize;
    } else {
      const uint64_t valueSize = sizeof(physicalType);
      return outerEncodingSize + valueSize;
    }
  }

  void reset() final {}

  void skip(uint32_t /* rowCount */) final {}

  void materialize(uint32_t rowCount, void* buffer) final {
    physicalType* castBuffer = static_cast<physicalType*>(buffer);
    for (uint32_t i = 0; i < rowCount; ++i) {
      castBuffer[i] = value_;
    }
  }

  void materializeBoolsAsBits(
      uint32_t /*rowCount*/,
      uint64_t* /*buffer*/,
      int /*begin*/) override {
    NIMBLE_UNREACHABLE("");
  }

  template <typename DecoderVisitor>
  void readWithVisitor(DecoderVisitor& visitor, ReadWithVisitorParams& params) {
    detail::readWithVisitorSlow(
        visitor, params, nullptr, [&] { return value_; });
  }

  bool dictionaryEnabled() const override {
    return true;
  }

  uint32_t dictionarySize() const override {
    return 1;
  }

  const void* dictionaryEntry(uint32_t index) const override {
    NIMBLE_DCHECK_EQ(index, 0);
    return &value_;
  }

  const void* dictionaryEntries() const override {
    return &value_;
  }

  void materializeIndices(uint32_t rowCount, uint32_t* buffer) override {
    std::fill(buffer, buffer + rowCount, 0);
  }

  /// Reads dictionary indices for a constant encoding. Every row maps to
  /// index 0 (the single dictionary entry).
  template <typename V>
  void readIndicesWithVisitor(V& visitor, ReadWithVisitorParams& params) {
    NIMBLE_CHECK(
        !V::kHasHook, "readIndicesWithVisitor does not support value hooks");
    const auto numReadRows =
        visitor.rowAt(visitor.numRows() - 1) - params.numScanned + 1;
    auto* rawNulls = visitor.reader().rawNullsInReadRange();
    const auto numNonNulls = rawNulls != nullptr
        ? velox::bits::countNonNulls(
              rawNulls, params.numScanned, params.numScanned + numReadRows)
        : numReadRows;

    if (V::dense) {
      NIMBLE_CHECK_EQ(
          visitor.rowAt(visitor.numRows() - 1),
          visitor.rowAt(0) + visitor.numRows() - 1,
          "Dense visitor must have contiguous rows");
      detail::readDenseMaterializedIndices(
          *this, visitor, params, rawNulls, numReadRows, numNonNulls);
      return;
    }

    // Sparse path is unlikely for constant encoding but supported for
    // correctness.
    auto indicesBuffer =
        velox::AlignedBuffer::allocate<uint32_t>(numNonNulls, this->pool_);
    auto* rawIndices = indicesBuffer->template asMutable<uint32_t>();
    detail::readSparseMaterializedIndices(
        *this,
        visitor,
        params.numScanned,
        params.prepareResultNulls,
        rawNulls,
        numReadRows,
        numNonNulls,
        rawIndices);
  }

  std::string debugString(int offset) const final {
    return fmt::format(
        "{}{}<{}> rowCount={} value={}",
        std::string(offset, ' '),
        toString(this->encodingType()),
        toString(this->dataType()),
        this->rowCount(),
        NIMBLE_AS_CONST(T, value_));
  }

  static std::string_view encode(
      EncodingSelection<physicalType>& selection,
      std::span<const physicalType> values,
      Buffer& buffer,
      const Encoding::Options& options = {}) {
    const bool useVarint = options.useVarintRowCount;
    if (values.empty()) {
      NIMBLE_INCOMPATIBLE_ENCODING("ConstantEncoding cannot be empty.");
    }

    if (!isConstant(values, selection.statistics(), options)) {
      NIMBLE_INCOMPATIBLE_ENCODING("ConstantEncoding requires constant data.");
    }

    const uint32_t rowCount = values.size();
    uint32_t encodingSize = Encoding::serializePrefixSize(rowCount, useVarint);
    if constexpr (isStringType<physicalType>()) {
      encodingSize += 4 + values[0].size();
    } else {
      encodingSize += sizeof(physicalType);
    }
    char* reserved = buffer.reserve(encodingSize);
    char* pos = reserved;
    Encoding::serializePrefix(
        EncodingType::Constant,
        TypeTraits<T>::dataType,
        rowCount,
        useVarint,
        pos);
    // Canonicalizing logically-equal floats is only valid under ALP (which
    // encodes floats by logical value); otherwise store the exact bits.
    encoding::write<physicalType>(
        options.allowNestedAlpSelection ? canonicalValue(values.front())
                                        : values.front(),
        pos);
    NIMBLE_CHECK_EQ(pos - reserved, encodingSize, "Encoding size mismatch.");
    return {reserved, encodingSize};
  }

  static std::string_view slice(
      std::string_view encoded,
      uint32_t offset,
      uint32_t length,
      Buffer& buffer,
      const Encoding::Options& options = {}) {
    const auto sourceRowCount =
        EncodingPrefix::readRowCount(encoded, options.useVarintRowCount);
    NIMBLE_CHECK_LE(offset, sourceRowCount);
    NIMBLE_CHECK_LE(length, sourceRowCount - offset);
    NIMBLE_CHECK_GT(length, 0, "Cannot slice zero rows.");

    const auto sourcePrefixSize =
        EncodingPrefix::prefixSize(encoded, options.useVarintRowCount);
    const std::string_view valueData = encoded.substr(sourcePrefixSize);
    const auto prefixSize =
        EncodingPrefix::serializedSize(length, options.useVarintRowCount);
    const auto encodingSize = prefixSize + valueData.size();
    char* reserved = buffer.reserve(encodingSize);
    char* pos = reserved;
    EncodingPrefix::serialize(
        EncodingType::Constant,
        TypeTraits<T>::dataType,
        length,
        options.useVarintRowCount,
        pos);
    encoding::writeBytes(valueData, pos);
    NIMBLE_CHECK_EQ(pos - reserved, encodingSize, "Encoding size mismatch.");
    return {reserved, encodingSize};
  }

 protected:
  static bool isConstant(
      std::span<const physicalType> values,
      const Statistics<physicalType>& statistics,
      const Encoding::Options& options) {
    NIMBLE_CHECK(
        !values.empty(), "ConstantEncoding requires non-empty values.");
    if (values.size() == 1) {
      return true;
    }

    // For integral types a unique count of one is equivalent to min == max.
    // Both are lazily populated, but populateMinMax() is a comparison scan
    // while populateUniques() inserts one hash entry per value. Encoding
    // selection evaluates ConstantEncoding on every stream, so going through
    // uniqueCounts() here builds a full unique-value map on high-cardinality
    // streams purely to discover that the count is not one.
    //
    // Only integral types qualify. Statistics has no min/max for booleans, and
    // their unique counts are bounded at two entries anyway. String min/max are
    // by length rather than lexicographic, so equal endpoints do not imply
    // equal values.
    if constexpr (isIntegralType<T>()) {
      return statistics.min() == statistics.max();
    }

    if (statistics.uniqueCounts().value().size() == 1) {
      return true;
    }

    if constexpr (!isFloatingPointType<T>()) {
      return false;
    }

    // Logical-equality constancy collapses physically-distinct but logically
    // equal floats (only -0.0/+0.0; NaN is excluded since NaN != NaN) to a
    // single canonical value. That is only sound when ALP is enabled, since ALP
    // encodes floats by logical value. Without ALP, encodings must preserve the
    // exact bits, so require physical (bit-exact) constancy.
    if (!options.allowNestedAlpSelection) {
      return false;
    }
    const auto logicalValue =
        EncodingPhysicalType<T>::asEncodingLogicalType(values.front());
    return std::all_of(values.begin(), values.end(), [&](physicalType value) {
      return EncodingPhysicalType<T>::asEncodingLogicalType(value) ==
          logicalValue;
    });
  }

  static physicalType canonicalValue(physicalType value) {
    if constexpr (!isFloatingPointType<T>()) {
      return value;
    }
    // Floating-point constant selection uses logical equality; store the
    // matching canonical physical value.
    const auto logical = EncodingPhysicalType<T>::asEncodingLogicalType(value);
    return EncodingPhysicalType<T>::asEncodingPhysicalType(logical);
  }

  physicalType value_;
};

template <typename T>
class ConstantEncoding : public ConstantEncodingBase<T> {
 public:
  using cppDataType = T;
  using physicalType = typename TypeTraits<T>::physicalType;

  ConstantEncoding(
      velox::memory::MemoryPool& pool,
      std::string_view data,
      std::function<void*(uint32_t)> stringBufferFactory,
      const Encoding::Options& options = {});
};

//
// End of public API. Implementation follows.
//

template <typename T>
ConstantEncoding<T>::ConstantEncoding(
    velox::memory::MemoryPool& pool,
    std::string_view data,
    std::function<void*(uint32_t)> /* stringBufferFactory */,
    const Encoding::Options& options)
    : ConstantEncodingBase<T>(pool, data, options) {
  const char* pos = data.data() + this->dataOffset();
  this->value_ = encoding::read<physicalType>(pos);
  NIMBLE_CHECK_EQ(pos, data.end(), "Unexpected constant encoding end");
}

// Specialization for bool to override materializeBoolsAsBits
template <>
class ConstantEncoding<bool> final : public ConstantEncodingBase<bool> {
 public:
  using cppDataType = bool;
  using physicalType = bool;

  ConstantEncoding(
      velox::memory::MemoryPool& pool,
      std::string_view data,
      std::function<void*(uint32_t)> /* stringBufferFactory */,
      const Encoding::Options& options = {})
      : ConstantEncodingBase<bool>(pool, data, options) {
    const char* pos = data.data() + this->dataOffset();
    this->value_ = encoding::read<physicalType>(pos);
    NIMBLE_CHECK_EQ(pos, data.end(), "Unexpected constant encoding end");
  }

  void materializeBoolsAsBits(uint32_t rowCount, uint64_t* buffer, int begin)
      final {
    velox::bits::fillBits(buffer, begin, begin + rowCount, this->value_);
  }
};

template <>
class ConstantEncoding<std::string_view> final
    : public ConstantEncodingBase<std::string_view> {
 public:
  using cppDataType = std::string_view;
  using physicalType = std::string_view;

  ConstantEncoding(
      velox::memory::MemoryPool& pool,
      std::string_view data,
      std::function<void*(uint32_t)> stringBufferFactory,
      const Encoding::Options& options = {});
};
} // namespace facebook::nimble
