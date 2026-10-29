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
#include <limits>
#include <optional>
#include <span>
#include "velox/common/base/SimdUtil.h"
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
    if (!isConstant(values, statistics)) {
      return std::nullopt;
    }
    return estimateSize(values.size(), statistics, options);
  }

  /// Estimates Constant size for a target row count using the supplied
  /// statistics, without requiring the target values.
  static std::optional<uint64_t> estimateSize(
      size_t rowCount,
      const Statistics<physicalType>& statistics,
      const Encoding::Options& options) {
    if (!statistics.isConstant()) {
      return std::nullopt;
    }
    NIMBLE_CHECK_LE(rowCount, std::numeric_limits<uint32_t>::max());
    const uint64_t outerEncodingSize = Encoding::serializePrefixSize(
        static_cast<uint32_t>(rowCount), options.useVarintRowCount);
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
    auto* nulls = visitor.reader().rawNullsInReadRange();
    if (params.scanFastPath &&
        velox::dwio::common::useFastPath(visitor, nulls)) {
      detail::readWithVisitorFast(*this, visitor, params, nulls);
      return;
    }
    detail::readWithVisitorSlow(
        visitor, params, nullptr, [&] { return value_; });
  }

  /// Reads the selected rows of a chunk in which every row holds 'value_'. The
  /// filter is evaluated once and its verdict applies to every selected row, so
  /// a failing chunk is skipped without touching the rows and a passing chunk
  /// is filled in bulk. Invoked through 'detail::readWithVisitorFast'.
  template <bool kScatter, typename V>
  void bulkScan(
      V& visitor,
      vector_size_t /*currentRow*/,
      const vector_size_t* selectedRows,
      vector_size_t numSelected,
      const vector_size_t* scatterRows) {
    using DataType = typename V::DataType;
    using ValueType = detail::ValueType<DataType>;
    constexpr bool kScatterValues = kScatter && !V::kHasFilter && !V::kHasHook;
    const auto data = detail::castFromPhysicalType<DataType>(value_);
    bool pass{true};
    if constexpr (V::kHasFilter) {
      pass = velox::common::applyFilter(visitor.filter(), data);
    }
    const auto numRows = visitor.numRows() - visitor.rowIndex();
    ValueType* values{nullptr};
    vector_size_t numHits{0};
    vector_size_t numValues{0};
    if (pass) {
      if constexpr (V::kHasFilter) {
        const auto* rows = kScatter ? scatterRows : selectedRows;
        std::copy(rows, rows + numSelected, visitor.outputRows(numSelected));
        numHits = numSelected;
      }
      if constexpr (!V::kFilterOnly) {
        values = detail::mutableValues<ValueType>(visitor, numRows);
        // Scattered values also fill the slots of null rows, which the null
        // bitmap masks.
        numValues = kScatterValues ? numRows : numSelected;
        velox::simd::simdFill(
            values,
            detail::dataToValue(visitor, data),
            static_cast<uint32_t>(numValues));
      }
    }
    visitor.setRowIndex(visitor.numRows());
    if constexpr (V::kHasHook) {
      if (pass) {
        NIMBLE_DCHECK_EQ(numValues, numSelected);
        visitor.hook().addValues(scatterRows, values, numSelected);
      }
    } else {
      visitor.addNumValues(V::kFilterOnly ? numHits : numValues);
    }
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

    if (!isConstant(values, selection.statistics())) {
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
    encoding::write<physicalType>(values.front(), pos);
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
      const Statistics<physicalType>& statistics) {
    NIMBLE_CHECK(
        !values.empty(), "ConstantEncoding requires non-empty values.");
    return values.size() == 1 || statistics.isConstant();
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
  NIMBLE_CHECK_EQ(
      pos, data.data() + data.size(), "Unexpected constant encoding end");
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
    NIMBLE_CHECK_EQ(
        pos, data.data() + data.size(), "Unexpected constant encoding end");
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
