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

#include <numeric>
#include <span>

#include "velox/common/memory/Memory.h"
#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/common/FixedBitArray.h"
#include "velox/dwio/nimble/common/Types.h"
#include "velox/dwio/nimble/compression/Compression.h"
#include "velox/dwio/nimble/encodings/common/Encoding.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrimitives.h"
#include "velox/dwio/nimble/encodings/common/EncodingType.h"
#include "velox/dwio/nimble/encodings/legacy/Encoding.h"
#include "velox/dwio/nimble/encodings/selection/EncodingIdentifier.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelection.h"

// Holds data in the the 'trivial' way for each data type.
// Namely as physicalType* for numerics, a packed set of offsets and a blob of
// characters for strings, and bit-packed for bools.

namespace facebook::nimble::legacy {

// Handles the numeric cases. Bools and strings are specialized below.
// Data layout is:
// EncodingPrefix::kFixedPrefixSize bytes: standard Encoding prefix
// 1 byte: lengths compression
// rowCount * sizeof(physicalType) bytes: the data
template <typename T>
class TrivialEncoding final
    : public TypedEncoding<T, typename TypeTraits<T>::physicalType> {
 public:
  using cppDataType = T;
  using physicalType = typename TypeTraits<T>::physicalType;

  static constexpr int kPrefixSize = 1;
  static constexpr int kCompressionTypeOffset =
      EncodingPrefix::kFixedPrefixSize;
  static constexpr int kDataOffset =
      EncodingPrefix::kFixedPrefixSize + TrivialEncoding<T>::kPrefixSize;

  TrivialEncoding(
      velox::memory::MemoryPool& pool,
      std::string_view data,
      std::function<void*(uint32_t)> stringBufferFactory);

  void reset() final;
  void skip(uint32_t rowCount) final;
  void materialize(uint32_t rowCount, void* buffer) final;

  template <typename DecoderVisitor>
  void readWithVisitor(DecoderVisitor& visitor, ReadWithVisitorParams& params);

  template <bool kScatter, typename Visitor>
  void bulkScan(
      Visitor& visitor,
      vector_size_t currentRow,
      const vector_size_t* selectedRows,
      vector_size_t numSelected,
      const vector_size_t* scatterRows);

  static std::string_view encode(
      EncodingSelection<physicalType>& selection,
      std::span<const physicalType> values,
      Buffer& buffer);

 private:
  uint32_t row_;
  const T* values_;
  velox::BufferPtr uncompressed_;
};

// For the string case the layout is:
// EncodingPrefix::kFixedPrefixSize bytes: standard Encoding prefix
// 1 byte: data compression
// 4 bytes: offset to start of blob
// XX bytes: bit packed lengths
// YY bytes: actual characters
template <>
class TrivialEncoding<std::string_view> final
    : public TypedEncoding<std::string_view, std::string_view> {
 public:
  using cppDataType = std::string_view;

  static constexpr int kPrefixSize = 5;
  static constexpr int kDataCompressionOffset =
      EncodingPrefix::kFixedPrefixSize;
  static constexpr int kBlobOffsetOffset = EncodingPrefix::kFixedPrefixSize + 1;
  static constexpr int kLengthOffset = EncodingPrefix::kFixedPrefixSize +
      TrivialEncoding<std::string_view>::kPrefixSize;

  TrivialEncoding(
      velox::memory::MemoryPool& pool,
      std::string_view data,
      std::function<void*(uint32_t)> stringBufferFactory);

  void reset() final;
  void skip(uint32_t rowCount) final;
  void materialize(uint32_t rowCount, void* buffer) final;

  template <typename DecoderVisitor>
  void readWithVisitor(DecoderVisitor& visitor, ReadWithVisitorParams& params);

  // Returns the total size of the characters payload in bytes
  uint64_t uncompressedDataBytes() const;

  static std::string_view encode(
      EncodingSelection<std::string_view>& selection,
      std::span<const std::string_view> values,
      Buffer& buffer);

 private:
  uint32_t row_;
  const char* blob_;
  const char* pos_;
  std::unique_ptr<Encoding> lengths_;
  Vector<uint32_t> buffer_;

  // The size of the uncompressed data in bytes.
  //
  // NOTE: This is not necessarily the size of 'dataUncompressed_' because the
  // data could come as uncompressed.
  uint64_t uncompressedDataBytes_;
  velox::BufferPtr dataUncompressed_;
};

// For the bool case the layout is:
// EncodingPrefix::kFixedPrefixSize bytes: standard Encoding prefix
// 1 byte: compression type
// FBA::BufferSize(rowCount, 1) bytes: the bitmap
template <>
class TrivialEncoding<bool> final : public TypedEncoding<bool, bool> {
 public:
  using cppDataType = bool;

  static constexpr int kPrefixSize = 1;
  static constexpr int kCompressionTypeOffset =
      EncodingPrefix::kFixedPrefixSize;
  static constexpr int kDataOffset =
      EncodingPrefix::kFixedPrefixSize + TrivialEncoding<bool>::kPrefixSize;

  TrivialEncoding(
      velox::memory::MemoryPool& pool,
      std::string_view data,
      std::function<void*(uint32_t)> stringBufferFactory);

  void reset() final;
  void skip(uint32_t rowCount) final;
  void materialize(uint32_t rowCount, void* buffer) final;

  void materializeBoolsAsBits(uint32_t rowCount, uint64_t* buffer, int begin)
      final;

  template <typename DecoderVisitor>
  void readWithVisitor(DecoderVisitor& visitor, ReadWithVisitorParams& params);

  static std::string_view encode(
      EncodingSelection<bool>& selection,
      std::span<const bool> values,
      Buffer& buffer);

 private:
  uint32_t row_{0};
  const char* bitmap_;
  velox::BufferPtr uncompressed_;
};

//
// End of public API. Implementations follow.
//

template <typename T>
TrivialEncoding<T>::TrivialEncoding(
    velox::memory::MemoryPool& pool,
    std::string_view data,
    std::function<void*(uint32_t)> /* stringBufferFactory */)
    : TypedEncoding<T, physicalType>{pool, data},
      row_{0},
      values_{reinterpret_cast<const T*>(data.data() + kDataOffset)} {
  auto compressionType =
      static_cast<CompressionType>(data[kCompressionTypeOffset]);
  if (compressionType != CompressionType::Uncompressed) {
    uncompressed_ = Compression::uncompress(
        pool,
        compressionType,
        TypeTraits<physicalType>::dataType,
        {data.data() + kDataOffset, data.size() - kDataOffset},
        /*decompressCounter=*/nullptr);
    values_ = reinterpret_cast<const T*>(uncompressed_->as<char>());
    NIMBLE_CHECK_EQ(
        reinterpret_cast<const char*>(values_ + this->rowCount()),
        uncompressed_->as<char>() + uncompressed_->size(),
        "Unexpected trivial encoding end");
  } else {
    NIMBLE_CHECK_EQ(
        reinterpret_cast<const char*>(values_ + this->rowCount()),
        data.data() + data.size(),
        "Unexpected trivial encoding end");
  }
}

template <typename T>
void TrivialEncoding<T>::reset() {
  row_ = 0;
}

template <typename T>
void TrivialEncoding<T>::skip(uint32_t rowCount) {
  row_ += rowCount;
}

template <typename T>
void TrivialEncoding<T>::materialize(uint32_t rowCount, void* buffer) {
  const auto start = values_ + row_;
  std::copy(start, start + rowCount, static_cast<T*>(buffer));
  row_ += rowCount;
}

template <typename T>
std::string_view TrivialEncoding<T>::encode(
    EncodingSelection<physicalType>& selection,
    std::span<const physicalType> values,
    Buffer& buffer) {
  const uint32_t rowCount = values.size();
  const uint32_t uncompressedSize = sizeof(T) * rowCount;

  auto compressionPolicy = selection.compressionPolicy();
  CompressionEncoder<T> compressionEncoder{
      buffer.getMemoryPool(),
      *compressionPolicy,
      TypeTraits<physicalType>::dataType,
      {reinterpret_cast<const char*>(values.data()), uncompressedSize}};

  const uint32_t encodingSize = kDataOffset + compressionEncoder.getSize();
  char* reserved = buffer.reserve(encodingSize);
  char* pos = reserved;
  Encoding::serializePrefix(
      EncodingType::Trivial, TypeTraits<T>::dataType, rowCount, false, pos);
  encoding::writeChar(
      static_cast<char>(compressionEncoder.compressionType()), pos);
  compressionEncoder.write(pos);
  return {reserved, encodingSize};
}

template <typename T>
template <bool kScatter, typename V>
void TrivialEncoding<T>::bulkScan(
    V& visitor,
    vector_size_t currentRow,
    const vector_size_t* selectedRows,
    vector_size_t numSelected,
    const vector_size_t* scatterRows) {
  const auto numRows = visitor.numRows() - visitor.rowIndex();
  T* values = detail::mutableValues<T>(visitor, numRows);
  const auto offset = static_cast<vector_size_t>(row_) - currentRow;
  if constexpr (V::dense) {
    memcpy(values, values_ + selectedRows[0] + offset, numSelected * sizeof(T));
  } else {
    for (vector_size_t i = 0; i < numSelected; ++i) {
      values[i] = values_[selectedRows[i] + offset];
    }
  }
  if constexpr (!V::kHasHook) {
    values = reinterpret_cast<T*>(visitor.reader().rawValues());
  }
  int32_t numValues = visitor.reader().numValues();
  int32_t* filterHits;
  if constexpr (V::kHasFilter) {
    NIMBLE_DCHECK_EQ(visitor.reader().numRows(), numValues, "");
    filterHits = visitor.outputRows(numSelected) - numValues;
  } else {
    filterHits = nullptr;
  }
  velox::dwio::common::
      processFixedWidthRun<T, V::kFilterOnly, kScatter, V::dense>(
          velox::RowSet(selectedRows, numSelected),
          0,
          numSelected,
          scatterRows,
          values,
          filterHits,
          numValues,
          visitor.filter(),
          visitor.hook());
  row_ += selectedRows[numSelected - 1] - currentRow + 1;
  if constexpr (!V::kHasHook) {
    visitor.addNumValues(
        V::kHasFilter ? numValues - visitor.reader().numValues() : numRows);
  }
  visitor.setRowIndex(visitor.numRows());
}

template <typename T>
template <typename V>
void TrivialEncoding<T>::readWithVisitor(
    V& visitor,
    ReadWithVisitorParams& params) {
  auto* nulls = visitor.reader().rawNullsInReadRange();
  if constexpr (sizeof(T) >= 2) {
    if (velox::dwio::common::useFastPath(visitor, nulls)) {
      detail::readWithVisitorFast(*this, visitor, params, nulls);
      return;
    }
  }
  detail::readWithVisitorSlow(
      visitor,
      params,
      [&](auto toSkip) { skip(toSkip); },
      [&] { return values_[row_++]; });
}

template <typename V>
void TrivialEncoding<std::string_view>::readWithVisitor(
    V& visitor,
    ReadWithVisitorParams& params) {
  const auto endRow = visitor.rowAt(visitor.numRows() - 1);
  auto numSelected = endRow + 1 - params.numScanned;
  if (auto& nulls = visitor.reader().nullsInReadRange()) {
    numSelected -= velox::bits::countNulls(
        nulls->template as<uint64_t>(), params.numScanned, endRow + 1);
  }
  buffer_.resize(numSelected);
  lengths_->materialize(numSelected, buffer_.data());
  auto* lengths = buffer_.data();
  detail::readWithVisitorSlow(
      visitor,
      params,
      [&](auto toSkip) {
        row_ += toSkip;
        pos_ += std::accumulate(lengths, lengths + toSkip, 0ull);
        lengths += toSkip;
      },
      [&] {
        auto len = *lengths++;
        std::string_view value(pos_, len);
        ++row_;
        pos_ += len;
        return value;
      });
}

template <typename V>
void TrivialEncoding<bool>::readWithVisitor(
    V& visitor,
    ReadWithVisitorParams& params) {
  detail::readWithVisitorSlow(
      visitor,
      params,
      [&](auto toSkip) { skip(toSkip); },
      [&] {
        return velox::bits::isBitSet(
            reinterpret_cast<const uint8_t*>(bitmap_), row_++);
      });
}

} // namespace facebook::nimble::legacy
