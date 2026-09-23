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
#include <limits>
#include <span>
#include <utility>
#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/common/Types.h"
#include "velox/dwio/nimble/common/Vector.h"
#include "velox/dwio/nimble/encodings/DictionaryEncoding.h"
#include "velox/dwio/nimble/encodings/common/Encoding.h"
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrefix.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrimitives.h"
#include "velox/dwio/nimble/encodings/common/EncodingType.h"
#include "velox/dwio/nimble/encodings/selection/EncodingIdentifier.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelection.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelectionPolicy.h"

// A nullable encoding holds a subencoding of non-null values and another
// subencoding of booleans representing whether each row was null.

namespace facebook::nimble {

// Data layout is:
// EncodingPrefix::kFixedPrefixSize bytes: standard Encoding prefix
// 4 bytes: non-null child encoding size (X)
// X bytes: non-null child encoding bytes
// Y bytes: null child encoding bytes
template <typename T>
class NullableEncoding final
    : public TypedEncoding<T, typename TypeTraits<T>::physicalType> {
 public:
  using cppDataType = T;
  using physicalType = typename TypeTraits<T>::physicalType;

  NullableEncoding(
      velox::memory::MemoryPool& pool,
      std::string_view data,
      std::function<void*(uint32_t)> stringBufferFactory,
      const Encoding::Options& options = {});

  ~NullableEncoding() override {
    this->releaseVectorBuffer(nullBuffer_);
    this->releaseVectorBuffer(nonNullBitsBuffer_);
  }

  NullableEncoding(const NullableEncoding&) = delete;
  NullableEncoding& operator=(const NullableEncoding&) = delete;
  NullableEncoding(NullableEncoding&&) = delete;
  NullableEncoding& operator=(NullableEncoding&&) = delete;

  uint32_t nullCount() const final;
  bool isNullable() const final;
  const Encoding* nonNulls() const;

  void reset() final;
  void skip(uint32_t rowCount) final;
  void materialize(uint32_t rowCount, void* buffer) final;
  uint32_t materializeNullable(
      uint32_t rowCount,
      void* buffer,
      std::function<void*()> getOutputNulls,
      const velox::bits::Bitmap* scatterOutputBitmap = nullptr,
      uint32_t offset = 0) final;

  template <typename DecoderVisitor>
  void readWithVisitor(DecoderVisitor& visitor, ReadWithVisitorParams& params);

  /// Reads dictionary indices from a NullableEncoding that wraps a
  /// DictionaryEncoding. Handles nulls, then delegates to the inner
  /// DictionaryEncoding's readIndicesWithVisitor().
  /// Non-legacy encodings only.
  template <typename IndicesVisitor>
  void readIndicesWithVisitor(
      IndicesVisitor& visitor,
      ReadWithVisitorParams& params);

  static std::string_view encodeNullable(
      EncodingSelection<physicalType>& selection,
      std::span<const physicalType> values,
      std::span<const bool> nulls,
      Buffer& buffer,
      const Encoding::Options& options = {});

  /// Wraps pre-serialized value and null child encodings in a NullableEncoding
  /// envelope. The value child may use either T or physicalType.
  static std::string_view encodeNullable(
      uint32_t rowCount,
      std::string_view serializedValues,
      std::string_view serializedNulls,
      Buffer& buffer,
      const Encoding::Options& options = {});

  static std::string_view slice(
      std::string_view encoded,
      uint32_t offset,
      uint32_t length,
      Buffer& buffer,
      const Encoding::Options& options = {});

  std::string debugString(int offset) const final;

  // Dictionary API — delegates to the inner (non-null) encoding.
  bool dictionaryEnabled() const override {
    return nonNulls()->dictionaryEnabled();
  }

  uint32_t dictionarySize() const override {
    return nonNulls()->dictionarySize();
  }

  const void* dictionaryEntry(uint32_t index) const override {
    return nonNulls()->dictionaryEntry(index);
  }

  const void* dictionaryEntries() const override {
    return nonNulls()->dictionaryEntries();
  }

 private:
  // Materializes consecutive rows while keeping the validity stream packed.
  uint32_t contiguousRead(
      uint32_t rowCount,
      std::function<void*()>& getOutputNulls,
      uint32_t offset,
      void* buffer);

  // Materializes rows into positions selected by the scatter bitmap.
  uint32_t scatteredRead(
      uint32_t rowCount,
      std::function<void*()>& getOutputNulls,
      const velox::bits::Bitmap& scatterOutputBitmap,
      uint32_t offset,
      void* buffer);

  /// Materializes null bits into the visitor's reader null bitmap.
  /// Shared by readWithVisitor and readIndicesWithVisitor.
  template <typename V>
  void materializeNullsForVisitor(V& visitor, ReadWithVisitorParams& params);

  static std::pair<uint32_t, uint32_t> countNonNullsForSlice(
      std::string_view encoded,
      uint32_t offset,
      uint32_t length,
      Buffer& buffer,
      const Encoding::Options& options);

  // One bit for each row. A true bit represents a row with a non-null value.
  const char* bitmap_{};
  std::unique_ptr<Encoding> nonNullValues_;
  std::unique_ptr<Encoding> nulls_;
  uint32_t row_ = 0;

  // Byte-per-row validity scratch used by materialize and scattered reads.
  Vector<bool> nullBuffer_;

  // Packed validity scratch used by contiguous nullable reads.
  Vector<uint64_t> nonNullBitsBuffer_;
};

//
// End of public API. Implementations follow.
//

template <typename T>
NullableEncoding<T>::NullableEncoding(
    velox::memory::MemoryPool& pool,
    std::string_view data,
    std::function<void*(uint32_t)> stringBufferFactory,
    const Encoding::Options& options)
    : TypedEncoding<T, physicalType>(pool, data, options),
      nullBuffer_(this->template getVectorBuffer<bool>()),
      nonNullBitsBuffer_(this->template getVectorBuffer<uint64_t>()) {
  const char* pos = data.data() + this->dataOffset();
  const uint32_t nonNullsBytes = encoding::readUint32(pos);
  nonNullValues_ = EncodingFactory().create(
      *this->pool_, {pos, nonNullsBytes}, stringBufferFactory, options);
  pos += nonNullsBytes;
  nulls_ = EncodingFactory().create(
      *this->pool_,
      {pos, static_cast<size_t>(data.data() + data.size() - pos)},
      stringBufferFactory,
      options);
  NIMBLE_DCHECK_EQ(
      Encoding::rowCount(), nulls_->rowCount(), "Nulls count mismatch.");
}

template <typename T>
uint32_t NullableEncoding<T>::nullCount() const {
  return nulls_->rowCount() - nonNullValues_->rowCount();
}

template <typename T>
const Encoding* NullableEncoding<T>::nonNulls() const {
  return nonNullValues_.get();
}

template <typename T>
bool NullableEncoding<T>::isNullable() const {
  return true;
}

template <typename T>
void NullableEncoding<T>::reset() {
  row_ = 0;
  nonNullValues_->reset();
  nulls_->reset();
}

template <typename T>
void NullableEncoding<T>::skip(uint32_t rowCount) {
  // Hrm this isn't ideal. We should return to this later -- a new
  // encoding func? Encoding::Accumulate to add up next N rows?
  nullBuffer_.resize(rowCount);
  nulls_->materialize(rowCount, nullBuffer_.data());
  const uint32_t nonNullCount =
      std::accumulate(nullBuffer_.begin(), nullBuffer_.end(), 0U);
  nonNullValues_->skip(nonNullCount);
}

template <typename T>
void NullableEncoding<T>::materialize(uint32_t rowCount, void* buffer) {
  // This too isn't ideal. We will want an Encoding::Indices method or
  // something our SparseBool can use, giving back just the set indices
  // rather than a materialization.
  nullBuffer_.resize(rowCount);
  nulls_->materialize(rowCount, nullBuffer_.data());
  const uint32_t nonNullCount =
      std::accumulate(nullBuffer_.begin(), nullBuffer_.end(), 0U);
  nonNullValues_->materialize(nonNullCount, buffer);

  if (nonNullCount != rowCount) {
    physicalType* output = static_cast<physicalType*>(buffer) + rowCount - 1;
    const physicalType* lastNonNull =
        static_cast<physicalType*>(buffer) + nonNullCount - 1;
    // This is a generic scatter -- should we have a common scatter func?
    uint32_t pos = rowCount - 1;
    while (output != lastNonNull) {
      if (nullBuffer_[pos]) {
        *output = *lastNonNull;
        --lastNonNull;
      } else {
        *output = physicalType();
      }
      --output;
      --pos;
    }
  }

  row_ += rowCount;
}

template <typename T>
uint32_t NullableEncoding<T>::materializeNullable(
    uint32_t rowCount,
    void* buffer,
    std::function<void*()> getOutputNulls,
    const velox::bits::Bitmap* scatterOutputBitmap,
    uint32_t offset) {
  const uint32_t nonNullCount = scatterOutputBitmap == nullptr
      ? contiguousRead(rowCount, getOutputNulls, offset, buffer)
      : scatteredRead(
            rowCount, getOutputNulls, *scatterOutputBitmap, offset, buffer);
  row_ += rowCount;
  return nonNullCount;
}

template <typename T>
uint32_t NullableEncoding<T>::contiguousRead(
    uint32_t rowCount,
    std::function<void*()>& getOutputNulls,
    uint32_t offset,
    void* buffer) {
  // Empty reads are valid and must not access either output buffer.
  if (rowCount == 0) {
    return 0;
  }

  const auto numWords = velox::bits::nwords(rowCount);
  nonNullBitsBuffer_.resize(numWords);
  auto* nonNullBits = nonNullBitsBuffer_.data();
  // Validity streams can only use bool encodings that implement this API;
  // SliceEncoding<bool> forwards it to one of those encodings.
  nulls_->materializeBoolsAsBits(rowCount, nonNullBits, 0);
  NIMBLE_CHECK_LE(rowCount, std::numeric_limits<int32_t>::max());
  const auto nonNullCount = static_cast<uint32_t>(
      velox::bits::countBits(nonNullBits, 0, static_cast<int32_t>(rowCount)));

  auto* output = static_cast<physicalType*>(buffer) + offset;
  nonNullValues_->materialize(nonNullCount, output);
  if (nonNullCount == rowCount) {
    return nonNullCount;
  }

  velox::bits::copyBits(
      nonNullBits,
      0,
      static_cast<uint64_t*>(getOutputNulls()),
      offset,
      rowCount);

  // Expand backward to avoid overwriting dense source values. Iterating set
  // bits keeps work proportional to non-null rows; the destinations are not
  // contiguous enough for SIMD stores.
  uint32_t sourceIndex = nonNullCount;
  const uint32_t remainingBits = rowCount % 64;
  for (auto wordIndex = numWords; wordIndex > 0;) {
    --wordIndex;
    uint64_t word = nonNullBits[wordIndex];
    if (wordIndex + 1 == numWords && remainingBits != 0) {
      word &= (uint64_t{1} << remainingBits) - 1;
    }
    while (word != 0) {
      const uint32_t bit = 63 - __builtin_clzll(word);
      const uint32_t outputIndex = wordIndex * 64 + bit;
      output[outputIndex] = output[--sourceIndex];
      word ^= uint64_t{1} << bit;
    }
  }
  NIMBLE_CHECK_EQ(sourceIndex, 0);
  return nonNullCount;
}

template <typename T>
uint32_t NullableEncoding<T>::scatteredRead(
    uint32_t rowCount,
    std::function<void*()>& getOutputNulls,
    const velox::bits::Bitmap& scatterOutputBitmap,
    uint32_t offset,
    void* buffer) {
  nullBuffer_.resize(rowCount);
  nulls_->materialize(rowCount, nullBuffer_.data());
  const uint32_t nonNullCount =
      std::accumulate(nullBuffer_.begin(), nullBuffer_.end(), 0U);

  if (offset > 0) {
    buffer = static_cast<physicalType*>(buffer) + offset;
  }
  nonNullValues_->materialize(nonNullCount, buffer);

  const auto scatterSize = scatterOutputBitmap.size() - offset;
  NIMBLE_CHECK_GE(
      scatterSize,
      rowCount,
      "Scattered output must have at least rowCount positions");

  // A column that is entirely null over this batch would still walk every
  // position in the scatter loop below to set nothing. Short-circuit to the
  // cleared bitmap.
  if (nonNullCount == 0 && scatterSize != 0) {
    velox::bits::BitmapBuilder nullBits{getOutputNulls(), offset + scatterSize};
    nullBits.clear(offset, offset + scatterSize);
    return 0;
  }

  if (nonNullCount != scatterSize) {
    void* nullBitmap = getOutputNulls();
    velox::bits::BitmapBuilder nullBits{nullBitmap, offset + scatterSize};
    nullBits.clear(offset, offset + scatterSize);

    uint32_t pos = offset + scatterSize - 1;
    physicalType* output = static_cast<physicalType*>(buffer) + scatterSize - 1;
    const physicalType* lastNonNull =
        static_cast<physicalType*>(buffer) + nonNullCount - 1;

    auto nonNullIt = nullBuffer_.begin() + rowCount - 1;
    if (scatterSize != rowCount) {
      // In scattered reads, spread the items into the right positions in
      // |buffer| and |nullBitmap| based on the bits set to 1 in
      // |scatterOutputBitmap|.
      while (output != lastNonNull) {
        if (scatterOutputBitmap.test(pos)) {
          if (*nonNullIt--) {
            nullBits.set(pos);
            *output = *lastNonNull;
            --lastNonNull;
          }
        }
        --output;
        --pos;
      }
    } else {
      while (output != lastNonNull) {
        if (*nonNullIt--) {
          nullBits.set(pos);
          *output = *lastNonNull;
          --lastNonNull;
        }
        --output;
        --pos;
      }
    }

    if (output >= buffer) {
      nullBits.set(offset, pos + 1);
    }
  }

  return nonNullCount;
}

template <typename T>
template <typename V>
void NullableEncoding<T>::materializeNullsForVisitor(
    V& visitor,
    ReadWithVisitorParams& params) {
  const auto endRow = visitor.rowAt(visitor.numRows() - 1);
  const auto rowCount = endRow - params.numScanned + 1;
  if (const uint64_t* incomingNulls = visitor.reader().rawNullsInReadRange()) {
    auto nwords = velox::bits::nwords(rowCount);
    if (params.numScanned % 64 == 0) {
      nullBuffer_.resize(nwords * sizeof(uint64_t));
    } else {
      nullBuffer_.resize(2 * nwords * sizeof(uint64_t));
    }
    auto* chunkNulls = reinterpret_cast<uint64_t*>(nullBuffer_.data());
    auto* chunkNullBytes = reinterpret_cast<char*>(nullBuffer_.data());
    if (params.numScanned % 64 == 0) {
      incomingNulls += params.numScanned / 64;
    } else {
      auto* incomingNullsCopy = chunkNulls + nwords;
      velox::bits::copyBits(
          incomingNulls, params.numScanned, incomingNullsCopy, 0, rowCount);
      incomingNulls = incomingNullsCopy;
    }
    const auto numInner =
        velox::bits::countNonNulls(incomingNulls, 0, rowCount);
    nulls_->materializeBoolsAsBits(numInner, chunkNulls, 0);
    velox::bits::scatterBits(
        numInner, rowCount, chunkNullBytes, incomingNulls, chunkNullBytes);
    auto* nulls = params.makeReaderNulls();
    velox::bits::copyBits(chunkNulls, 0, nulls, params.numScanned, rowCount);
  } else {
    auto* nulls = params.makeReaderNulls();
    nulls_->materializeBoolsAsBits(rowCount, nulls, params.numScanned);
  }
  params.setReturnNullsMode();
}

template <typename T>
template <typename V>
void NullableEncoding<T>::readWithVisitor(
    V& visitor,
    ReadWithVisitorParams& params) {
  materializeNullsForVisitor(visitor, params);
  callReadWithVisitor(*nonNullValues_, visitor, params);
}

template <typename T>
template <typename V>
void NullableEncoding<T>::readIndicesWithVisitor(
    V& visitor,
    ReadWithVisitorParams& params) {
  NIMBLE_CHECK(
      this->dictionaryEnabled(),
      "readIndicesWithVisitor requires dictionary-enabled inner encoding");
  NIMBLE_CHECK(
      !V::kHasHook, "readIndicesWithVisitor does not support value hooks");
  materializeNullsForVisitor(visitor, params);
  if (nonNullValues_->dataType() == TypeTraits<T>::dataType) {
    callReadIndicesWithVisitor<T>(*nonNullValues_, visitor, params);
  } else {
    callReadIndicesWithVisitor<physicalType>(*nonNullValues_, visitor, params);
  }
}

template <typename T>
std::string_view NullableEncoding<T>::encodeNullable(
    EncodingSelection<physicalType>& selection,
    std::span<const physicalType> values,
    std::span<const bool> nulls,
    Buffer& buffer,
    const Encoding::Options& options) {
  NIMBLE_CHECK_LE(
      values.size(),
      nulls.size(),
      "Nullable value count cannot exceed null count.");
  NIMBLE_CHECK_LE(nulls.size(), std::numeric_limits<uint32_t>::max());
  const auto rowCount = static_cast<uint32_t>(nulls.size());

  auto* pool = &buffer.getMemoryPool();
  ScopedEncodingBuffer scopedBuffer{pool, options.encodingBufferPool};
  std::string_view serializedValues =
      selection.template encodeNested<physicalType>(
          EncodingIdentifiers::Nullable::Data,
          values,
          scopedBuffer.get(),
          options);
  std::string_view serializedNulls = selection.template encodeNested<bool>(
      EncodingIdentifiers::Nullable::Nulls, nulls, scopedBuffer.get(), options);

  return encodeNullable(
      rowCount, serializedValues, serializedNulls, buffer, options);
}

template <typename T>
std::string_view NullableEncoding<T>::encodeNullable(
    uint32_t rowCount,
    std::string_view serializedValues,
    std::string_view serializedNulls,
    Buffer& buffer,
    const Encoding::Options& options) {
  const bool useVarint = options.useVarintRowCount;
  const auto valueDataType = EncodingPrefix::readDataType(serializedValues);
  NIMBLE_CHECK(
      valueDataType == TypeTraits<T>::dataType ||
          valueDataType == TypeTraits<physicalType>::dataType,
      "Nullable value child data type must match the parent or its physical type.");
  NIMBLE_CHECK_EQ(
      EncodingPrefix::readDataType(serializedNulls),
      DataType::Bool,
      "Nullable null child must be bool.");
  const auto valueRowCount =
      EncodingPrefix::readRowCount(serializedValues, useVarint);
  const auto nullRowCount =
      EncodingPrefix::readRowCount(serializedNulls, useVarint);
  NIMBLE_CHECK_LE(
      valueRowCount,
      rowCount,
      "Nullable value child row count cannot exceed null child row count.");
  NIMBLE_CHECK_EQ(
      nullRowCount,
      rowCount,
      "Nullable null child row count must match parent.");

  const size_t encodingSize =
      TypedEncoding<T, physicalType>::serializePrefixSize(rowCount, useVarint) +
      4 + serializedValues.size() + serializedNulls.size();
  char* reserved = buffer.reserve(encodingSize);
  char* pos = reserved;
  TypedEncoding<T, physicalType>::serializePrefix(
      EncodingType::Nullable,
      TypeTraits<T>::dataType,
      rowCount,
      useVarint,
      pos);
  encoding::writeString(serializedValues, pos);
  encoding::writeBytes(serializedNulls, pos);
  NIMBLE_DCHECK_EQ(pos - reserved, encodingSize, "Encoding size mismatch.");
  return {reserved, encodingSize};
}

template <typename T>
std::pair<uint32_t, uint32_t> NullableEncoding<T>::countNonNullsForSlice(
    std::string_view encoded,
    uint32_t offset,
    uint32_t length,
    Buffer& buffer,
    const Encoding::Options& options) {
  const auto rowEnd = offset + length;
  if (rowEnd == 0) {
    return {0, 0};
  }

  auto* pool = &buffer.getMemoryPool();
  auto encoding = EncodingFactory{options}.create(
      *pool, encoded, [](uint32_t /*size*/) -> void* { return nullptr; });

  ScopedVector<bool> values{rowEnd, pool, options.bufferPool};
  encoding->materialize(rowEnd, values.data());
  const auto sliceBegin = values.begin() + offset;
  return {
      std::count(values.begin(), sliceBegin, true),
      std::count(sliceBegin, values.end(), true)};
}

template <typename T>
std::string_view NullableEncoding<T>::slice(
    std::string_view encoded,
    uint32_t offset,
    uint32_t length,
    Buffer& buffer,
    const Encoding::Options& options) {
  const auto sourceRowCount =
      EncodingPrefix::readRowCount(encoded, options.useVarintRowCount);
  NIMBLE_CHECK_LE(offset, sourceRowCount);
  NIMBLE_CHECK_LE(length, sourceRowCount - offset);
  NIMBLE_CHECK_GT(length, 0, "Cannot slice zero rows.");

  const char* pos = encoded.data() +
      EncodingPrefix::prefixSize(encoded, options.useVarintRowCount);
  const uint32_t valuesSize = encoding::readUint32(pos);
  const std::string_view values{pos, valuesSize};
  pos += valuesSize;
  const std::string_view nulls{pos, encoded.data() + encoded.size()};

  const auto [nonNullOffset, nonNullCount] =
      countNonNullsForSlice(nulls, offset, length, buffer, options);

  auto* pool = &buffer.getMemoryPool();
  ScopedEncodingBuffer scopedBuffer{pool, options.encodingBufferPool};
  const auto slicedValues = nonNullCount == 0
      ? EncodingFactory::encode<physicalType>(
            std::make_unique<ManualEncodingSelectionPolicy<physicalType>>(
                std::vector<std::pair<EncodingType, float>>{},
                std::nullopt,
                std::nullopt),
            std::span<const physicalType>{},
            scopedBuffer.get(),
            options)
      : EncodingFactory::slice(
            values, nonNullOffset, nonNullCount, scopedBuffer.get(), options);
  const auto slicedNulls = EncodingFactory::slice(
      nulls, offset, length, scopedBuffer.get(), options);

  return encodeNullable(length, slicedValues, slicedNulls, buffer, options);
}

template <typename T>
std::string NullableEncoding<T>::debugString(int offset) const {
  std::string log = Encoding::debugString(offset);
  log += fmt::format(
      "\n{}non-null child:\n{}",
      std::string(offset + 2, ' '),
      nonNullValues_->debugString(offset + 4));
  log += fmt::format(
      "\n{}null child:\n{}",
      std::string(offset + 2, ' '),
      nulls_->debugString(offset + 4));
  return log;
}

} // namespace facebook::nimble
