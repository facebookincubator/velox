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

#include <span>
#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/common/FixedBitArray.h"
#include "velox/dwio/nimble/common/Types.h"
#include "velox/dwio/nimble/common/Vector.h"
#include "velox/dwio/nimble/encodings/common/Encoding.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrimitives.h"
#include "velox/dwio/nimble/encodings/common/EncodingType.h"
#include "velox/dwio/nimble/encodings/legacy/Encoding.h"
#include "velox/dwio/nimble/encodings/legacy/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/selection/EncodingIdentifier.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelection.h"

// A nullable encoding holds a subencoding of non-null values and another
// subencoding of booleans representing whether each row was null.

namespace facebook::nimble::legacy {

// Data layout is:
// EncodingPrefix::kFixedPrefixSize bytes: standard Encoding prefix
// 4 bytes: non-null child encoding size (X)
// X bytes: non-null child encoding bytes
// Y byes: null child encoding bytes
template <typename T>
class NullableEncoding final
    : public TypedEncoding<T, typename TypeTraits<T>::physicalType> {
 public:
  using cppDataType = T;
  using physicalType = typename TypeTraits<T>::physicalType;

  NullableEncoding(
      velox::memory::MemoryPool& memoryPool,
      std::string_view data,
      std::function<void*(uint32_t)> stringBufferFactory);

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

  static std::string_view encodeNullable(
      EncodingSelection<physicalType>& selection,
      std::span<const physicalType> values,
      std::span<const bool> nulls,
      Buffer& buffer);

  std::string debugString(int offset) const final;

 private:
  // One bit for each row. A true bit represents a row with a non-null value.
  const char* bitmap_{};
  std::unique_ptr<Encoding> nonNullValues_;
  std::unique_ptr<Encoding> nulls_;
  uint32_t row_ = 0;

  // Temporary buffers.
  Vector<uint32_t> indicesBuffer_;
  Vector<bool> nullBuffer_;
};

//
// End of public API. Implementations follow.
//

template <typename T>
NullableEncoding<T>::NullableEncoding(
    velox::memory::MemoryPool& memoryPool,
    std::string_view data,
    std::function<void*(uint32_t)> stringBufferFactory)
    : TypedEncoding<T, physicalType>(memoryPool, data),
      indicesBuffer_(this->pool_),
      nullBuffer_(this->pool_) {
  const EncodingFactory factory;
  const char* pos = data.data() + EncodingPrefix::kFixedPrefixSize;
  const uint32_t nonNullsBytes = encoding::readUint32(pos);
  nonNullValues_ = factory.create(
      *this->pool_,
      {pos, nonNullsBytes},
      stringBufferFactory,
      Encoding::Options{});
  pos += nonNullsBytes;
  nulls_ = factory.create(
      *this->pool_,
      {pos, static_cast<size_t>(data.data() + data.size() - pos)},
      stringBufferFactory,
      Encoding::Options{});
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
  nullBuffer_.resize(rowCount);
  nulls_->materialize(rowCount, nullBuffer_.data());
  const uint32_t nonNullCount =
      std::accumulate(nullBuffer_.begin(), nullBuffer_.end(), 0U);

  if (offset > 0) {
    buffer = static_cast<physicalType*>(buffer) + offset;
  }
  nonNullValues_->materialize(nonNullCount, buffer);

  const auto scatterSize =
      scatterOutputBitmap ? scatterOutputBitmap->size() - offset : rowCount;

  // A column that is entirely null over this batch would still walk every
  // position in the scatter loop below to set nothing. Short-circuit to the
  // cleared bitmap.
  if (nonNullCount == 0 && scatterSize != 0) {
    velox::bits::BitmapBuilder nullBits{getOutputNulls(), offset + scatterSize};
    nullBits.clear(offset, offset + scatterSize);
    row_ += rowCount;
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
        if (scatterOutputBitmap->test(pos)) {
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

  row_ += rowCount;
  return nonNullCount;
}

template <typename T>
template <typename V>
void NullableEncoding<T>::readWithVisitor(
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
  legacy::callReadWithVisitor(*nonNullValues_, visitor, params);
}

template <typename T>
std::string_view NullableEncoding<T>::encodeNullable(
    EncodingSelection<physicalType>& selection,
    std::span<const physicalType> values,
    std::span<const bool> nulls,
    Buffer& buffer) {
  const uint32_t rowCount = nulls.size();

  Buffer tempBuffer{buffer.getMemoryPool()};
  std::string_view serializedValues =
      selection.template encodeNested<physicalType>(
          EncodingIdentifiers::Nullable::Data, values, tempBuffer);
  std::string_view serializedNulls = selection.template encodeNested<bool>(
      EncodingIdentifiers::Nullable::Nulls, nulls, tempBuffer);

  const uint32_t encodingSize = EncodingPrefix::kFixedPrefixSize + 4 +
      serializedValues.size() + serializedNulls.size();
  char* reserved = buffer.reserve(encodingSize);
  char* pos = reserved;
  Encoding::serializePrefix(
      EncodingType::Nullable, TypeTraits<T>::dataType, rowCount, false, pos);
  encoding::writeString(serializedValues, pos);
  encoding::writeBytes(serializedNulls, pos);
  NIMBLE_DCHECK_EQ(pos - reserved, encodingSize, "Encoding size mismatch.");
  return {reserved, encodingSize};
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

} // namespace facebook::nimble::legacy
