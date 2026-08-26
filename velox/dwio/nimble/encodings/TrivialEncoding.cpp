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
#include "velox/dwio/nimble/encodings/TrivialEncoding.h"
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"

namespace facebook::nimble {
namespace {

struct UncompressedData {
  const char* data;
  velox::BufferPtr buffer{};
};

struct StringPayload {
  CompressionType compressionType;
  std::string_view lengths;
  std::string_view blob;
};

StringPayload readStringPayload(std::string_view encoded, uint32_t dataOffset) {
  const char* pos = encoded.data() + dataOffset;
  const auto compressionType =
      static_cast<CompressionType>(encoding::readChar(pos));
  const uint32_t lengthsSize = encoding::readUint32(pos);
  const std::string_view lengths{pos, lengthsSize};
  pos += lengthsSize;
  return {
      .compressionType = compressionType,
      .lengths = lengths,
      .blob = {pos, static_cast<size_t>(encoded.end() - pos)}};
}

void writeStringHeader(
    uint32_t rowCount,
    bool useVarint,
    CompressionType compressionType,
    std::string_view serializedLengths,
    char*& pos) {
  EncodingPrefix::serialize(
      EncodingType::Trivial, DataType::String, rowCount, useVarint, pos);
  encoding::writeChar(static_cast<char>(compressionType), pos);
  encoding::writeUint32(serializedLengths.size(), pos);
  encoding::writeBytes(serializedLengths, pos);
}

UncompressedData uncompressIfNeeded(
    velox::memory::MemoryPool& pool,
    CompressionType compressionType,
    DataType dataType,
    std::string_view data,
    const Encoding::Options& options) {
  if (compressionType == CompressionType::Uncompressed) {
    return {.data = data.data()};
  }

  auto buffer = Compression::uncompress(
      pool,
      compressionType,
      dataType,
      data,
      options.decompressCounter(),
      options.bufferPool);
  return {.data = buffer->as<char>(), .buffer = std::move(buffer)};
}

template <typename T>
Vector<T> makeScratchVector(
    velox::memory::MemoryPool& pool,
    const Encoding::Options& options,
    uint64_t capacity) {
  if (auto* bufferPool = options.bufferPool) {
    if (auto buffer = bufferPool->get(capacity * sizeof(T))) {
      return Vector<T>{std::move(buffer)};
    }
  }
  return Vector<T>{&pool};
}

template <typename T>
void releaseScratchVector(Vector<T>& vector, const Encoding::Options& options) {
  if (auto* bufferPool = options.bufferPool) {
    bufferPool->release(vector.releaseBuffer());
  }
}

} // namespace

TrivialEncoding<std::string_view>::TrivialEncoding(
    velox::memory::MemoryPool& pool,
    std::string_view data,
    std::function<void*(uint32_t)> stringBufferFactory,
    const Encoding::Options& options)
    : TypedEncoding<std::string_view, std::string_view>{pool, data, options},
      row_{0},
      buffer_{&pool} {
  const auto payload = readStringPayload(data, this->dataOffset());
  lengths_ = EncodingFactory().create(
      pool, payload.lengths, stringBufferFactory, options);
  blob_ = payload.blob.data();

  if (payload.compressionType != CompressionType::Uncompressed) {
    dataUncompressed_ = Compression::uncompress(
        pool,
        payload.compressionType,
        DataType::String,
        payload.blob,
        options.decompressCounter(),
        options.bufferPool);
    blob_ = dataUncompressed_->as<char>();
    uncompressedDataBytes_ = dataUncompressed_->size();
  } else {
    uncompressedDataBytes_ = payload.blob.size();
  }
  // TODO(huamengjiang): if we want to reduce the temporary memory peak, we can
  // pass the string buffer factory into the compression api.
  auto stringBuffer = stringBufferFactory(uncompressedDataBytes_);
  // TODO(huamengjiang): in a follow up, we will let the factory return a smart
  // pointer that can also be held by the encoding.
  std::memcpy(stringBuffer, blob_, uncompressedDataBytes_);
  releaseBuffer(dataUncompressed_);
  blob_ = static_cast<char*>(stringBuffer);
  pos_ = blob_;
}

void TrivialEncoding<std::string_view>::reset() {
  row_ = 0;
  pos_ = blob_;
  lengths_->reset();
}

void TrivialEncoding<std::string_view>::skip(uint32_t rowCount) {
  buffer_.resize(rowCount);
  lengths_->materialize(rowCount, buffer_.data());
  row_ += rowCount;
  pos_ += std::accumulate(buffer_.begin(), buffer_.end(), 0U);
}

void TrivialEncoding<std::string_view>::materialize(
    uint32_t rowCount,
    void* buffer) {
  buffer_.resize(rowCount);
  lengths_->materialize(rowCount, buffer_.data());
  const char* pos = pos_;
  const uint32_t* data = buffer_.data();
  for (int i = 0; i < rowCount; ++i) {
    static_cast<std::string_view*>(buffer)[i] = std::string_view(pos, data[i]);
    pos += data[i];
  }
  pos_ = pos;
  row_ += rowCount;
}

uint64_t TrivialEncoding<std::string_view>::uncompressedDataBytes() const {
  return uncompressedDataBytes_;
}

std::string_view TrivialEncoding<std::string_view>::encode(
    EncodingSelection<std::string_view>& selection,
    std::span<const std::string_view> values,
    Buffer& buffer,
    const Encoding::Options& options) {
  const bool useVarint = options.useVarintRowCount;
  const uint32_t valueCount = values.size();
  std::vector<uint32_t> lengths;
  lengths.reserve(valueCount);
  for (auto value : values) {
    lengths.push_back(value.size());
  }

  auto* pool = &buffer.getMemoryPool();
  ScopedEncodingBuffer scopedBuffer{pool, options.encodingBufferPool};
  std::string_view serializedLengths =
      selection.template encodeNested<uint32_t>(
          EncodingIdentifiers::Trivial::Lengths,
          {lengths},
          scopedBuffer.get(),
          options);

  auto dataCompressionPolicy = selection.compressionPolicy();
  auto uncompressedSize = selection.statistics().totalStringsLength();

  Vector<char> vector{pool};

  CompressionEncoder<std::string_view> compressionEncoder{
      *pool,
      *dataCompressionPolicy,
      DataType::String,
      /*bitWidth=*/0,
      uncompressedSize,
      [&]() {
        vector.resize(uncompressedSize);
        return std::span<char>{vector.data(), uncompressedSize};
      },
      [&](char*& pos) {
        for (auto value : values) {
          std::copy(value.cbegin(), value.cend(), pos);
          pos += value.size();
        }
      }};

  const uint32_t encodingSize =
      Encoding::serializePrefixSize(valueCount, useVarint) +
      TrivialEncoding<std::string_view>::kPrefixSize +
      serializedLengths.size() + compressionEncoder.getSize();

  char* reserved = buffer.reserve(encodingSize);
  char* pos = reserved;
  writeStringHeader(
      valueCount,
      useVarint,
      compressionEncoder.compressionType(),
      serializedLengths,
      pos);
  compressionEncoder.write(pos);

  NIMBLE_CHECK_EQ(pos - reserved, encodingSize, "Encoding size mismatch.");
  return {reserved, encodingSize};
}

std::string_view TrivialEncoding<std::string_view>::slice(
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

  const auto sourcePrefixSize =
      EncodingPrefix::prefixSize(encoded, options.useVarintRowCount);
  const auto payload = readStringPayload(encoded, sourcePrefixSize);
  const auto blob = uncompressIfNeeded(
      buffer.getMemoryPool(),
      payload.compressionType,
      DataType::String,
      payload.blob,
      options);
  const char* blobData = blob.data;

  const auto slicedLengths =
      EncodingFactory::slice(payload.lengths, offset, length, buffer, options);
  const auto rowEnd = offset + length;
  auto materializedLengths =
      makeScratchVector<uint32_t>(buffer.getMemoryPool(), options, rowEnd);
  materializedLengths.resize(rowEnd);
  auto lengthsEncoding = EncodingFactory{options}.create(
      buffer.getMemoryPool(),
      payload.lengths,
      [](uint32_t /*totalLength*/) -> void* { return nullptr; });
  lengthsEncoding->materialize(rowEnd, materializedLengths.data());
  const auto sliceBegin = materializedLengths.begin() + offset;
  const auto blobOffset =
      std::accumulate(materializedLengths.begin(), sliceBegin, uint32_t{0});
  const auto blobBytes =
      std::accumulate(sliceBegin, materializedLengths.end(), uint32_t{0});

  const auto prefixSize =
      EncodingPrefix::serializedSize(length, options.useVarintRowCount);
  const auto encodingSize =
      prefixSize + kPrefixSize + slicedLengths.size() + blobBytes;
  char* reserved = buffer.reserve(encodingSize);
  char* pos = reserved;
  writeStringHeader(
      length,
      options.useVarintRowCount,
      CompressionType::Uncompressed,
      slicedLengths,
      pos);
  encoding::writeBytes({blobData + blobOffset, blobBytes}, pos);
  NIMBLE_CHECK_EQ(pos - reserved, encodingSize, "Encoding size mismatch.");
  releaseScratchVector(materializedLengths, options);
  return std::string_view{reserved, encodingSize};
}

TrivialEncoding<bool>::TrivialEncoding(
    velox::memory::MemoryPool& pool,
    std::string_view data,
    std::function<void*(uint32_t)> /* stringBufferFactory */,
    const Encoding::Options& options)
    : TypedEncoding<bool, bool>{pool, data, options},
      bitmap_{data.data() + this->dataOffset() + kPrefixSize} {
  const auto compressionType =
      static_cast<CompressionType>(data[this->dataOffset()]);
  if (compressionType != CompressionType::Uncompressed) {
    uncompressed_ = Compression::uncompress(
        pool,
        compressionType,
        DataType::Undefined,
        {bitmap_, static_cast<size_t>(data.end() - bitmap_)},
        options.decompressCounter(),
        options.bufferPool);
    bitmap_ = uncompressed_->as<char>();
    NIMBLE_CHECK_EQ(
        bitmap_ + FixedBitArray::bufferSize(rowCount(), 1),
        uncompressed_->as<char>() + uncompressed_->size(),
        "Unexpected trivial encoding end");
  } else {
    NIMBLE_CHECK_EQ(
        bitmap_ + FixedBitArray::bufferSize(rowCount(), 1),
        data.end(),
        "Unexpected trivial encoding end");
  }
}

void TrivialEncoding<bool>::reset() {
  row_ = 0;
}

void TrivialEncoding<bool>::skip(uint32_t rowCount) {
  row_ += rowCount;
}

void TrivialEncoding<bool>::materialize(uint32_t rowCount, void* buffer) {
  // Align to word boundary, go fast over words, then do remainder.
  bool* output = static_cast<bool*>(buffer);
  const uint32_t rowsToWord = (row_ & 63) == 0 ? 0 : 64 - (row_ & 63);
  if (rowsToWord >= rowCount) {
    for (int i = 0; i < rowCount; ++i) {
      *output = velox::bits::isBitSet(
          reinterpret_cast<const uint8_t*>(bitmap_), row_);
      ++output;
      ++row_;
    }
    return;
  }
  for (uint32_t i = 0; i < rowsToWord; ++i) {
    *output =
        velox::bits::isBitSet(reinterpret_cast<const uint8_t*>(bitmap_), row_);
    ++output;
    ++row_;
  }
  const uint32_t rowsRemaining = rowCount - rowsToWord;
  const uint32_t numWords = rowsRemaining >> 6;
  const uint64_t* nextWord =
      reinterpret_cast<const uint64_t*>(bitmap_ + (row_ >> 3));
  for (uint32_t i = 0; i < numWords; ++i) {
    uint64_t word = nextWord[i];
    for (int j = 0; j < 64; ++j) {
      *output = word & 1;
      word >>= 1;
      ++output;
    }
    row_ += 64;
  }
  const uint32_t remainder = rowsRemaining - (numWords << 6);
  for (uint32_t i = 0; i < remainder; ++i) {
    *output =
        velox::bits::isBitSet(reinterpret_cast<const uint8_t*>(bitmap_), row_);
    ++output;
    ++row_;
  }
}

void TrivialEncoding<bool>::materializeBoolsAsBits(
    uint32_t rowCount,
    uint64_t* buffer,
    int begin) {
  velox::bits::copyBits(
      reinterpret_cast<const uint64_t*>(bitmap_),
      row_,
      buffer,
      begin,
      rowCount);
  row_ += rowCount;
}

std::string_view TrivialEncoding<bool>::encode(
    EncodingSelection<bool>& selection,
    std::span<const bool> values,
    Buffer& buffer,
    const Encoding::Options& options) {
  const bool useVarint = options.useVarintRowCount;
  const uint32_t valueCount = values.size();
  const uint32_t bitmapBytes = FixedBitArray::bufferSize(valueCount, 1);

  auto* pool = &buffer.getMemoryPool();
  Vector<char> vector{pool};

  auto dataCompressionPolicy = selection.compressionPolicy();
  CompressionEncoder<std::string_view> compressionEncoder{
      *pool,
      *dataCompressionPolicy,
      DataType::Undefined,
      /*bitWidth=*/1,
      bitmapBytes,
      [&]() {
        vector.resize(bitmapBytes);
        return std::span<char>{vector};
      },
      [&](char*& pos) {
        memset(pos, 0, bitmapBytes);
        for (size_t i = 0; i < values.size(); ++i) {
          velox::bits::maybeSetBit(pos, i, values[i]);
        }
        pos += bitmapBytes;
      }};

  const uint32_t encodingSize =
      Encoding::serializePrefixSize(valueCount, useVarint) + kPrefixSize +
      compressionEncoder.getSize();
  char* reserved = buffer.reserve(encodingSize);
  char* pos = reserved;
  Encoding::serializePrefix(
      EncodingType::Trivial, DataType::Bool, valueCount, useVarint, pos);
  encoding::writeChar(
      static_cast<char>(compressionEncoder.compressionType()), pos);
  compressionEncoder.write(pos);

  NIMBLE_CHECK_EQ(pos - reserved, encodingSize, "Encoding size mismatch.");
  return {reserved, encodingSize};
}

std::string_view TrivialEncoding<bool>::slice(
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

  const auto sourcePrefixSize =
      EncodingPrefix::prefixSize(encoded, options.useVarintRowCount);
  const char* sourcePos = encoded.data() + sourcePrefixSize;
  const auto compressionType =
      static_cast<CompressionType>(encoding::readChar(sourcePos));
  const auto sourceData = uncompressIfNeeded(
      buffer.getMemoryPool(),
      compressionType,
      DataType::Undefined,
      {sourcePos, static_cast<size_t>(encoded.end() - sourcePos)},
      options);
  sourcePos = sourceData.data;

  const auto prefixSize =
      EncodingPrefix::serializedSize(length, options.useVarintRowCount);
  const auto bitmapBytes = FixedBitArray::bufferSize(length, 1);
  const auto encodingSize = prefixSize + kPrefixSize + bitmapBytes;
  char* reserved = buffer.reserve(encodingSize);
  char* pos = reserved;
  EncodingPrefix::serialize(
      EncodingType::Trivial,
      DataType::Bool,
      length,
      options.useVarintRowCount,
      pos);
  encoding::writeChar(static_cast<char>(CompressionType::Uncompressed), pos);
  std::memset(pos, 0, bitmapBytes);
  velox::bits::copyBits(
      reinterpret_cast<const uint64_t*>(sourcePos),
      offset,
      reinterpret_cast<uint64_t*>(pos),
      0,
      length);
  pos += bitmapBytes;
  NIMBLE_CHECK_EQ(pos - reserved, encodingSize, "Encoding size mismatch.");
  return std::string_view{reserved, encodingSize};
}

} // namespace facebook::nimble
