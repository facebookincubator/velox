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
#include "velox/dwio/nimble/encodings/legacy/TrivialEncoding.h"
#include "velox/dwio/nimble/encodings/legacy/EncodingFactory.h"

namespace facebook::nimble::legacy {

TrivialEncoding<std::string_view>::TrivialEncoding(
    velox::memory::MemoryPool& pool,
    std::string_view data,
    std::function<void*(uint32_t)> stringBufferFactory)
    : TypedEncoding<std::string_view, std::string_view>{pool, data},
      row_{0},
      buffer_{&pool} {
  auto pos = data.data() + kDataCompressionOffset;
  auto dataCompressionType =
      static_cast<CompressionType>(encoding::readChar(pos));
  auto lengthsSize = encoding::readUint32(pos);
  lengths_ =
      EncodingFactory().create(pool, {pos, lengthsSize}, stringBufferFactory);
  blob_ = pos + lengthsSize;

  if (dataCompressionType != CompressionType::Uncompressed) {
    dataUncompressed_ = Compression::uncompress(
        pool,
        dataCompressionType,
        DataType::String,
        {blob_, static_cast<size_t>(data.end() - blob_)},
        /*decompressCounter=*/nullptr);
    blob_ = dataUncompressed_->as<char>();
    uncompressedDataBytes_ = dataUncompressed_->size();
  } else {
    uncompressedDataBytes_ = data.size() - std::distance(blob_, data.data());
  }
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
    Buffer& buffer) {
  const uint32_t valueCount = values.size();
  std::vector<uint32_t> lengths;
  lengths.reserve(valueCount);
  for (auto value : values) {
    lengths.push_back(value.size());
  }

  Buffer tempBuffer{buffer.getMemoryPool()};
  std::string_view serializedLengths =
      selection.template encodeNested<uint32_t>(
          EncodingIdentifiers::Trivial::Lengths, {lengths}, tempBuffer);

  auto dataCompressionPolicy = selection.compressionPolicy();
  auto uncompressedSize = selection.statistics().totalStringsLength();

  Vector<char> vector{&buffer.getMemoryPool()};

  CompressionEncoder<std::string_view> compressionEncoder{
      buffer.getMemoryPool(),
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

  const uint32_t encodingSize = EncodingPrefix::kFixedPrefixSize +
      TrivialEncoding<std::string_view>::kPrefixSize +
      serializedLengths.size() + compressionEncoder.getSize();

  char* reserved = buffer.reserve(encodingSize);
  char* pos = reserved;
  Encoding::serializePrefix(
      EncodingType::Trivial, DataType::String, valueCount, false, pos);
  encoding::writeChar(
      static_cast<char>(compressionEncoder.compressionType()), pos);
  encoding::writeUint32(serializedLengths.size(), pos);
  encoding::writeBytes(serializedLengths, pos);
  compressionEncoder.write(pos);

  NIMBLE_CHECK_EQ(pos - reserved, encodingSize, "Encoding size mismatch.");
  return {reserved, encodingSize};
}

TrivialEncoding<bool>::TrivialEncoding(
    velox::memory::MemoryPool& pool,
    std::string_view data,
    std::function<void*(uint32_t)> /* stringBufferFactory */)
    : TypedEncoding<bool, bool>{pool, data},
      bitmap_{data.data() + kDataOffset} {
  const auto compressionType =
      static_cast<CompressionType>(data[kCompressionTypeOffset]);
  if (compressionType != CompressionType::Uncompressed) {
    uncompressed_ = Compression::uncompress(
        pool,
        compressionType,
        DataType::Undefined,
        {bitmap_, static_cast<size_t>(data.end() - bitmap_)},
        /*decompressCounter=*/nullptr);
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
    Buffer& buffer) {
  const uint32_t valueCount = values.size();
  const uint32_t bitmapBytes = FixedBitArray::bufferSize(valueCount, 1);

  Vector<char> vector{&buffer.getMemoryPool()};

  auto dataCompressionPolicy = selection.compressionPolicy();
  CompressionEncoder<std::string_view> compressionEncoder{
      buffer.getMemoryPool(),
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

  const uint32_t encodingSize = kDataOffset + compressionEncoder.getSize();
  char* reserved = buffer.reserve(encodingSize);
  char* pos = reserved;
  Encoding::serializePrefix(
      EncodingType::Trivial, DataType::Bool, valueCount, false, pos);
  encoding::writeChar(
      static_cast<char>(compressionEncoder.compressionType()), pos);
  compressionEncoder.write(pos);

  NIMBLE_CHECK_EQ(pos - reserved, encodingSize, "Encoding size mismatch.");
  return {reserved, encodingSize};
}

} // namespace facebook::nimble::legacy
