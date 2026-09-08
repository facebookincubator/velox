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

#include <array>
#include <span>
#include <string_view>
#include <type_traits>
#include <vector>

#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/common/FixedBitArray.h"
#include "velox/dwio/nimble/common/Varint.h"
#include "velox/dwio/nimble/common/Vector.h"
#include "velox/dwio/nimble/compression/Compression.h"
#include "velox/dwio/nimble/encodings/TrivialEncoding.h"
#include "velox/dwio/nimble/encodings/common/Encoding.h"
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/common/EncodingLayout.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrimitives.h"
#include "velox/dwio/nimble/encodings/common/EncodingType.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelection.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelectionPolicy.h"

// Frame of Reference encoding. Divides data into fixed-size frames, storing
// a reference value per frame and bit-packing the exceptions (value - ref).
// Stores per-frame bit offsets for O(1) random access.
//
// Data layout is:
// Standard Encoding prefix
// 1 byte: compression type
// varint: frame size
// varint: number of frames
// varint: first frame rows
// varint + XX bytes: BitWidths array encoding (nested)
// varint + YY bytes: References array encoding (nested)
// varint + ZZ bytes: BitOffsets array encoding (nested)
// varint + WW bytes: bit-packed exceptions data

namespace facebook::nimble {

template <typename T>
class ForEncoding final
    : public TypedEncoding<T, typename TypeTraits<T>::physicalType> {
 public:
  using cppDataType = T;
  using physicalType = typename TypeTraits<T>::physicalType;

  static const int kCompressionOffset = Encoding::kPrefixSize;

  ForEncoding(
      velox::memory::MemoryPool& memoryPool,
      std::string_view data,
      std::function<void*(uint32_t)> stringBufferFactory = nullptr,
      const Encoding::Options& options = {});

  ~ForEncoding() override {
    this->releaseBuffer(uncompressedData_);
  }

  void reset() final;
  void skip(uint32_t rowCount) final;
  void materialize(uint32_t rowCount, void* buffer) final;

  template <typename DecoderVisitor>
  void readWithVisitor(DecoderVisitor& visitor, ReadWithVisitorParams& params);

  static std::string_view encode(
      EncodingSelection<physicalType>& selection,
      std::span<const physicalType> values,
      Buffer& buffer,
      const Encoding::Options& options = {});

  static std::string_view slice(
      std::string_view encoded,
      uint32_t offset,
      uint32_t length,
      Buffer& buffer,
      const Encoding::Options& options = {});

  std::string debugString(int offset) const final;

 private:
  struct FrameInfo {
    physicalType reference;
    uint8_t bitWidth;
    uint64_t bitOffset;
    uint32_t size;
  };

  // Counts frames when the first frame may be shorter than frameSize.
  static uint32_t
  numFrames(uint32_t rowCount, uint32_t frameSize, uint32_t firstFrameRows) {
    if (rowCount <= firstFrameRows) {
      return 1;
    }
    return 1 + velox::bits::divRoundUp(rowCount - firstFrameRows, frameSize);
  }

  // Parsed stream metadata shared by normal decode and native slice.
  struct Header {
    Header() = default;

    Header(
        uint32_t rowCount,
        uint32_t frameSize,
        uint32_t numFrames,
        uint32_t firstFrameRows,
        CompressionType compressionType = CompressionType::Uncompressed)
        : rowCount{rowCount},
          compressionType{compressionType},
          frameSize{frameSize},
          numFrames{numFrames},
          firstFrameRows{firstFrameRows} {
      NIMBLE_CHECK_GT(frameSize, 0, "FOR frame size must be positive.");
      NIMBLE_CHECK_GT(numFrames, 0, "FOR stream must contain frames.");
      NIMBLE_CHECK_GT(firstFrameRows, 0, "FOR first frame must contain rows.");
      NIMBLE_CHECK_LE(
          firstFrameRows,
          frameSize,
          "FOR first frame cannot exceed frame size.");
      NIMBLE_CHECK_EQ(
          numFrames,
          ForEncoding<T>::numFrames(rowCount, frameSize, firstFrameRows),
          "FOR frame count mismatch.");

      const auto lastFrameOffset = numFrames == 1
          ? uint32_t{0}
          : firstFrameRows + (numFrames - 2) * frameSize;
      lastFrameRows = rowCount - lastFrameOffset;
    }

    // Logical rows represented by this FOR stream.
    uint32_t rowCount{0};
    CompressionType compressionType{CompressionType::Uncompressed};
    // Target frame size for full interior frames.
    uint32_t frameSize{0};
    // Number of frame metadata entries and packed frame segments.
    uint32_t numFrames{0};
    // Sliced streams may start with a partial frame.
    uint32_t firstFrameRows{0};
    // Regular streams only have a partial final frame; sliced streams may have
    // both partial first and last frames.
    uint32_t lastFrameRows{0};
    // Serialized child streams used by encode and native slice output.
    std::string_view serializedBitWidths;
    std::string_view serializedReferences;
    std::string_view serializedBitOffsets;
    // Byte length of the packed residual payload.
    uint32_t packedDataSize{0};
  };

  static std::string_view readVarintString(const char*& pos) {
    const uint32_t size = varint::readVarint32(&pos);
    std::string_view result{pos, size};
    pos += size;
    return result;
  }

  static Header parseHeader(
      std::string_view encoded,
      const Encoding::Options& options,
      const char*& pos) {
    pos = encoded.data() +
        EncodingPrefix::prefixSize(encoded, options.useVarintRowCount);
    const auto compressionType =
        static_cast<CompressionType>(encoding::readChar(pos));
    const auto frameSize = varint::readVarint32(&pos);
    const auto numFrames = varint::readVarint32(&pos);
    const auto firstFrameRows = varint::readVarint32(&pos);
    Header header{
        EncodingPrefix::readRowCount(encoded, options.useVarintRowCount),
        frameSize,
        numFrames,
        firstFrameRows,
        compressionType};
    header.serializedBitWidths = readVarintString(pos);
    header.serializedReferences = readVarintString(pos);
    header.serializedBitOffsets = readVarintString(pos);
    header.packedDataSize = varint::readVarint32(&pos);
    return header;
  }

  static uint32_t headerSize(const Header& header) {
    return sizeof(uint8_t) /*compressionType*/ +
        varint::varintSize(header.frameSize) +
        varint::varintSize(header.numFrames) +
        varint::varintSize(header.firstFrameRows) +
        varint::varintSize(header.serializedBitWidths.size()) +
        header.serializedBitWidths.size() +
        varint::varintSize(header.serializedReferences.size()) +
        header.serializedReferences.size() +
        varint::varintSize(header.serializedBitOffsets.size()) +
        header.serializedBitOffsets.size() +
        varint::varintSize(header.packedDataSize) + header.packedDataSize;
  }

  template <typename MetadataType>
  static std::string_view encodeMetadata(
      std::span<const MetadataType> values,
      Buffer& buffer,
      const Encoding::Options& options) {
    EncodingSelectionResult result{.encodingType = EncodingType::Trivial};
    EncodingSelection<MetadataType> selection{
        std::move(result), Statistics<MetadataType>::create(values), nullptr};
    return TrivialEncoding<MetadataType>::encode(
        selection, values, buffer, options);
  }

  // FOR stores residual widths using a fixed set of wire-supported widths.
  // velox::bits::bitsRequired returns the exact bit count; this rounds that
  // count up to the nearest width the FOR payload writer can encode.
  static uint8_t minBitWidth(uint64_t maxValue) {
    if (maxValue == 0) {
      return 1;
    }

    constexpr std::array<uint8_t, 7> kBitWidths = {1, 2, 4, 8, 16, 32, 64};
    const auto bitsNeeded = velox::bits::bitsRequired(maxValue);
    for (const auto width : kBitWidths) {
      if (width >= bitsNeeded) {
        return width;
      }
    }
    return 64;
  }

  static void prepareSliceHeaderMetadata(
      const Header& header,
      uint32_t firstFrame,
      uint32_t numFrames,
      uint32_t offset,
      uint32_t length,
      Header& sliceHeader,
      Vector<uint8_t>& sourceBitWidths,
      Vector<uint64_t>& sourceBitOffsets,
      Vector<uint64_t>& sliceBitOffsets,
      Buffer& buffer,
      const Encoding::Options& options);

  static uint32_t frameRowOffset(const Header& header, uint32_t frameIndex) {
    if (frameIndex == 0) {
      return 0;
    }
    return header.firstFrameRows + (frameIndex - 1) * header.frameSize;
  }

  // Uses cached lastFrameRows because sliced streams may have partial first and
  // last frames.
  static uint32_t frameRowCount(const Header& header, uint32_t frameIndex) {
    if (frameIndex == header.numFrames - 1) {
      return header.lastFrameRows;
    }
    if (frameIndex == 0) {
      return header.firstFrameRows;
    }
    return header.frameSize;
  }

  static uint32_t frameIndex(const Header& header, uint32_t row) {
    if (row < header.firstFrameRows) {
      return 0;
    }
    const auto frameIndex =
        1 + (row - header.firstFrameRows) / header.frameSize;
    NIMBLE_CHECK_LT(frameIndex, header.numFrames);
    return frameIndex;
  }

  static uint32_t rowOffsetInFrame(const Header& header, uint32_t row) {
    if (header.firstFrameRows == header.frameSize) {
      return row % header.frameSize;
    }
    return row - frameRowOffset(header, frameIndex(header, row));
  }

  uint32_t frameIndex(uint32_t row) const {
    return frameIndex(header_, row);
  }

  uint32_t rowOffsetInFrame(uint32_t row) const {
    return rowOffsetInFrame(header_, row);
  }

  void decodeRange(uint32_t startRow, uint32_t rowCount, physicalType* output)
      const;

  physicalType decodeValue(uint32_t row) const;

  Header header_;
  Vector<FrameInfo> frames_;
  const char* packedData_;
  uint32_t currentRow_;
  velox::BufferPtr uncompressedData_;
  Vector<physicalType> buffer_;
};
//
// Implementation
//

template <typename T>
ForEncoding<T>::ForEncoding(
    velox::memory::MemoryPool& memoryPool,
    std::string_view data,
    std::function<void*(uint32_t)> stringBufferFactory,
    const Encoding::Options& options)
    : TypedEncoding<T, physicalType>{memoryPool, data, options},
      frames_{&memoryPool},
      currentRow_(0),
      buffer_{&memoryPool} {
  static_assert(
      std::is_integral_v<physicalType>,
      "ForEncoding only supports integral types");

  const char* pos;
  const EncodingFactory encodingFactory(options);
  const auto nullStringBufferFactory = [](uint32_t /*size*/) -> void* {
    return nullptr;
  };
  const auto& decodeStringBufferFactory =
      stringBufferFactory ? stringBufferFactory : nullStringBufferFactory;

  header_ = parseHeader(data, options, pos);
  NIMBLE_CHECK_EQ(header_.rowCount, this->rowCount_);

  auto bitWidthsEncoding = encodingFactory.create(
      *this->pool_, header_.serializedBitWidths, decodeStringBufferFactory);

  Vector<uint8_t> bitWidths(&memoryPool, header_.numFrames);
  bitWidthsEncoding->materialize(header_.numFrames, bitWidths.data());

  auto referencesEncoding = encodingFactory.create(
      *this->pool_, header_.serializedReferences, decodeStringBufferFactory);

  Vector<physicalType> references(&memoryPool, header_.numFrames);
  referencesEncoding->materialize(header_.numFrames, references.data());

  auto bitOffsetsEncoding = encodingFactory.create(
      *this->pool_, header_.serializedBitOffsets, decodeStringBufferFactory);
  Vector<uint64_t> bitOffsets(&memoryPool, header_.numFrames);
  bitOffsetsEncoding->materialize(header_.numFrames, bitOffsets.data());

  std::string_view packedDataView{pos, header_.packedDataSize};

  if (header_.compressionType != CompressionType::Uncompressed) {
    uncompressedData_ = Compression::uncompress(
        *this->pool_,
        header_.compressionType,
        DataType::Undefined,
        packedDataView,
        options.decompressCounter());
    packedData_ = uncompressedData_->as<char>();
  } else {
    packedData_ = packedDataView.data();
  }

  // Build frame metadata
  frames_.reserve(header_.numFrames);
  for (uint32_t i = 0; i < header_.numFrames; ++i) {
    FrameInfo frame{};
    frame.reference = references[i];
    frame.bitWidth = bitWidths[i];
    frame.bitOffset = bitOffsets[i];

    frame.size = frameRowCount(header_, i);

    frames_.push_back(frame);
  }
}

template <typename T>
void ForEncoding<T>::reset() {
  currentRow_ = 0;
}

template <typename T>
void ForEncoding<T>::skip(uint32_t rowCount) {
  currentRow_ += rowCount;
  NIMBLE_DCHECK(
      currentRow_ <= this->rowCount_, "Skipping past end of encoding");
}

template <typename T>
void ForEncoding<T>::decodeRange(
    uint32_t startRow,
    uint32_t rowCount,
    physicalType* output) const {
  NIMBLE_DCHECK(
      startRow + rowCount <= this->rowCount_, "Decoding past end of encoding");

  // Streaming bit-cursor decoder: reads rowsToDecode values of frame.bitWidth
  // bits from byteCursor at bitOffsetInByte, writing results via
  // decodeException.
  auto decodeBitStream = [](const uint8_t* byteCursor,
                            uint8_t bitOffsetInByte,
                            uint8_t bitWidth,
                            uint32_t rowsToDecode,
                            auto&& decodeException) {
    uint64_t bitBuffer = 0;
    uint32_t bitsInBuffer = 0;

    // Consume the partial first byte if not byte-aligned
    if (bitOffsetInByte != 0) {
      bitBuffer = static_cast<uint8_t>(*byteCursor) >> bitOffsetInByte;
      bitsInBuffer = 8u - bitOffsetInByte;
      ++byteCursor;
    }

    const uint64_t mask = (bitWidth == 64) ? ~0ULL : ((1ULL << bitWidth) - 1);

    for (uint32_t i = 0; i < rowsToDecode; ++i) {
      while (bitsInBuffer < bitWidth) {
        bitBuffer |= static_cast<uint64_t>(static_cast<uint8_t>(*byteCursor))
            << bitsInBuffer;
        ++byteCursor;
        bitsInBuffer += 8;
      }
      decodeException(i, bitBuffer & mask);
      bitBuffer >>= bitWidth;
      bitsInBuffer -= bitWidth;
    }
  };

  uint32_t currentRow = startRow;
  uint32_t outputOffset = 0;
  uint32_t remainingRowCount = rowCount;

  while (remainingRowCount > 0) {
    const uint32_t index = frameIndex(currentRow);
    const auto& frame = frames_[index];
    const uint32_t rowOffset = rowOffsetInFrame(currentRow);
    const uint32_t rowsInFrame = frame.size - rowOffset;
    const uint32_t rowsToDecode = std::min(remainingRowCount, rowsInFrame);

    // Per-frame reference application — captures frame by ref
    auto decodeException = [&](uint32_t i, uint64_t residual) {
      if constexpr (std::is_signed_v<physicalType>) {
        output[outputOffset + i] = static_cast<physicalType>(
            static_cast<int64_t>(residual) +
            static_cast<int64_t>(frame.reference));
      } else {
        output[outputOffset + i] =
            static_cast<physicalType>(residual + frame.reference);
      }
    };

    if (frame.bitWidth == 0) {
      // All values in frame are identical to reference
      std::fill(
          output + outputOffset,
          output + outputOffset + rowsToDecode,
          frame.reference);

    } else {
      const uint64_t bitPosition =
          frame.bitOffset + static_cast<uint64_t>(rowOffset) * frame.bitWidth;
      const uint8_t bitOffsetInByte = static_cast<uint8_t>(bitPosition & 7U);
      const uint8_t* byteCursor =
          reinterpret_cast<const uint8_t*>(packedData_) + (bitPosition / 8);

      if (bitOffsetInByte == 0) {
        // Byte-aligned: use typed loads for power-of-two widths, bit-stream
        // otherwise
        switch (frame.bitWidth) {
          case 8:
            for (uint32_t i = 0; i < rowsToDecode; ++i) {
              decodeException(i, byteCursor[i]);
            }
            break;
          case 16:
            for (uint32_t i = 0; i < rowsToDecode; ++i) {
              uint16_t v;
              std::memcpy(
                  &v, byteCursor + i * sizeof(uint16_t), sizeof(uint16_t));
              decodeException(i, v);
            }
            break;
          case 32:
            for (uint32_t i = 0; i < rowsToDecode; ++i) {
              uint32_t v;
              std::memcpy(
                  &v, byteCursor + i * sizeof(uint32_t), sizeof(uint32_t));
              decodeException(i, v);
            }
            break;
          case 64:
            for (uint32_t i = 0; i < rowsToDecode; ++i) {
              uint64_t v;
              std::memcpy(
                  &v, byteCursor + i * sizeof(uint64_t), sizeof(uint64_t));
              decodeException(i, v);
            }
            break;
          default:
            // Non-power-of-two width, byte-aligned start: bitOffsetInByte == 0
            decodeBitStream(
                byteCursor, 0, frame.bitWidth, rowsToDecode, decodeException);
            break;
        }
      } else {
        // Unaligned: always use the bit-streaming path
        decodeBitStream(
            byteCursor,
            bitOffsetInByte,
            frame.bitWidth,
            rowsToDecode,
            decodeException);
      }
    }

    currentRow += rowsToDecode;
    outputOffset += rowsToDecode;
    remainingRowCount -= rowsToDecode;
  }
}

template <typename T>
typename ForEncoding<T>::physicalType ForEncoding<T>::decodeValue(
    uint32_t row) const {
  physicalType value{};
  decodeRange(row, 1, &value);
  return value;
}

template <typename T>
void ForEncoding<T>::materialize(uint32_t rowCount, void* buffer) {
  physicalType* output = static_cast<physicalType*>(buffer);

  decodeRange(currentRow_, rowCount, output);

  currentRow_ += rowCount;
}

template <typename T>
template <typename V>
void ForEncoding<T>::readWithVisitor(
    V& visitor,
    ReadWithVisitorParams& params) {
  // Provide a skip function so readWithVisitorSlow can advance currentRow_ past
  // non-selected non-null rows when the visitor's row set is sparse (e.g. due
  // to chunk-index seeks or filter-driven row group skipping). The skip lambda
  // receives the count of non-null rows to bypass, matching how ForEncoding
  // stores only non-null values sequentially.  Using currentRow_ keeps state
  // consistent across multiple readWithVisitor calls and with interleaved
  // skip()/materialize() calls on the same instance.
  detail::readWithVisitorSlow(
      visitor,
      params,
      [&](auto numNonNullsToSkip) { skip(numNonNullsToSkip); },
      [&] { return decodeValue(currentRow_++); });
}

template <typename T>
std::string ForEncoding<T>::debugString(int offset) const {
  std::string log = Encoding::debugString(offset);
  log += fmt::format(
      "\n{}frameSize={}, numFrames={}",
      std::string(offset, ' '),
      header_.frameSize,
      header_.numFrames);

  std::map<uint8_t, uint32_t> bitWidthCounts;
  for (const auto& frame : frames_) {
    bitWidthCounts[frame.bitWidth]++;
  }

  log +=
      fmt::format("\n{}bitWidth distribution: ", std::string(offset + 2, ' '));
  for (const auto& [width, count] : bitWidthCounts) {
    log += fmt::format("{}b:{} ", width, count);
  }

  return log;
}

// Static encode method
template <typename T>
std::string_view ForEncoding<T>::encode(
    EncodingSelection<physicalType>& selection,
    std::span<const physicalType> values,
    Buffer& buffer,
    const Encoding::Options& options) {
  static_assert(
      std::is_integral_v<physicalType>,
      "ForEncoding only supports integral types");

  const bool useVarint = options.useVarintRowCount;
  const uint32_t rowCount = values.size();
  NIMBLE_CHECK_GT(rowCount, 0, "Cannot encode empty data with ForEncoding.");

  // Default frame size (TODO: adaptive selection)
  const uint32_t frameSize = 128;

  const auto firstFrameRows = std::min(rowCount, frameSize);
  Header header{
      rowCount,
      frameSize,
      ForEncoding<T>::numFrames(rowCount, frameSize, firstFrameRows),
      firstFrameRows};

  auto* pool = &buffer.getMemoryPool();
  Vector<uint8_t> bitWidths(pool, header.numFrames);
  Vector<physicalType> references(pool, header.numFrames);
  Vector<uint64_t> bitOffsets(pool, header.numFrames);

  Vector<char> packedData(pool);
  uint64_t bitBuffer = 0;
  size_t bitBufferLen = 0;

  auto writeBits = [&](uint64_t value, uint8_t numBits) {
    size_t bitsToWrite = numBits;

    while (bitsToWrite > 0) {
      size_t spaceInBuffer = 64 - bitBufferLen;
      size_t bitsThisRound = std::min(bitsToWrite, spaceInBuffer);

      uint64_t mask =
          (bitsThisRound == 64) ? ~0ULL : ((1ULL << bitsThisRound) - 1);
      uint64_t chunk = value & mask;

      bitBuffer |= (chunk << bitBufferLen);
      bitBufferLen += bitsThisRound;

      // Shifting by the full width (64) is undefined behavior, so guard it the
      // same way as the mask computation above.
      value = (bitsThisRound == 64) ? 0 : (value >> bitsThisRound);
      bitsToWrite -= bitsThisRound;

      while (bitBufferLen >= 8) {
        packedData.push_back(static_cast<char>(bitBuffer & 0xFF));
        bitBuffer >>= 8;
        bitBufferLen -= 8;
      }
    }
  };

  uint64_t totalBits = 0;

  for (uint32_t frameIdx = 0; frameIdx < header.numFrames; ++frameIdx) {
    uint32_t frameStart = frameRowOffset(header, frameIdx);
    uint32_t frameLength = frameRowCount(header, frameIdx);
    uint32_t frameEnd = frameStart + frameLength;

    physicalType minValue = values[frameStart];
    physicalType maxValue = values[frameStart];

    for (uint32_t i = frameStart; i < frameEnd; ++i) {
      minValue = std::min(minValue, values[i]);
      maxValue = std::max(maxValue, values[i]);
    }

    uint64_t maxException = 0;
    if constexpr (std::is_signed_v<physicalType>) {
      maxException = static_cast<uint64_t>(
          static_cast<int64_t>(maxValue) - static_cast<int64_t>(minValue));
    } else {
      maxException = static_cast<uint64_t>(maxValue - minValue);
    }

    const uint8_t bitWidth = minBitWidth(maxException);

    references[frameIdx] = minValue;
    bitWidths[frameIdx] = bitWidth;

    bitOffsets[frameIdx] = totalBits;
    for (uint32_t i = frameStart; i < frameEnd; ++i) {
      uint64_t exception;
      if constexpr (std::is_signed_v<physicalType>) {
        exception = static_cast<uint64_t>(
            static_cast<int64_t>(values[i]) - static_cast<int64_t>(minValue));
      } else {
        exception = static_cast<uint64_t>(values[i] - minValue);
      }
      writeBits(exception, bitWidth);
    }

    totalBits += frameLength * bitWidth;
  }

  if (bitBufferLen > 0) {
    packedData.push_back(static_cast<char>(bitBuffer & 0xFF));
  }

  ScopedEncodingBuffer scopedBuffer{pool, options.encodingBufferPool};

  header.serializedBitWidths = selection.template encodeNested<uint8_t>(
      EncodingIdentifiers::For::BitWidths,
      {bitWidths},
      scopedBuffer.get(),
      options);

  header.serializedReferences = selection.template encodeNested<physicalType>(
      EncodingIdentifiers::For::References,
      {references},
      scopedBuffer.get(),
      options);

  header.serializedBitOffsets = selection.template encodeNested<uint64_t>(
      EncodingIdentifiers::For::BitOffsets,
      {bitOffsets},
      scopedBuffer.get(),
      options);

  auto dataCompressionPolicy = selection.compressionPolicy();
  CompressionEncoder<char> compressionEncoder{
      *pool,
      *dataCompressionPolicy,
      DataType::Undefined,
      /*bitWidth=*/8,
      static_cast<uint32_t>(packedData.size()),
      [&]() { return std::span<char>{packedData}; },
      [&](char*& pos) {
        std::memcpy(pos, packedData.data(), packedData.size());
        pos += packedData.size();
        return pos;
      }};
  header.compressionType = compressionEncoder.compressionType();
  header.packedDataSize = compressionEncoder.getSize();

  uint32_t encodingSize =
      Encoding::serializePrefixSize(rowCount, useVarint) + headerSize(header);

  char* reserved = buffer.reserve(encodingSize);
  char* pos = reserved;

  Encoding::serializePrefix(
      EncodingType::FOR, TypeTraits<T>::dataType, rowCount, useVarint, pos);

  encoding::writeChar(static_cast<char>(header.compressionType), pos);
  varint::writeVarint(header.frameSize, &pos);
  varint::writeVarint(header.numFrames, &pos);
  varint::writeVarint(header.firstFrameRows, &pos);

  encoding::writeVarintString(header.serializedBitWidths, pos);
  encoding::writeVarintString(header.serializedReferences, pos);

  encoding::writeVarintString(header.serializedBitOffsets, pos);

  varint::writeVarint(header.packedDataSize, &pos);
  compressionEncoder.write(pos);

  NIMBLE_DCHECK_EQ(encodingSize, pos - reserved, "Encoding size mismatch.");
  return {reserved, encodingSize};
}

template <typename T>
void ForEncoding<T>::prepareSliceHeaderMetadata(
    const Header& header,
    uint32_t firstFrame,
    uint32_t numFrames,
    uint32_t offset,
    uint32_t length,
    Header& sliceHeader,
    Vector<uint8_t>& sourceBitWidths,
    Vector<uint64_t>& sourceBitOffsets,
    Vector<uint64_t>& sliceBitOffsets,
    Buffer& buffer,
    const Encoding::Options& options) {
  const auto noStringBufferFactory = [](uint32_t /*size*/) -> void* {
    return nullptr;
  };
  sliceHeader.serializedBitWidths = EncodingFactory::slice(
      header.serializedBitWidths, firstFrame, numFrames, buffer, options);
  sliceHeader.serializedReferences = EncodingFactory::slice(
      header.serializedReferences, firstFrame, numFrames, buffer, options);
  const auto sourceSerializedBitOffsets = EncodingFactory::slice(
      header.serializedBitOffsets, firstFrame, numFrames, buffer, options);

  // Bit widths and references are frame-local child streams, so the slice keeps
  // the source encoding layout for the contiguous frame range. Bit offsets
  // point into the packed payload and must be regenerated because the sliced
  // payload starts at bit offset 0.
  EncodingFactory encodingFactory{options};
  auto bitWidthsEncoding = encodingFactory.create(
      buffer.getMemoryPool(),
      sliceHeader.serializedBitWidths,
      noStringBufferFactory);
  bitWidthsEncoding->materialize(numFrames, sourceBitWidths.data());

  auto bitOffsetsEncoding = encodingFactory.create(
      buffer.getMemoryPool(),
      sourceSerializedBitOffsets,
      noStringBufferFactory);
  bitOffsetsEncoding->materialize(numFrames, sourceBitOffsets.data());

  uint64_t totalBits{0};
  uint32_t currentRow{offset};
  const uint32_t endRow = offset + length;
  for (uint32_t sliceFrame = 0; sliceFrame < numFrames; ++sliceFrame) {
    const auto sourceFrame = firstFrame + sliceFrame;
    const auto rowOffset = currentRow - frameRowOffset(header, sourceFrame);
    const auto rowCount = std::min<uint32_t>(
        endRow - currentRow, frameRowCount(header, sourceFrame) - rowOffset);

    sliceBitOffsets[sliceFrame] = totalBits;
    totalBits += static_cast<uint64_t>(rowCount) * sourceBitWidths[sliceFrame];
    currentRow += rowCount;
  }
  NIMBLE_CHECK_EQ(currentRow, endRow, "FOR slice frame planning mismatch.");

  sliceHeader.serializedBitOffsets =
      EncodingFactory::encodeWithCapturedLayout<uint64_t>(
          header.serializedBitOffsets,
          {sliceBitOffsets.data(), sliceBitOffsets.size()},
          buffer,
          options,
          "Captured FOR bit offset layout");
  sliceHeader.packedDataSize = velox::bits::nbytes(totalBits);
}

template <typename T>
std::string_view ForEncoding<T>::slice(
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

  auto* pool = &buffer.getMemoryPool();
  ScopedEncodingBuffer scopedBuffer{pool, options.encodingBufferPool};
  const char* packedDataPos{nullptr};
  const auto header = parseHeader(encoded, options, packedDataPos);
  NIMBLE_CHECK_EQ(header.rowCount, sourceRowCount);
  const auto firstFrame = frameIndex(header, offset);
  const auto lastFrame = frameIndex(header, offset + length - 1);
  const auto numFrames = lastFrame - firstFrame + 1;
  const auto firstFrameRows = std::min<uint32_t>(
      length,
      frameRowCount(header, firstFrame) - rowOffsetInFrame(header, offset));
  Header sliceHeader{length, header.frameSize, numFrames, firstFrameRows};

  ScopedVector<uint8_t> sourceBitWidths{numFrames, pool, options.bufferPool};
  ScopedVector<uint64_t> sourceBitOffsets{numFrames, pool, options.bufferPool};
  ScopedVector<uint64_t> sliceBitOffsets{numFrames, pool, options.bufferPool};
  prepareSliceHeaderMetadata(
      header,
      firstFrame,
      numFrames,
      offset,
      length,
      sliceHeader,
      sourceBitWidths,
      sourceBitOffsets,
      sliceBitOffsets,
      scopedBuffer.get(),
      options);

  std::string_view packedData{packedDataPos, header.packedDataSize};
  velox::BufferPtr uncompressedData;
  if (header.compressionType != CompressionType::Uncompressed) {
    uncompressedData = Compression::uncompress(
        *pool,
        header.compressionType,
        DataType::Undefined,
        packedData,
        options.decompressCounter());
    packedData = {uncompressedData->as<char>(), uncompressedData->size()};
  }

  uint32_t currentRow{offset};
  const uint32_t endRow = offset + length;
  const bool useVarint = options.useVarintRowCount;
  const uint32_t encodingSize =
      Encoding::serializePrefixSize(length, useVarint) +
      headerSize(sliceHeader);

  char* reserved = buffer.reserve(encodingSize);
  char* pos = reserved;
  Encoding::serializePrefix(
      EncodingType::FOR, TypeTraits<T>::dataType, length, useVarint, pos);
  encoding::writeChar(static_cast<char>(CompressionType::Uncompressed), pos);
  varint::writeVarint(sliceHeader.frameSize, &pos);
  varint::writeVarint(sliceHeader.numFrames, &pos);
  varint::writeVarint(sliceHeader.firstFrameRows, &pos);
  encoding::writeVarintString(sliceHeader.serializedBitWidths, pos);
  encoding::writeVarintString(sliceHeader.serializedReferences, pos);
  encoding::writeVarintString(sliceHeader.serializedBitOffsets, pos);
  varint::writeVarint(sliceHeader.packedDataSize, &pos);

  std::memset(pos, 0, sliceHeader.packedDataSize);
  currentRow = offset;
  for (uint32_t sliceFrame = 0; sliceFrame < numFrames; ++sliceFrame) {
    const auto sourceFrame = firstFrame + sliceFrame;
    const auto rowOffset = currentRow - frameRowOffset(header, sourceFrame);
    const auto rowCount = std::min<uint32_t>(
        endRow - currentRow, frameRowCount(header, sourceFrame) - rowOffset);
    const auto bitWidth = sourceBitWidths[sliceFrame];
    const auto bitCount = static_cast<uint64_t>(rowCount) * bitWidth;
    if (bitCount > 0) {
      velox::bits::copyBits(
          reinterpret_cast<const uint64_t*>(packedData.data()),
          sourceBitOffsets[sliceFrame] +
              static_cast<uint64_t>(rowOffset) * bitWidth,
          reinterpret_cast<uint64_t*>(pos),
          sliceBitOffsets[sliceFrame],
          bitCount);
    }
    currentRow += rowCount;
  }
  pos += sliceHeader.packedDataSize;

  NIMBLE_CHECK_EQ(encodingSize, pos - reserved, "Encoding size mismatch.");
  return {reserved, encodingSize};
}

} // namespace facebook::nimble
