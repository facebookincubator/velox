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
#include <cstdint>
#include <cstring>

#include "velox/dwio/nimble/common/Varint.h"
#include "velox/dwio/nimble/common/Vector.h"
#include "velox/dwio/nimble/encodings/HuffmanEncoding.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrimitives.h"
#include "velox/dwio/nimble/encodings/views/EncodingView.h"

namespace facebook::nimble {

template <typename T>
class HuffmanEncodingView final : public TypedEncodingView<T> {
 public:
  using physicalType = typename TypedEncodingView<T>::physicalType;

  HuffmanEncodingView(
      std::string_view data,
      velox::memory::MemoryPool* pool,
      const Encoding::Options& options)
      : TypedEncodingView<T>{data, pool, options},
        alphabet_{this->template getVectorBuffer<physicalType>()},
        decodeTable_{this->template getVectorBuffer<DecodeEntry>()},
        checkpoints_{this->template getVectorBuffer<uint32_t>()} {
    static_assert(
        std::is_integral_v<physicalType> &&
        !std::is_same_v<physicalType, bool>);
    NIMBLE_CHECK_EQ(this->encodingType_, EncodingType::Huffman);

    const char* pos = data.data() + this->dataOffset_;
    const uint32_t symbolCount = varint::readVarint32(&pos);
    tableLog_ = static_cast<uint8_t>(encoding::readChar(pos));
    NIMBLE_CHECK_GE(symbolCount, 2);
    NIMBLE_CHECK_LE(symbolCount, HuffmanEncoding<T>::kMaxSymbols);
    NIMBLE_CHECK_LE(tableLog_, HuffmanEncoding<T>::kMaxCodeBits);

    alphabet_.resize(symbolCount);
    std::memcpy(alphabet_.data(), pos, symbolCount * sizeof(physicalType));
    pos += symbolCount * sizeof(physicalType);

    const auto* lengths = reinterpret_cast<const uint8_t*>(pos);
    pos += symbolCount;
    decodeTable_.resize(1u << tableLog_);
    std::fill(decodeTable_.begin(), decodeTable_.end(), DecodeEntry{});

    std::array<uint16_t, HuffmanEncoding<T>::kMaxCodeBits + 1> counts{};
    for (uint32_t symbol = 0; symbol < symbolCount; ++symbol) {
      NIMBLE_CHECK_GT(lengths[symbol], 0);
      NIMBLE_CHECK_LE(lengths[symbol], tableLog_);
      ++counts[lengths[symbol]];
    }
    std::array<uint16_t, HuffmanEncoding<T>::kMaxCodeBits + 1> nextCode{};
    uint16_t code = 0;
    for (uint8_t bits = 1; bits <= tableLog_; ++bits) {
      code = static_cast<uint16_t>((code + counts[bits - 1]) << 1);
      nextCode[bits] = code;
    }
    for (uint32_t symbol = 0; symbol < symbolCount; ++symbol) {
      const uint8_t bits = lengths[symbol];
      const uint16_t reversed = reverseBits(nextCode[bits]++, bits);
      for (uint32_t slot = reversed; slot < decodeTable_.size();
           slot += 1u << bits) {
        decodeTable_[slot] = DecodeEntry{static_cast<uint16_t>(symbol), bits};
      }
    }

    const uint32_t checkpointCount = varint::readVarint32(&pos);
    NIMBLE_CHECK_EQ(
        checkpointCount,
        velox::bits::divRoundUp(
            this->rowCount_, HuffmanEncoding<T>::kCheckpointStride));
    checkpoints_.resize(checkpointCount);
    uint32_t bitOffset = 0;
    for (uint32_t checkpoint = 0; checkpoint < checkpointCount; ++checkpoint) {
      bitOffset += varint::readVarint32(&pos);
      checkpoints_[checkpoint] = bitOffset;
    }
    bitstreamBytes_ = varint::readVarint32(&pos);
    bitstream_ = reinterpret_cast<const uint8_t*>(pos);
    NIMBLE_CHECK_EQ(pos + bitstreamBytes_, data.end());
  }

  ~HuffmanEncodingView() override {
    this->releaseVectorBuffer(checkpoints_);
    this->releaseVectorBuffer(decodeTable_);
    this->releaseVectorBuffer(alphabet_);
  }

 private:
  struct DecodeEntry {
    uint16_t symbol{0};
    uint8_t bits{0};
  };

  static uint16_t reverseBits(uint16_t value, uint8_t bits) {
    uint16_t reversed = 0;
    for (uint8_t i = 0; i < bits; ++i) {
      reversed = static_cast<uint16_t>((reversed << 1) | (value & 1));
      value >>= 1;
    }
    return reversed;
  }

  T readTypedAt(uint32_t index) const final {
    NIMBLE_CHECK_LT(index, this->rowCount_);
    const uint32_t checkpoint = index / HuffmanEncoding<T>::kCheckpointStride;
    uint32_t bitOffset = checkpoints_[checkpoint];
    uint32_t current = checkpoint * HuffmanEncoding<T>::kCheckpointStride;
    while (current <= index) {
      uint32_t lookahead = 0;
      const uint32_t byteOffset = bitOffset >> 3;
      NIMBLE_CHECK_LT(byteOffset, bitstreamBytes_);
      const uint32_t available =
          std::min<uint32_t>(4, bitstreamBytes_ - byteOffset);
      std::memcpy(&lookahead, bitstream_ + byteOffset, available);
      lookahead >>= bitOffset & 7;
      const auto entry = decodeTable_[lookahead & ((1u << tableLog_) - 1)];
      NIMBLE_CHECK_GT(entry.bits, 0);
      bitOffset += entry.bits;
      if (current++ == index) {
        return detail::castFromPhysicalType<T>(alphabet_[entry.symbol]);
      }
    }
    NIMBLE_UNREACHABLE("Invalid Huffman row {}", index);
  }

  void readPhysical(uint32_t offset, uint32_t length, physicalType* output)
      const final {
    this->checkReadRange(offset, length);
    if (length == 0) {
      return;
    }

    const uint32_t checkpoint = offset / HuffmanEncoding<T>::kCheckpointStride;
    uint32_t bitOffset = checkpoints_[checkpoint];
    uint32_t current = checkpoint * HuffmanEncoding<T>::kCheckpointStride;
    const auto end = offset + length;
    while (current < end) {
      uint32_t lookahead = 0;
      const uint32_t byteOffset = bitOffset >> 3;
      NIMBLE_CHECK_LT(byteOffset, bitstreamBytes_);
      const uint32_t available =
          std::min<uint32_t>(4, bitstreamBytes_ - byteOffset);
      std::memcpy(&lookahead, bitstream_ + byteOffset, available);
      lookahead >>= bitOffset & 7;
      const auto entry = decodeTable_[lookahead & ((1u << tableLog_) - 1)];
      NIMBLE_CHECK_GT(entry.bits, 0);
      bitOffset += entry.bits;
      if (current >= offset) {
        output[current - offset] = alphabet_[entry.symbol];
      }
      ++current;
    }
  }

  Vector<physicalType> alphabet_;
  Vector<DecodeEntry> decodeTable_;
  Vector<uint32_t> checkpoints_;
  const uint8_t* bitstream_{nullptr};
  uint32_t bitstreamBytes_{0};
  uint8_t tableLog_{0};
};

} // namespace facebook::nimble
