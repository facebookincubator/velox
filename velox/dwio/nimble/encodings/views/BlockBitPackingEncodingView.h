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
#include <cstring>
#include <type_traits>
#include <vector>

#include "velox/common/base/BitUtil.h"
#include "velox/dwio/nimble/common/FixedBitArray.h"
#include "velox/dwio/nimble/common/Varint.h"
#include "velox/dwio/nimble/common/Vector.h"
#include "velox/dwio/nimble/encodings/BlockBitPackingEncoding.h"
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrimitives.h"
#include "velox/dwio/nimble/encodings/views/EncodingView.h"

namespace facebook::nimble {

template <typename T>
class BlockBitPackingEncodingView final : public TypedEncodingView<T> {
 public:
  using physicalType = typename TypedEncodingView<T>::physicalType;

  BlockBitPackingEncodingView(
      std::string_view data,
      velox::memory::MemoryPool* pool,
      const Encoding::Options& options)
      : TypedEncodingView<T>{data, pool, options},
        blocks_{this->template getVectorBuffer<BlockMeta>()},
        blockRowOffsets_{this->template getVectorBuffer<uint32_t>()} {
    NIMBLE_CHECK_EQ(this->encodingType_, EncodingType::BlockBitPacking);
    const auto source = BlockBitPackingEncoding<T>::parseHeader(data, options);
    NIMBLE_CHECK_EQ(
        source.compressionType,
        CompressionType::Uncompressed,
        "EncodingView does not support compressed BlockBitPacking streams.");
    NIMBLE_CHECK_GT(source.blockSize, 0);
    NIMBLE_CHECK_GT(source.numBlocks, 0);
    blockSize_ = source.blockSize;
    firstBlockRows_ = source.firstBlockRows;
    numBlocks_ = source.numBlocks;

    auto noStringBufferFactory = [](uint32_t) -> void* { return nullptr; };
    auto baselinesEncoding = EncodingFactory().create(
        *this->pool_, source.encodedBaselines, noStringBufferFactory, options);
    NIMBLE_CHECK_NOT_NULL(baselinesEncoding);
    auto baselines = this->template getVectorBuffer<physicalType>();
    baselines.resize(source.numBlocks);
    baselinesEncoding->materialize(source.numBlocks, baselines.data());

    auto bitWidthsEncoding = EncodingFactory().create(
        *this->pool_, source.encodedBitWidths, noStringBufferFactory, options);
    NIMBLE_CHECK_NOT_NULL(bitWidthsEncoding);
    auto bitWidths = this->template getVectorBuffer<uint8_t>();
    bitWidths.resize(source.numBlocks);
    bitWidthsEncoding->materialize(source.numBlocks, bitWidths.data());

    auto offsetsEncoding = EncodingFactory().create(
        *this->pool_,
        source.encodedBlockOffsets,
        noStringBufferFactory,
        options);
    NIMBLE_CHECK_NOT_NULL(offsetsEncoding);
    auto offsets = this->template getVectorBuffer<uint32_t>();
    offsets.resize(source.numBlocks);
    offsetsEncoding->materialize(source.numBlocks, offsets.data());

    blockRowOffsets_.resize(source.numBlocks + 1);
    uint32_t rowOffset{0};
    blocks_.reserve(source.numBlocks);
    for (uint16_t i = 0; i < source.numBlocks; ++i) {
      blockRowOffsets_[i] = rowOffset;
      const auto rowCount =
          BlockBitPackingEncoding<T>::blockRowCount(source, /*blockIndex=*/i);
      NIMBLE_CHECK_LE(rowCount, source.blockSize);
      rowOffset += rowCount;
      const auto bitWidth = bitWidths[i];
      blocks_.push_back(
          BlockMeta{
              .baseline = baselines[i],
              .bitWidth = bitWidth,
              .offset = offsets[i],
              .rowCount = rowCount,
          });
    }
    blockRowOffsets_[source.numBlocks] = rowOffset;
    NIMBLE_CHECK_EQ(rowOffset, this->rowCount_);
    this->releaseVectorBuffer(offsets);
    this->releaseVectorBuffer(bitWidths);
    this->releaseVectorBuffer(baselines);
    packedData_ = source.packedData.data();
  }

  ~BlockBitPackingEncodingView() override {
    this->releaseVectorBuffer(blockRowOffsets_);
    this->releaseVectorBuffer(blocks_);
  }

 private:
  T readTypedAt(uint32_t index) const final {
    NIMBLE_CHECK_LT(index, this->rowCount_);
    const auto indexBlock = blockIndex(index);
    const auto blockOffset = index - blockRowOffsets_[indexBlock];
    const auto& block = blocks_[indexBlock];
    if (block.bitWidth == BlockBitPackingEncoding<T>::kRawBlockBitWidth) {
      const auto* values =
          reinterpret_cast<const physicalType*>(packedData_ + block.offset);
      return detail::castFromPhysicalType<T>(values[blockOffset]);
    }
    if (block.bitWidth == 0) {
      return detail::castFromPhysicalType<T>(block.baseline);
    }
    const auto numRows = blockRowCount(indexBlock);
    FixedBitArray fba{
        {packedData_ + block.offset,
         FixedBitArray::bufferSize(numRows, block.bitWidth)},
        block.bitWidth};
    return detail::castFromPhysicalType<T>(
        static_cast<physicalType>(fba.get(blockOffset) + block.baseline));
  }

  void readPhysical(uint32_t offset, uint32_t length, physicalType* output)
      const final {
    this->checkReadRange(offset, length);
    uint32_t outputOffset{0};
    while (outputOffset < length) {
      const auto indexBlock = blockIndex(offset);
      const auto blockOffset = offset - blockRowOffsets_[indexBlock];
      const auto count = std::min<uint32_t>(
          length - outputOffset, blocks_[indexBlock].rowCount - blockOffset);
      readBlockRange(indexBlock, blockOffset, count, output + outputOffset);
      outputOffset += count;
      offset += count;
    }
  }

  uint32_t blockRowCount(uint32_t blockIndex) const {
    return blocks_[blockIndex].rowCount;
  }

  uint32_t blockIndex(uint32_t row) const {
    if (row < firstBlockRows_) {
      return 0;
    }
    const auto blockIndex = 1 + (row - firstBlockRows_) / blockSize_;
    NIMBLE_CHECK_LT(blockIndex, numBlocks_);
    return blockIndex;
  }

  void readBlockRange(
      uint32_t blockIndex,
      uint32_t blockOffset,
      uint32_t length,
      physicalType* output) const {
    const auto& block = blocks_[blockIndex];
    const auto bitWidth = block.bitWidth;
    const auto baseline = block.baseline;
    const auto* blockData = packedData_ + block.offset;

    if (bitWidth == BlockBitPackingEncoding<T>::kRawBlockBitWidth) {
      const auto* rawValues = reinterpret_cast<const physicalType*>(blockData);
      std::memcpy(
          output, rawValues + blockOffset, length * sizeof(physicalType));
      return;
    }

    if (bitWidth == 0) {
      std::fill(output, output + length, baseline);
      return;
    }

    const auto* inputBytes = reinterpret_cast<const uint8_t*>(blockData);
    if (blockOffset == 0) {
      BlockBitPackingEncoding<T>::fullUnpack(
          inputBytes, output, length, bitWidth, baseline);
      return;
    }

    readPartialBlockRange(
        inputBytes,
        blockIndex,
        blockOffset,
        length,
        bitWidth,
        baseline,
        output);
  }

  void readPartialBlockRange(
      const uint8_t* input,
      uint32_t blockIndex,
      uint32_t blockOffset,
      uint32_t length,
      uint8_t bitWidth,
      physicalType baseline,
      physicalType* output) const {
    constexpr uint32_t kGroupSize = packingGroupSize();
    const auto numRows = blockRowCount(blockIndex);
    const auto numFullGroupRows = numRows - (numRows % kGroupSize);
    const auto groupBytes = bitWidth * kGroupSize / 8;
    const auto* remainderInput = reinterpret_cast<const char*>(input) +
        (numFullGroupRows / kGroupSize) * groupBytes;

    // Rows after the full fastpack groups are stored as a FixedBitArray tail.
    if (blockOffset >= numFullGroupRows) {
      FixedBitArray fba{
          {remainderInput,
           FixedBitArray::bufferSize(numRows - numFullGroupRows, bitWidth)},
          bitWidth};
      const auto remainderOffset = blockOffset - numFullGroupRows;
      fba.bulkGetWithBaseline(remainderOffset, length, output, baseline);
      return;
    }

    uint32_t outputOffset{0};
    if (const auto groupOffset = blockOffset % kGroupSize; groupOffset != 0) {
      std::array<physicalType, kGroupSize> values;
      const auto group = blockOffset / kGroupSize;
      BlockBitPackingEncoding<T>::fullUnpack(
          input + group * groupBytes,
          values.data(),
          kGroupSize,
          bitWidth,
          baseline);
      const auto count = std::min<uint32_t>(length, kGroupSize - groupOffset);
      std::copy(
          values.data() + groupOffset,
          values.data() + groupOffset + count,
          output);
      outputOffset += count;
      blockOffset += count;
    }

    const auto packedLength = std::min<uint32_t>(
        length - outputOffset, numFullGroupRows - blockOffset);
    if (packedLength != 0) {
      BlockBitPackingEncoding<T>::fullUnpack(
          input + (blockOffset / kGroupSize) * groupBytes,
          output + outputOffset,
          packedLength,
          bitWidth,
          baseline);
      outputOffset += packedLength;
      blockOffset += packedLength;
    }

    if (outputOffset < length) {
      FixedBitArray fba{
          {remainderInput,
           FixedBitArray::bufferSize(numRows - numFullGroupRows, bitWidth)},
          bitWidth};
      fba.bulkGetWithBaseline(
          blockOffset - numFullGroupRows,
          length - outputOffset,
          output + outputOffset,
          baseline);
    }
  }

  static constexpr uint32_t packingGroupSize() {
    if constexpr (sizeof(physicalType) == 1) {
      return 8u;
    } else if constexpr (sizeof(physicalType) == 2) {
      return 16u;
    } else {
      return 32u;
    }
  }

  struct BlockMeta {
    physicalType baseline;
    uint8_t bitWidth;
    uint32_t offset;
    uint32_t rowCount;
  };

  Vector<BlockMeta> blocks_;
  Vector<uint32_t> blockRowOffsets_;
  const char* packedData_{nullptr};
  uint16_t blockSize_{0};
  uint16_t numBlocks_{0};
  uint32_t firstBlockRows_{0};
};

} // namespace facebook::nimble
