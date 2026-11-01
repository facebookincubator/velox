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

#include "velox/common/base/BitUtil.h"
#include "velox/dwio/nimble/common/FixedBitArray.h"
#include "velox/dwio/nimble/common/Varint.h"
#include "velox/dwio/nimble/common/Vector.h"
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrimitives.h"
#include "velox/dwio/nimble/encodings/deprecated/BlockBitPackingEncoding.h"
#include "velox/dwio/nimble/encodings/views/EncodingView.h"
#include "velox/dwio/nimble/encodings/views/EncodingViewFactory.h"

// Reads the first BlockBitPacking layout, whose first block may be shorter
// than the block size.

namespace facebook::nimble::deprecated {

template <typename T>
class BlockBitPackingEncodingView final : public TypedEncodingView<T> {
 public:
  using physicalType = typename TypedEncodingView<T>::physicalType;

  BlockBitPackingEncodingView(const BlockBitPackingEncodingView&) = delete;
  BlockBitPackingEncodingView& operator=(const BlockBitPackingEncodingView&) =
      delete;
  BlockBitPackingEncodingView(BlockBitPackingEncodingView&&) = delete;
  BlockBitPackingEncodingView& operator=(BlockBitPackingEncodingView&&) =
      delete;

  BlockBitPackingEncodingView(
      std::string_view data,
      velox::memory::MemoryPool* pool,
      const Encoding::Options& options)
      : TypedEncodingView<T>{data, pool, options},
        blocks_{this->template getVectorBuffer<BlockMeta>()},
        blockRowOffsets_{this->template getVectorBuffer<uint32_t>()} {
    NIMBLE_CHECK_EQ(this->encodingType_, EncodingType::BlockBitPacking);
    const auto source = BlockBitPackingEncoding<T>::parseHeader(data, options);
    NIMBLE_CHECK_GT(source.blockSize, 0);
    NIMBLE_CHECK_GT(source.numBlocks, 0);
    blockSize_ = source.blockSize;
    firstBlockRows_ = source.firstBlockRows;
    numBlocks_ = source.numBlocks;
    header_ = source;

    // The view keeps the decompressed payload, so it stays valid with the view.
    const auto packedData = this->decompressPayload(
        source.compressionType, DataType::Undefined, source.packedData);
    packedData_ = packedData.data();
    packedSize_ = packedData.size();
    if (options.cacheEncodingView) {
      initializeCached(source);
    } else {
      initializeUncached(source);
    }
  }

  ~BlockBitPackingEncodingView() override {
    this->releaseVectorBuffer(blockRowOffsets_);
    this->releaseVectorBuffer(blocks_);
  }

 private:
  struct BlockMeta {
    physicalType baseline;
    uint8_t bitWidth;
    uint32_t offset;
    uint32_t rowCount;
  };

  // Reads the metadata of every block once, so that reads need no stream
  // access, and checks every block against the packed payload.
  void initializeCached(
      const typename BlockBitPackingEncoding<T>::Header& source) {
    auto noStringBufferFactory = [](uint32_t) -> void* { return nullptr; };
    auto baselinesEncoding = EncodingFactory().create(
        *this->pool_,
        source.encodedBaselines,
        noStringBufferFactory,
        this->options_);
    NIMBLE_CHECK_NOT_NULL(baselinesEncoding);
    auto baselines = this->template getVectorBuffer<physicalType>();
    baselines.resize(source.numBlocks);
    baselinesEncoding->materialize(source.numBlocks, baselines.data());

    auto bitWidthsEncoding = EncodingFactory().create(
        *this->pool_,
        source.encodedBitWidths,
        noStringBufferFactory,
        this->options_);
    NIMBLE_CHECK_NOT_NULL(bitWidthsEncoding);
    auto bitWidths = this->template getVectorBuffer<uint8_t>();
    bitWidths.resize(source.numBlocks);
    bitWidthsEncoding->materialize(source.numBlocks, bitWidths.data());

    auto offsetsEncoding = EncodingFactory().create(
        *this->pool_,
        source.encodedBlockOffsets,
        noStringBufferFactory,
        this->options_);
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
      const BlockMeta block{
          .baseline = baselines[i],
          .bitWidth = bitWidths[i],
          .offset = offsets[i],
          .rowCount = rowCount,
      };
      checkBlockBounds(block);
      blocks_.push_back(block);
    }
    blockRowOffsets_[source.numBlocks] = rowOffset;
    NIMBLE_CHECK_EQ(rowOffset, this->rowCount_);
    this->releaseVectorBuffer(offsets);
    this->releaseVectorBuffer(bitWidths);
    this->releaseVectorBuffer(baselines);
  }

  // Keeps the metadata streams as views, so that a read touches only the
  // metadata of its own block.
  void initializeUncached(
      const typename BlockBitPackingEncoding<T>::Header& source) {
    baselinesView_ = detail::createTypedEncodingView<physicalType>(
        source.encodedBaselines, this->pool_, this->options_);
    NIMBLE_CHECK_NOT_NULL(baselinesView_);
    NIMBLE_CHECK_EQ(baselinesView_->rowCount(), source.numBlocks);
    bitWidthsView_ = detail::createTypedEncodingView<uint8_t>(
        source.encodedBitWidths, this->pool_, this->options_);
    NIMBLE_CHECK_NOT_NULL(bitWidthsView_);
    NIMBLE_CHECK_EQ(bitWidthsView_->rowCount(), source.numBlocks);
    offsetsView_ = detail::createTypedEncodingView<uint32_t>(
        source.encodedBlockOffsets, this->pool_, this->options_);
    NIMBLE_CHECK_NOT_NULL(offsetsView_);
    NIMBLE_CHECK_EQ(offsetsView_->rowCount(), source.numBlocks);
    NIMBLE_CHECK_EQ(source.rowCount, this->rowCount_);
  }

  // Returns whether the view holds the metadata of every block.
  bool cachesBlockMetadata() const {
    return this->options_.cacheEncodingView;
  }

  // Checks that the packed values of a block lie within the packed payload. A
  // corrupt stream can declare a block that extends past it.
  void checkBlockBounds(const BlockMeta& block) const {
    NIMBLE_CHECK_FILE_LE(
        block.offset,
        packedSize_,
        "BlockBitPacking block offset exceeds packed payload");
    const auto blockBytes = BlockBitPackingEncoding<T>::getPackedSize(
        block.rowCount, block.bitWidth);
    NIMBLE_CHECK_FILE_LE(
        blockBytes,
        packedSize_ - block.offset,
        "BlockBitPacking block payload exceeds packed payload");
  }

  T readTypedAt(uint32_t index) const final {
    return detail::castFromPhysicalType<T>(readPhysicalAt(index));
  }

  physicalType readPhysicalAt(uint32_t index) const final {
    NIMBLE_CHECK_LT(index, this->rowCount_);
    const auto indexBlock = blockIndex(index);
    const auto blockOffset = index - rowOffsetOf(indexBlock);
    return readBlockValue(indexBlock, blockOffset);
  }

  physicalType readBlockValue(uint32_t blockIndex, uint32_t blockOffset) const {
    const auto block = metaOf(blockIndex);
    if (block.bitWidth == BlockBitPackingEncoding<T>::kRawBlockBitWidth) {
      const auto* values =
          reinterpret_cast<const physicalType*>(packedData_ + block.offset);
      return values[blockOffset];
    }
    if (block.bitWidth == 0) {
      return block.baseline;
    }
    FixedBitArray fixedBitArray{packedData_ + block.offset, block.bitWidth};
    return static_cast<physicalType>(
        fixedBitArray.get(blockOffset) + block.baseline);
  }

  void readPhysical(uint32_t offset, uint32_t length, physicalType* output)
      const final {
    this->checkReadRange(offset, length);
    uint32_t outputOffset{0};
    while (outputOffset < length) {
      const auto indexBlock = blockIndex(offset);
      const auto blockOffset = offset - rowOffsetOf(indexBlock);
      const auto count = std::min<uint32_t>(
          length - outputOffset, blockRowCount(indexBlock) - blockOffset);
      readBlockRange(indexBlock, blockOffset, count, output + outputOffset);
      outputOffset += count;
      offset += count;
    }
  }

  uint32_t blockRowCount(uint32_t blockIndex) const {
    return BlockBitPackingEncoding<T>::blockRowCount(header_, blockIndex);
  }

  uint32_t rowOffsetOf(uint32_t blockIndex) const {
    return cachesBlockMetadata()
        ? blockRowOffsets_[blockIndex]
        : BlockBitPackingEncoding<T>::blockRowOffset(header_, blockIndex);
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
    const auto block = metaOf(blockIndex);
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
      FixedBitArray fba{remainderInput, bitWidth};
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
      FixedBitArray fba{remainderInput, bitWidth};
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

  // Returns the metadata of a block, from the cached array or from the
  // metadata streams otherwise. Checks a block read from the streams against
  // the packed payload.
  BlockMeta metaOf(uint32_t blockIndex) const {
    if (cachesBlockMetadata()) {
      return blocks_[blockIndex];
    }
    const BlockMeta block{
        .baseline = baselinesView_->readAt(blockIndex),
        .bitWidth = bitWidthsView_->readAt(blockIndex),
        .offset = offsetsView_->readAt(blockIndex),
        .rowCount = blockRowCount(blockIndex),
    };
    checkBlockBounds(block);
    return block;
  }

  // The parsed header, which gives the row count and row offset of any block.
  typename BlockBitPackingEncoding<T>::Header header_;
  // Hold the metadata of every block. Empty without cacheEncodingView.
  Vector<BlockMeta> blocks_;
  Vector<uint32_t> blockRowOffsets_;
  // Read the metadata of one block at a time. Set only without
  // cacheEncodingView.
  std::unique_ptr<TypedEncodingView<physicalType>> baselinesView_;
  std::unique_ptr<TypedEncodingView<uint8_t>> bitWidthsView_;
  std::unique_ptr<TypedEncodingView<uint32_t>> offsetsView_;
  // The uncompressed packed values of every block, and their size in bytes.
  const char* packedData_{nullptr};
  size_t packedSize_{0};
  uint16_t blockSize_{0};
  uint16_t numBlocks_{0};
  uint32_t firstBlockRows_{0};
};

} // namespace facebook::nimble::deprecated
