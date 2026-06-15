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
#include <cstring>
#include <limits>
#include <optional>
#include <span>
#include <type_traits>

#include "folly/ScopeGuard.h"
#include "velox/common/base/BitUtil.h"
#include "velox/dwio/common/DecoderUtil.h"
#include "velox/dwio/common/Lemire/BitPacking/bitpackinghelpers.h"
#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/common/Constants.h"
#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/common/FixedBitArray.h"
#include "velox/dwio/nimble/common/Varint.h"
#include "velox/dwio/nimble/common/Vector.h"
#include "velox/dwio/nimble/compression/Compression.h"
#include "velox/dwio/nimble/encodings/FixedBitWidthEncoding.h"
#include "velox/dwio/nimble/encodings/TrivialEncoding.h"
#include "velox/dwio/nimble/encodings/common/Encoding.h"
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrimitives.h"
#include "velox/dwio/nimble/encodings/common/EncodingType.h"
#include "velox/dwio/nimble/encodings/selection/EncodingIdentifier.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelection.h"
#include "velox/dwio/nimble/encodings/views/EncodingViewFactory.h"

// Stores integer data in fixed-size blocks (default 1024 rows), each with its
// own baseline and bit width. Adapted from Impulse's per-block bitpacking
// approach for better compression when value ranges vary across the stream.

namespace facebook::nimble {

template <typename T>
class BlockBitPackingEncodingView;

template <typename T>
class BlockBitPackingEncoding final
    : public TypedEncoding<T, typename TypeTraits<T>::physicalType> {
 public:
  using cppDataType = T;
  using physicalType = typename TypeTraits<T>::physicalType;

  /// Maximum rows in one block; bounded so stack decode buffers stay fixed.
  static constexpr uint16_t kMaxBlockSize = kBlockBitPackingBlockSize;
  /// Serialized bit-width marker for blocks stored as raw physical values.
  static constexpr uint8_t kRawBlockBitWidth{255};
  /// The per-block metadata is indexed by a uint16_t, so a single stream can
  /// hold at most this many blocks. encode() enforces this, and estimateSize()
  /// returns nullopt past it so encoding selection skips this encoding.
  static constexpr uint32_t kMaxBlockCount =
      std::numeric_limits<uint16_t>::max();

  BlockBitPackingEncoding(
      velox::memory::MemoryPool& pool,
      std::string_view data,
      const std::function<void*(uint32_t)>& stringBufferFactory,
      const Encoding::Options& options = {});

  ~BlockBitPackingEncoding() override {
    this->releaseBuffer(uncompressedData_);
    this->releaseVectorBuffer(buffer_);
    this->releaseVectorBuffer(bitWidths_);
    this->releaseVectorBuffer(baselines_);
    this->releaseVectorBuffer(blockOffsets_);
  }

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
      Buffer& buffer,
      const Encoding::Options& options = {});

  static std::string_view slice(
      std::string_view encoded,
      uint32_t offset,
      uint32_t length,
      Buffer& buffer,
      const Encoding::Options& options = {});

  /// Estimates the uncompressed encoded size using the same per-block packing
  /// decisions as encode(). The per-block metadata is routed through nested
  /// encoding selection (data-dependent size), so it is estimated as Trivial
  /// sub-encodings; the estimate is approximate, not exact. Returns nullopt
  /// when the stream has more blocks than the uint16_t block index can address,
  /// so encoding selection skips it instead of hard-failing in encode().
  static std::optional<uint64_t> estimateSize(
      std::span<const physicalType> values,
      uint16_t blockSize = kBlockBitPackingBlockSize);

  static std::optional<uint64_t> estimateSize(
      const Statistics<physicalType>& statistics,
      uint16_t blockSize = kBlockBitPackingBlockSize) {
    const auto& blocks = statistics.minMaxBlocks(blockSize);
    const auto numBlocks = static_cast<uint32_t>(blocks.size());
    if (numBlocks > kMaxBlockCount) {
      return std::nullopt;
    }

    uint32_t rowCount{0};
    uint64_t packedSize = 0;
    for (const auto& block : blocks) {
      rowCount += block.count;
      const auto rawSize = block.count * sizeof(physicalType);
      const auto range = block.max - block.min;
      const auto bw = range == 0 ? 0 : velox::bits::bitsRequired(range);
      if (bw >= sizeof(physicalType) * 8) {
        packedSize += rawSize;
        continue;
      }
      const auto blockPackedSize =
          static_cast<uint64_t>(FixedBitArray::bufferSize(block.count, bw));
      if (blockPackedSize < rawSize) {
        packedSize += blockPackedSize;
      } else {
        packedSize += rawSize;
      }
    }

    const auto baselinesSize = estimateMetadataSize<physicalType>(numBlocks);
    const auto bitWidthsSize = estimateMetadataSize<uint8_t>(numBlocks);
    const auto offsetsSize = estimateMetadataSize<uint32_t>(numBlocks);
    const auto firstBlockRows = std::min<uint32_t>(blockSize, rowCount);
    const uint64_t metadataSize = headerSize(
                                      blockSize,
                                      numBlocks,
                                      static_cast<uint32_t>(baselinesSize),
                                      static_cast<uint32_t>(bitWidthsSize),
                                      static_cast<uint32_t>(offsetsSize),
                                      firstBlockRows) +
        baselinesSize + bitWidthsSize + offsetsSize;
    return EncodingPrefix::serializedSize(rowCount, /*useVarint=*/false) +
        metadataSize + packedSize;
  }

#ifdef NIMBLE_ENABLE_EXPERIMENTAL_ENCODINGS
  /// Statistics-only overload for general encoding selection (e.g. as a
  /// SubIntSplit segment candidate), where only `Statistics<physicalType>` --
  /// not the raw values -- is available. Without raw values, true per-block
  /// locality can't be observed; this approximates a "typical" per-block bit
  /// width as the ~50%-coverage point of `statistics.bucketCounts()` (median
  /// bit width of value - min), analogous to how PFOREncoding's
  /// `selectBaseBitWidth` picks a 90%-coverage baseline. This is a coarser
  /// approximation than the span-based overload above.
  static uint64_t estimateSize(
      uint64_t rowCount,
      const Statistics<physicalType>& statistics,
      uint16_t blockSize = kBlockBitPackingBlockSize) {
    if (rowCount == 0) {
      return EncodingPrefix::kFixedPrefixSize;
    }
    const auto& bucketCounts = statistics.bucketCounts();
    const auto fullRange =
        static_cast<physicalType>(statistics.max() - statistics.min());
    const uint8_t maxBitWidth =
        static_cast<uint8_t>(velox::bits::bitsRequired(fullRange));

    constexpr double kCoverageThreshold = 0.5;
    const uint64_t threshold = static_cast<uint64_t>(
        static_cast<double>(rowCount) * kCoverageThreshold);
    uint8_t typicalBitWidth = maxBitWidth;
    uint64_t cumulative = 0;
    for (size_t k = 0; k < bucketCounts.size(); ++k) {
      cumulative += bucketCounts[k];
      if (cumulative >= threshold) {
        typicalBitWidth = std::min<uint8_t>(
            static_cast<uint8_t>(std::min<size_t>((k + 1) * 7, 64)),
            maxBitWidth);
        break;
      }
    }

    const uint32_t numBlocks =
        velox::bits::divRoundUp(static_cast<uint32_t>(rowCount), blockSize);
    const uint64_t packedSize =
        FixedBitArray::bufferSize(rowCount, typicalBitWidth);
    const uint64_t metadataSize = kMetadataHeaderSize +
        TrivialEncoding<physicalType>::estimateSize(numBlocks) +
        TrivialEncoding<uint8_t>::estimateSize(numBlocks) +
        TrivialEncoding<uint32_t>::estimateSize(numBlocks);
    return EncodingPrefix::kFixedPrefixSize + metadataSize + packedSize;
  }
#endif

  std::string debugString(int offset) const final;

 private:
  friend class BlockBitPackingEncodingView<T>;

  struct BlockInfo {
    physicalType baseline;
    uint8_t bitWidth;
    uint32_t packedSize;
    uint32_t start;
    uint32_t count;
    bool skipEncoding;
  };

  static void writeBlockPayload(
      std::span<const physicalType> values,
      const BlockInfo& block,
      char* output) {
    if (block.skipEncoding) {
      std::memcpy(output, values.data(), block.count * sizeof(physicalType));
      return;
    }

    if (block.bitWidth == 0) {
      return;
    }

    if constexpr (sizeof(physicalType) == 4) {
      // The 32-bit path uses Lemire fastpack for full 32-value groups; a
      // trailing partial group falls back to FixedBitArray so callers can use
      // any block row count.
      constexpr uint32_t kGroupSize = 32;
      const auto baseline32 = static_cast<uint32_t>(block.baseline);
      auto* dst = reinterpret_cast<uint32_t*>(output);
      const auto* src = reinterpret_cast<const uint32_t*>(values.data());

      const auto fullGroups = block.count / kGroupSize;
      const auto remainder = block.count % kGroupSize;

      uint32_t tmp[kGroupSize];
      for (uint32_t group = 0; group < fullGroups; ++group) {
        for (uint32_t i = 0; i < kGroupSize; ++i) {
          tmp[i] = src[group * kGroupSize + i] - baseline32;
        }
        velox::fastpforlib::fastpack(tmp, dst, block.bitWidth);
        dst += block.bitWidth;
      }
      if (remainder > 0) {
        const auto remainderOffset =
            fullGroups * kGroupSize * block.bitWidth / 8;
        FixedBitArray fba(output + remainderOffset, block.bitWidth);
        for (uint32_t i = 0; i < remainder; ++i) {
          fba.set(i, values[fullGroups * kGroupSize + i] - block.baseline);
        }
      }
    } else {
      FixedBitArray fba(output, block.bitWidth);
      fba.bulkSetWithBaseline(0, block.count, values.data(), block.baseline);
    }
  }

  static void writePayload(
      std::span<const physicalType> values,
      std::span<const BlockInfo> blocks,
      uint32_t totalPackedSize,
      char*& pos) {
    std::memset(pos, 0, totalPackedSize);
    uint32_t dataOffset{0};
    for (const auto& block : blocks) {
      writeBlockPayload(
          values.subspan(block.start, block.count), block, pos + dataOffset);
      dataOffset += block.packedSize;
    }
    pos += totalPackedSize;
  }

  static uint32_t headerSize(
      uint16_t blockSize,
      uint32_t numBlocks,
      uint32_t baselinesSize,
      uint32_t bitWidthsSize,
      uint32_t offsetsSize,
      uint32_t firstBlockRows) {
    return /*compressionType=*/sizeof(uint8_t) + varint::varintSize(blockSize) +
        varint::varintSize(numBlocks) + varint::varintSize(baselinesSize) +
        varint::varintSize(bitWidthsSize) + varint::varintSize(offsetsSize) +
        varint::varintSize(firstBlockRows);
  }

  template <typename MetadataType>
  static uint64_t estimateMetadataSize(uint32_t rowCount) {
    const auto trivialSize =
        TrivialEncoding<MetadataType>::estimateSize(rowCount);
    const auto fixedBitWidthSize =
        FixedBitWidthEncoding<MetadataType>::estimateSize(
            rowCount,
            /*minValue=*/0,
            /*maxValue=*/
            std::numeric_limits<
                typename TypeTraits<MetadataType>::physicalType>::max(),
            Encoding::Options{});
    return std::min(trivialSize, fixedBitWidthSize);
  }

  struct BlockSliceInfo {
    // Source block that contributes rows to this output block.
    uint32_t blockIndex{0};
    // Byte offset of the source block payload, adjusted for raw partial blocks.
    uint32_t packedOffset{0};
    // First row copied from the source block.
    uint32_t rowOffset{0};
    // Bytes written for this output block.
    uint32_t packedSize{0};
    // Rows written for this output block.
    uint32_t rowCount{0};
    // Output bit width; zero also represents constant blocks.
    uint8_t bitWidth{0};
    // True when the payload is copied as raw physical values.
    bool skipEncoding{false};
  };

  // Parsed stream metadata. lastBlockRows is derived from rowCount and the
  // first-block override; it is not serialized.
  struct Header {
    uint32_t rowCount{0};
    uint16_t blockSize{0};
    uint16_t numBlocks{0};
    CompressionType compressionType{CompressionType::Uncompressed};
    std::string_view encodedBaselines;
    std::string_view encodedBitWidths;
    std::string_view encodedBlockOffsets;
    uint32_t firstBlockRows{0};
    uint32_t lastBlockRows{0};
    std::string_view packedData;
  };

  struct BlockPosition {
    uint32_t blockIndex{0};
    uint32_t rowOffset{0};
    uint32_t rowCount{0};
  };

  static Header parseHeader(
      std::string_view encoded,
      const Encoding::Options& options) {
    Header source;
    source.rowCount =
        EncodingPrefix::readRowCount(encoded, options.useVarintRowCount);
    const char* pos = encoded.data() +
        EncodingPrefix::prefixSize(encoded, options.useVarintRowCount);
    source.compressionType =
        static_cast<CompressionType>(encoding::readChar(pos));
    source.blockSize = static_cast<uint16_t>(varint::readVarint32(&pos));
    source.numBlocks = static_cast<uint16_t>(varint::readVarint32(&pos));
    NIMBLE_CHECK_GT(source.blockSize, 0);
    NIMBLE_CHECK_GT(source.numBlocks, 0);
    NIMBLE_CHECK_GT(source.rowCount, 0);

    const auto baselinesSize = varint::readVarint32(&pos);
    source.encodedBaselines = {pos, baselinesSize};
    pos += baselinesSize;
    const auto bitWidthsSize = varint::readVarint32(&pos);
    source.encodedBitWidths = {pos, bitWidthsSize};
    pos += bitWidthsSize;
    const auto blockOffsetsSize = varint::readVarint32(&pos);
    source.encodedBlockOffsets = {pos, blockOffsetsSize};
    pos += blockOffsetsSize;
    source.firstBlockRows = varint::readVarint32(&pos);
    NIMBLE_CHECK_GT(source.firstBlockRows, 0);
    NIMBLE_CHECK_LE(source.firstBlockRows, source.blockSize);
    NIMBLE_CHECK_LE(source.firstBlockRows, source.rowCount);
    source.lastBlockRows = source.numBlocks == 1 ? source.firstBlockRows
                                                 : source.rowCount -
            source.firstBlockRows - (source.numBlocks - 2) * source.blockSize;
    NIMBLE_CHECK_GT(source.lastBlockRows, 0);
    NIMBLE_CHECK_LE(source.lastBlockRows, source.blockSize);

    source.packedData = {pos, static_cast<size_t>(encoded.end() - pos)};
    return source;
  }

  static std::string_view packedDataSource(
      const Header& header,
      velox::BufferPtr& uncompressedData,
      velox::memory::MemoryPool* pool,
      const Encoding::Options& options) {
    if (header.compressionType == CompressionType::Uncompressed) {
      return header.packedData;
    }
    NIMBLE_CHECK_NOT_NULL(pool);
    uncompressedData = Compression::uncompress(
        *pool,
        header.compressionType,
        DataType::Undefined,
        header.packedData,
        options.decompressCounter(),
        options.bufferPool);
    return {uncompressedData->as<char>(), uncompressedData->size()};
  }

  static void readBlockOffsets(
      const Header& header,
      uint32_t blockOffset,
      uint32_t blockCount,
      uint32_t packedDataSize,
      Vector<uint32_t>& output,
      Buffer& scratchBuffer,
      const Encoding::Options& options) {
    // Each block size is derived from its start offset and the next block's
    // start offset. For the final source block, the next offset is the packed
    // data size because there is no stored successor offset.
    const auto numReadBlocks =
        blockCount + (blockOffset + blockCount < header.numBlocks ? 1 : 0);
    readMetadataRange(
        header.encodedBlockOffsets,
        blockOffset,
        numReadBlocks,
        output,
        scratchBuffer,
        options);
    if (numReadBlocks == blockCount) {
      output[blockCount] = packedDataSize;
    }
  }

  // Sliced streams can preserve a partial first block, so row counts are based
  // on the parsed source header instead of blockIndex * blockSize.
  static uint32_t blockRowCount(const Header& header, uint32_t blockIndex) {
    if (blockIndex == 0) {
      return header.firstBlockRows;
    }
    if (blockIndex == header.numBlocks - 1) {
      return header.lastBlockRows;
    }
    return header.blockSize;
  }

  // Returns the number of rows in the given block (may be less than
  // blockSize_ for the last block).
  uint32_t blockRowCount(uint32_t blockIndex) const {
    if (blockIndex == 0) {
      return firstBlockRows_;
    }
    if (blockIndex == numBlocks_ - 1) {
      return lastBlockRows_;
    }
    return blockSize_;
  }

  // Row offset for sliced streams where the first block may not be full.
  static uint32_t blockRowOffset(const Header& header, uint32_t blockIndex) {
    if (blockIndex == 0) {
      return 0;
    }
    return header.firstBlockRows + (blockIndex - 1) * header.blockSize;
  }

  static BlockPosition blockPosition(const Header& header, uint32_t row) {
    if (row < header.firstBlockRows) {
      return {
          .blockIndex = 0,
          .rowOffset = row,
          .rowCount = header.firstBlockRows,
      };
    }
    const auto relativeRow = row - header.firstBlockRows;
    const auto index = 1 + relativeRow / header.blockSize;
    NIMBLE_CHECK_LT(index, header.numBlocks);
    return {
        .blockIndex = index,
        .rowOffset = relativeRow % header.blockSize,
        .rowCount = index == header.numBlocks - 1 ? header.lastBlockRows
                                                  : header.blockSize,
    };
  }

  uint32_t blockIndex(uint32_t row) const {
    if (row < firstBlockRows_) {
      return 0;
    }
    const auto blockIndex = 1 + (row - firstBlockRows_) / blockSize_;
    NIMBLE_CHECK_LT(blockIndex, numBlocks_);
    return blockIndex;
  }

  uint32_t blockRowOffset(uint32_t blockIndex) const {
    if (blockIndex == 0) {
      return 0;
    }
    return firstBlockRows_ + (blockIndex - 1) * blockSize_;
  }

  // Returns the byte size of a block payload for the selected storage mode.
  static uint32_t getPackedSize(uint32_t rowCount, uint8_t bitWidth) {
    if (bitWidth == kRawBlockBitWidth) {
      return static_cast<uint32_t>(rowCount * sizeof(physicalType));
    }
    if (bitWidth == 0) {
      return uint32_t{0};
    }
    return static_cast<uint32_t>(FixedBitArray::bufferSize(rowCount, bitWidth));
  }

  static void writeBlockSlicesPayload(
      const Header& header,
      std::string_view packedData,
      std::span<const BlockSliceInfo> blockSlices,
      std::span<const uint32_t> sliceOffsets,
      uint32_t totalPackedSize,
      char*& pos);

  // Reads a single decoded value at the given absolute row index.
  physicalType readSingleValue(uint32_t row) const;

  // Decodes a contiguous range within one block into 'output'.
  void materializeBlockRange(
      uint32_t blockIndex,
      uint32_t blockValueOffset,
      uint32_t blockValueCount,
      physicalType* output) const;

  // Unpacks 'numRows' bit-packed values in fixed-size groups, falling
  // back to FixedBitArray for any trailing remainder.
  static void fullUnpack(
      const uint8_t* input,
      physicalType* output,
      uint32_t numRows,
      uint8_t bitWidth,
      physicalType baseline);

  // Builds the encoding plan for a single block.
  static BlockInfo makeBlockInfo(
      std::span<const physicalType> values,
      uint32_t start,
      uint32_t count);

  // Materializes one serialized per-block metadata child stream.
  template <typename MetadataType>
  void readMetadataStream(
      std::string_view encoded,
      Vector<MetadataType>& output,
      velox::memory::MemoryPool& pool,
      const std::function<void*(uint32_t)>& stringBufferFactory,
      const Encoding::Options& options) const;

  template <typename OutputVector>
  static void readMetadataRange(
      std::string_view encoded,
      uint32_t offset,
      uint32_t count,
      OutputVector& output,
      Buffer& scratchBuffer,
      const Encoding::Options& options);

  uint16_t blockSize_;
  uint16_t numBlocks_;
  Vector<physicalType> baselines_;
  Vector<uint8_t> bitWidths_;
  Vector<uint32_t> blockOffsets_;
  uint32_t firstBlockRows_{0};
  uint32_t lastBlockRows_{0};
  const char* packedData_;
  uint32_t row_ = 0;
  velox::BufferPtr uncompressedData_;
  Vector<physicalType> buffer_;
};

//
// End of class declaration. Implementations follow.
//

template <typename T>
std::optional<uint64_t> BlockBitPackingEncoding<T>::estimateSize(
    std::span<const physicalType> values,
    uint16_t blockSize) {
  const auto rowCount = static_cast<uint32_t>(values.size());
  const uint32_t numBlocks = velox::bits::divRoundUp(rowCount, blockSize);
  if (numBlocks > kMaxBlockCount) {
    return std::nullopt;
  }
  uint64_t packedSize = 0;
  for (uint32_t blockIndex = 0; blockIndex < numBlocks; ++blockIndex) {
    const auto start = blockIndex * blockSize;
    const auto end = std::min<uint32_t>(start + blockSize, rowCount);
    packedSize += makeBlockInfo(values, start, end - start).packedSize;
  }
  const auto baselinesSize = estimateMetadataSize<physicalType>(numBlocks);
  const auto bitWidthsSize = estimateMetadataSize<uint8_t>(numBlocks);
  const auto offsetsSize = estimateMetadataSize<uint32_t>(numBlocks);
  const auto firstBlockRows = std::min<uint32_t>(blockSize, rowCount);
  const uint64_t metadataSize = headerSize(
                                    blockSize,
                                    numBlocks,
                                    static_cast<uint32_t>(baselinesSize),
                                    static_cast<uint32_t>(bitWidthsSize),
                                    static_cast<uint32_t>(offsetsSize),
                                    firstBlockRows) +
      baselinesSize + bitWidthsSize + offsetsSize;
  return EncodingPrefix::serializedSize(rowCount, /*useVarint=*/false) +
      metadataSize + packedSize;
}

template <typename T>
typename BlockBitPackingEncoding<T>::BlockInfo
BlockBitPackingEncoding<T>::makeBlockInfo(
    std::span<const physicalType> values,
    uint32_t start,
    uint32_t count) {
  auto blockValues = values.subspan(start, count);
  auto [minValue, maxValue] =
      std::minmax_element(blockValues.begin(), blockValues.end());
  const auto range = *maxValue - *minValue;
  const auto rawSize = count * sizeof(physicalType);

  const auto bitsRequired = range == 0 ? 0 : velox::bits::bitsRequired(range);
  NIMBLE_DCHECK_LE(
      bitsRequired,
      sizeof(physicalType) * 8,
      "bitsRequired cannot exceed type width.");
  if (bitsRequired == sizeof(physicalType) * 8) {
    return {0, 0, static_cast<uint32_t>(rawSize), start, count, true};
  }

  // Skip encoding when packing doesn't reduce size.
  const auto packedSize =
      static_cast<uint32_t>(FixedBitArray::bufferSize(count, bitsRequired));
  if (packedSize >= rawSize) {
    return {0, 0, static_cast<uint32_t>(rawSize), start, count, true};
  }

  return {
      *minValue,
      static_cast<uint8_t>(bitsRequired),
      packedSize,
      start,
      count,
      false};
}

template <typename T>
void BlockBitPackingEncoding<T>::writeBlockSlicesPayload(
    const Header& header,
    std::string_view packedData,
    std::span<const BlockSliceInfo> blockSlices,
    std::span<const uint32_t> sliceOffsets,
    uint32_t totalPackedSize,
    char*& pos) {
  std::memset(pos, 0, totalPackedSize);
  uint32_t dataOffset{0};
  const auto needsBitCopy = [&](const BlockSliceInfo& blockSlice) {
    return !blockSlice.skipEncoding && blockSlice.bitWidth != 0 &&
        (blockSlice.rowOffset != 0 ||
         blockSlice.rowCount != blockRowCount(header, blockSlice.blockIndex));
  };
  const auto copyBlockBitsRange = [&](uint32_t blockIndex) {
    const auto& blockSlice = blockSlices[blockIndex];

    const auto sourceBitOffset =
        static_cast<uint64_t>(blockSlice.rowOffset) * blockSlice.bitWidth;
    const auto sliceBits =
        static_cast<uint64_t>(blockSlice.rowCount) * blockSlice.bitWidth;
    encoding::copyPackedBits(
        packedData.substr(blockSlice.packedOffset),
        sourceBitOffset,
        sliceBits,
        pos + dataOffset);
    dataOffset += blockSlice.packedSize;
  };

  const auto numBlocks = static_cast<uint32_t>(blockSlices.size());
  const bool firstNeedsBitCopy = needsBitCopy(blockSlices[0]);
  const bool lastNeedsBitCopy =
      numBlocks > 1 && needsBitCopy(blockSlices[numBlocks - 1]);
  if (firstNeedsBitCopy) {
    copyBlockBitsRange(/*blockIndex=*/0);
  }

  uint32_t packedBegin = firstNeedsBitCopy ? 1 : 0;
  const uint32_t packedEnd = lastNeedsBitCopy ? numBlocks - 1 : numBlocks;
  // Constant blocks have rows but no packed payload (bit width 0). They still
  // participate in slice offsets, but cannot anchor the source byte range for
  // the coalesced copy.
  while (packedBegin < packedEnd && blockSlices[packedBegin].packedSize == 0) {
    ++packedBegin;
  }
  if (packedBegin < packedEnd) {
    NIMBLE_CHECK_EQ(dataOffset, sliceOffsets[packedBegin]);
    const auto packedEndOffset = sliceOffsets[packedEnd];
    const auto packedBytes = packedEndOffset - sliceOffsets[packedBegin];
    std::memcpy(
        pos + dataOffset,
        packedData.data() + blockSlices[packedBegin].packedOffset,
        packedBytes);
    dataOffset += packedBytes;
  }

  if (lastNeedsBitCopy) {
    copyBlockBitsRange(numBlocks - 1);
  }
  pos += totalPackedSize;
}

template <typename T>
BlockBitPackingEncoding<T>::BlockBitPackingEncoding(
    velox::memory::MemoryPool& pool,
    std::string_view data,
    const std::function<void*(uint32_t)>& stringBufferFactory,
    const Encoding::Options& options)
    : TypedEncoding<T, physicalType>{pool, data, options},
      baselines_{this->template getVectorBuffer<physicalType>()},
      bitWidths_{this->template getVectorBuffer<uint8_t>()},
      blockOffsets_{this->template getVectorBuffer<uint32_t>()},
      packedData_{nullptr},
      buffer_{&pool} {
  const auto source = parseHeader(data, options);
  blockSize_ = source.blockSize;
  numBlocks_ = source.numBlocks;
  firstBlockRows_ = source.firstBlockRows;
  lastBlockRows_ = source.lastBlockRows;

  NIMBLE_CHECK(blockSize_ > 0 && blockSize_ <= kMaxBlockSize);

  baselines_.resize(numBlocks_);
  bitWidths_.resize(numBlocks_);
  blockOffsets_.resize(numBlocks_);

  readMetadataStream(
      source.encodedBaselines, baselines_, pool, stringBufferFactory, options);
  readMetadataStream(
      source.encodedBitWidths, bitWidths_, pool, stringBufferFactory, options);
  readMetadataStream(
      source.encodedBlockOffsets,
      blockOffsets_,
      pool,
      stringBufferFactory,
      options);

  packedData_ =
      packedDataSource(source, uncompressedData_, &pool, options).data();
}

template <typename T>
template <typename MetadataType>
void BlockBitPackingEncoding<T>::readMetadataStream(
    std::string_view encoded,
    Vector<MetadataType>& output,
    velox::memory::MemoryPool& pool,
    const std::function<void*(uint32_t)>& stringBufferFactory,
    const Encoding::Options& options) const {
  NIMBLE_CHECK_GT(numBlocks_, 0);
  EncodingFactory()
      .create(pool, encoded, stringBufferFactory, options)
      ->materialize(numBlocks_, output.data());
}

template <typename T>
template <typename OutputVector>
void BlockBitPackingEncoding<T>::readMetadataRange(
    std::string_view encoded,
    uint32_t offset,
    uint32_t count,
    OutputVector& output,
    Buffer& scratchBuffer,
    const Encoding::Options& options) {
  NIMBLE_CHECK_GE(output.size(), count, "Metadata output buffer is too small.");
  auto* const pool = &scratchBuffer.getMemoryPool();
  auto encoding = EncodingFactory{options}.create(
      *pool, encoded, [&scratchBuffer](uint32_t size) -> void* {
        return scratchBuffer.reserve(size);
      });
  encoding->skip(offset);
  encoding->materialize(count, output.data());
}

template <typename T>
void BlockBitPackingEncoding<T>::reset() {
  row_ = 0;
}

template <typename T>
void BlockBitPackingEncoding<T>::skip(uint32_t rowCount) {
  row_ += rowCount;
}

template <typename T>
typename BlockBitPackingEncoding<T>::physicalType
BlockBitPackingEncoding<T>::readSingleValue(uint32_t row) const {
  const auto blockIndex = this->blockIndex(row);
  const auto blockOffset = row - blockRowOffset(blockIndex);
  const auto bitWidth = bitWidths_[blockIndex];
  const auto baseline = baselines_[blockIndex];
  if (bitWidth == kRawBlockBitWidth) {
    const auto* rawValues = reinterpret_cast<const physicalType*>(
        packedData_ + blockOffsets_[blockIndex]);
    return rawValues[blockOffset];
  }
  if (bitWidth == 0) {
    return baseline;
  }
  const auto numRows = blockRowCount(blockIndex);
  FixedBitArray fba{
      {packedData_ + blockOffsets_[blockIndex],
       FixedBitArray::bufferSize(numRows, bitWidth)},
      bitWidth};
  return static_cast<physicalType>(fba.get(blockOffset)) + baseline;
}

template <typename T>
void BlockBitPackingEncoding<T>::materializeBlockRange(
    uint32_t blockIndex,
    uint32_t blockValueOffset,
    uint32_t blockValueCount,
    physicalType* output) const {
  const auto bitWidth = bitWidths_[blockIndex];
  const auto baseline = baselines_[blockIndex];
  const auto* blockData = packedData_ + blockOffsets_[blockIndex];

  if (bitWidth == kRawBlockBitWidth) {
    const auto* rawValues = reinterpret_cast<const physicalType*>(blockData);
    std::memcpy(
        output,
        rawValues + blockValueOffset,
        blockValueCount * sizeof(physicalType));
    return;
  }

  if (bitWidth == 0) {
    std::fill(output, output + blockValueCount, baseline);
    return;
  }

  const auto* inputBytes = reinterpret_cast<const uint8_t*>(blockData);

  if (blockValueOffset == 0) {
    fullUnpack(inputBytes, output, blockValueCount, bitWidth, baseline);
  } else {
    physicalType tmp[kMaxBlockSize];
    fullUnpack(
        inputBytes,
        tmp,
        blockValueOffset + blockValueCount,
        bitWidth,
        baseline);
    std::memcpy(
        output, tmp + blockValueOffset, blockValueCount * sizeof(physicalType));
  }
}

template <typename T>
void BlockBitPackingEncoding<T>::fullUnpack(
    const uint8_t* input,
    physicalType* output,
    uint32_t numRows,
    uint8_t bitWidth,
    physicalType baseline) {
  constexpr uint32_t kGroupSize = [] {
    if constexpr (sizeof(physicalType) == 1) {
      return 8u;
    } else if constexpr (sizeof(physicalType) == 2) {
      return 16u;
    } else {
      return 32u;
    }
  }();

  uint32_t currentRow = 0;

  if constexpr (std::is_same_v<physicalType, uint8_t>) {
    for (; currentRow + kGroupSize <= numRows; currentRow += kGroupSize) {
      velox::fastpforlib::internal::fastunpack_quarter(
          input, output + currentRow, bitWidth);
      input += bitWidth;
    }
  } else if constexpr (std::is_same_v<physicalType, uint16_t>) {
    for (; currentRow + kGroupSize <= numRows; currentRow += kGroupSize) {
      velox::fastpforlib::internal::fastunpack_half(
          reinterpret_cast<const uint16_t*>(input),
          output + currentRow,
          bitWidth);
      input += bitWidth * 2;
    }
  } else if constexpr (isFourByteIntegralType<physicalType>()) {
    for (; currentRow + kGroupSize <= numRows; currentRow += kGroupSize) {
      velox::fastpforlib::fastunpack(
          reinterpret_cast<const uint32_t*>(input),
          reinterpret_cast<uint32_t*>(output + currentRow),
          bitWidth);
      input += bitWidth * 4;
    }
  } else if constexpr (isEightByteIntegralType<physicalType>()) {
    for (; currentRow + kGroupSize <= numRows; currentRow += kGroupSize) {
      velox::fastpforlib::fastunpack(
          reinterpret_cast<const uint32_t*>(input),
          reinterpret_cast<uint64_t*>(output + currentRow),
          bitWidth);
      input += bitWidth * 4;
    }
  }

  if (currentRow < numRows) {
    const auto tailCount = numRows - currentRow;
    FixedBitArray fba(
        {reinterpret_cast<const char*>(input),
         FixedBitArray::bufferSize(tailCount, bitWidth)},
        bitWidth);
    for (uint32_t i = 0; i < tailCount; ++i) {
      output[currentRow + i] = static_cast<physicalType>(fba.get(i));
    }
  }

  if (baseline != 0) {
    for (uint32_t i = 0; i < numRows; ++i) {
      output[i] += baseline;
    }
  }
}

template <typename T>
void BlockBitPackingEncoding<T>::materialize(uint32_t rowCount, void* buffer) {
  auto* output = static_cast<physicalType*>(buffer);
  uint32_t remaining = rowCount;
  uint32_t currentRow = row_;

  while (remaining > 0) {
    const auto blockIndex = this->blockIndex(currentRow);
    const auto blockOffset = currentRow - blockRowOffset(blockIndex);
    const auto blockRows =
        std::min(remaining, blockRowCount(blockIndex) - blockOffset);

    materializeBlockRange(blockIndex, blockOffset, blockRows, output);

    output += blockRows;
    currentRow += blockRows;
    remaining -= blockRows;
  }
  row_ += rowCount;
}

template <typename T>
template <typename V>
void BlockBitPackingEncoding<T>::readWithVisitor(
    V& visitor,
    ReadWithVisitorParams& params) {
  using OutputType = detail::ValueType<typename V::DataType>;
  constexpr bool kIsWideType =
      (isFourByteIntegralType<physicalType>() ||
       isEightByteIntegralType<physicalType>());
  constexpr bool kIsNarrowType =
      std::is_integral_v<physicalType> && !kIsWideType;
  constexpr bool kIsFluidCast = sizeof(OutputType) >= sizeof(physicalType) &&
      std::is_integral_v<OutputType> && std::is_integral_v<physicalType>;
  // Wide types (4/8 byte): full fast path with filter/scatter/hook support.
  // Narrow types (1/2 byte): fast path only without filter/scatter/hook AND
  // without nulls, because processFixedWidthRun requires >= 4-byte types
  // and the scatter path (used for nullable columns) calls it.
  constexpr bool kCanUseFastPath =
      (kIsWideType || (kIsNarrowType && !V::kHasFilter && !V::kHasHook));
  if constexpr (
      kCanUseFastPath &&
      std::is_same_v<
          typename V::Extract,
          velox::dwio::common::ExtractToReader> &&
      kIsFluidCast) {
    auto* nulls = visitor.reader().rawNullsInReadRange();
    if (velox::dwio::common::useFastPath(visitor, nulls) &&
        (kIsWideType || nulls == nullptr)) {
      detail::readWithVisitorFast(*this, visitor, params, nulls);
      return;
    }
  }
  detail::readWithVisitorSlow(
      visitor,
      params,
      [&](auto toSkip) { skip(toSkip); },
      [&] {
        physicalType value = readSingleValue(row_++);
        return value;
      });
}

template <typename T>
template <bool kScatter, typename V>
void BlockBitPackingEncoding<T>::bulkScan(
    V& visitor,
    vector_size_t currentRow,
    const vector_size_t* selectedRows,
    vector_size_t numSelected,
    const vector_size_t* scatterRows) {
  using OutputType = detail::ValueType<typename V::DataType>;

  if (numSelected == 0) {
    return;
  }

  const auto numRows = visitor.numRows() - visitor.rowIndex();
  const auto offset =
      static_cast<int64_t>(row_) - static_cast<int64_t>(currentRow);

  auto* values = detail::mutableValues<OutputType>(visitor, numRows);

  constexpr bool kSameSize = sizeof(physicalType) == sizeof(OutputType);
  constexpr bool kIsUpcast = sizeof(OutputType) > sizeof(physicalType) &&
      std::is_integral_v<OutputType> && std::is_integral_v<physicalType>;

  if constexpr (V::dense) {
    const auto startRow = selectedRows[0] + offset;

    if constexpr (kSameSize) {
      // Materialize directly into output — no temp buffer needed.
      auto* dst = reinterpret_cast<physicalType*>(values);
      uint32_t decodedRows = 0;
      uint32_t currentAbsRow = startRow;
      while (decodedRows < static_cast<uint32_t>(numSelected)) {
        const auto blockIndex = this->blockIndex(currentAbsRow);
        const auto blockOffset = currentAbsRow - blockRowOffset(blockIndex);
        const auto toDecode = std::min(
            static_cast<uint32_t>(numSelected) - decodedRows,
            blockRowCount(blockIndex) - blockOffset);
        materializeBlockRange(
            blockIndex, blockOffset, toDecode, dst + decodedRows);
        decodedRows += toDecode;
        currentAbsRow += toDecode;
      }
    } else if constexpr (kIsUpcast) {
      if (buffer_.capacity() == 0) {
        buffer_ = this->template getVectorBuffer<physicalType>();
      }
      buffer_.resize(numSelected);
      auto* rawBuf = buffer_.data();
      uint32_t decodedRows = 0;
      uint32_t currentAbsRow = startRow;
      while (decodedRows < static_cast<uint32_t>(numSelected)) {
        const auto blockIndex = this->blockIndex(currentAbsRow);
        const auto blockOffset = currentAbsRow - blockRowOffset(blockIndex);
        const auto toDecode = std::min(
            static_cast<uint32_t>(numSelected) - decodedRows,
            blockRowCount(blockIndex) - blockOffset);
        materializeBlockRange(
            blockIndex, blockOffset, toDecode, rawBuf + decodedRows);
        decodedRows += toDecode;
        currentAbsRow += toDecode;
      }
      for (vector_size_t i = 0; i < numSelected; ++i) {
        values[i] = static_cast<OutputType>(rawBuf[i]);
      }
    }
  } else {
    // Sparse: per-block strategy based on block type and selectivity:
    //  - skipEncoding: direct array access, no decode
    //  - bitWidth == 0: constant block, return baseline
    //  - low selectivity (<5%): FBA per-element for selected rows only
    //  - high selectivity: full block decode, pick rows
    constexpr uint32_t kMaxFbaSelectivityPct = 5;

    vector_size_t i = 0;
    while (i < numSelected) {
      const auto absRow = static_cast<uint32_t>(selectedRows[i]) +
          static_cast<uint32_t>(offset);
      const auto blockIndex = this->blockIndex(absRow);
      const auto blockStart = blockRowOffset(blockIndex);
      const auto bitWidth = bitWidths_[blockIndex];
      const auto baseline = baselines_[blockIndex];

      // Find all selected rows belonging to this block.
      vector_size_t runEnd = i + 1;
      while (runEnd < numSelected) {
        const auto nextAbsRow = static_cast<uint32_t>(selectedRows[runEnd]) +
            static_cast<uint32_t>(offset);
        if (this->blockIndex(nextAbsRow) != blockIndex) {
          break;
        }
        ++runEnd;
      }

      const auto runLength = static_cast<uint32_t>(runEnd - i);
      const auto numBlockRows = blockRowCount(blockIndex);

      // Raw: direct index, no decode.
      if (bitWidth == kRawBlockBitWidth) {
        const auto* rawValues = reinterpret_cast<const physicalType*>(
            packedData_ + blockOffsets_[blockIndex]);
        for (vector_size_t j = i; j < runEnd; ++j) {
          const auto blockOffset = static_cast<uint32_t>(selectedRows[j]) +
              static_cast<uint32_t>(offset) - blockStart;
          values[j] = static_cast<OutputType>(rawValues[blockOffset]);
        }
        // Constant: every value equals baseline.
      } else if (bitWidth == 0) {
        for (vector_size_t j = i; j < runEnd; ++j) {
          values[j] = static_cast<OutputType>(baseline);
        }
      } else if (runLength * 100 < numBlockRows * kMaxFbaSelectivityPct) {
        // Low selectivity: FBA per-element decode for selected rows only.
        FixedBitArray fba{
            {packedData_ + blockOffsets_[blockIndex],
             FixedBitArray::bufferSize(numBlockRows, bitWidth)},
            bitWidth};
        for (vector_size_t j = i; j < runEnd; ++j) {
          const auto blockOffset = static_cast<uint32_t>(selectedRows[j]) +
              static_cast<uint32_t>(offset) - blockStart;
          values[j] = static_cast<OutputType>(
              static_cast<physicalType>(fba.get(blockOffset)) + baseline);
        }
      } else {
        // High selectivity: full block decode, pick rows.
        physicalType tmp[kMaxBlockSize];
        materializeBlockRange(blockIndex, 0, numBlockRows, tmp);
        for (vector_size_t j = i; j < runEnd; ++j) {
          const auto blockOffset = static_cast<uint32_t>(selectedRows[j]) +
              static_cast<uint32_t>(offset) - blockStart;
          values[j] = static_cast<OutputType>(tmp[blockOffset]);
        }
      }
      i = runEnd;
    }
  }

  row_ += selectedRows[numSelected - 1] - currentRow + 1;

  if constexpr (!kScatter && !V::kHasFilter && !V::kHasHook) {
    visitor.addNumValues(numRows);
    visitor.setRowIndex(visitor.numRows());
    return;
  }

  // processFixedWidthRun requires >= 4-byte OutputType. Narrow types
  // with filters/hooks are routed to the slow path by readWithVisitor,
  // so this branch is unreachable for them.
  if constexpr (sizeof(OutputType) >= 4) {
    if constexpr (!V::kHasHook) {
      values = reinterpret_cast<OutputType*>(visitor.reader().rawValues());
    }

    auto numValues = visitor.reader().numValues();
    int32_t* filterHits = nullptr;
    if constexpr (V::kHasFilter) {
      filterHits = visitor.outputRows(numSelected) - numValues;
    }

    velox::dwio::common::
        processFixedWidthRun<OutputType, V::kFilterOnly, kScatter, V::dense>(
            velox::RowSet(selectedRows, numSelected),
            0,
            numSelected,
            scatterRows,
            values,
            filterHits,
            numValues,
            visitor.filter(),
            visitor.hook());

    if constexpr (!V::kHasHook) {
      visitor.addNumValues(
          V::kHasFilter ? numValues - visitor.reader().numValues() : numRows);
    }
  } else {
    NIMBLE_UNREACHABLE(
        "Narrow-type bulkScan with filter/hook should use the slow path.");
  }
  visitor.setRowIndex(visitor.numRows());
}

template <typename T>
std::string_view BlockBitPackingEncoding<T>::encode(
    EncodingSelection<physicalType>& selection,
    std::span<const physicalType> values,
    Buffer& buffer,
    const Encoding::Options& options) {
  static_assert(
      std::is_unsigned_v<physicalType>, "Physical type must be unsigned.");

  auto* pool = &buffer.getMemoryPool();
  const uint16_t blockSize = options.blockBitPackingBlockSize;
  const auto rowCount = static_cast<uint32_t>(values.size());
  NIMBLE_CHECK_GT(rowCount, 0, "BlockBitPacking cannot encode empty streams.");
  const uint32_t numBlocks = velox::bits::divRoundUp(rowCount, blockSize);
  NIMBLE_CHECK_LE(
      numBlocks,
      kMaxBlockCount,
      "Row count too large for BlockBitPacking encoding.");

  ScopedVector<BlockInfo> blocks{numBlocks, pool, options.bufferPool};

  uint32_t totalPackedSize = 0;
  for (uint16_t blockIndex = 0; blockIndex < numBlocks; ++blockIndex) {
    const auto start = static_cast<uint32_t>(blockIndex) * blockSize;
    const auto end = std::min(start + blockSize, rowCount);
    blocks[blockIndex] = makeBlockInfo(values, start, end - start);
    totalPackedSize += blocks[blockIndex].packedSize;
  }

  // Note: if the encoded size (packed data + nested metadata sub-streams +
  // header) >= rawSize, this encoding is not beneficial. The encoding selection
  // policy should avoid selecting it in that case. We still encode correctly so
  // the round-trip contract holds.
  ScopedVector<char> packedVector{/*size=*/0, pool, options.bufferPool};
  auto dataCompressionPolicy = selection.compressionPolicy();
  CompressionEncoder<T> compressionEncoder{
      *pool,
      *dataCompressionPolicy,
      DataType::Undefined,
      /*bitWidth=*/8,
      /*uncompressedSize=*/totalPackedSize,
      [&]() {
        packedVector.resize(totalPackedSize);
        return std::span<char>{packedVector.data(), packedVector.size()};
      },
      [&](char*& pos) {
        writePayload(
            values,
            std::span<const BlockInfo>{blocks.data(), blocks.size()},
            totalPackedSize,
            pos);
        return pos;
      }};

  // BlockBitPacking routes the per-block metadata (baselines, bit widths, data
  // offsets) through recursive encoding selection so Nimble can pick the best
  // encoding for each (e.g. Constant/Delta for correlated baselines, Delta for
  // the ascending offsets). The packed block data stays raw (still optionally
  // Zstd-compressed via the compression encoder).
  ScopedVector<physicalType> baselines{numBlocks, pool, options.bufferPool};
  ScopedVector<uint8_t> bitWidths{numBlocks, pool, options.bufferPool};
  ScopedVector<uint32_t> blockOffsets{numBlocks, pool, options.bufferPool};
  uint32_t runningOffset{0};
  for (uint16_t blockIndex = 0; blockIndex < numBlocks; ++blockIndex) {
    baselines[blockIndex] = blocks[blockIndex].baseline;
    bitWidths[blockIndex] = blocks[blockIndex].skipEncoding
        ? kRawBlockBitWidth
        : blocks[blockIndex].bitWidth;
    blockOffsets[blockIndex] = runningOffset;
    runningOffset += blocks[blockIndex].packedSize;
  }

  ScopedEncodingBuffer tempBuffer{pool, options.encodingBufferPool};
  const std::string_view encodedBaselines =
      selection.template encodeNested<physicalType>(
          EncodingIdentifiers::BlockBitPacking::Baselines,
          std::span<const physicalType>{baselines.data(), baselines.size()},
          tempBuffer.get(),
          options);
  const std::string_view encodedBitWidths =
      selection.template encodeNested<uint8_t>(
          EncodingIdentifiers::BlockBitPacking::BitWidths,
          std::span<const uint8_t>{bitWidths.data(), bitWidths.size()},
          tempBuffer.get(),
          options);
  const std::string_view encodedOffsets =
      selection.template encodeNested<uint32_t>(
          EncodingIdentifiers::BlockBitPacking::Offsets,
          std::span<const uint32_t>{blockOffsets.data(), blockOffsets.size()},
          tempBuffer.get(),
          options);
  const auto firstBlockRows = std::min<uint32_t>(blockSize, rowCount);

  const uint32_t headerSize = BlockBitPackingEncoding<T>::headerSize(
      blockSize,
      numBlocks,
      static_cast<uint32_t>(encodedBaselines.size()),
      static_cast<uint32_t>(encodedBitWidths.size()),
      static_cast<uint32_t>(encodedOffsets.size()),
      firstBlockRows);
  const uint32_t encodingSize = Encoding::serializePrefixSize(
                                    rowCount, options.useVarintRowCount) +
      headerSize +
      static_cast<uint32_t>(encodedBaselines.size() + encodedBitWidths.size() +
                            encodedOffsets.size()) +
      compressionEncoder.getSize();

  char* reserved = buffer.reserve(encodingSize);
  char* pos = reserved;
  Encoding::serializePrefix(
      EncodingType::BlockBitPacking,
      TypeTraits<T>::dataType,
      rowCount,
      options.useVarintRowCount,
      pos);
  encoding::writeChar(
      static_cast<char>(compressionEncoder.compressionType()), pos);
  varint::writeVarint(blockSize, &pos);
  varint::writeVarint(numBlocks, &pos);
  encoding::writeVarintString(encodedBaselines, pos);
  encoding::writeVarintString(encodedBitWidths, pos);
  encoding::writeVarintString(encodedOffsets, pos);
  varint::writeVarint(firstBlockRows, &pos);
  compressionEncoder.write(pos);

  NIMBLE_CHECK_EQ(encodingSize, pos - reserved, "Encoding size mismatch.");
  return {reserved, encodingSize};
}

template <typename T>
std::string_view BlockBitPackingEncoding<T>::slice(
    std::string_view encoded,
    uint32_t offset,
    uint32_t length,
    Buffer& buffer,
    const Encoding::Options& options) {
  auto* pool = &buffer.getMemoryPool();

  velox::BufferPtr uncompressedSourceData;
  SCOPE_EXIT {
    if (options.bufferPool != nullptr && uncompressedSourceData != nullptr) {
      options.bufferPool->release(std::move(uncompressedSourceData));
    }
  };
  const auto source = parseHeader(encoded, options);
  NIMBLE_CHECK_LE(offset, source.rowCount);
  NIMBLE_CHECK_LE(length, source.rowCount - offset);
  NIMBLE_CHECK_GT(length, 0, "Cannot slice zero rows.");

  const auto packedData =
      packedDataSource(source, uncompressedSourceData, pool, options);

  const auto firstBlock = blockPosition(source, offset).blockIndex;
  const auto lastBlock = blockPosition(source, offset + length - 1).blockIndex;
  const auto numBlocks = lastBlock - firstBlock + 1;
  NIMBLE_CHECK_LE(
      numBlocks,
      kMaxBlockCount,
      "Row count too large for BlockBitPacking encoding.");

  ScopedVector<uint32_t> blockOffsets{
      static_cast<uint64_t>(numBlocks) + 1, pool, options.bufferPool};
  ScopedEncodingBuffer scopedBuffer{pool, options.encodingBufferPool};
  readBlockOffsets(
      source,
      firstBlock,
      numBlocks,
      static_cast<uint32_t>(packedData.size()),
      blockOffsets,
      scopedBuffer.get(),
      options);

  const auto getPackedSize = [&](uint32_t blockIndex) {
    const auto blockOffset = blockIndex - firstBlock;
    return blockOffsets[blockOffset + 1] - blockOffsets[blockOffset];
  };
  ScopedVector<uint8_t> bitWidths{numBlocks, pool, options.bufferPool};
  readMetadataRange(
      source.encodedBitWidths,
      firstBlock,
      numBlocks,
      bitWidths,
      scopedBuffer.get(),
      options);
  const auto getBlockBitWidth = [&](uint32_t blockIndex) {
    return bitWidths[blockIndex - firstBlock];
  };

  ScopedVector<BlockSliceInfo> blockSlices{numBlocks, pool, options.bufferPool};
  ScopedVector<uint32_t> sliceOffsets{
      static_cast<uint64_t>(numBlocks) + 1, pool, options.bufferPool};
  uint32_t totalPackedSize{0};
  uint32_t curRow{offset};
  const auto endRow = offset + length;
  for (uint16_t sliceIndex = 0; sliceIndex < numBlocks; ++sliceIndex) {
    const auto position = blockPosition(source, curRow);
    const auto blockIndex = position.blockIndex;
    NIMBLE_CHECK_EQ(blockIndex, firstBlock + sliceIndex);
    const auto rowOffset = position.rowOffset;
    const auto rowCount =
        std::min<uint32_t>(endRow - curRow, position.rowCount - rowOffset);
    const auto fullBlock = rowOffset == 0 && rowCount == position.rowCount;
    const auto bitWidth = fullBlock ? uint8_t{0} : getBlockBitWidth(blockIndex);
    const auto skipEncoding = bitWidth == kRawBlockBitWidth;
    const uint32_t packedOffsetAdjustment = skipEncoding
        ? static_cast<uint32_t>(rowOffset * sizeof(physicalType))
        : uint32_t{0};
    blockSlices[sliceIndex] = {
        .blockIndex = blockIndex,
        .packedOffset =
            blockOffsets[blockIndex - firstBlock] + packedOffsetAdjustment,
        .rowOffset = rowOffset,
        .packedSize = fullBlock
            ? getPackedSize(blockIndex)
            : BlockBitPackingEncoding::getPackedSize(rowCount, bitWidth),
        .rowCount = rowCount,
        .bitWidth = skipEncoding ? uint8_t{0} : bitWidth,
        .skipEncoding = skipEncoding};
    sliceOffsets[sliceIndex] = totalPackedSize;
    totalPackedSize += blockSlices[sliceIndex].packedSize;
    curRow += rowCount;
  }
  sliceOffsets[numBlocks] = totalPackedSize;
  NIMBLE_CHECK_EQ(curRow, endRow);

  const auto encodedBaselines = EncodingFactory::slice(
      source.encodedBaselines,
      firstBlock,
      numBlocks,
      scopedBuffer.get(),
      options);
  const auto encodedBitWidths = EncodingFactory::slice(
      source.encodedBitWidths,
      firstBlock,
      numBlocks,
      scopedBuffer.get(),
      options);
  const auto encodedOffsets =
      EncodingFactory::encodeWithCapturedLayout<uint32_t>(
          source.encodedBlockOffsets,
          std::span<const uint32_t>{sliceOffsets.data(), numBlocks},
          scopedBuffer.get(),
          options,
          "Captured BlockBitPacking offset layout");

  const auto firstBlockRows = blockSlices[0].rowCount;
  const uint32_t headerSize = BlockBitPackingEncoding<T>::headerSize(
      source.blockSize,
      numBlocks,
      static_cast<uint32_t>(encodedBaselines.size()),
      static_cast<uint32_t>(encodedBitWidths.size()),
      static_cast<uint32_t>(encodedOffsets.size()),
      firstBlockRows);
  const uint32_t encodingSize = Encoding::serializePrefixSize(
                                    length, options.useVarintRowCount) +
      headerSize +
      static_cast<uint32_t>(encodedBaselines.size() + encodedBitWidths.size() +
                            encodedOffsets.size()) +
      totalPackedSize;

  char* reserved = buffer.reserve(encodingSize);
  char* pos = reserved;
  Encoding::serializePrefix(
      EncodingType::BlockBitPacking,
      TypeTraits<T>::dataType,
      length,
      options.useVarintRowCount,
      pos);
  encoding::writeChar(static_cast<char>(CompressionType::Uncompressed), pos);
  varint::writeVarint(source.blockSize, &pos);
  varint::writeVarint(numBlocks, &pos);
  encoding::writeVarintString(encodedBaselines, pos);
  encoding::writeVarintString(encodedBitWidths, pos);
  encoding::writeVarintString(encodedOffsets, pos);
  varint::writeVarint(firstBlockRows, &pos);

  writeBlockSlicesPayload(
      source,
      packedData,
      std::span<const BlockSliceInfo>{blockSlices.data(), blockSlices.size()},
      std::span<const uint32_t>{sliceOffsets.data(), sliceOffsets.size()},
      totalPackedSize,
      pos);

  NIMBLE_CHECK_EQ(encodingSize, pos - reserved, "Encoding size mismatch.");
  return {reserved, encodingSize};
}

template <typename T>
std::string BlockBitPackingEncoding<T>::debugString(int offset) const {
  return fmt::format(
      "{}{}<{}> rowCount={} blockSize={} numBlocks={}",
      std::string(offset, ' '),
      toString(Encoding::encodingType()),
      toString(Encoding::dataType()),
      Encoding::rowCount(),
      blockSize_,
      numBlocks_);
}

} // namespace facebook::nimble
