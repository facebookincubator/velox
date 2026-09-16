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
#include <cstring>
#include <limits>
#include <optional>
#include <span>
#include <type_traits>

#include <fmt/format.h>
#include <folly/CPortability.h>

#include "velox/common/base/BitUtil.h"
#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/common/FixedBitArray.h"
#include "velox/dwio/nimble/common/Varint.h"
#include "velox/dwio/nimble/common/Vector.h"
#include "velox/dwio/nimble/encodings/common/Encoding.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrimitives.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelection.h"

namespace facebook::nimble {

template <typename T>
class DeltaBlockEncodingView;

/// Stores sorted integer streams as block checkpoints plus fixed-width deltas
/// inside each block. This is intended for sorted dictionary alphabets, where
/// values are monotonic and per-block gaps are substantially narrower than the
/// original integer width.
///
/// DeltaEncoding handles arbitrary integer streams by writing decreasing values
/// as restatements in a child stream. DeltaBlock has no restatement stream; it
/// requires non-decreasing values and stores one base value per block plus
/// deltas to subsequent rows in that block.
template <typename T>
class DeltaBlockEncoding final
    : public TypedEncoding<T, typename TypeTraits<T>::physicalType> {
 public:
  using cppDataType = T;
  using physicalType = typename TypeTraits<T>::physicalType;

  DeltaBlockEncoding(
      velox::memory::MemoryPool& pool,
      std::string_view data,
      const std::function<void*(uint32_t)>& stringBufferFactory,
      const Encoding::Options& options = {});

  ~DeltaBlockEncoding() override {
    this->releaseVectorBuffer(deltaChunk_);
    this->releaseVectorBuffer(blockBases_);
    this->releaseVectorBuffer(bitWidths_);
    this->releaseVectorBuffer(blockOffsets_);
  }

  DeltaBlockEncoding(const DeltaBlockEncoding&) = delete;
  DeltaBlockEncoding& operator=(const DeltaBlockEncoding&) = delete;
  DeltaBlockEncoding(DeltaBlockEncoding&&) = delete;
  DeltaBlockEncoding& operator=(DeltaBlockEncoding&&) = delete;

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

  static std::optional<uint64_t> estimateSize(
      std::span<const physicalType> values,
      const Encoding::Options& options);

  std::string debugString(int offset) const final;

 private:
  static_assert(
      std::is_integral_v<T> && !std::is_same_v<T, bool>,
      "DeltaBlockEncoding only supports non-bool integer types.");

  friend class DeltaBlockEncodingView<T>;

  static constexpr uint64_t kSignMask{
      uint64_t{1} << (sizeof(physicalType) * 8 - 1)};
  static constexpr uint32_t kDeltaChunkSize{1024};

  static uint64_t toOrdered(physicalType value) {
    if constexpr (std::is_signed_v<T>) {
      return static_cast<uint64_t>(value) ^ kSignMask;
    } else {
      return static_cast<uint64_t>(value);
    }
  }

  static physicalType fromOrdered(uint64_t value) {
    if constexpr (std::is_signed_v<T>) {
      return static_cast<physicalType>(value ^ kSignMask);
    } else {
      return static_cast<physicalType>(value);
    }
  }

  static uint32_t
  blockRowCount(uint32_t rowCount, uint16_t blockSize, uint32_t blockIndex) {
    const auto start = blockIndex * blockSize;
    return std::min<uint32_t>(blockSize, rowCount - start);
  }

  static uint8_t bitsRequired(uint64_t value) {
    return value == 0 ? 0
                      : static_cast<uint8_t>(velox::bits::bitsRequired(value));
  }

  struct ReadWithVisitorState {
    std::optional<uint32_t> cachedBlockIndex;
    uint32_t cachedPosition{0};
    uint8_t cachedBitWidth{0};
    uint64_t cachedOrdered{0};
    FixedBitArray cachedDeltas{};
  };

  // Returns the number of rows in the given block, handling the final partial
  // block.
  uint32_t blockRowCount(uint32_t blockIndex) const;

  // Decodes a slice within a single block by reconstructing values from its
  // base and deltas.
  void materializeBlock(
      uint32_t blockIndex,
      uint32_t position,
      uint32_t rowCount,
      physicalType* output,
      uint64_t* deltaChunk);

  // Decodes a contiguous range that may span multiple blocks.
  void materializeContiguous(
      uint32_t startRow,
      uint32_t rowCount,
      physicalType* output);

  void seekReadWithVisitorState(ReadWithVisitorState& state, uint32_t row)
      const;

  physicalType decodeWithVisitorState(ReadWithVisitorState& state);

  // Number of rows per block, except possibly the last partial block.
  uint16_t blockSize_{0};
  // Total number of blocks in the encoding.
  uint32_t numBlocks_{0};
  // Per-block base values that anchor ordered delta reconstruction.
  Vector<physicalType> blockBases_;
  // Per-block bit width for delta storage. Zero indicates a constant block.
  Vector<uint8_t> bitWidths_;
  // Per-block byte offsets into packedData_ for the delta bit streams.
  Vector<uint32_t> blockOffsets_;
  Vector<uint64_t> deltaChunk_;
  // Pointer into the encoded payload that holds the packed delta bits.
  const char* packedData_{nullptr};
  // Current read cursor used by materialize() and skip().
  uint32_t row_{0};
};

template <typename T>
DeltaBlockEncoding<T>::DeltaBlockEncoding(
    velox::memory::MemoryPool& pool,
    std::string_view data,
    const std::function<void*(uint32_t)>& /*stringBufferFactory*/,
    const Encoding::Options& options)
    : TypedEncoding<T, physicalType>{pool, data, options},
      blockBases_{this->template getVectorBuffer<physicalType>()},
      bitWidths_{this->template getVectorBuffer<uint8_t>()},
      blockOffsets_{this->template getVectorBuffer<uint32_t>()},
      deltaChunk_{this->template getVectorBuffer<uint64_t>()} {
  const char* pos = data.data() + this->dataOffset();
  blockSize_ = static_cast<uint16_t>(varint::readVarint32(&pos));
  numBlocks_ = varint::readVarint32(&pos);
  NIMBLE_CHECK_GT(blockSize_, 0);
  NIMBLE_CHECK_EQ(
      numBlocks_,
      velox::bits::divRoundUp(this->rowCount(), blockSize_),
      "Invalid DeltaBlock block count.");

  blockBases_.resize(numBlocks_);
  for (uint32_t i = 0; i < numBlocks_; ++i) {
    blockBases_[i] =
        DeltaBlockEncoding<T>::fromOrdered(varint::readVarint64(&pos));
  }

  bitWidths_.resize(numBlocks_);
  std::memcpy(bitWidths_.data(), pos, numBlocks_ * sizeof(uint8_t));
  pos += numBlocks_ * sizeof(uint8_t);

  blockOffsets_.resize(numBlocks_);
  for (uint32_t i = 0; i < numBlocks_; ++i) {
    blockOffsets_[i] = varint::readVarint32(&pos);
  }
  packedData_ = pos;
}

template <typename T>
void DeltaBlockEncoding<T>::reset() {
  row_ = 0;
}

template <typename T>
void DeltaBlockEncoding<T>::skip(uint32_t rowCount) {
  NIMBLE_CHECK_LE(row_, this->rowCount(), "Invalid encoding position.");
  NIMBLE_CHECK_LE(
      rowCount, this->rowCount() - row_, "Skipping past end of encoding.");
  row_ += rowCount;
}

template <typename T>
uint32_t DeltaBlockEncoding<T>::blockRowCount(uint32_t blockIndex) const {
  return DeltaBlockEncoding<T>::blockRowCount(
      this->rowCount(), blockSize_, blockIndex);
}

template <typename T>
void DeltaBlockEncoding<T>::materializeBlock(
    uint32_t blockIndex,
    uint32_t position,
    uint32_t rowCount,
    physicalType* output,
    uint64_t* deltaChunk) {
  NIMBLE_CHECK_LE(position + rowCount, blockRowCount(blockIndex));
  if (rowCount == 0) {
    return;
  }

  uint64_t ordered = DeltaBlockEncoding<T>::toOrdered(blockBases_[blockIndex]);
  const auto bitWidth = bitWidths_[blockIndex];
  if (bitWidth == 0) {
    std::fill_n(output, rowCount, DeltaBlockEncoding<T>::fromOrdered(ordered));
    return;
  }

  FixedBitArray deltas{
      {packedData_ + blockOffsets_[blockIndex],
       FixedBitArray::bufferSize(blockRowCount(blockIndex) - 1, bitWidth)},
      bitWidth};
  // Apply deltas before `position` so `ordered` is the first requested value.
  for (uint32_t offset = 0; offset < position; offset += kDeltaChunkSize) {
    const auto chunkSize =
        std::min<uint32_t>(kDeltaChunkSize, position - offset);
    deltas.bulkGetWithBaseline(
        offset, chunkSize, deltaChunk, /*baseline=*/uint64_t{0});
    for (uint32_t i = 0; i < chunkSize; ++i) {
      ordered += deltaChunk[i];
    }
  }

  // Deltas are stored between adjacent rows. After reconstructing the value at
  // `position`, delta `position + offset` advances to the next output row.
  output[0] = DeltaBlockEncoding<T>::fromOrdered(ordered);
  for (uint32_t offset = 0; offset + 1 < rowCount;) {
    const auto chunkSize =
        std::min<uint32_t>(kDeltaChunkSize, rowCount - offset - 1);
    deltas.bulkGetWithBaseline(
        position + offset, chunkSize, deltaChunk, /*baseline=*/uint64_t{0});
    for (uint32_t i = 0; i < chunkSize; ++i) {
      ordered += deltaChunk[i];
      output[offset + i + 1] = DeltaBlockEncoding<T>::fromOrdered(ordered);
    }
    offset += chunkSize;
  }
}

template <typename T>
void DeltaBlockEncoding<T>::materializeContiguous(
    uint32_t startRow,
    uint32_t rowCount,
    physicalType* output) {
  deltaChunk_.resize(kDeltaChunkSize);
  uint32_t written{0};
  while (written < rowCount) {
    const auto row = startRow + written;
    const auto blockIndex = row / blockSize_;
    const auto position = row % blockSize_;
    const auto rowsInBlock = std::min<uint32_t>(
        rowCount - written, blockRowCount(blockIndex) - position);
    materializeBlock(
        blockIndex,
        position,
        rowsInBlock,
        output + written,
        deltaChunk_.data());
    written += rowsInBlock;
  }
}

template <typename T>
void DeltaBlockEncoding<T>::materialize(uint32_t rowCount, void* buffer) {
  NIMBLE_CHECK_LE(row_, this->rowCount(), "Invalid encoding position.");
  NIMBLE_CHECK_LE(
      rowCount, this->rowCount() - row_, "Reading past end of encoding.");
  auto* output = static_cast<physicalType*>(buffer);
  materializeContiguous(row_, rowCount, output);
  row_ += rowCount;
}

template <typename T>
void DeltaBlockEncoding<T>::seekReadWithVisitorState(
    ReadWithVisitorState& state,
    uint32_t row) const {
  const auto blockIndex = row / blockSize_;
  const auto position = row % blockSize_;
  state.cachedBlockIndex = blockIndex;
  state.cachedPosition = 0;
  state.cachedBitWidth = bitWidths_[blockIndex];
  state.cachedOrdered =
      DeltaBlockEncoding<T>::toOrdered(blockBases_[blockIndex]);
  if (state.cachedBitWidth != 0) {
    state.cachedDeltas = FixedBitArray{
        {packedData_ + blockOffsets_[blockIndex],
         FixedBitArray::bufferSize(
             blockRowCount(blockIndex) - 1, state.cachedBitWidth)},
        state.cachedBitWidth};
    for (uint32_t i = 0; i < position; ++i) {
      state.cachedOrdered += state.cachedDeltas.get(i);
    }
  }
  state.cachedPosition = position;
}

template <typename T>
typename DeltaBlockEncoding<T>::physicalType
DeltaBlockEncoding<T>::decodeWithVisitorState(ReadWithVisitorState& state) {
  const auto blockIndex = row_ / blockSize_;
  const auto position = row_ % blockSize_;
  if (!state.cachedBlockIndex.has_value() ||
      blockIndex != state.cachedBlockIndex.value() ||
      position < state.cachedPosition) {
    seekReadWithVisitorState(state, row_);
  } else if (position > state.cachedPosition) {
    if (state.cachedBitWidth != 0) {
      for (uint32_t i = state.cachedPosition; i < position; ++i) {
        state.cachedOrdered += state.cachedDeltas.get(i);
      }
    }
    state.cachedPosition = position;
  }
  ++row_;
  return DeltaBlockEncoding<T>::fromOrdered(state.cachedOrdered);
}

template <typename T>
template <typename DecoderVisitor>
void DeltaBlockEncoding<T>::readWithVisitor(
    DecoderVisitor& visitor,
    ReadWithVisitorParams& params) {
  ReadWithVisitorState state;

  detail::readWithVisitorSlow(
      visitor,
      params,
      [&](auto toSkip) { skip(toSkip); },
      [&] { return decodeWithVisitorState(state); });
}

template <typename T>
std::optional<uint64_t> DeltaBlockEncoding<T>::estimateSize(
    std::span<const physicalType> values,
    const Encoding::Options& options) {
  const auto blockSize = options.deltaBlockSize;
  if (values.empty() || blockSize == 0) {
    return std::nullopt;
  }

  const auto rowCount = static_cast<uint32_t>(values.size());
  const auto numBlocks = velox::bits::divRoundUp(rowCount, blockSize);
  uint64_t packedSize{0};
  uint64_t blockBasesSize{0};
  uint64_t blockOffsetsSize{0};
  for (uint32_t blockIndex = 0; blockIndex < numBlocks; ++blockIndex) {
    const uint32_t start = blockIndex * blockSize;
    const uint32_t end = std::min<uint32_t>(start + blockSize, rowCount);
    blockBasesSize +=
        varint::varintSize(DeltaBlockEncoding<T>::toOrdered(values[start]));
    uint64_t prev = DeltaBlockEncoding<T>::toOrdered(values[start]);
    uint64_t maxDelta{0};
    for (uint32_t i = start + 1; i < end; ++i) {
      const auto curr = DeltaBlockEncoding<T>::toOrdered(values[i]);
      if (curr < prev) {
        return std::nullopt;
      }
      maxDelta = std::max(maxDelta, curr - prev);
      prev = curr;
    }
    // Each block stores the first row as a base, then one delta per remaining
    // row.
    const auto deltaCount = end - start - 1;
    blockOffsetsSize += varint::varintSize(packedSize);
    if (deltaCount != 0 && maxDelta != 0) {
      packedSize +=
          FixedBitArray::bufferSize(deltaCount, bitsRequired(maxDelta));
    }
  }

  return Encoding::serializePrefixSize(rowCount, options.useVarintRowCount) +
      varint::varintSize(blockSize) + varint::varintSize(numBlocks) +
      blockBasesSize + /*bitWidths=*/numBlocks * sizeof(uint8_t) +
      blockOffsetsSize + packedSize;
}

template <typename T>
std::string_view DeltaBlockEncoding<T>::encode(
    EncodingSelection<physicalType>& /*selection*/,
    std::span<const physicalType> values,
    Buffer& buffer,
    const Encoding::Options& options) {
  NIMBLE_CHECK(!values.empty(), "DeltaBlock cannot encode empty streams.");

  const auto blockSize = options.deltaBlockSize;
  NIMBLE_CHECK_GT(blockSize, 0);
  const auto rowCount = static_cast<uint32_t>(values.size());
  const auto numBlocks = velox::bits::divRoundUp(rowCount, blockSize);
  auto* pool = &buffer.getMemoryPool();
  Vector<physicalType> blockBases{pool, numBlocks};
  Vector<uint8_t> bitWidths{pool, numBlocks};
  Vector<uint32_t> blockOffsets{pool, numBlocks};
  uint64_t packedSize{0};
  uint64_t blockBasesSize{0};
  uint64_t blockOffsetsSize{0};

  for (uint32_t blockIndex = 0; blockIndex < numBlocks; ++blockIndex) {
    const uint32_t start = blockIndex * blockSize;
    const uint32_t end = std::min<uint32_t>(start + blockSize, rowCount);
    const uint32_t count = end - start;

    blockBases[blockIndex] = values[start];
    blockBasesSize +=
        varint::varintSize(DeltaBlockEncoding<T>::toOrdered(values[start]));
    uint64_t prev = DeltaBlockEncoding<T>::toOrdered(values[start]);
    uint64_t maxDelta{0};
    for (uint32_t i = start + 1; i < end; ++i) {
      const auto curr = DeltaBlockEncoding<T>::toOrdered(values[i]);
      if (FOLLY_UNLIKELY(curr < prev)) {
        NIMBLE_INCOMPATIBLE_ENCODING(
            "DeltaBlock requires non-decreasing values.");
      }
      maxDelta = std::max(maxDelta, curr - prev);
      prev = curr;
    }

    const auto bitWidth = bitsRequired(maxDelta);
    bitWidths[blockIndex] = bitWidth;
    blockOffsets[blockIndex] = static_cast<uint32_t>(packedSize);
    blockOffsetsSize += varint::varintSize(packedSize);
    if (count <= 1 || bitWidth == 0) {
      continue;
    }

    const auto deltaCount = count - 1;
    const auto bytes = FixedBitArray::bufferSize(deltaCount, bitWidth);
    NIMBLE_CHECK_LE(
        packedSize + bytes,
        std::numeric_limits<uint32_t>::max(),
        "DeltaBlock payload exceeds uint32 offsets.");
    packedSize += bytes;
  }

  ScopedEncodingBuffer scopedBuffer{pool, options.encodingBufferPool};
  char* packedData{nullptr};
  if (packedSize != 0) {
    packedData = scopedBuffer.get().reserve(packedSize);
    std::memset(packedData, 0, packedSize);
  }

  for (uint32_t blockIndex = 0; blockIndex < numBlocks; ++blockIndex) {
    const auto bitWidth = bitWidths[blockIndex];
    if (bitWidth == 0) {
      continue;
    }

    const uint32_t start = blockIndex * blockSize;
    const uint32_t end = std::min<uint32_t>(start + blockSize, rowCount);
    if (end - start <= 1) {
      continue;
    }

    FixedBitArray deltas{packedData + blockOffsets[blockIndex], bitWidth};
    uint64_t prev = DeltaBlockEncoding<T>::toOrdered(values[start]);
    for (uint32_t i = start + 1; i < end; ++i) {
      const auto curr = DeltaBlockEncoding<T>::toOrdered(values[i]);
      deltas.set(i - start - 1, curr - prev);
      prev = curr;
    }
  }

  const uint64_t encodingSize =
      Encoding::serializePrefixSize(rowCount, options.useVarintRowCount) +
      varint::varintSize(blockSize) + varint::varintSize(numBlocks) +
      /*blockBases=*/blockBasesSize +
      /*bitWidths=*/numBlocks * sizeof(uint8_t) +
      /*blockOffsets=*/blockOffsetsSize + /*packedData=*/packedSize;
  NIMBLE_CHECK_LE(
      encodingSize,
      std::numeric_limits<uint32_t>::max(),
      "DeltaBlock encoding exceeds uint32 size.");

  char* reserved = buffer.reserve(encodingSize);
  char* pos = reserved;
  Encoding::serializePrefix(
      EncodingType::DeltaBlock,
      TypeTraits<T>::dataType,
      rowCount,
      options.useVarintRowCount,
      pos);
  varint::writeVarint(blockSize, &pos);
  varint::writeVarint(numBlocks, &pos);
  for (uint32_t i = 0; i < numBlocks; ++i) {
    varint::writeVarint(DeltaBlockEncoding<T>::toOrdered(blockBases[i]), &pos);
  }
  std::memcpy(pos, bitWidths.data(), numBlocks * sizeof(uint8_t));
  pos += numBlocks * sizeof(uint8_t);
  for (uint32_t i = 0; i < numBlocks; ++i) {
    varint::writeVarint(blockOffsets[i], &pos);
  }
  if (packedSize != 0) {
    std::memcpy(pos, packedData, packedSize);
    pos += packedSize;
  }
  NIMBLE_CHECK_EQ(pos - reserved, encodingSize, "Encoding size mismatch.");
  return {reserved, static_cast<size_t>(encodingSize)};
}

template <typename T>
std::string DeltaBlockEncoding<T>::debugString(int offset) const {
  return Encoding::debugString(offset) +
      fmt::format(
             "\n{}blockSize={}, numBlocks={}",
             std::string(offset, ' '),
             blockSize_,
             numBlocks_);
}

} // namespace facebook::nimble
