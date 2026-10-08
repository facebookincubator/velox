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

#include <folly/Synchronized.h>
#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "velox/dwio/nimble/encodings/FsstEncoding.h"
#include "velox/dwio/nimble/encodings/views/EncodingView.h"

namespace facebook::nimble {

/// A random-access FSST view that decodes strings lazily in cached chunks.
///
/// FSST's native decoder already unrolls four codes and uses wide symbol
/// stores. Chunking amortizes view bookkeeping without forcing callers to
/// materialize the whole stream, and cached chunks keep returned string_views
/// stable.
class FsstEncodingView final : public TypedEncodingView<std::string_view> {
 public:
  FsstEncodingView(
      std::string_view data,
      velox::memory::MemoryPool* pool,
      const Encoding::Options& options)
      : TypedEncodingView<std::string_view>{
            FsstEncoding::validateEncodedPrefix(data, options),
            pool,
            options} {
    NIMBLE_CHECK_EQ(this->encodingType_, EncodingType::Fsst);
    const auto header = FsstEncoding::parseHeader(data, this->dataOffset_);
    FsstEncoding::validateSymbolTable(header.symbolTable);
    const auto bytesConsumed = nimble_fsst_import(
        &decoder_,
        const_cast<unsigned char*>(
            reinterpret_cast<const unsigned char*>(header.symbolTable.data())));
    NIMBLE_CHECK_FILE_EQ(
        static_cast<size_t>(bytesConsumed),
        header.symbolTable.size(),
        "FSST symbol table import size mismatch.");

    auto noStringBufferFactory = [](uint32_t) -> void* { return nullptr; };
    auto lengthsEncoding = EncodingFactory().create(
        *this->pool_, header.lengths, noStringBufferFactory, options);
    NIMBLE_CHECK_NOT_NULL(lengthsEncoding);
    NIMBLE_CHECK_FILE_EQ(
        lengthsEncoding->dataType(),
        DataType::Uint32,
        "FSST lengths encoding must contain Uint32 values.");
    NIMBLE_CHECK_FILE(
        !lengthsEncoding->isNullable(),
        "FSST lengths encoding must not be nullable.");
    NIMBLE_CHECK_FILE_EQ(
        lengthsEncoding->rowCount(),
        this->rowCount_,
        "FSST lengths row count does not match the parent encoding.");

    if (header.compressionType == CompressionType::Uncompressed) {
      blob_ = header.blob;
    } else {
      blob_ = this->decompressPayload(
          header.compressionType, DataType::String, header.blob);
    }

    std::vector<uint32_t> lengths(this->rowCount_);
    if (!lengths.empty()) {
      lengthsEncoding->materialize(
          static_cast<uint32_t>(lengths.size()), lengths.data());
    }
    const auto compressedBytes = FsstEncoding::validateCompressedLengths(
        lengths, blob_, /*blobOffset=*/0);
    NIMBLE_CHECK_FILE_EQ(
        compressedBytes,
        blob_.size(),
        "FSST compressed lengths do not match the blob size.");

    compressedOffsets_.resize(this->rowCount_ + 1);
    for (uint32_t row = 0; row < this->rowCount_; ++row) {
      compressedOffsets_[row + 1] = compressedOffsets_[row] + lengths[row];
    }
    decodedChunks_.wlock()->resize(
        (this->rowCount_ + kRowsPerChunk - 1) / kRowsPerChunk);
  }

 private:
  static constexpr uint32_t kRowsPerChunk = 1'024;
  static constexpr size_t kMaxSymbolLength = 8;

  struct DecodedChunk {
    std::string data;
    std::vector<size_t> offsets;
  };

  std::string_view readTypedAt(uint32_t index) const final {
    NIMBLE_CHECK_LT(index, this->rowCount_);
    const auto chunkIndex = index / kRowsPerChunk;
    const auto chunk = getChunk(chunkIndex);
    const auto rowInChunk = index % kRowsPerChunk;
    return valueAt(*chunk, rowInChunk);
  }

  void readPhysical(uint32_t offset, uint32_t length, std::string_view* output)
      const final {
    this->checkReadRange(offset, length);
    uint32_t outputOffset{0};
    while (outputOffset < length) {
      const auto row = offset + outputOffset;
      const auto chunkIndex = row / kRowsPerChunk;
      const auto rowInChunk = row % kRowsPerChunk;
      const auto chunk = getChunk(chunkIndex);
      const auto rowsInChunk = static_cast<uint32_t>(chunk->offsets.size() - 1);
      const auto rowsToCopy =
          std::min(length - outputOffset, rowsInChunk - rowInChunk);
      for (uint32_t i = 0; i < rowsToCopy; ++i) {
        output[outputOffset + i] = valueAt(*chunk, rowInChunk + i);
      }
      outputOffset += rowsToCopy;
    }
  }

  void readPhysicalAt(
      std::span<const uint32_t> indices,
      std::string_view* output) const final {
    uint32_t cachedChunkIndex = std::numeric_limits<uint32_t>::max();
    std::shared_ptr<const DecodedChunk> cachedChunk;
    for (size_t i = 0; i < indices.size(); ++i) {
      const auto row = indices[i];
      NIMBLE_CHECK_LT(row, this->rowCount_);
      const auto chunkIndex = row / kRowsPerChunk;
      if (chunkIndex != cachedChunkIndex) {
        cachedChunk = getChunk(chunkIndex);
        cachedChunkIndex = chunkIndex;
      }
      output[i] = valueAt(*cachedChunk, row % kRowsPerChunk);
    }
  }

  static std::string_view valueAt(
      const DecodedChunk& chunk,
      uint32_t rowInChunk) {
    const auto begin = chunk.offsets[rowInChunk];
    const auto end = chunk.offsets[rowInChunk + 1];
    return {chunk.data.data() + begin, end - begin};
  }

  std::shared_ptr<const DecodedChunk> getChunk(uint32_t chunkIndex) const {
    {
      const auto chunks = decodedChunks_.rlock();
      if ((*chunks)[chunkIndex]) {
        return (*chunks)[chunkIndex];
      }
    }

    auto decoded = decodeChunk(chunkIndex);
    auto chunks = decodedChunks_.wlock();
    auto& cached = (*chunks)[chunkIndex];
    if (!cached) {
      cached = std::move(decoded);
    }
    return cached;
  }

  std::shared_ptr<const DecodedChunk> decodeChunk(uint32_t chunkIndex) const {
    const auto firstRow = chunkIndex * kRowsPerChunk;
    const auto endRow = std::min(this->rowCount_, firstRow + kRowsPerChunk);
    auto chunk = std::make_shared<DecodedChunk>();
    chunk->offsets.resize(endRow - firstRow + 1);

    size_t maxOutputBytes{0};
    for (auto row = firstRow; row < endRow; ++row) {
      const auto compressedSize =
          compressedOffsets_[row + 1] - compressedOffsets_[row];
      const size_t padding = compressedSize == 0 ? 0 : 1;
      NIMBLE_CHECK_FILE_LE(
          maxOutputBytes,
          std::numeric_limits<size_t>::max() - padding,
          "FSST decoded chunk exceeds the supported size.");
      NIMBLE_CHECK_FILE_LE(
          compressedSize,
          (std::numeric_limits<size_t>::max() - maxOutputBytes - padding) /
              kMaxSymbolLength,
          "FSST decoded chunk exceeds the supported size.");
      maxOutputBytes += compressedSize * kMaxSymbolLength + padding;
    }
    chunk->data.resize(maxOutputBytes);

    size_t outputOffset{0};
    for (auto row = firstRow; row < endRow; ++row) {
      const auto compressedBegin = compressedOffsets_[row];
      const auto compressedSize = compressedOffsets_[row + 1] - compressedBegin;
      if (compressedSize != 0) {
        const auto maxDecodedSize = compressedSize * kMaxSymbolLength;
        const auto decodedSize = nimble_fsst_decompress(
            &decoder_,
            compressedSize,
            reinterpret_cast<const unsigned char*>(
                blob_.data() + compressedBegin),
            maxDecodedSize + 1,
            reinterpret_cast<unsigned char*>(
                chunk->data.data() + outputOffset));
        NIMBLE_CHECK_GT(
            decodedSize,
            0,
            "FSST decompression failed for non-empty compressed string.");
        NIMBLE_CHECK_FILE_LE(
            decodedSize,
            maxDecodedSize,
            "FSST decompressed string exceeds the maximum expansion bound.");
        outputOffset += decodedSize;
      }
      chunk->offsets[row - firstRow + 1] = outputOffset;
    }
    chunk->data.resize(outputOffset);
    return chunk;
  }

  fsst_decoder_t decoder_{};
  std::string_view blob_;
  std::vector<size_t> compressedOffsets_;
  // TODO: Limit the number of chunks cached by each FSST view.
  mutable folly::Synchronized<std::vector<std::shared_ptr<const DecodedChunk>>>
      decodedChunks_;
};

} // namespace facebook::nimble
