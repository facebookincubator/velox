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

#include "velox/dwio/nimble/encodings/views/FsstEncodingView.h"

#include <algorithm>
#include <array>
#include <limits>
#include <string>
#include <utility>

namespace facebook::nimble {

struct FsstEncodingView::DecodedChunk {
  std::string data;
  std::vector<size_t> offsets;
};

FsstEncodingView::FsstEncodingView(
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

  if (header.compressionType == CompressionType::Uncompressed) {
    blob_ = header.blob;
  } else {
    blob_ = this->decompressPayload(
        header.compressionType, DataType::String, header.blob);
  }

  if (options.cacheEncodingView) {
    initializeCached(header.lengths);
  } else {
    initializeUncached(header.lengths);
  }
}

void FsstEncodingView::initializeUncached(std::string_view encodedLengths) {
  NIMBLE_CHECK_FILE_EQ(
      EncodingPrefix::readDataType(encodedLengths),
      DataType::Uint32,
      "FSST lengths encoding must contain Uint32 values.");
  lengths_ = detail::createTypedEncodingView<uint32_t>(
      encodedLengths, this->pool_, this->options_);
  NIMBLE_CHECK_NOT_NULL(lengths_);
  NIMBLE_CHECK_FILE(
      lengths_->encodingType() != EncodingType::Nullable &&
          lengths_->encodingType() != EncodingType::Sentinel,
      "FSST lengths encoding must not be nullable: {}",
      lengths_->encodingType());
  NIMBLE_CHECK_FILE_EQ(
      lengths_->rowCount(),
      this->rowCount_,
      "FSST lengths row count does not match the parent encoding.");
}

void FsstEncodingView::initializeCached(std::string_view encodedLengths) {
  auto noStringBufferFactory = [](uint32_t) -> void* { return nullptr; };
  auto lengthsEncoding = EncodingFactory().create(
      *this->pool_, encodedLengths, noStringBufferFactory, this->options_);
  NIMBLE_CHECK_NOT_NULL(lengthsEncoding);
  NIMBLE_CHECK_FILE_EQ(
      lengthsEncoding->dataType(),
      DataType::Uint32,
      "FSST lengths encoding must contain Uint32 values.");
  NIMBLE_CHECK_FILE(
      !lengthsEncoding->isNullable(),
      "FSST lengths encoding must not be nullable: {}",
      lengthsEncoding->encodingType());
  NIMBLE_CHECK_FILE_EQ(
      lengthsEncoding->rowCount(),
      this->rowCount_,
      "FSST lengths row count does not match the parent encoding.");

  std::vector<uint32_t> lengths(this->rowCount_);
  if (!lengths.empty()) {
    lengthsEncoding->materialize(
        static_cast<uint32_t>(lengths.size()), lengths.data());
  }
  const auto compressedBytes =
      FsstEncoding::validateCompressedLengths(lengths, blob_, /*blobOffset=*/0);
  NIMBLE_CHECK_FILE_EQ(
      compressedBytes,
      blob_.size(),
      "FSST compressed lengths do not match the blob size.");

  compressedOffsets_.resize(this->rowCount_ + 1);
  for (uint32_t row = 0; row < this->rowCount_; ++row) {
    compressedOffsets_[row + 1] = compressedOffsets_[row] + lengths[row];
  }
  decodedChunks_.wlock()->resize(
      this->rowCount_ / kRowsPerChunk + (this->rowCount_ % kRowsPerChunk != 0));
}

std::string_view FsstEncodingView::readTypedAt(uint32_t index) const {
  NIMBLE_CHECK_LT(index, this->rowCount_);
  if (!cachesDecodedChunks()) {
    const auto size = lengths_->readAt(index);
    return decodeValue(index, lengths_->sum(0, index), size);
  }
  const auto chunkIndex = index / kRowsPerChunk;
  const auto chunk = getChunk(chunkIndex);
  const auto rowInChunk = index % kRowsPerChunk;
  return valueAt(*chunk, rowInChunk);
}

void FsstEncodingView::readPhysical(
    uint32_t offset,
    uint32_t length,
    std::string_view* output) const {
  this->checkReadRange(offset, length);
  if (!cachesDecodedChunks()) {
    // Sums the lengths before the range once and then walks the range.
    ScopedVector<uint32_t> sizes{
        length, this->pool_, this->options_.bufferPool};
    lengths_->read(offset, length, sizes.data());
    auto begin = lengths_->sum(0, offset);
    for (uint32_t i = 0; i < length; ++i) {
      output[i] = decodeValue(offset + i, begin, sizes[i]);
      begin += sizes[i];
    }
    return;
  }
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

void FsstEncodingView::readPhysicalAt(
    std::span<const uint32_t> indices,
    std::string_view* output) const {
  if (!cachesDecodedChunks()) {
    // Sums the lengths once for all indices, not once for each of them.
    ScopedVector<uint64_t> begins{
        indices.size(), this->pool_, this->options_.bufferPool};
    lengths_->sum(indices, begins.data());
    for (size_t i = 0; i < indices.size(); ++i) {
      output[i] =
          decodeValue(indices[i], begins[i], lengths_->readAt(indices[i]));
    }
    return;
  }
  uint32_t cachedChunkIndex = std::numeric_limits<uint32_t>::max();
  std::shared_ptr<const DecodedChunk> cachedChunk;
  for (size_t i = 0; i < indices.size(); ++i) {
    const auto row = indices[i];
    NIMBLE_DCHECK_LT(row, this->rowCount_);
    const auto chunkIndex = row / kRowsPerChunk;
    if (chunkIndex != cachedChunkIndex) {
      cachedChunk = getChunk(chunkIndex);
      cachedChunkIndex = chunkIndex;
    }
    output[i] = valueAt(*cachedChunk, row % kRowsPerChunk);
  }
}

std::string_view FsstEncodingView::decodeValue(
    uint32_t row,
    uint64_t compressedBegin,
    uint32_t compressedSize) const {
  if (const auto decoded = decodedStrings_.find(row); decoded.has_value()) {
    return *decoded;
  }

  // Validates only this string because the view skips the whole-chunk
  // validation that a cached view performs at construction.
  const std::array<uint32_t, 1> compressedSizes{compressedSize};
  FsstEncoding::validateCompressedLengths(
      compressedSizes, blob_, static_cast<size_t>(compressedBegin));
  std::string decoded;
  if (compressedSize != 0) {
    const auto maxDecodedSize =
        static_cast<size_t>(compressedSize) * kMaxSymbolLength;
    decoded.resize(maxDecodedSize + 1);
    const auto decodedSize = nimble_fsst_decompress(
        &decoder_,
        compressedSize,
        reinterpret_cast<const unsigned char*>(blob_.data() + compressedBegin),
        maxDecodedSize + 1,
        reinterpret_cast<unsigned char*>(decoded.data()));
    NIMBLE_CHECK_GT(
        decodedSize,
        0,
        "FSST decompression failed for non-empty compressed string.");
    NIMBLE_CHECK_FILE_LE(
        decodedSize,
        maxDecodedSize,
        "FSST decompressed string exceeds the maximum expansion bound.");
    decoded.resize(decodedSize);
  }

  return decodedStrings_.add(row, std::move(decoded));
}

std::optional<std::string_view> FsstEncodingView::DecodedStrings::find(
    uint32_t row) const {
  const std::shared_lock<std::shared_mutex> guard{lock_};
  const auto it = positions_.find(row);
  if (it == positions_.end()) {
    return std::nullopt;
  }
  return values_[it->second];
}

std::string_view FsstEncodingView::DecodedStrings::add(
    uint32_t row,
    std::string decoded) {
  const std::unique_lock<std::shared_mutex> guard{lock_};
  // Appending never moves the strings that are already there, so views of them
  // stay valid for the readers that hold them.
  const auto [it, inserted] = positions_.try_emplace(row, values_.size());
  if (inserted) {
    values_.push_back(std::move(decoded));
  }
  return values_[it->second];
}

std::string_view FsstEncodingView::valueAt(
    const DecodedChunk& chunk,
    uint32_t rowInChunk) {
  const auto begin = chunk.offsets[rowInChunk];
  const auto end = chunk.offsets[rowInChunk + 1];
  return {chunk.data.data() + begin, end - begin};
}

std::shared_ptr<const FsstEncodingView::DecodedChunk>
FsstEncodingView::getChunk(uint32_t chunkIndex) const {
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

std::shared_ptr<const FsstEncodingView::DecodedChunk>
FsstEncodingView::decodeChunk(uint32_t chunkIndex) const {
  const auto firstRow = chunkIndex * kRowsPerChunk;
  const auto endRow = firstRow +
      std::min(kRowsPerChunk,
               static_cast<uint32_t>(this->rowCount_ - firstRow));
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
          reinterpret_cast<unsigned char*>(chunk->data.data() + outputOffset));
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

} // namespace facebook::nimble
