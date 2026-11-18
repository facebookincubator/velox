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
#include <memory>
#include <numeric>
#include <utility>

#include "velox/buffer/Buffer.h"

namespace facebook::nimble {

struct FsstEncodingView::DecodedChunk {
  // Pool-owned decompressed strings remain valid until the chunk is released.
  velox::BufferPtr data;
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
  const auto numChunks =
      this->rowCount_ / kRowsPerChunk + (this->rowCount_ % kRowsPerChunk != 0);
  decodedChunks_.wlock()->resize(numChunks);
  rowsReadAlone_ = std::make_unique<std::atomic_uint32_t[]>(numChunks);
}

std::string_view FsstEncodingView::readTypedAt(uint32_t index) const {
  NIMBLE_CHECK_LT(index, this->rowCount_);
  if (!cachesDecodedChunks()) {
    const auto size = lengths_->readAt(index);
    const auto begin = lengths_->sum(0, index);
    return decodeValue(index, begin, size, *decodedStrings_.wlock());
  }
  const auto chunk = chunkForRead(rowChunk(index), /*numRows=*/1);
  if (chunk != nullptr) {
    return valueAt(*chunk, index % kRowsPerChunk);
  }
  std::string_view value;
  decodeRows(std::span<const uint32_t>{&index, 1}, &value);
  return value;
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
    auto strings = decodedStrings_.wlock();
    for (uint32_t i = 0; i < length; ++i) {
      output[i] = decodeValue(offset + i, begin, sizes[i], *strings);
      begin += sizes[i];
    }
    return;
  }
  uint32_t outputOffset{0};
  while (outputOffset < length) {
    const auto row = offset + outputOffset;
    const auto chunkIndex = rowChunk(row);
    const auto rowInChunk = row % kRowsPerChunk;
    const auto rowsToCopy =
        std::min(length - outputOffset, chunkRows(chunkIndex) - rowInChunk);
    const auto chunk = chunkForRead(chunkIndex, rowsToCopy);
    if (chunk == nullptr) {
      ScopedVector<uint32_t> rows{
          rowsToCopy, this->pool_, this->options_.bufferPool};
      std::iota(rows.begin(), rows.end(), row);
      decodeRows({rows.data(), rows.size()}, output + outputOffset);
    } else {
      for (uint32_t i = 0; i < rowsToCopy; ++i) {
        output[outputOffset + i] = valueAt(*chunk, rowInChunk + i);
      }
    }
    outputOffset += rowsToCopy;
  }
}

void FsstEncodingView::readPhysicalAt(
    std::span<const uint32_t> indices,
    std::string_view* output) const {
  if (indices.empty()) {
    return;
  }
  this->checkReadIndices(indices);
  if (!cachesDecodedChunks()) {
    // Sums the lengths once for all indices, not once for each of them.
    ScopedVector<uint64_t> begins{
        indices.size(), this->pool_, this->options_.bufferPool};
    lengths_->sum(indices, begins.data());
    ScopedVector<uint32_t> sizes{
        indices.size(), this->pool_, this->options_.bufferPool};
    lengths_->readAt(indices, sizes.data());
    auto strings = decodedStrings_.wlock();
    for (size_t i = 0; i < indices.size(); ++i) {
      output[i] = decodeValue(indices[i], begins[i], sizes[i], *strings);
    }
    return;
  }

  // Reads each run of indices in one chunk from the chunk or alone, as
  // chunkForRead() decides for the run.
  size_t runBegin{0};
  while (runBegin < indices.size()) {
    const auto chunkIndex = rowChunk(indices[runBegin]);
    auto runEnd = runBegin;
    while (runEnd < indices.size() && rowChunk(indices[runEnd]) == chunkIndex) {
      ++runEnd;
    }
    const auto chunk =
        chunkForRead(chunkIndex, static_cast<uint32_t>(runEnd - runBegin));
    if (chunk == nullptr) {
      decodeRows(
          indices.subspan(runBegin, runEnd - runBegin), output + runBegin);
    } else {
      for (auto i = runBegin; i < runEnd; ++i) {
        output[i] = valueAt(*chunk, indices[i] % kRowsPerChunk);
      }
    }
    runBegin = runEnd;
  }
}

uint32_t FsstEncodingView::read(
    std::span<const RowRange> ranges,
    const std::function<void(uint32_t)>& setNull,
    void* output) const {
  if (!cachesDecodedChunks() || ranges.empty()) {
    return TypedEncodingView<std::string_view>::read(ranges, setNull, output);
  }
  const auto numRows = this->checkReadRanges(ranges);
  const auto firstChunkIndex = rowChunk(ranges.front().startRow);
  const auto lastChunkIndex = rowChunk(ranges.back().endRow - 1);
  ScopedVector<uint32_t> numChunkRows{
      lastChunkIndex - firstChunkIndex + 1,
      this->pool_,
      this->options_.bufferPool};
  std::fill(numChunkRows.begin(), numChunkRows.end(), 0);
  for (const auto& range : ranges) {
    auto row = range.startRow;
    while (row < range.endRow) {
      const auto chunkIndex = rowChunk(row);
      const auto end = std::min(
          range.endRow, chunkIndex * kRowsPerChunk + chunkRows(chunkIndex));
      numChunkRows[chunkIndex - firstChunkIndex] += end - row;
      row = end;
    }
  }

  auto* values = static_cast<std::string_view*>(output);

  // The rows of the current chunk that the read decodes alone, as one batch.
  // decodeRowBatch() decodes them under one lock when the read leaves the
  // chunk. Their outputs are adjacent, from 'rowBatchOutputOffset'. The batch
  // stays empty for a chunk that the read reads from the chunk's decoded
  // strings.
  ScopedVector<uint32_t> rowBatch{
      std::min(numRows, kRowsPerChunk), this->pool_, this->options_.bufferPool};

  // Starts empty, and keeps the capacity, which a batch never exceeds.
  rowBatch.resize(0);
  uint32_t rowBatchOutputOffset{0};

  // Decodes the rows of the batch, if there are any, since decodeRows() needs
  // rows, and empties the batch.
  const auto decodeRowBatch = [&] {
    if (!rowBatch.empty()) {
      decodeRows(
          {rowBatch.data(), rowBatch.size()}, values + rowBatchOutputOffset);
      rowBatch.resize(0);
    }
  };
  uint32_t outputOffset{0};
  auto chunkIndex = std::numeric_limits<uint32_t>::max();
  std::shared_ptr<const DecodedChunk> chunk;
  for (const auto& range : ranges) {
    for (auto row = range.startRow; row < range.endRow; ++row) {
      if (rowChunk(row) != chunkIndex) {
        decodeRowBatch();
        rowBatchOutputOffset = outputOffset;
        chunkIndex = rowChunk(row);
        chunk = chunkForRead(
            chunkIndex, numChunkRows[chunkIndex - firstChunkIndex]);
      }
      if (chunk == nullptr) {
        rowBatch.push_back(row);
      } else {
        values[outputOffset] = valueAt(*chunk, row % kRowsPerChunk);
      }
      ++outputOffset;
    }
  }
  decodeRowBatch();
  return numRows;
}

std::string_view FsstEncodingView::decodeValue(
    uint32_t row,
    uint64_t compressedBegin,
    uint32_t compressedSize,
    DecodedStrings& strings) const {
  if (const auto decoded = strings.find(row); decoded.has_value()) {
    return *decoded;
  }

  // Validates only this string because a view without cacheEncodingView skips
  // the validation of every length that a cached view performs at
  // construction.
  if (!cachesDecodedChunks()) {
    const std::array<uint32_t, 1> compressedSizes{compressedSize};
    FsstEncoding::validateCompressedLengths(
        compressedSizes, blob_, static_cast<size_t>(compressedBegin));
  }
  if (compressedSize == 0) {
    return {};
  }
  const auto maxDecodedSize =
      static_cast<size_t>(compressedSize) * kMaxSymbolLength;
  auto* decoded = strings.reserve(maxDecodedSize + 1, this->pool_);
  const auto decodedSize = nimble_fsst_decompress(
      &decoder_,
      compressedSize,
      reinterpret_cast<const unsigned char*>(blob_.data() + compressedBegin),
      maxDecodedSize + 1,
      reinterpret_cast<unsigned char*>(decoded));
  NIMBLE_CHECK_GT(
      decodedSize,
      0,
      "FSST decompression failed for non-empty compressed string.");
  NIMBLE_CHECK_FILE_LE(
      decodedSize,
      maxDecodedSize,
      "FSST decompressed string exceeds the maximum expansion bound.");
  return strings.add(row, decodedSize);
}

void FsstEncodingView::decodeRows(
    std::span<const uint32_t> rows,
    std::string_view* output) const {
  NIMBLE_CHECK_GT(rows.size(), 0, "Rows to decode must not be empty.");
  auto strings = decodedStrings_.wlock();
  for (size_t i = 0; i < rows.size(); ++i) {
    const auto compressedBegin = compressedOffsets_[rows[i]];
    output[i] = decodeValue(
        rows[i],
        compressedBegin,
        static_cast<uint32_t>(
            compressedOffsets_[rows[i] + 1] - compressedBegin),
        *strings);
  }
}

std::optional<std::string_view> FsstEncodingView::DecodedStrings::find(
    uint32_t row) const {
  const auto it = strings_.find(row);
  if (it == strings_.end()) {
    return std::nullopt;
  }
  return it->second;
}

char* FsstEncodingView::DecodedStrings::reserve(
    size_t maxSize,
    velox::memory::MemoryPool* pool) {
  if (freeBytes_ < maxSize) {
    // A new block never moves the strings that are already there, so views of
    // them stay valid for the readers that hold them.
    const auto blockSize = std::max(kBlockSize, maxSize);
    const auto& block = blocks_.emplace_back(
        velox::AlignedBuffer::allocate<char>(blockSize, pool));
    free_ = block->asMutable<char>();
    freeBytes_ = blockSize;
  }
  return free_;
}

std::string_view FsstEncodingView::DecodedStrings::add(
    uint32_t row,
    size_t size) {
  NIMBLE_CHECK_LE(size, freeBytes_);
  const std::string_view value{free_, size};
  strings_.emplace(row, value);
  free_ += size;
  freeBytes_ -= size;
  return value;
}

void FsstEncodingView::DecodedStrings::addBlocks(
    std::vector<velox::BufferPtr>& buffers) const {
  buffers.insert(buffers.end(), blocks_.begin(), blocks_.end());
}

void FsstEncodingView::addValueBuffers(
    std::vector<velox::BufferPtr>& buffers) const {
  TypedEncodingView<std::string_view>::addValueBuffers(buffers);
  {
    const auto chunks = decodedChunks_.rlock();
    for (const auto& chunk : *chunks) {
      if (chunk != nullptr && chunk->data != nullptr) {
        buffers.push_back(chunk->data);
      }
    }
  }
  decodedStrings_.rlock()->addBlocks(buffers);
}

std::string_view FsstEncodingView::valueAt(
    const DecodedChunk& chunk,
    uint32_t rowInChunk) {
  const auto begin = chunk.offsets[rowInChunk];
  const auto end = chunk.offsets[rowInChunk + 1];
  if (begin == end) {
    return {};
  }
  return {chunk.data->as<char>() + begin, end - begin};
}

uint32_t FsstEncodingView::chunkRows(uint32_t chunkIndex) const {
  return std::min(
      kRowsPerChunk,
      static_cast<uint32_t>(this->rowCount_ - chunkIndex * kRowsPerChunk));
}

std::shared_ptr<const FsstEncodingView::DecodedChunk>
FsstEncodingView::chunkForRead(uint32_t chunkIndex, uint32_t numRows) const {
  {
    const auto chunks = decodedChunks_.rlock();
    if ((*chunks)[chunkIndex] != nullptr) {
      return (*chunks)[chunkIndex];
    }
  }
  const auto minRowsToDecode =
      chunkRows(chunkIndex) * kMinRowsToDecodeChunk / kRowsPerChunk;

  // Counts only reads of fewer rows than the minimum, so the count cannot
  // overflow before a read decodes the chunk.
  if (numRows < minRowsToDecode) {
    auto& rowsReadAlone = rowsReadAlone_[chunkIndex];
    const auto rowsRead =
        rowsReadAlone.fetch_add(numRows, std::memory_order_relaxed) + numRows;
    if (rowsRead < minRowsToDecode) {
      return nullptr;
    }
  }

  auto decoded = decodeChunk(chunkIndex);
  auto chunks = decodedChunks_.wlock();
  auto& cached = (*chunks)[chunkIndex];

  // A concurrent read may have published the chunk first and returned views of
  // its strings, so the chunk it published stays.
  if (cached == nullptr) {
    cached = std::move(decoded);
  }
  return cached;
}

std::shared_ptr<const FsstEncodingView::DecodedChunk>
FsstEncodingView::decodeChunk(uint32_t chunkIndex) const {
  const auto firstRow = chunkIndex * kRowsPerChunk;
  const auto endRow = firstRow + chunkRows(chunkIndex);
  auto chunk = std::make_shared<DecodedChunk>();
  chunk->offsets.resize(endRow - firstRow + 1);

  // Decodes each string once, straight after the previous one, without a pass
  // that sums the decoded lengths first. The view validated every compressed
  // string, its trailing escape included, when it was created. A code decodes
  // to at most kMaxSymbolLength bytes, so room for that many bytes per code
  // plus the decoder's padding keeps fsst_decompress in
  // velox/external/fsst/fsst.h bounded by its input rather than its output:
  // its four-symbol loop and three-code tail then never stop early and read an
  // escape as a symbol. They guard their 8-byte stores with posOut + 32 <= size
  // and posOut + 24 <= size, so a final one-byte symbol needs 31 spare bytes.
  constexpr size_t kMaxSymbolLength{8};
  constexpr size_t kDecoderPadding{31};

  // Starts from a typical FSST ratio and doubles, so a cached chunk does not
  // keep the worst case of kMaxSymbolLength bytes per code.
  constexpr size_t kInitialExpansion{3};
  const size_t compressedBytes =
      compressedOffsets_[endRow] - compressedOffsets_[firstRow];
  size_t decodedBytes{0};
  for (auto row = firstRow; row < endRow; ++row) {
    const auto compressedBegin = compressedOffsets_[row];
    const size_t compressedSize = compressedOffsets_[row + 1] - compressedBegin;
    if (compressedSize != 0) {
      const auto* compressed = blob_.data() + compressedBegin;
      const size_t requiredBytes =
          decodedBytes + compressedSize * kMaxSymbolLength + kDecoderPadding;
      if (chunk->data == nullptr) {
        chunk->data = velox::AlignedBuffer::allocate<char>(
            std::max(
                requiredBytes,
                compressedBytes * kInitialExpansion + kDecoderPadding),
            this->pool_);
      } else if (chunk->data->size() < requiredBytes) {
        velox::AlignedBuffer::reallocate<char>(
            &chunk->data, std::max(requiredBytes, 2 * chunk->data->size()));
      }
      const auto decodedSize = nimble_fsst_decompress(
          &decoder_,
          compressedSize,
          reinterpret_cast<const unsigned char*>(compressed),
          chunk->data->size() - decodedBytes,
          reinterpret_cast<unsigned char*>(
              chunk->data->asMutable<char>() + decodedBytes));
      NIMBLE_CHECK_GT(
          decodedSize,
          0,
          "FSST decompression failed for non-empty compressed string.");
      decodedBytes += decodedSize;
    }
    chunk->offsets[row - firstRow + 1] = decodedBytes;
  }
  return chunk;
}

} // namespace facebook::nimble
