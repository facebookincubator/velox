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

#include <folly/container/F14Set.h>
#include <algorithm>
#include <array>
#include <limits>
#include <memory>
#include <numeric>
#include <span>
#include <utility>

#include "velox/buffer/Buffer.h"
#include "velox/common/Casts.h"
#include "velox/dwio/nimble/encodings/views/DecodedStringBuffers.h"

namespace facebook::nimble {

struct FsstEncodingView::DecodedChunk {
  // Pool-owned decompressed strings of this chunk remain valid until every
  // holder of the buffer releases it. The other chunks that one read decoded
  // with it share the buffer, and so do the chunks of the views that share
  // Encoding::Options::decodedStringBuffers.
  velox::BufferPtr data;

  // Where the strings of the chunk start in 'data'. Seldom its first byte: a
  // decode writes after the strings of earlier decodes of a shared buffer, and
  // the chunks of one decode follow each other.
  const char* strings{nullptr};

  // Where the string of each row of the chunk starts after 'strings', and
  // where the last one ends.
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

  // Counts the rows of each chunk that a range across chunks asks for, so that
  // the chunks it decodes share one buffer.
  if (length > 0 && rowChunk(offset) != rowChunk(offset + length - 1)) {
    const auto firstChunkIndex = rowChunk(offset);
    const auto lastChunkIndex = rowChunk(offset + length - 1);
    ScopedVector<uint32_t> numChunkRows{
        lastChunkIndex - firstChunkIndex + 1,
        this->pool_,
        this->options_.bufferPool};
    for (auto chunkIndex = firstChunkIndex; chunkIndex <= lastChunkIndex;
         ++chunkIndex) {
      const auto chunkBegin = chunkIndex * kRowsPerChunk;
      numChunkRows[chunkIndex - firstChunkIndex] =
          std::min(offset + length, chunkBegin + chunkRows(chunkIndex)) -
          std::max(offset, chunkBegin);
    }
    prepareChunks(firstChunkIndex, {numChunkRows.data(), numChunkRows.size()});
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
    std::span<const uint32_t> indices,
    const std::function<void(uint32_t)>& setNull,
    const std::function<void(velox::BufferPtr)>& addStringBuffer,
    void* output) const {
  if (addStringBuffer == nullptr || indices.empty()) {
    return TypedEncodingView<std::string_view>::read(
        indices, setNull, addStringBuffer, output);
  }
  NIMBLE_CHECK_LE(indices.size(), std::numeric_limits<uint32_t>::max());
  this->checkReadIndices(indices);
  auto* values = static_cast<std::string_view*>(output);
  if (cachesDecodedChunks()) {
    decodeRows(indices, addStringBuffer, values);
    return static_cast<uint32_t>(indices.size());
  }

  // Sums the lengths once for all indices, not once for each of them.
  ScopedVector<uint64_t> offsets{
      indices.size(), this->pool_, this->options_.bufferPool};
  lengths_->sum(indices, offsets.data());
  ScopedVector<uint32_t> sizes{
      indices.size(), this->pool_, this->options_.bufferPool};
  lengths_->readAt(indices, sizes.data());
  decodeRows(
      {offsets.data(), offsets.size()},
      {sizes.data(), sizes.size()},
      addStringBuffer,
      values);
  return static_cast<uint32_t>(indices.size());
}

uint32_t FsstEncodingView::read(
    std::span<const RowRange> ranges,
    const std::function<void(uint32_t)>& setNull,
    const std::function<void(velox::BufferPtr)>& addStringBuffer,
    void* output) const {
  if (ranges.empty() ||
      (!cachesDecodedChunks() && addStringBuffer == nullptr)) {
    return TypedEncodingView<std::string_view>::read(
        ranges, setNull, addStringBuffer, output);
  }
  const auto numRows = this->checkReadRanges(ranges);
  auto* values = static_cast<std::string_view*>(output);
  if (!cachesDecodedChunks()) {
    // Reads the lengths of each range, and sums the lengths before it once.
    ScopedVector<uint64_t> offsets{
        numRows, this->pool_, this->options_.bufferPool};
    ScopedVector<uint32_t> sizes{
        numRows, this->pool_, this->options_.bufferPool};
    uint32_t outputOffset{0};
    for (const auto& range : ranges) {
      lengths_->read(
          range.startRow, range.numRows(), sizes.data() + outputOffset);
      auto offset = lengths_->sum(0, range.startRow);
      for (uint32_t i{0}; i < range.numRows(); ++i) {
        offsets[outputOffset + i] = offset;
        offset += sizes[outputOffset + i];
      }
      outputOffset += range.numRows();
    }
    decodeRows(
        {offsets.data(), offsets.size()},
        {sizes.data(), sizes.size()},
        addStringBuffer,
        values);
    return numRows;
  }

  // Returns whether a chunk that the ranges ask for rows of is decoded, so
  // that the read takes the strings of those rows from the chunk.
  const auto spansDecodedChunk = [&] {
    const auto chunks = decodedChunks_.rlock();
    for (const auto& range : ranges) {
      for (auto chunkIndex = rowChunk(range.startRow);
           chunkIndex <= rowChunk(range.endRow - 1);
           ++chunkIndex) {
        if ((*chunks)[chunkIndex] != nullptr) {
          return true;
        }
      }
    }
    return false;
  };
  if (addStringBuffer != nullptr && !spansDecodedChunk()) {
    decodeRanges(ranges, /*chunks=*/{}, addStringBuffer, values);
    return numRows;
  }

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
  prepareChunks(firstChunkIndex, {numChunkRows.data(), numChunkRows.size()});
  if (addStringBuffer != nullptr) {
    // Decides for each chunk once, as the loop below does, whether the read
    // takes its rows from the chunk.
    std::vector<std::shared_ptr<const DecodedChunk>> chunks(
        numChunkRows.size());
    for (uint32_t i{0}; i < numChunkRows.size(); ++i) {
      if (numChunkRows[i] > 0) {
        chunks[i] = chunkForRead(firstChunkIndex + i, numChunkRows[i]);
      }
    }
    decodeRanges(ranges, chunks, addStringBuffer, values);
    return numRows;
  }

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

std::string_view FsstEncodingView::valueAt(
    const DecodedChunk& chunk,
    uint32_t rowInChunk) {
  const auto begin = chunk.offsets[rowInChunk];
  const auto end = chunk.offsets[rowInChunk + 1];
  if (begin == end) {
    return {};
  }
  return {chunk.strings + begin, end - begin};
}

uint32_t FsstEncodingView::chunkRows(uint32_t chunkIndex) const {
  return std::min(
      kRowsPerChunk,
      static_cast<uint32_t>(this->rowCount_ - chunkIndex * kRowsPerChunk));
}

uint32_t FsstEncodingView::minRowsToDecodeChunk(uint32_t chunkIndex) const {
  return chunkRows(chunkIndex) * kMinRowsToDecodeChunk / kRowsPerChunk;
}

std::shared_ptr<const FsstEncodingView::DecodedChunk>
FsstEncodingView::chunkForRead(uint32_t chunkIndex, uint32_t numRows) const {
  {
    const auto chunks = decodedChunks_.rlock();
    if ((*chunks)[chunkIndex] != nullptr) {
      return (*chunks)[chunkIndex];
    }
  }
  const auto minRowsToDecode = minRowsToDecodeChunk(chunkIndex);

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

  auto decoded = std::move(
      decodeChunks(std::span<const uint32_t>{&chunkIndex, 1}).front());
  auto chunks = decodedChunks_.wlock();
  auto& cached = (*chunks)[chunkIndex];

  // A concurrent read may have published the chunk first and returned views of
  // its strings, so the chunk it published stays.
  if (cached == nullptr) {
    cached = std::move(decoded);
  }
  return cached;
}

void FsstEncodingView::prepareChunks(
    uint32_t chunkIndex,
    std::span<const uint32_t> numChunkRows) const {
  // Leaves the chunk of a read within one chunk to chunkForRead(), since no
  // other chunk of the read can share its buffer.
  if (numChunkRows.size() < 2) {
    return;
  }
  ScopedVector<uint32_t> chunkIndices{
      numChunkRows.size(), this->pool_, this->options_.bufferPool};

  // Starts empty, and keeps the capacity, which the chunks never exceed.
  chunkIndices.resize(0);
  {
    const auto chunks = decodedChunks_.rlock();
    for (uint32_t i = 0; i < numChunkRows.size(); ++i) {
      if ((*chunks)[chunkIndex + i] == nullptr &&
          numChunkRows[i] >= minRowsToDecodeChunk(chunkIndex + i)) {
        chunkIndices.push_back(chunkIndex + i);
      }
    }
  }

  // Leaves a single chunk that needs decoding to chunkForRead() too. The read
  // calls chunkForRead() for each of its chunks next, which decodes the chunk
  // alone with the same decodeChunks() call.
  if (chunkIndices.size() < 2) {
    return;
  }
  auto decodedChunks = decodeChunks({chunkIndices.data(), chunkIndices.size()});
  auto chunks = decodedChunks_.wlock();
  for (size_t i = 0; i < decodedChunks.size(); ++i) {
    auto& cached = (*chunks)[chunkIndices[i]];

    // A concurrent read may have published the chunk first and returned views
    // of its strings, so the chunk it published stays.
    if (cached == nullptr) {
      cached = std::move(decodedChunks[i]);
    }
  }
}

namespace {

// Returns how many bytes 'compressed' decodes to with 'decoder', in one pass
// over its codes and without decoding it. A code below FSST_ESC stands for the
// symbol of decoder.len[code] bytes. FSST_ESC is followed by a literal byte,
// which decodes to itself. 'compressed' can hold several validated strings one
// after another: a validated string does not end inside an escape, so the pass
// counts each string as a pass over it alone would.
size_t getDecodedBytes(
    const fsst_decoder_t& decoder,
    std::string_view compressed) {
  size_t numBytes{0};
  for (size_t i{0}; i < compressed.size(); ++i) {
    const auto code = static_cast<uint8_t>(compressed[i]);
    if (code == FSST_ESC) {
      // An escape is followed by a byte that decodes to itself.
      ++numBytes;
      ++i;
    } else {
      numBytes += decoder.len[code];
    }
  }
  return numBytes;
}

} // namespace

namespace detail {

// Decodes FSST strings one after another into buffers: rooms of
// 'sharedBuffers', which the view shares with other views, when it is not
// null, or else buffers of its own.
//
// The decoder writes into the rest of a buffer without asking for more until
// the next string may not fit. It then finishes the buffer, passes it to
// 'addBuffer', and reserves a new one, so that the strings of a decode can lie
// in several buffers and none is copied. A room of the shared buffers is the
// rest of one of them, and later decodes take the bytes that a decode does not
// use. A buffer of its own takes exactly the bytes reserved, so a caller
// without shared buffers reserves the bytes that its strings decode to before
// it decodes them: the buffer then holds no more than they take. A caller
// that needs some strings in one buffer, such as the rows of a chunk, reserves
// room for all of them first too. A string that may not fit takes room for
// the bytes that it decodes to, not the most bytes that it can decode to,
// which can exceed a room of the shared buffers.
class BulkStringDecoder {
 public:
  // 'maxSymbolLength' is the most bytes that one code decodes to. 'pool'
  // allocates the buffers of its own. 'addBuffer' takes each buffer that holds
  // strings when the decoder finishes it, and must outlive the decoder.
  BulkStringDecoder(
      const fsst_decoder_t& decoder,
      size_t maxSymbolLength,
      DecodedStringBuffers* sharedBuffers,
      velox::memory::MemoryPool* pool,
      const std::function<void(velox::BufferPtr)>& addBuffer)
      : decoder_{decoder},
        maxSymbolLength_{maxSymbolLength},
        sharedBuffers_{sharedBuffers},
        pool_{velox::checkedNotNull(pool)},
        addBuffer_{addBuffer} {
    NIMBLE_CHECK_NOT_NULL(
        addBuffer_, "A decoder must get a callback that takes its buffers.");
  }

  BulkStringDecoder(const BulkStringDecoder&) = delete;
  BulkStringDecoder& operator=(const BulkStringDecoder&) = delete;
  BulkStringDecoder(BulkStringDecoder&&) = delete;
  BulkStringDecoder& operator=(BulkStringDecoder&&) = delete;

  // Ends the room of the shared buffers that a decode leaves open when it
  // throws between a reservation and finish(), such as when an allocation in
  // decodeChunks() fails. 'buffer_' is empty otherwise, since every caller
  // calls finish() after a decode that succeeds, and finish() empties it. The
  // room ends with no bytes used, since its strings reach no holder before
  // finish(). The shared buffers outlive the decoder, and their reserve()
  // fails while a room is open, which their clear() does not end, so without
  // this every later decode with them would fail. A buffer of its own needs
  // nothing, since 'ownBuffer_' frees it.
  ~BulkStringDecoder() {
    if (buffer_.data() != nullptr && sharedBuffers_ != nullptr) {
      sharedBuffers_->finish(0);
    }
  }

  // Returns whether the decoder takes its buffers from shared buffers.
  bool sharesBuffers() const {
    return sharedBuffers_ != nullptr;
  }

  // Returns the most bytes that strings of 'numCompressedBytes' compressed
  // bytes decode to.
  size_t maxDecompressedBytes(size_t numCompressedBytes) const {
    return numCompressedBytes * maxSymbolLength_;
  }

  // Makes sure that the buffer has room after its strings for strings that
  // decode to 'numBytes' bytes: finishes the buffer and reserves a new one
  // when it does not. Strings that decode to no bytes need no room.
  void reserve(size_t numBytes) {
    if (numBytes > 0 &&
        buffer_.size() - usedBytes_ < numBytes + kDecoderPadding) {
      newBuffer(numBytes + kDecoderPadding);
    }
  }

  // Decodes 'compressed' right after the strings in the buffer, and returns
  // the string, or an empty string for empty 'compressed'. 'compressed' must
  // be validated, as a cached view validates every string, its trailing escape
  // included, when it is created. Always inlined, since the callers decode one
  // string at a time, and a call per string is a measurable part of decoding
  // short strings.
  FOLLY_ALWAYS_INLINE std::string_view decode(std::string_view compressed) {
    if (compressed.empty()) {
      return {};
    }
    if (buffer_.size() - usedBytes_ <
        maxDecompressedBytes(compressed.size()) + kDecoderPadding) {
      // Reserves the bytes that the string decodes to. The rest of a
      // reservation of the caller for exactly the bytes of its strings then
      // holds its last strings, and a new room or buffer takes no more than
      // the string needs.
      reserve(getDecodedBytes(decoder_, compressed));
    }
    auto* decoded = buffer_.data() + usedBytes_;
    const auto decodedSize = nimble_fsst_decompress(
        &decoder_,
        compressed.size(),
        reinterpret_cast<const unsigned char*>(compressed.data()),
        buffer_.size() - usedBytes_,
        reinterpret_cast<unsigned char*>(decoded));
    NIMBLE_CHECK_GT(
        decodedSize,
        0,
        "FSST decompression failed for non-empty compressed string.");
    usedBytes_ += decodedSize;
    return {decoded, decodedSize};
  }

  // Returns where the next string goes, right after the strings in the
  // buffer, or null before the first buffer and after finish().
  const char* position() const {
    return buffer_.data() + usedBytes_;
  }

  // Finishes the buffer, and passes it to 'addBuffer' when a string lies in
  // it.
  void finish() {
    if (buffer_.data() == nullptr) {
      return;
    }

    // Empties the state before 'addBuffer' runs, so that the destructor does
    // not finish the buffer again when 'addBuffer' throws.
    buffer_ = {};
    auto buffer = sharedBuffers_ != nullptr ? sharedBuffers_->finish(usedBytes_)
                                            : std::move(ownBuffer_);
    if (usedBytes_ > 0) {
      usedBytes_ = 0;
      addBuffer_(std::move(buffer));
    }
  }

 private:
  // Finishes the buffer and reserves a new one with room for at least
  // 'numBytes' bytes: a room of the shared buffers, or else a buffer of its
  // own of that many bytes. Kept out of decode(), which rarely calls it.
  FOLLY_NOINLINE void newBuffer(size_t numBytes) {
    finish();
    if (sharedBuffers_ != nullptr) {
      buffer_ = sharedBuffers_->reserve(numBytes);
      return;
    }
    ownBuffer_ = velox::AlignedBuffer::allocate<char>(numBytes, pool_);
    buffer_ = {ownBuffer_->asMutable<char>(), ownBuffer_->size()};
  }

  // Room for this many bytes past the bytes that the strings decode to keeps
  // fsst_decompress in velox/external/fsst/fsst.h bounded by its input rather
  // than its output: its four-symbol loop and three-code tail then never stop
  // early and read an escape as a symbol. They guard their 8-byte stores with
  // posOut + 32 <= size and posOut + 24 <= size, so a final one-byte symbol
  // needs 31 spare bytes.
  static constexpr size_t kDecoderPadding{31};

  // The symbol table of the view.
  const fsst_decoder_t& decoder_;

  const size_t maxSymbolLength_;

  // Gives the rooms when not null.
  DecodedStringBuffers* const sharedBuffers_;

  // Allocates the buffers of its own, without shared buffers.
  velox::memory::MemoryPool* const pool_;

  const std::function<void(velox::BufferPtr)>& addBuffer_;

  // The buffer of its own that 'buffer_' lies in, without shared buffers,
  // until it finishes.
  velox::BufferPtr ownBuffer_;

  // The rest of the buffer that the strings go into: a room of the shared
  // buffers, or 'ownBuffer_'. Empty before the first buffer and after
  // finish().
  std::span<char> buffer_;

  // The bytes of the strings in 'buffer_', from its start.
  size_t usedBytes_{0};
};

} // namespace detail

namespace {

// Passes 'buffer', the buffer of a decoded chunk that a read takes strings
// from, to 'addStringBuffer' unless 'chunkBufferSet' holds it, and adds it
// there: chunks can share a buffer, which a read passes once.
void addChunkBuffer(
    const velox::BufferPtr& buffer,
    const std::function<void(velox::BufferPtr)>& addStringBuffer,
    folly::F14FastSet<const velox::Buffer*>& chunkBufferSet) {
  if (buffer != nullptr && chunkBufferSet.insert(buffer.get()).second) {
    addStringBuffer(buffer);
  }
}

} // namespace

size_t FsstEncodingView::decodedBytes(uint32_t startRow, uint32_t endRow)
    const {
  const auto startOffset = compressedOffsets_[startRow];
  return getDecodedBytes(
      decoder_,
      {blob_.data() + startOffset, compressedOffsets_[endRow] - startOffset});
}

detail::BulkStringDecoder FsstEncodingView::makeBulkStringDecoder(
    const std::function<void(velox::BufferPtr)>& addBuffer) const {
  return detail::BulkStringDecoder{
      decoder_,
      kMaxSymbolLength,
      this->options_.decodedStringBuffers,
      this->pool_,
      addBuffer};
}

void FsstEncodingView::reserveDecodeBuffer(
    std::span<const RowRange> ranges,
    std::span<const std::shared_ptr<const DecodedChunk>> chunks,
    detail::BulkStringDecoder& stringDecoder) const {
  if (stringDecoder.sharesBuffers()) {
    return;
  }
  const auto firstChunkIndex = rowChunk(ranges.front().startRow);
  size_t numBytes{0};
  for (const auto& range : ranges) {
    // Adds up the rows of the range chunk by chunk, but for the rows of the
    // chunks that 'chunks' holds, which the read takes from them.
    auto startRow = range.startRow;
    while (startRow < range.endRow) {
      const auto chunkIndex = rowChunk(startRow);
      const auto endRow = std::min(
          range.endRow, chunkIndex * kRowsPerChunk + chunkRows(chunkIndex));
      if (chunks.empty() || chunks[chunkIndex - firstChunkIndex] == nullptr) {
        numBytes += decodedBytes(startRow, endRow);
      }
      startRow = endRow;
    }
  }
  stringDecoder.reserve(numBytes);
}

void FsstEncodingView::decodeRanges(
    std::span<const RowRange> ranges,
    std::span<const std::shared_ptr<const DecodedChunk>> chunks,
    const std::function<void(velox::BufferPtr)>& addStringBuffer,
    std::string_view* output) const {
  const auto firstChunkIndex = rowChunk(ranges.front().startRow);
  if (!chunks.empty()) {
    NIMBLE_CHECK_EQ(
        chunks.size(),
        rowChunk(ranges.back().endRow - 1) - firstChunkIndex + 1,
        "Chunks must hold an entry for each chunk that the ranges span.");
  }
  auto stringDecoder = makeBulkStringDecoder(addStringBuffer);
  reserveDecodeBuffer(ranges, chunks, stringDecoder);
  if (chunks.empty()) {
    uint32_t outputOffset{0};
    for (const auto& range : ranges) {
      for (auto row = range.startRow; row < range.endRow; ++row) {
        output[outputOffset++] = stringDecoder.decode(compressedValue(row));
      }
    }
    stringDecoder.finish();
    return;
  }

  folly::F14FastSet<const velox::Buffer*> chunkBufferSet;
  const DecodedChunk* prevChunk{nullptr};
  uint32_t outputOffset{0};
  for (const auto& range : ranges) {
    for (auto row = range.startRow; row < range.endRow; ++row) {
      const auto& chunk = chunks[rowChunk(row) - firstChunkIndex];
      if (chunk == nullptr) {
        output[outputOffset++] = stringDecoder.decode(compressedValue(row));
        continue;
      }
      if (chunk.get() != prevChunk) {
        addChunkBuffer(chunk->data, addStringBuffer, chunkBufferSet);
        prevChunk = chunk.get();
      }
      output[outputOffset++] = valueAt(*chunk, row % kRowsPerChunk);
    }
  }
  stringDecoder.finish();
}

void FsstEncodingView::reserveDecodeBuffer(
    std::span<const uint32_t> indices,
    std::span<const size_t> runEnds,
    std::span<const DecodedChunk* const> runChunks,
    detail::BulkStringDecoder& stringDecoder) const {
  if (stringDecoder.sharesBuffers()) {
    return;
  }
  size_t numBytes{0};
  size_t runBegin{0};
  for (size_t run{0}; run < runEnds.size(); ++run) {
    if (runChunks[run] == nullptr) {
      for (auto i = runBegin; i < runEnds[run]; ++i) {
        numBytes += decodedBytes(indices[i], indices[i] + 1);
      }
    }
    runBegin = runEnds[run];
  }
  stringDecoder.reserve(numBytes);
}

void FsstEncodingView::decodeRows(
    std::span<const uint32_t> indices,
    const std::function<void(velox::BufferPtr)>& addStringBuffer,
    std::string_view* output) const {
  // Splits the indices into runs in one chunk, as readPhysicalAt() does, and
  // decides for each run whether the read takes its strings from the chunk
  // before it decodes any string, since a chunk that chunkForRead() decodes
  // takes a room of the shared buffers. Keeps where each run ends, and the
  // chunk of the run, or null when the read decodes its strings itself. The
  // cache keeps every chunk that chunkForRead() returns while the view lives.
  ScopedVector<size_t> runEnds{
      indices.size(), this->pool_, this->options_.bufferPool};
  ScopedVector<const DecodedChunk*> runChunks{
      indices.size(), this->pool_, this->options_.bufferPool};

  // Both start empty, and keep the capacity, which the runs never exceed.
  runEnds.resize(0);
  runChunks.resize(0);
  size_t runBegin{0};
  while (runBegin < indices.size()) {
    const auto chunkIndex = rowChunk(indices[runBegin]);
    auto runEnd = runBegin;
    while (runEnd < indices.size() && rowChunk(indices[runEnd]) == chunkIndex) {
      ++runEnd;
    }
    runEnds.push_back(runEnd);
    runChunks.push_back(
        chunkForRead(chunkIndex, static_cast<uint32_t>(runEnd - runBegin))
            .get());
    runBegin = runEnd;
  }
  auto stringDecoder = makeBulkStringDecoder(addStringBuffer);
  reserveDecodeBuffer(
      indices,
      {runEnds.data(), runEnds.size()},
      {runChunks.data(), runChunks.size()},
      stringDecoder);
  folly::F14FastSet<const velox::Buffer*> chunkBufferSet;
  size_t outputIndex{0};
  for (size_t run{0}; run < runEnds.size(); ++run) {
    // @lint-ignore CLANGTIDY facebook-hte-LocalUncheckedArrayBounds
    const auto* chunk = runChunks[run];
    if (chunk != nullptr) {
      addChunkBuffer(chunk->data, addStringBuffer, chunkBufferSet);
    }
    for (; outputIndex < runEnds[run]; ++outputIndex) {
      const auto row = indices[outputIndex];
      output[outputIndex] = chunk != nullptr
          ? valueAt(*chunk, row % kRowsPerChunk)
          : stringDecoder.decode(compressedValue(row));
    }
  }
  stringDecoder.finish();
}

void FsstEncodingView::reserveDecodeBuffer(
    std::span<const uint64_t> offsets,
    std::span<const uint32_t> sizes,
    detail::BulkStringDecoder& stringDecoder) const {
  if (stringDecoder.sharesBuffers()) {
    return;
  }
  size_t numBytes{0};
  for (size_t i{0}; i < sizes.size(); ++i) {
    numBytes +=
        getDecodedBytes(decoder_, {blob_.data() + offsets[i], sizes[i]});
  }
  stringDecoder.reserve(numBytes);
}

void FsstEncodingView::decodeRows(
    std::span<const uint64_t> offsets,
    std::span<const uint32_t> sizes,
    const std::function<void(velox::BufferPtr)>& addStringBuffer,
    std::string_view* output) const {
  // Validates the strings one at a time because a view without
  // cacheEncodingView skips the validation of every length that a cached view
  // performs at construction.
  for (size_t i{0}; i < sizes.size(); ++i) {
    FsstEncoding::validateCompressedLengths(
        sizes.subspan(i, 1), blob_, static_cast<size_t>(offsets[i]));
  }
  auto stringDecoder = makeBulkStringDecoder(addStringBuffer);
  reserveDecodeBuffer(offsets, sizes, stringDecoder);
  for (size_t i{0}; i < sizes.size(); ++i) {
    output[i] = stringDecoder.decode({blob_.data() + offsets[i], sizes[i]});
  }
  stringDecoder.finish();
}

void FsstEncodingView::reserveDecodeBuffer(
    std::span<const uint32_t> chunkIndices,
    size_t numCompressedBytes,
    detail::BulkStringDecoder& stringDecoder) const {
  const auto maxDecompressedBytes =
      stringDecoder.maxDecompressedBytes(numCompressedBytes);
  if (stringDecoder.sharesBuffers() &&
      maxDecompressedBytes <= DecodedStringBuffers::kMaxBufferBytes) {
    stringDecoder.reserve(maxDecompressedBytes);
    return;
  }
  size_t numBytes{0};
  for (const auto chunkIndex : chunkIndices) {
    const auto firstRow = chunkIndex * kRowsPerChunk;
    numBytes += decodedBytes(firstRow, firstRow + chunkRows(chunkIndex));
  }
  stringDecoder.reserve(numBytes);
}

std::vector<std::shared_ptr<FsstEncodingView::DecodedChunk>>
FsstEncodingView::decodeChunks(std::span<const uint32_t> chunkIndices) const {
  NIMBLE_CHECK(!chunkIndices.empty(), "Chunks to decode must not be empty.");
  size_t numCompressedBytes{0};
  for (const auto chunkIndex : chunkIndices) {
    const auto firstRow = chunkIndex * kRowsPerChunk;
    numCompressedBytes += compressedOffsets_[firstRow + chunkRows(chunkIndex)] -
        compressedOffsets_[firstRow];
  }

  // Takes the one buffer that the decoder passes: the reservation below fits
  // the strings of every chunk in one buffer, which every chunk holds.
  velox::BufferPtr buffer;
  const std::function<void(velox::BufferPtr)> setBuffer =
      [&buffer](velox::BufferPtr _buffer) {
        NIMBLE_CHECK_NULL(buffer, "Decoded chunks must lie in one buffer.");
        buffer = std::move(_buffer);
      };
  auto stringDecoder = makeBulkStringDecoder(setBuffer);

  // Reserves room for every string of the chunks, so that they lie in the one
  // buffer that every chunk holds.
  reserveDecodeBuffer(chunkIndices, numCompressedBytes, stringDecoder);
  std::vector<std::shared_ptr<DecodedChunk>> chunks;
  chunks.reserve(chunkIndices.size());
  for (const auto chunkIndex : chunkIndices) {
    const auto firstRow = chunkIndex * kRowsPerChunk;
    const auto endRow = firstRow + chunkRows(chunkIndex);
    auto& chunk = chunks.emplace_back(std::make_shared<DecodedChunk>());
    chunk->strings = stringDecoder.position();
    chunk->offsets.resize(endRow - firstRow + 1);
    size_t offset{0};
    for (auto row = firstRow; row < endRow; ++row) {
      offset += stringDecoder.decode(compressedValue(row)).size();
      chunk->offsets[row - firstRow + 1] = offset;
    }
  }
  stringDecoder.finish();

  // Gives each chunk the buffer, which the decoder passes only when it
  // finishes the buffer. It passes none when every string of the chunks is
  // empty, and valueAt() reads no bytes of an empty string.
  NIMBLE_CHECK_EQ(
      buffer != nullptr,
      numCompressedBytes > 0,
      "Decoded chunks must have a buffer exactly when they have strings.");
  for (auto& chunk : chunks) {
    chunk->data = buffer;
  }
  return chunks;
}

} // namespace facebook::nimble
