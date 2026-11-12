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

#include "velox/dwio/nimble/serializer/EncodingViewDecoder.h"

#include <algorithm>
#include <cstring>
#include <limits>

#include <folly/ScopeGuard.h>

#include "velox/common/Casts.h"
#include "velox/dwio/nimble/common/NimbleException.h"
#include "velox/dwio/nimble/common/Vector.h"

namespace facebook::nimble {
namespace {

// Copies selected strings into tracked storage that can outlive the decoder.
void retainStringContent(
    const void* notNulls,
    uint32_t count,
    velox::memory::MemoryPool* pool,
    void* values,
    std::vector<velox::BufferPtr>& stringBuffers) {
  auto* source = static_cast<std::string_view*>(values);
  uint64_t totalBytes{0};
  for (uint32_t i{0}; i < count; ++i) {
    if (notNulls == nullptr ||
        velox::bits::isBitSet(static_cast<const uint8_t*>(notNulls), i)) {
      totalBytes += source[i].size();
    }
  }
  if (totalBytes == 0) {
    return;
  }

  auto buffer = velox::AlignedBuffer::allocate<char>(totalBytes, pool);
  NIMBLE_CHECK_NOT_NULL(buffer);
  auto* rawBuffer = buffer->asMutable<char>();
  NIMBLE_CHECK_NOT_NULL(rawBuffer);
  for (uint32_t i{0}; i < count; ++i) {
    if (notNulls != nullptr &&
        !velox::bits::isBitSet(static_cast<const uint8_t*>(notNulls), i)) {
      continue;
    }
    if (!source[i].empty()) {
      std::copy(source[i].cbegin(), source[i].cend(), rawBuffer);
    }
    source[i] = std::string_view{rawBuffer, source[i].size()};
    rawBuffer += source[i].size();
  }
  stringBuffers.push_back(std::move(buffer));
}

uint32_t countRows(std::span<const RowRange> ranges) {
  uint64_t numRows{0};
  for (size_t i{0}; i < ranges.size(); ++i) {
    const auto& range = ranges[i];
    NIMBLE_CHECK(!range.empty(), "Read range must not be empty");
    if (i > 0) {
      NIMBLE_CHECK_LE(
          ranges[i - 1].endRow,
          range.startRow,
          "Read ranges must be ordered and disjoint");
    }
    numRows += range.numRows();
  }
  NIMBLE_CHECK_LE(
      numRows,
      static_cast<uint64_t>(std::numeric_limits<velox::vector_size_t>::max()));
  return static_cast<uint32_t>(numRows);
}

template <typename ReadFunction>
uint32_t readSelected(
    uint32_t numRows,
    DataType dataType,
    velox::memory::MemoryPool* pool,
    void* output,
    const std::function<void*()>& getOutputNulls,
    std::vector<velox::BufferPtr>& stringBuffers,
    ReadFunction readFunction) {
  if (numRows == 0) {
    return 0;
  }
  NIMBLE_CHECK_NOT_NULL(output);

  void* outputNulls{nullptr};
  const auto setNull = [&](uint32_t outputIndex) {
    NIMBLE_CHECK_NOT_NULL(
        getOutputNulls, "Nullable selected read requires output nulls");
    if (outputNulls == nullptr) {
      outputNulls = getOutputNulls();
      velox::bits::fillBits(
          static_cast<uint64_t*>(outputNulls),
          0,
          static_cast<velox::vector_size_t>(numRows),
          velox::bits::kNotNull);
    }
    velox::bits::clearBit(static_cast<uint64_t*>(outputNulls), outputIndex);
  };

  const auto numNonNulls = readFunction(setNull);
  if (dataType == DataType::String) {
    // TODO: Avoid this copy when EncodingView can transfer an ownership handle
    // for its encoded bytes to the output vector.
    retainStringContent(outputNulls, numRows, pool, output, stringBuffers);
  }
  return numNonNulls;
}

} // namespace

EncodingViewDecoder::EncodingViewDecoder(
    std::span<const std::string_view> chunks,
    velox::memory::MemoryPool* pool,
    velox::BufferPool* bufferPool,
    const EncodingViewFactory& encodingViewFactory)
    : pool_{velox::checkedNotNull(pool)},
      encodingViewFactory_{encodingViewFactory} {
  NIMBLE_CHECK(
      encodingViewFactory_ != nullptr,
      "EncodingViewDecoder requires an EncodingView factory");
  rebind(chunks, bufferPool);
}

void EncodingViewDecoder::rebind(
    std::span<const std::string_view> chunks,
    velox::BufferPool* bufferPool) {
  release();
  bufferPool_ = bufferPool;
  NIMBLE_CHECK(
      !chunks.empty(), "EncodingViewDecoder requires at least one chunk");
  SCOPE_FAIL {
    release();
  };
  uint32_t endRow{0};
  for (const auto chunk : chunks) {
    auto encodingView = encodingViewFactory_(chunk);
    NIMBLE_CHECK_NOT_NULL(encodingView);
    // @lint-ignore NULLSAFECLANG nullable-dereference
    const auto& view = *encodingView;
    if (!chunks_.empty()) {
      NIMBLE_CHECK_FILE_EQ(
          view.dataType(),
          dataType(),
          "Encoding chunks of a stream must store the same data type");
    }
    NIMBLE_CHECK_FILE_LE(
        view.rowCount(),
        std::numeric_limits<uint32_t>::max() - endRow,
        "Stream rows must fit in uint32_t");
    endRow += view.rowCount();
    chunks_.push_back({std::move(encodingView), endRow});
  }
}

void EncodingViewDecoder::release() {
  chunks_.clear();
  nextRow_ = 0;
}

uint32_t EncodingViewDecoder::next(
    uint32_t count,
    void* output,
    std::function<void*()> getOutputNulls,
    std::vector<velox::BufferPtr>& stringBuffers,
    const velox::bits::Bitmap* scatterOutputBitmap) {
  NIMBLE_CHECK_LE(nextRow_, numRows());
  NIMBLE_CHECK_LE(count, numRows() - nextRow_);
  const auto scatterSize = scatterOutputBitmap == nullptr
      ? 0
      : static_cast<int32_t>(scatterOutputBitmap->size());
  if (count == 0) {
    if (scatterOutputBitmap != nullptr) {
      NIMBLE_CHECK_EQ(
          velox::bits::countBits(
              static_cast<const uint64_t*>(scatterOutputBitmap->bits()),
              0,
              scatterSize),
          0,
          "Empty scattered reads require an empty scatter bitmap");
      NIMBLE_CHECK_NOT_NULL(
          getOutputNulls, "Scattered reads require output nulls");
      velox::bits::fillBits(
          static_cast<uint64_t*>(getOutputNulls()),
          0,
          scatterSize,
          velox::bits::kNull);
    }
    return 0;
  }

  NIMBLE_CHECK_NOT_NULL(output);
  const std::array<RowRange, 1> range{{{nextRow_, nextRow_ + count}}};
  if (scatterOutputBitmap == nullptr) {
    const auto numNonNulls = read(
        range, dataType(), output, std::move(getOutputNulls), stringBuffers);
    nextRow_ += count;
    return numNonNulls;
  }

  const auto* scatterBits =
      static_cast<const uint64_t*>(scatterOutputBitmap->bits());
  NIMBLE_CHECK_EQ(
      velox::bits::countBits(scatterBits, 0, scatterSize),
      count,
      "Scatter bitmap must contain one position per decoded row");
  if (scatterOutputBitmap->size() == count) {
    const auto numNonNulls = read(
        range, dataType(), output, std::move(getOutputNulls), stringBuffers);
    nextRow_ += count;
    return numNonNulls;
  }
  NIMBLE_CHECK_NOT_NULL(getOutputNulls, "Scattered reads require output nulls");
  const auto numNonNulls = scatterNext(
      count, output, getOutputNulls(), stringBuffers, scatterBits, scatterSize);
  nextRow_ += count;
  return numNonNulls;
}

uint32_t EncodingViewDecoder::remainingRows() {
  NIMBLE_CHECK_LE(nextRow_, numRows());
  if (nextRow_ == numRows()) {
    return 0;
  }
  return chunks_[findChunkIndex(nextRow_)].endRow - nextRow_;
}

uint32_t EncodingViewDecoder::scatterNext(
    uint32_t count,
    void* output,
    void* outputNulls,
    std::vector<velox::BufferPtr>& stringBuffers,
    const uint64_t* scatterBits,
    int32_t scatterSize) {
  const auto valueWidth = decodedValueWidth(dataType());
  ScopedVector<char> values{
      static_cast<size_t>(count) * valueWidth, pool_, bufferPool_};
  std::memset(values->data(), 0, values->size());
  ScopedVector<uint64_t> decodedNulls{
      velox::bits::nwords(count), pool_, bufferPool_};
  const std::array<RowRange, 1> range{{{nextRow_, nextRow_ + count}}};
  const auto numNonNulls = read(
      range,
      dataType(),
      values->data(),
      [&]() { return decodedNulls->data(); },
      stringBuffers);

  auto* rawOutputNulls = static_cast<uint64_t*>(outputNulls);
  velox::bits::fillBits(rawOutputNulls, 0, scatterSize, velox::bits::kNull);
  auto* outputBytes = static_cast<char*>(output);

  // Copy one contiguous run of selected positions at a time. forEachSetBit
  // steps over unselected gaps a word at a time, so the cost tracks the number
  // of decoded rows rather than the output size.
  int32_t sourceIndex{0};
  int32_t runBegin{0};
  int32_t runSize{0};
  const auto scatterRun = [&]() {
    if (runSize == 0) {
      return;
    }
    std::memcpy(
        outputBytes + static_cast<size_t>(runBegin) * valueWidth,
        values->data() + static_cast<size_t>(sourceIndex) * valueWidth,
        static_cast<size_t>(runSize) * valueWidth);
    if (numNonNulls == count) {
      velox::bits::fillBits(
          rawOutputNulls, runBegin, runBegin + runSize, velox::bits::kNotNull);
    } else {
      velox::bits::copyBits(
          decodedNulls->data(), sourceIndex, rawOutputNulls, runBegin, runSize);
    }
    sourceIndex += runSize;
  };
  velox::bits::forEachSetBit(
      scatterBits, 0, scatterSize, [&](int32_t outputIndex) {
        if (outputIndex == runBegin + runSize) {
          ++runSize;
          return;
        }
        scatterRun();
        runBegin = outputIndex;
        runSize = 1;
      });
  scatterRun();
  NIMBLE_CHECK_EQ(sourceIndex, static_cast<int32_t>(count));
  return numNonNulls;
}

uint32_t EncodingViewDecoder::read(
    std::span<const uint32_t> rows,
    DataType dataType,
    void* output,
    std::function<void*()> getOutputNulls,
    std::vector<velox::BufferPtr>& stringBuffers) {
  NIMBLE_CHECK_LE(
      rows.size(),
      static_cast<size_t>(std::numeric_limits<velox::vector_size_t>::max()));
  NIMBLE_CHECK_EQ(
      this->dataType(),
      dataType,
      "EncodingView data type does not match FieldReader");
  return readSelected(
      static_cast<uint32_t>(rows.size()),
      dataType,
      pool_,
      output,
      getOutputNulls,
      stringBuffers,
      [&](const auto& setNull) { return readRows(rows, setNull, output); });
}

uint32_t EncodingViewDecoder::read(
    std::span<const RowRange> ranges,
    DataType dataType,
    void* output,
    std::function<void*()> getOutputNulls,
    std::vector<velox::BufferPtr>& stringBuffers) {
  const auto numRows = countRows(ranges);
  NIMBLE_CHECK_EQ(
      this->dataType(),
      dataType,
      "EncodingView data type does not match FieldReader");
  return readSelected(
      numRows,
      dataType,
      pool_,
      output,
      getOutputNulls,
      stringBuffers,
      [&](const auto& setNull) { return readRanges(ranges, setNull, output); });
}

uint32_t EncodingViewDecoder::readRows(
    std::span<const uint32_t> rows,
    const std::function<void(uint32_t)>& setNull,
    void* output) {
  // A single chunk starts at stream row zero, so its rows are the stream rows.
  if (chunks_.size() == 1) {
    return chunks_.front().view->read(rows, setNull, output);
  }

  const auto valueWidth = decodedValueWidth(dataType());

  // Holds the part of 'rows' that falls in one chunk, converted to rows of that
  // chunk.
  std::vector<uint32_t> chunkRows;
  chunkRows.reserve(rows.size());
  uint32_t numNonNulls{0};
  size_t rowIndex{0};
  while (rowIndex < rows.size()) {
    // Reads each run of adjacent entries of 'rows' that fall in one chunk with
    // a single call. Rows can come in any order, so a chunk can be visited more
    // than once.
    const auto chunkIndex = findChunkIndex(rows[rowIndex]);
    const auto startRow = chunkStartRow(chunkIndex);
    const auto endRow = chunks_[chunkIndex].endRow;
    const auto outputOffset = static_cast<uint32_t>(rowIndex);
    chunkRows.clear();
    while (rowIndex < rows.size() && rows[rowIndex] >= startRow &&
           rows[rowIndex] < endRow) {
      chunkRows.push_back(rows[rowIndex] - startRow);
      ++rowIndex;
    }
    numNonNulls += chunks_[chunkIndex].view->read(
        chunkRows,
        [&setNull, outputOffset](uint32_t index) {
          setNull(outputOffset + index);
        },
        static_cast<char*>(output) +
            static_cast<size_t>(outputOffset) * valueWidth);
  }
  return numNonNulls;
}

uint32_t EncodingViewDecoder::readRanges(
    std::span<const RowRange> ranges,
    const std::function<void(uint32_t)>& setNull,
    void* output) {
  // A single chunk starts at stream row zero, so its rows are the stream rows.
  if (chunks_.size() == 1) {
    return chunks_.front().view->read(ranges, setNull, output);
  }

  const auto valueWidth = decodedValueWidth(dataType());

  // Holds the parts of 'ranges' that fall in one chunk, converted to rows of
  // that chunk. A chunk gets at most one part of each range.
  std::vector<RowRange> chunkRanges;
  chunkRanges.reserve(ranges.size());
  uint32_t numNonNulls{0};
  uint32_t outputOffset{0};
  size_t rangeIndex{0};

  // First row of ranges[rangeIndex] that is not read yet. A range that crosses
  // a chunk boundary is read in parts, one per chunk.
  uint32_t nextRangeRow{ranges[rangeIndex].startRow};
  while (rangeIndex < ranges.size()) {
    // Reads the parts of the ranges that fall in one chunk with a single call.
    const auto chunkIndex = findChunkIndex(nextRangeRow);
    const auto startRow = chunkStartRow(chunkIndex);
    const auto endRow = chunks_[chunkIndex].endRow;
    const auto chunkOutputOffset = outputOffset;
    chunkRanges.clear();
    while (rangeIndex < ranges.size() && nextRangeRow < endRow) {
      const auto& range = ranges[rangeIndex];
      const auto rangeEndRow = std::min(range.endRow, endRow);
      chunkRanges.emplace_back(nextRangeRow - startRow, rangeEndRow - startRow);
      outputOffset += rangeEndRow - nextRangeRow;
      if (rangeEndRow < range.endRow) {
        nextRangeRow = rangeEndRow;
        break;
      }
      if (++rangeIndex < ranges.size()) {
        nextRangeRow = ranges[rangeIndex].startRow;
      }
    }
    numNonNulls += chunks_[chunkIndex].view->read(
        chunkRanges,
        [&setNull, chunkOutputOffset](uint32_t index) {
          setNull(chunkOutputOffset + index);
        },
        static_cast<char*>(output) +
            static_cast<size_t>(chunkOutputOffset) * valueWidth);
  }
  return numNonNulls;
}

size_t EncodingViewDecoder::findChunkIndex(uint32_t row) const {
  NIMBLE_CHECK_LT(row, numRows(), "Read rows must be within the stream");
  return static_cast<size_t>(
      std::ranges::upper_bound(
          chunks_, row, std::ranges::less{}, &Chunk::endRow) -
      chunks_.begin());
}

void EncodingViewDecoder::skip(uint32_t count) {
  NIMBLE_CHECK_LE(nextRow_, numRows());
  NIMBLE_CHECK_LE(count, numRows() - nextRow_);
  nextRow_ += count;
}

void EncodingViewDecoder::reset() {
  nextRow_ = 0;
}

// The decoder reads through EncodingViews and never materializes an
// Encoding, so callers must treat the result as absent.
// @lint-ignore CLANGTIDY facebook-hte-NullableReturn
const Encoding* EncodingViewDecoder::encoding() const {
  return nullptr;
}

} // namespace facebook::nimble
