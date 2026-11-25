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
#include <functional>
#include <limits>
#include <span>
#include <vector>

#include <folly/ScopeGuard.h>

#include "velox/common/Casts.h"
#include "velox/dwio/nimble/common/NimbleException.h"
#include "velox/dwio/nimble/common/Vector.h"

namespace facebook::nimble {
namespace {

// Copies the selected strings into one new buffer, which it adds to
// 'stringBuffers', so that they stay valid after the decoder is gone.
void copyStringContent(
    const void* notNulls,
    uint32_t count,
    velox::memory::MemoryPool* pool,
    void* outputValues,
    std::vector<velox::BufferPtr>& stringBuffers) {
  auto* source = static_cast<std::string_view*>(outputValues);
  const auto selected = [&](uint32_t row) {
    return notNulls == nullptr ||
        velox::bits::isBitSet(static_cast<const uint8_t*>(notNulls), row);
  };
  uint64_t totalBytes{0};
  for (uint32_t i{0}; i < count; ++i) {
    if (selected(i)) {
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
    if (!selected(i)) {
      continue;
    }
    std::copy(source[i].cbegin(), source[i].cend(), rawBuffer);
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

// Reads with readFunction, then, for strings, keeps them valid with
// retainFunction, which takes the output nulls, or nullptr when no row is null.
template <typename ReadFunction, typename RetainFunction>
uint32_t readSelected(
    uint32_t numRows,
    DataType dataType,
    void* output,
    const std::function<void*()>& getOutputNulls,
    ReadFunction readFunction,
    RetainFunction retainFunction) {
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
    retainFunction(outputNulls);
  }
  return numNonNulls;
}

#ifndef NDEBUG
// Checks that each selected non-empty string that a read wrote to
// 'outputValues' lies in one of 'buffers', the buffers that the views passed to
// vouch for the strings they returned. Finding the buffer of each string costs
// as much as the lookup that the views spare the read, so only debug builds
// check.
void checkStringsInBuffers(
    const void* notNulls,
    uint32_t count,
    std::span<const velox::BufferPtr> buffers,
    const void* outputValues) {
  const auto* values = static_cast<const std::string_view*>(outputValues);
  const std::less<const char*> less;
  for (uint32_t i{0}; i < count; ++i) {
    if (values[i].empty() ||
        (notNulls != nullptr &&
         !velox::bits::isBitSet(static_cast<const uint8_t*>(notNulls), i))) {
      continue;
    }
    const bool held =
        std::any_of(buffers.begin(), buffers.end(), [&](const auto& buffer) {
          const auto* begin = buffer->template as<char>();
          return !less(values[i].data(), begin) &&
              !less(
                  begin + buffer->size(), values[i].data() + values[i].size());
        });
    NIMBLE_DCHECK(
        held,
        "A view must pass buffers that hold every string it returns: output {}",
        i);
  }
}
#endif

} // namespace

namespace detail {

// Collects the buffers that hold the strings of a read that holds them instead
// of copying them, such as a scan: the buffers that the views of the chunks
// pass to the addStringBuffer of their reads, which hold every non-empty string
// that the reads return. The decoder keeps one for its reads, and resets it
// after each.
class StringRetainer {
 public:
  StringRetainer() = default;

  ~StringRetainer() = default;

  // addBuffer_ points to this object, so it must not move.
  StringRetainer(const StringRetainer&) = delete;
  StringRetainer& operator=(const StringRetainer&) = delete;
  StringRetainer(StringRetainer&&) = delete;
  StringRetainer& operator=(StringRetainer&&) = delete;

  // Returns the addStringBuffer that a read passes to the reads of the views.
  const std::function<void(velox::BufferPtr)>& addBuffer() const {
    return addBuffer_;
  }

  // Returns the buffers that the views passed since the last reset().
  const std::vector<velox::BufferPtr>& buffers() const {
    return buffers_;
  }

  // Moves the buffers that the views passed since the last reset() to
  // 'buffers'.
  void retain(std::vector<velox::BufferPtr>& buffers) {
    buffers.insert(
        buffers.end(),
        std::make_move_iterator(buffers_.begin()),
        std::make_move_iterator(buffers_.end()));
  }

  // Drops the buffers of the last read, keeping the capacity of the list.
  void reset() {
    buffers_.clear();
  }

 private:
  // Keeps each buffer once, since the views of several chunks can pass the
  // same shared buffer.
  const std::function<void(velox::BufferPtr)> addBuffer_{
      [this](velox::BufferPtr buffer) {
        NIMBLE_CHECK_NOT_NULL(buffer, "A view must not pass a null buffer");
        NIMBLE_CHECK_GT(
            buffer->size(), 0, "A view must not pass an empty buffer");
        if (std::find(buffers_.begin(), buffers_.end(), buffer) ==
            buffers_.end()) {
          buffers_.emplace_back(std::move(buffer));
        }
      }};

  // The buffers that the views passed to addBuffer_, each once.
  std::vector<velox::BufferPtr> buffers_;
};

} // namespace detail

namespace {

// Returns the addStringBuffer that a read passes to the reads of the views:
// that of 'retainer', or none when it is null, for a read that copies its
// strings, so that the views need not pass buffers.
const std::function<void(velox::BufferPtr)>& addStringBufferOf(
    const detail::StringRetainer* retainer) {
  static const std::function<void(velox::BufferPtr)> kNoAddStringBuffer;
  return retainer == nullptr ? kNoAddStringBuffer : retainer->addBuffer();
}

// Keeps the selected strings that a read wrote to 'outputValues' valid after
// the decoder is gone: holds the buffers that 'retainer' collected, or copies
// every string when it is null, as for a sparse read.
void retainStrings(
    const void* notNulls,
    uint32_t count,
    velox::memory::MemoryPool* pool,
    detail::StringRetainer* retainer,
    void* outputValues,
    std::vector<velox::BufferPtr>& stringBuffers) {
  if (retainer == nullptr) {
    copyStringContent(notNulls, count, pool, outputValues, stringBuffers);
    return;
  }
#ifndef NDEBUG
  checkStringsInBuffers(notNulls, count, retainer->buffers(), outputValues);
#endif
  retainer->retain(stringBuffers);
}

// Checks that 'bufferHolder' holds all of the bytes of 'chunk'.
void checkHoldsChunk(
    const velox::Buffer& bufferHolder,
    std::string_view chunk) {
  const std::less<const char*> less;
  const auto* begin = bufferHolder.as<char>();
  NIMBLE_USER_CHECK(
      !less(chunk.data(), begin),
      "A chunk buffer holder must not start after its chunk");
  NIMBLE_USER_CHECK(
      !less(begin + bufferHolder.size(), chunk.data() + chunk.size()),
      "A chunk buffer holder must not end before its chunk");
}

// Creates the view of each of 'chunks' with 'encodingViewFactory', which it
// passes the chunk and the buffer that holds it: the chunk's entry of
// 'chunkBufferHolders', or null when that is empty. Appends each view, with
// that buffer and the stream row where its chunk ends, to 'boundChunks', which
// must be empty.
void bindChunks(
    std::span<const std::string_view> chunks,
    std::span<const velox::BufferPtr> chunkBufferHolders,
    const EncodingViewDecoder::EncodingViewFactory& encodingViewFactory,
    std::vector<detail::BoundChunk>& boundChunks) {
  NIMBLE_CHECK(boundChunks.empty(), "Chunks must be bound to an empty list");
  if (!chunkBufferHolders.empty()) {
    NIMBLE_USER_CHECK_EQ(
        chunkBufferHolders.size(),
        chunks.size(),
        "Chunk buffer holders must align with the chunks");
  }
  static const velox::BufferPtr kNoBufferHolder;
  uint32_t endRow{0};
  for (size_t i{0}; i < chunks.size(); ++i) {
    const auto chunk = chunks[i];
    const auto& bufferHolder =
        chunkBufferHolders.empty() ? kNoBufferHolder : chunkBufferHolders[i];
    if (bufferHolder != nullptr) {
      checkHoldsChunk(*bufferHolder, chunk);
    }
    auto encodingView = encodingViewFactory(chunk, bufferHolder);
    NIMBLE_CHECK_NOT_NULL(encodingView);
    // @lint-ignore NULLSAFECLANG nullable-dereference
    const auto& view = *encodingView;
    if (!boundChunks.empty()) {
      NIMBLE_CHECK_FILE_EQ(
          view.dataType(),
          boundChunks.front().view->dataType(),
          "Encoding chunks of a stream must store the same data type");
    }
    NIMBLE_CHECK_FILE_LE(
        view.rowCount(),
        std::numeric_limits<uint32_t>::max() - endRow,
        "Stream rows must fit in uint32_t");
    endRow += view.rowCount();
    boundChunks.push_back({bufferHolder, std::move(encodingView), endRow});
  }
}

} // namespace

EncodingViewDecoder::EncodingViewDecoder(
    std::span<const std::string_view> chunks,
    std::span<const velox::BufferPtr> chunkBufferHolders,
    velox::memory::MemoryPool* pool,
    velox::BufferPool* bufferPool,
    const EncodingViewFactory& encodingViewFactory)
    : pool_{velox::checkedNotNull(pool)},
      encodingViewFactory_{encodingViewFactory} {
  NIMBLE_CHECK(
      encodingViewFactory_ != nullptr,
      "EncodingViewDecoder requires an EncodingView factory");
  rebind(chunks, chunkBufferHolders, bufferPool);
}

EncodingViewDecoder::~EncodingViewDecoder() = default;

void EncodingViewDecoder::rebind(
    std::span<const std::string_view> chunks,
    std::span<const velox::BufferPtr> chunkBufferHolders,
    velox::BufferPool* bufferPool) {
  release();
  bufferPool_ = bufferPool;
  NIMBLE_CHECK(
      !chunks.empty(), "EncodingViewDecoder requires at least one chunk");
  SCOPE_FAIL {
    release();
  };
  bindChunks(chunks, chunkBufferHolders, encodingViewFactory_, chunks_);
}

void EncodingViewDecoder::release() {
  chunks_.clear();
  nextRow_ = 0;
}

bool EncodingViewDecoder::holdsStringBuffers(
    uint32_t numSelectedRows,
    uint32_t firstRow,
    uint32_t lastRow) const {
  // A read must select at least one in this many of the rows of the chunks it
  // spans to hold their buffers instead of copying its values.
  constexpr uint64_t kMaxSpannedRowsPerSelectedRow{4};

  // The rows from firstRow to lastRow are fewer than the rows of the chunks
  // they span, so this skips looking up the chunks of a sparse read.
  if (numSelectedRows * kMaxSpannedRowsPerSelectedRow <
      static_cast<uint64_t>(lastRow - firstRow) + 1) {
    return false;
  }
  const uint64_t numSpannedRows = chunks_[findChunkIndex(lastRow)].endRow -
      chunkStartRow(findChunkIndex(firstRow));
  return numSelectedRows * kMaxSpannedRowsPerSelectedRow >= numSpannedRows;
}

// @lint-ignore CLANGTIDY facebook-hte-NullableReturn
detail::StringRetainer* EncodingViewDecoder::stringRetainer(
    uint32_t numSelectedRows,
    uint32_t firstRow,
    uint32_t lastRow) {
  if (!holdsStringBuffers(numSelectedRows, firstRow, lastRow)) {
    return nullptr;
  }
  if (stringRetainer_ == nullptr) {
    stringRetainer_ = std::make_unique<detail::StringRetainer>();
  }
  return stringRetainer_.get();
}

// @lint-ignore CLANGTIDY facebook-hte-NullableReturn
detail::StringRetainer* EncodingViewDecoder::stringRetainer(
    DataType dataType,
    std::span<const uint32_t> rows) {
  if (dataType != DataType::String || rows.empty()) {
    return nullptr;
  }

  // The rows can come in any order. Sorted rows give their lowest and highest
  // without a search.
  const auto [first, last] = std::is_sorted(rows.begin(), rows.end())
      ? std::pair{rows.begin(), rows.end() - 1}
      : std::minmax_element(rows.begin(), rows.end());
  return stringRetainer(static_cast<uint32_t>(rows.size()), *first, *last);
}

// @lint-ignore CLANGTIDY facebook-hte-NullableReturn
detail::StringRetainer* EncodingViewDecoder::stringRetainer(
    DataType dataType,
    std::span<const RowRange> ranges,
    uint32_t numRows) {
  if (dataType != DataType::String || numRows == 0) {
    return nullptr;
  }
  return stringRetainer(
      numRows, ranges.front().startRow, ranges.back().endRow - 1);
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
  const auto numRows = static_cast<uint32_t>(rows.size());
  auto* retainer = stringRetainer(dataType, rows);
  SCOPE_EXIT {
    if (retainer != nullptr) {
      retainer->reset();
    }
  };
  return readSelected(
      numRows,
      dataType,
      output,
      getOutputNulls,
      [&](const auto& setNull) {
        return readRows(rows, setNull, addStringBufferOf(retainer), output);
      },
      [&](const void* outputNulls) {
        retainStrings(
            outputNulls, numRows, pool_, retainer, output, stringBuffers);
      });
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
  auto* retainer = stringRetainer(dataType, ranges, numRows);
  SCOPE_EXIT {
    if (retainer != nullptr) {
      retainer->reset();
    }
  };
  return readSelected(
      numRows,
      dataType,
      output,
      getOutputNulls,
      [&](const auto& setNull) {
        return readRanges(ranges, setNull, addStringBufferOf(retainer), output);
      },
      [&](const void* outputNulls) {
        retainStrings(
            outputNulls, numRows, pool_, retainer, output, stringBuffers);
      });
}

uint32_t EncodingViewDecoder::readRows(
    std::span<const uint32_t> rows,
    const std::function<void(uint32_t)>& setNull,
    const std::function<void(velox::BufferPtr)>& addStringBuffer,
    void* output) {
  // A single chunk starts at stream row zero, so its rows are the stream rows.
  if (chunks_.size() == 1) {
    return chunks_.front().view->read(rows, setNull, addStringBuffer, output);
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
        addStringBuffer,
        static_cast<char*>(output) +
            static_cast<size_t>(outputOffset) * valueWidth);
  }
  return numNonNulls;
}

uint32_t EncodingViewDecoder::readRanges(
    std::span<const RowRange> ranges,
    const std::function<void(uint32_t)>& setNull,
    const std::function<void(velox::BufferPtr)>& addStringBuffer,
    void* output) {
  // A single chunk starts at stream row zero, so its rows are the stream rows.
  if (chunks_.size() == 1) {
    return chunks_.front().view->read(ranges, setNull, addStringBuffer, output);
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
        addStringBuffer,
        static_cast<char*>(output) +
            static_cast<size_t>(chunkOutputOffset) * valueWidth);
  }
  return numNonNulls;
}

size_t EncodingViewDecoder::findChunkIndex(uint32_t row) const {
  NIMBLE_CHECK_LT(row, numRows(), "Read rows must be within the stream");
  return static_cast<size_t>(
      std::ranges::upper_bound(
          chunks_, row, std::ranges::less{}, &detail::BoundChunk::endRow) -
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
