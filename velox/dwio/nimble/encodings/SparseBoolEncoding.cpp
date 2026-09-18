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
#include "velox/dwio/nimble/encodings/SparseBoolEncoding.h"

#include <cstring>
#include <tuple>
#include <utility>

#include "velox/dwio/nimble/common/Types.h"
#include "velox/dwio/nimble/encodings/SliceEncoding.h"
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrimitives.h"
#include "velox/dwio/nimble/encodings/common/SortedPositionSlots.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelectionPolicy.h"

namespace facebook::nimble {

namespace {

// Return value of sliceSortedPositionStream: the sliced+wrapped position
// bytes plus the two counts the caller needs to fill in RangeCounts without
// walking the stream a second time.
struct SlicedSparsePositions {
  // Encoded SliceEncoding-wrapped position stream for the requested row range.
  std::string_view slicedPositions;
  // Sparse-position count inside [offset, offset + length).
  uint32_t sparseInRange;
  // Sparse-position count strictly before |offset|.
  uint32_t sparseBefore;
};

// Slices a SparseBool position stream (strictly ascending, terminated by a
// sentinel at value sourceRowCount) over rows [offset, offset + length),
// rebasing every retained position by -offset. Locates the slot bounds
// covering the range via detail::findSortedPositionSlots, slices the
// retained range plus one terminator slot positionally through
// EncodingFactory::slice, and wraps the sliced bytes with SliceEncoding.
// When the sliced inner encoding supports push-down (FixedBitWidth / PFOR /
// Constant / Nullable / Trivial), the wrapper folds the -offset shift into
// the inner bytes at write time and the on-wire delta is zero. Otherwise
// the delta rides on the wire and the reader applies it.
SlicedSparsePositions sliceSortedPositionStream(
    std::string_view encodedPositions,
    uint32_t offset,
    uint32_t length,
    Buffer& scratchBuffer,
    const Encoding::Options& options) {
  auto& pool = scratchBuffer.getMemoryPool();
  const auto rangeEnd = offset + length;
  const int64_t valueDelta = -static_cast<int64_t>(offset);

  const uint32_t positionCount =
      EncodingPrefix::readRowCount(encodedPositions, options.useVarintRowCount);
  const auto [slotStart, slotEnd] = detail::findSortedPositionSlots(
      encodedPositions, positionCount, offset, rangeEnd, pool, options);
  NIMBLE_CHECK_LT(
      slotEnd,
      positionCount,
      "SparseBool positions must end in a sentinel past the slice end.");
  const uint32_t sparseInRange = slotEnd - slotStart;
  const uint32_t retainedCount = sparseInRange + 1;
  const auto slicedPositions = EncodingFactory::slice(
      encodedPositions, slotStart, retainedCount, scratchBuffer, options);
  return {
      .slicedPositions = SliceEncoding<uint32_t>::wrap(
          slicedPositions,
          /*offset=*/0,
          /*length=*/retainedCount,
          scratchBuffer,
          valueDelta,
          options),
      .sparseInRange = sparseInRange,
      .sparseBefore = slotStart,
  };
}

} // namespace

SparseBoolEncoding::SparseBoolEncoding(
    velox::memory::MemoryPool& pool,
    std::string_view data,
    std::function<void*(uint32_t)> stringBufferFactory,
    const Encoding::Options& options)
    : TypedEncoding<bool, bool>{pool, data, options},
      sparseValue_{static_cast<bool>(data[this->dataOffset()])},
      indicesUncompressed_{&pool},
      indices_{EncodingFactory().create(
          pool,
          {data.data() + this->dataOffset() + kPrefixSize,
           data.size() - this->dataOffset() - kPrefixSize},
          stringBufferFactory,
          options)} {
  reset();
}

void SparseBoolEncoding::reset() {
  row_ = 0;
  indices_.reset();
  nextIndex_ = indices_.nextValue();
}

void SparseBoolEncoding::skip(uint32_t rowCount) {
  const uint32_t end = row_ + rowCount;
  while (nextIndex_ < end) {
    nextIndex_ = indices_.nextValue();
  }
  row_ = end;
}

void SparseBoolEncoding::materialize(uint32_t rowCount, void* buffer) {
  if (rowCount == 0) {
    return;
  }
  const uint32_t end = row_ + rowCount;
  if (sparseValue_) {
    memset(buffer, 0, rowCount);
    while (nextIndex_ < end) {
      static_cast<bool*>(buffer)[nextIndex_ - row_] = true;
      nextIndex_ = indices_.nextValue();
    }
  } else {
    memset(buffer, 1, rowCount);
    while (nextIndex_ < end) {
      static_cast<bool*>(buffer)[nextIndex_ - row_] = false;
      nextIndex_ = indices_.nextValue();
    }
  }
  row_ = end;
}

void SparseBoolEncoding::materializeBoolsAsBits(
    uint32_t rowCount,
    uint64_t* buffer,
    int begin) {
  if (rowCount == 0) {
    return;
  }
  velox::bits::fillBits(buffer, begin, begin + rowCount, !sparseValue_);
  const auto end = row_ + rowCount;
  if (sparseValue_) {
    while (nextIndex_ < end) {
      velox::bits::setBit(buffer, begin + nextIndex_ - row_);
      nextIndex_ = indices_.nextValue();
    }
  } else {
    while (nextIndex_ < end) {
      velox::bits::clearBit(buffer, begin + nextIndex_ - row_);
      nextIndex_ = indices_.nextValue();
    }
  }
  row_ = end;
}

uint32_t SparseBoolEncoding::materializeSparseIndices(
    uint32_t rowCount,
    Vector<uint32_t>& buffer) {
  const uint32_t maxSparsePositions =
      std::min(rowCount, indices_.rowCount() - 1);
  buffer.reserve(maxSparsePositions);
  const uint32_t begin = row_;
  const uint32_t end = row_ + rowCount;
  uint32_t count{0};
  auto* positions = buffer.data();
  while (nextIndex_ < end) {
    NIMBLE_DCHECK_LT(
        count,
        maxSparsePositions,
        "SparseBool sparse position count exceeds its upper bound.");
    positions[count++] = nextIndex_ - begin;
    nextIndex_ = indices_.nextValue();
  }
  buffer.update_size(count);
  row_ = end;
  return count;
}

uint32_t SparseBoolEncoding::skipSparseIndices(uint32_t rowCount) {
  uint32_t count{0};
  const uint32_t end = row_ + rowCount;
  while (nextIndex_ < end) {
    ++count;
    nextIndex_ = indices_.nextValue();
  }
  row_ = end;
  return count;
}

void SparseBoolEncoding::countTrue(
    std::string_view encoded,
    uint32_t offset,
    uint32_t length,
    velox::memory::MemoryPool* pool,
    RangeCounts& counts,
    const Encoding::Options& options) {
  counts = {};
  NIMBLE_CHECK_NOT_NULL(pool, "Memory pool cannot be null");
  const auto rowCount =
      EncodingPrefix::readRowCount(encoded, options.useVarintRowCount);
  NIMBLE_CHECK_LE(offset, rowCount);
  NIMBLE_CHECK_LE(length, rowCount - offset);
  NIMBLE_CHECK_GT(length, 0, "Cannot count zero rows.");

  SparseBoolEncoding sparseBool{
      *pool,
      encoded,
      [](uint32_t /*totalLength*/) -> void* { return nullptr; },
      options};
  const auto numSkippedSparseBeforeRange = sparseBool.skipSparseIndices(offset);
  const auto numSkippedSparseInRange = sparseBool.skipSparseIndices(length);
  if (sparseBool.sparseValue()) {
    counts = {
        .numTrueBeforeRange = numSkippedSparseBeforeRange,
        .numTrueInRange = numSkippedSparseInRange,
    };
    return;
  }
  counts = {
      .numTrueBeforeRange = offset - numSkippedSparseBeforeRange,
      .numTrueInRange = length - numSkippedSparseInRange,
  };
}

uint32_t SparseBoolEncoding::countTrue(
    std::string_view encoded,
    uint32_t offset,
    uint32_t length,
    velox::memory::MemoryPool* pool,
    const Encoding::Options& options) {
  RangeCounts counts;
  countTrue(encoded, offset, length, pool, counts, options);
  return counts.numTrueInRange;
}

std::string_view SparseBoolEncoding::encode(
    EncodingSelection<bool>& selection,
    std::span<const bool> values,
    Buffer& buffer,
    const Encoding::Options& options) {
  const bool useVarint = options.useVarintRowCount;
  // Decide the polarity of the encoding.
  const uint64_t valueCount = values.size();
  const uint64_t setCount =
      selection.statistics().uniqueCounts().value().at(true);
  bool sparseValue;
  uint64_t indexCount;
  if (setCount > (valueCount >> 1)) {
    sparseValue = false;
    indexCount = valueCount - setCount;
  } else {
    sparseValue = true;
    indexCount = setCount;
  }

  auto* pool = &buffer.getMemoryPool();
  Vector<uint32_t> indices{pool};
  indices.reserve(indexCount + 1);
  if (sparseValue) {
    for (auto i = 0; i < values.size(); ++i) {
      if (values[i]) {
        indices.push_back(i);
      }
    }
  } else {
    for (auto i = 0; i < values.size(); ++i) {
      if (!values[i]) {
        indices.push_back(i);
      }
    }
  }

  // Pushing rowCount as the last item. Materialize relies on finding this value
  // in order to stop looping as this value is greater than any possible index.
  indices.push_back(valueCount);

  ScopedEncodingBuffer scopedBuffer{pool, options.encodingBufferPool};
  std::string_view serializedIndices =
      selection.template encodeNested<uint32_t>(
          EncodingIdentifiers::SparseBool::Indices,
          indices,
          scopedBuffer.get(),
          options);

  const uint32_t encodingSize =
      Encoding::serializePrefixSize(valueCount, useVarint) +
      SparseBoolEncoding::kPrefixSize + serializedIndices.size();
  char* reserved = buffer.reserve(encodingSize);
  char* pos = reserved;
  Encoding::serializePrefix(
      EncodingType::SparseBool, DataType::Bool, valueCount, useVarint, pos);
  encoding::writeChar(sparseValue, pos);
  encoding::writeBytes(serializedIndices, pos);

  NIMBLE_DCHECK_EQ(pos - reserved, encodingSize, "Encoding size mismatch.");
  return {reserved, encodingSize};
}

std::string_view SparseBoolEncoding::slice(
    std::string_view encoded,
    uint32_t offset,
    uint32_t length,
    Buffer& buffer,
    const Encoding::Options& options) {
  return sliceAndCount(encoded, offset, length, buffer, options).sliced;
}

SparseBoolEncoding::SliceResult SparseBoolEncoding::sliceAndCount(
    std::string_view encoded,
    uint32_t offset,
    uint32_t length,
    Buffer& buffer,
    const Encoding::Options& options) {
  const auto sourceRowCount =
      EncodingPrefix::readRowCount(encoded, options.useVarintRowCount);
  NIMBLE_CHECK_LE(offset, sourceRowCount);
  NIMBLE_CHECK_LE(length, sourceRowCount - offset);
  NIMBLE_CHECK_GT(length, 0, "Cannot slice zero rows.");

  const auto payload = encoded.substr(
      EncodingPrefix::prefixSize(encoded, options.useVarintRowCount));
  NIMBLE_CHECK(
      !payload.empty(), "SparseBool stream is missing its polarity byte.");
  const bool sparseValue = static_cast<bool>(payload.front());
  const std::string_view encodedPositions = payload.substr(sizeof(uint8_t));

  // The sliced positions are serialized into scratch space first because the
  // outer encoding size depends on their size.
  ScopedEncodingBuffer scopedBuffer{
      &buffer.getMemoryPool(), options.encodingBufferPool};
  const auto [slicedPositions, sparseInRange, sparseBefore] =
      sliceSortedPositionStream(
          encodedPositions, offset, length, scopedBuffer.get(), options);

  const auto prefixSize =
      EncodingPrefix::serializedSize(length, options.useVarintRowCount);
  const auto encodingSize =
      prefixSize + SparseBoolEncoding::kPrefixSize + slicedPositions.size();
  char* reserved = buffer.reserve(encodingSize);
  char* pos = reserved;
  EncodingPrefix::serialize(
      EncodingType::SparseBool,
      DataType::Bool,
      length,
      options.useVarintRowCount,
      pos);
  encoding::writeChar(static_cast<char>(sparseValue), pos);
  encoding::writeBytes(slicedPositions, pos);
  NIMBLE_CHECK_EQ(pos - reserved, encodingSize, "Encoding size mismatch.");

  // True positions carry the value indicated by |sparseValue|, so the true
  // counts are either the sparse counts or their complement within the range.
  return {
      .sliced = {reserved, encodingSize},
      .counts = sparseValue
          ? RangeCounts{
                .numTrueBeforeRange = sparseBefore,
                .numTrueInRange = sparseInRange,
            }
          : RangeCounts{
                .numTrueBeforeRange = offset - sparseBefore,
                .numTrueInRange = length - sparseInRange,
            },
  };
}

} // namespace facebook::nimble
