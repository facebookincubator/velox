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

#include <array>
#include <utility>

#include "velox/dwio/nimble/common/Types.h"
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrimitives.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelectionPolicy.h"

namespace facebook::nimble {

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

std::string_view SparseBoolEncoding::encodeWithSlicedIndices(
    std::string_view encoded,
    uint32_t length,
    std::span<const uint32_t> slicedIndicesWithSentinel,
    Buffer& buffer,
    const Encoding::Options& options) {
  NIMBLE_CHECK_GT(length, 0, "Cannot slice zero rows.");
  NIMBLE_CHECK(
      !slicedIndicesWithSentinel.empty(),
      "SparseBool slice indices must include row-count sentinel.");
  NIMBLE_CHECK_EQ(
      slicedIndicesWithSentinel.back(),
      length,
      "SparseBool slice sentinel must match slice length.");

  const char* readPos = encoded.data() +
      EncodingPrefix::prefixSize(encoded, options.useVarintRowCount);
  const auto sparseValue = encoding::readChar(readPos);
  const std::string_view encodedIndices{
      readPos, static_cast<size_t>(encoded.end() - readPos)};

  const auto serializedIndices =
      EncodingFactory::encodeWithCapturedLayout<uint32_t>(
          encodedIndices,
          slicedIndicesWithSentinel,
          buffer,
          options,
          "Captured SparseBool index layout");

  const auto prefixSize =
      EncodingPrefix::serializedSize(length, options.useVarintRowCount);
  const auto encodingSize =
      prefixSize + SparseBoolEncoding::kPrefixSize + serializedIndices.size();
  char* reserved = buffer.reserve(encodingSize);
  char* pos = reserved;
  EncodingPrefix::serialize(
      EncodingType::SparseBool,
      DataType::Bool,
      length,
      options.useVarintRowCount,
      pos);
  encoding::writeChar(sparseValue, pos);
  encoding::writeBytes(serializedIndices, pos);
  NIMBLE_CHECK_EQ(pos - reserved, encodingSize, "Encoding size mismatch.");
  return {reserved, encodingSize};
}

SparseBoolEncoding::SliceResult SparseBoolEncoding::sliceAndCount(
    std::string_view encoded,
    uint32_t offset,
    uint32_t length,
    Buffer& buffer,
    const Encoding::Options& options) {
  auto* pool = &buffer.getMemoryPool();
  const auto sourceRowCount =
      EncodingPrefix::readRowCount(encoded, options.useVarintRowCount);
  NIMBLE_CHECK_LE(offset, sourceRowCount);
  NIMBLE_CHECK_LE(length, sourceRowCount - offset);
  NIMBLE_CHECK_GT(length, 0, "Cannot slice zero rows.");

  SparseBoolEncoding source{
      *pool,
      encoded,
      [](uint32_t /*totalLength*/) -> void* { return nullptr; },
      options};
  const auto numSkippedSparse = source.skipSparseIndices(offset);

  const auto rowEnd = source.row_ + length;
  if (source.nextIndex_ >= rowEnd) {
    const RangeCounts counts = source.sparseValue()
        ? RangeCounts{
              .numTrueBeforeRange = numSkippedSparse,
              .numTrueInRange = 0,
          }
        : RangeCounts{
              .numTrueBeforeRange = offset - numSkippedSparse,
              .numTrueInRange = length,
          };
    const std::array<uint32_t, 1> slicedIndicesWithSentinel{length};
    return {
        .sliced = encodeWithSlicedIndices(
            encoded, length, slicedIndicesWithSentinel, buffer, options),
        .counts = counts,
    };
  }

  const auto maxSlicePositions =
      std::min<uint32_t>(length, source.indices_.rowCount() - 1);
  ScopedVector<uint32_t> slicedIndicesWithSentinel{
      maxSlicePositions + 1, pool, options.bufferPool};
  slicedIndicesWithSentinel->resize(0);
  const auto numSparseInSlice =
      source.materializeSparseIndices(length, *slicedIndicesWithSentinel);
  slicedIndicesWithSentinel->push_back(length);

  const RangeCounts counts = source.sparseValue()
      ? RangeCounts{
            .numTrueBeforeRange = numSkippedSparse,
            .numTrueInRange = numSparseInSlice,
        }
      : RangeCounts{
            .numTrueBeforeRange = offset - numSkippedSparse,
            .numTrueInRange = length - numSparseInSlice,
        };
  return {
      .sliced = encodeWithSlicedIndices(
          encoded,
          length,
          std::span<const uint32_t>{
              slicedIndicesWithSentinel->data(),
              slicedIndicesWithSentinel->size()},
          buffer,
          options),
      .counts = counts,
  };
}

} // namespace facebook::nimble
