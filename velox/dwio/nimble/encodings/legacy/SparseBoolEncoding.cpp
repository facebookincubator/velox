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
#include <numeric>

#include "velox/dwio/nimble/encodings/legacy/SparseBoolEncoding.h"

#include "velox/dwio/nimble/common/Types.h"
#include "velox/dwio/nimble/compression/Compression.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrimitives.h"
#include "velox/dwio/nimble/encodings/legacy/EncodingFactory.h"

namespace facebook::nimble::legacy {

SparseBoolEncoding::SparseBoolEncoding(
    velox::memory::MemoryPool& memoryPool,
    std::string_view data,
    std::function<void*(uint32_t)> stringBufferFactory)
    : TypedEncoding<bool, bool>{memoryPool, data},
      sparseValue_{static_cast<bool>(data[kSparseValueOffset])},
      indicesUncompressed_{&memoryPool},
      indices_{EncodingFactory().create(
          memoryPool,
          {data.data() + kIndicesOffset, data.size() - kIndicesOffset},
          stringBufferFactory,
          Encoding::Options{})} {
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

std::string_view SparseBoolEncoding::encode(
    EncodingSelection<bool>& selection,
    std::span<const bool> values,
    Buffer& buffer) {
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

  Vector<uint32_t> indices{&buffer.getMemoryPool()};
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

  Buffer tempBuffer{buffer.getMemoryPool()};
  std::string_view serializedIndices =
      selection.template encodeNested<uint32_t>(
          EncodingIdentifiers::SparseBool::Indices, indices, tempBuffer);

  const uint32_t encodingSize = EncodingPrefix::kFixedPrefixSize +
      SparseBoolEncoding::kPrefixSize + serializedIndices.size();
  char* reserved = buffer.reserve(encodingSize);
  char* pos = reserved;
  Encoding::serializePrefix(
      EncodingType::SparseBool, DataType::Bool, valueCount, false, pos);
  encoding::writeChar(sparseValue, pos);
  encoding::writeBytes(serializedIndices, pos);

  NIMBLE_DCHECK_EQ(pos - reserved, encodingSize, "Encoding size mismatch.");
  return {reserved, encodingSize};
}

} // namespace facebook::nimble::legacy
