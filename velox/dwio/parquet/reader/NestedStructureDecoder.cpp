/*
 * Copyright (c) Facebook, Inc. and its affiliates.
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

#include "velox/dwio/parquet/reader/NestedStructureDecoder.h"

#include "velox/common/base/Exceptions.h"
#include "velox/dwio/common/BufferUtil.h"

#include <iostream>

namespace facebook::velox::parquet {

void NestedStructureDecoder::readOffsetsAndNulls(
    const uint8_t* repetitionLevels,
    const uint8_t* definitionLevels,
    uint64_t numValues,
    uint32_t maxRepeat,
    uint32_t maxDefinition,
    int64_t& lastOffset,
    bool& wasLastCollectionNull,
    uint32_t* offsets,
    uint64_t* nulls,
    uint64_t& numNonEmptyCollections,
    uint64_t& numNonNullCollections,
    memory::MemoryPool& pool) {
  // Use the child level's max repetition value, which is +1 to the current
  // level's value
  auto childMaxRepeat = maxRepeat + 1;

  int64_t offset = lastOffset;
  for (int64_t i = 0; i < numValues; ++i) {
    uint8_t definitionLevel = definitionLevels[i];
    uint8_t repetitionLevel = repetitionLevels[i];

    // empty means it belongs to a row that is null in one of its ancestor
    // levels.
    bool isEmpty = definitionLevel < (maxDefinition - 1);
    bool isNull = definitionLevel == (maxDefinition - 1);
    bool isNotNull = definitionLevel >= maxDefinition;
    bool isCollectionBegin = (repetitionLevel < childMaxRepeat) & !isEmpty;
    bool isEntryBegin = (repetitionLevel <= childMaxRepeat) & !isEmpty;

    offset += isEntryBegin & !wasLastCollectionNull;
    offsets[numNonEmptyCollections] = offset;
    bits::setNull(nulls, numNonEmptyCollections, isNull);

    // Always update the outputs, but only increase the outputIndex when the
    // current entry is the begin of a new collection, and it's not empty.
    // Benchmark shows skipping non-collection-begin rows is worse than this
    // solution by nearly 2x because of extra branchings added for skipping.
    numNonNullCollections += isNotNull & isCollectionBegin;
    numNonEmptyCollections += isCollectionBegin;

    wasLastCollectionNull = isEmpty ? wasLastCollectionNull : isNull;
    lastOffset = isCollectionBegin ? offset : lastOffset;
  }

  lastOffset = offset;
}

int64_t NestedStructureDecoder::readOffsetsAndNulls(
    const uint8_t* repetitionLevels,
    const uint8_t* definitionLevels,
    uint64_t numValues,
    uint8_t maxRepeat,
    uint8_t maxDefinition,
    BufferPtr& offsetsBuffer,
    BufferPtr& lengthsBuffer,
    BufferPtr& nullsBuffer,
    uint64_t& numNonEmptyCollections,
    uint64_t& numNonNullCollections,
    memory::MemoryPool& pool) {
  dwio::common::ensureCapacity<uint8_t>(
      nullsBuffer, bits::nbytes(numValues), &pool);
  dwio::common::ensureCapacity<vector_size_t>(
      offsetsBuffer, numValues + 1, &pool);
  dwio::common::ensureCapacity<vector_size_t>(lengthsBuffer, numValues, &pool);

  auto offsets = offsetsBuffer->asMutable<uint32_t>();
  auto lengths = lengthsBuffer->asMutable<uint32_t>();
  auto nulls = nullsBuffer->asMutable<uint64_t>();

  bool wasLastCollectionNull = false;
  int64_t lastOffset = -1;
  numNonEmptyCollections = 0;
  numNonNullCollections = 0;
  readOffsetsAndNulls(
      repetitionLevels,
      definitionLevels,
      numValues,
      maxRepeat,
      maxDefinition,
      lastOffset,
      wasLastCollectionNull,
      offsets,
      nulls,
      numNonEmptyCollections,
      numNonNullCollections,
      pool);

  auto endOffset = lastOffset + !wasLastCollectionNull;
  offsets[numNonEmptyCollections] = endOffset;

  for (int i = 0; i < numNonEmptyCollections; i++) {
    lengths[i] = offsets[i + 1] - offsets[i];
  }

  return numNonEmptyCollections;
}

} // namespace facebook::velox::parquet
