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

int64_t NestedStructureDecoder::readOffsetsAndNulls(
    const uint8_t* repetitionLevels,
    const uint8_t* definitionLevels,
    int64_t numValues,
    uint8_t maxRepeat,
    uint8_t maxDefinition,
    BufferPtr& offsetsBuffer,
    BufferPtr& lengthsBuffer,
    BufferPtr& nullsBuffer,
    int64_t& numNonNulls,
    memory::MemoryPool& pool) {
  dwio::common::ensureCapacity<uint8_t>(
      nullsBuffer, bits::nbytes(numValues), &pool);
  dwio::common::ensureCapacity<vector_size_t>(
      offsetsBuffer, numValues + 1, &pool);
  dwio::common::ensureCapacity<vector_size_t>(lengthsBuffer, numValues, &pool);

  auto offsets = offsetsBuffer->asMutable<vector_size_t>();
  auto lengths = lengthsBuffer->asMutable<vector_size_t>();
  auto nulls = nullsBuffer->asMutable<uint64_t>();

  numNonNulls = 0;
  int64_t offset = 0;
  int64_t lastOffset = 0;
  bool wasLastCollectionNull = definitionLevels[0] == (maxDefinition - 1);
  bits::setNull(nulls, 0, wasLastCollectionNull);
  numNonNulls += !wasLastCollectionNull;
  offsets[0] = 0;
  int64_t outputIndex = 1;
  for (int64_t i = 1; i < numValues; ++i) {
    uint8_t definitionLevel = definitionLevels[i];
    uint8_t repetitionLevel = repetitionLevels[i];

    // empty means it belongs to a row that is null in one of its ancestor
    // levels.
    bool isEmpty = definitionLevel < (maxDefinition - 1);
    bool isNull = definitionLevel == (maxDefinition - 1);
    bool isCollectionBegin = (repetitionLevel < maxRepeat) & !isEmpty;
    bool isEntryBegin = (repetitionLevel <= maxRepeat) & !isEmpty;

    offset += isEntryBegin & !wasLastCollectionNull;
    offsets[outputIndex] = offset;
    lengths[outputIndex - 1] = offset - offsets[outputIndex - 1];
    bits::setNull(nulls, outputIndex, isNull);
    numNonNulls += !isNull & isCollectionBegin;

    // Always update the outputs, but only increase the outputIndex when the
    // current entry is the begin of a new collection, and it's not empty.
    // Benchmark shows skipping non-collection-begin rows is worse than this
    // solution by nearly 2x because of extra branchings added for skipping.
    outputIndex += isCollectionBegin;
    wasLastCollectionNull = isEmpty ? wasLastCollectionNull : isNull;
    lastOffset = isCollectionBegin ? offset : lastOffset;
  }

  offset += !wasLastCollectionNull;
  offsets[outputIndex] = offset;
  lengths[outputIndex - 1] = offset - lastOffset;

  offsetsBuffer->setSize((outputIndex + 1) * 4);
  lengthsBuffer->setSize(outputIndex * 4);
  nullsBuffer->setSize(bits::nbytes(outputIndex));

  return outputIndex;
}

std::vector<std::shared_ptr<NestedData>>
NestedStructureDecoder::readOffsetsAndNullsForAllLevels(
    const uint8_t* repetitionLevels,
    const uint8_t* definitionLevels,
    int64_t numValues,
    std::vector<uint32_t> maxRepeats_,
    std::vector<uint32_t> maxDefines_,
    memory::MemoryPool& pool) {
  std::vector<std::shared_ptr<NestedData>> nestedStructures;

  for (int i = 0; i < maxRepeats_.size(); i++) {
    std::shared_ptr<NestedData> nestedData = std::make_shared<NestedData>();
    NestedStructureDecoder::readOffsetsAndNulls(
        repetitionLevels,
        definitionLevels,
        numValues,
        maxRepeats_[i],
        maxDefines_[i],
        nestedData->offsets,
        nestedData->lengths,
        nestedData->nulls,
        nestedData->numNonNulls,
        pool);
    nestedStructures.push_back(nestedData);
  }

  return nestedStructures;
}

} // namespace facebook::velox::parquet
