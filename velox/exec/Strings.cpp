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

#include "velox/exec/Strings.h"

namespace facebook::velox::aggregate::prestosql {

StringView Strings::append(StringView value, HashStringAllocator& allocator) {
  // A string in Strings needs 8 bytes below itself for the last 8
  // bytes of the previous string, which copied to the next string
  // when setting a continued pointer. It also needs 8 bytes at its
  // tail for the same continue pointer to the next.
  constexpr int32_t kOverhead =
      HashStringAllocator::Header::kContinuedPtrSize + 8;
  VELOX_DCHECK(!value.isInline());

  maxStringSize = std::max<int32_t>(maxStringSize, value.size());

  // Request sufficient amount of memory to store the whole string
  // (value.size()) and allow some memory left for bookkeeping (8 last from
  // previous block + link to next block).
  const int32_t requiredBytes = value.size() + kOverhead;
  int32_t tail = 0;
  ByteStream stream(&allocator);
  if (firstBlock == nullptr) {
    // Allocate first block.
    currentBlock = allocator.newWrite(stream, requiredBytes);
    firstBlock = currentBlock.header;
    auto minSize = HashStringAllocator::freeListSizes()[0];
    tail = requiredBytes < minSize ? minSize - requiredBytes : 0;
  } else {
    allocator.extendWrite(currentBlock, stream);
    tail = stream.ranges().back().size;
  }

  // Check if there is enough space left.
  int32_t available = stream.ranges().back().size;
  if (available <
      value.size() + HashStringAllocator::Header::kContinuedPtrSize) {
    // Not enough space. Allocate new block.
    ByteRange newRange;
    auto sizes = HashStringAllocator::freeListSizes();
    int32_t targetBytes = kOverhead +
        (maxStringSize < 200      ? maxStringSize * 4
             : value.size() < 200 ? value.size() * 4
                                  : value.size());
    int32_t roundedUpBytes = targetBytes < sizes[0] ? sizes[0]
        : targetBytes < sizes[1]                    ? sizes[1]
                                                    : targetBytes;
    tail = roundedUpBytes;
    allocator.newContiguousRange(roundedUpBytes, &newRange);

    stream.setRange(newRange);
  }

  VELOX_DCHECK_LE(
      value.size() + HashStringAllocator::Header::kContinuedPtrSize,
      stream.ranges().back().size);

  // Copy the string and return a StringView over the copy.
  char* start = stream.writePosition();
  stream.appendStringPiece(folly::StringPiece(value.data(), value.size()));
  // There will always be at least enough space for the continue
  // pointer so the tail of the string does not get overwritten.
  currentBlock =
      allocator
          .finishWrite(
              stream, HashStringAllocator::Header::kContinuedPtrSize + tail)
          .second;
  return StringView(start, value.size());
}

void Strings::free(HashStringAllocator& allocator) {
  if (firstBlock != nullptr) {
    allocator.free(firstBlock);
    firstBlock = nullptr;
    currentBlock = {nullptr, nullptr};
  }
}
} // namespace facebook::velox::aggregate::prestosql
