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
#include "PrefixSort.h"

namespace facebook::velox::exec {
void PrefixSort::extractRowToPrefix(
    char* FOLLY_NONNULL row,
    char* FOLLY_NONNULL prefix) {
  // extract key
  for (int32_t index = 0; index < sortLayout_.numPrefixKeys_; index++) {
    VELOX_DYNAMIC_TYPE_DISPATCH(
        rowToPrefix,
        rowContainer_->keyTypes()[index]->kind(),
        index,
        rowContainer_->columnAt(index),
        row,
        prefix);
  }
  // Set address of row.
  *reinterpret_cast<char**>(prefix + sortLayout_.keySize) = row;
}

void PrefixSort::preparePrefix() {
  // Compute prefix offsets for sort columns.
  uint32_t offset = 0;
  for (int i = 0; i < sortLayout_.numPrefixKeys_; i++) {
    prefixOffsets.push_back(offset);
    offset += prefixKeySize(rowContainer_->keyTypes()[i]->kind());
  }
  int32_t numRows = numInputRows_;
  // Allocate prefixes_ data.
  constexpr auto kPageSize = memory::AllocationTraits::kPageSize;
  auto numPages =
      bits::roundUp(numRows * sortLayout_.entrySize, kPageSize) / kPageSize;
  rowContainer_->pool()->allocateContiguous(numPages, prefixAllocation);
  prefixes_ = prefixAllocation.data<char>();

  RowContainerIterator tmp;
  RowContainerIterator* iter = &tmp;
  int32_t count = 0;
  auto numAllocations = rowContainer_->rows_.numRanges();
  if (iter->allocationIndex == 0 && iter->rowOffset == 0) {
    iter->normalizedKeysLeft = rowContainer_->numRowsWithNormalizedKey_;
    iter->normalizedKeySize = rowContainer_->originalNormalizedKeySize_;
  }
  int32_t rowSize = rowContainer_->fixedRowSize_ +
      (iter->normalizedKeysLeft > 0 ? rowContainer_->originalNormalizedKeySize_
                                    : 0);
  char* prefix = prefixes_;
  char* address = nullptr;
  for (auto i = iter->allocationIndex; i < numAllocations; ++i) {
    auto range = rowContainer_->rows_.rangeAt(i);
    auto* data = range.data() +
        memory::alignmentPadding(range.data(), rowContainer_->alignment_);
    auto limit = range.size() -
        (reinterpret_cast<uintptr_t>(data) -
         reinterpret_cast<uintptr_t>(range.data()));
    auto row = iter->rowOffset;
    while (row + rowSize <= limit) {
      address = data + row +
          (iter->normalizedKeysLeft > 0
               ? rowContainer_->originalNormalizedKeySize_
               : 0);
      VELOX_DCHECK_EQ(
          reinterpret_cast<uintptr_t>(address) % rowContainer_->alignment_, 0);
      row += rowSize;
      if (--iter->normalizedKeysLeft == 0) {
        rowSize -= rowContainer_->originalNormalizedKeySize_;
      }
      if (bits::isBitSet(address, rowContainer_->freeFlagOffset_)) {
        continue;
      }
      prefix = prefixes_ + sortLayout_.entrySize * count;
      extractRowToPrefix(address, prefix);
      *(reinterpret_cast<char**>(prefix + sortLayout_.keySize)) = address;
      count++;
    }
    iter->rowOffset = 0;
  }
}

int PrefixSort::compare(
    const PrefixSortIterator& left,
    const PrefixSortIterator& right) {
  if (!sortLayout_.needSortData) {
    return FastMemcmp(*left, *right, (size_t)sortLayout_.keySize);
  } else {
    int result = FastMemcmp(*left, *right, (size_t)sortLayout_.keySize);
    if (result != 0) {
      return result;
    }
    char* leftAddress = getAddressFromPrefix(left);
    char* rightAddress = getAddressFromPrefix(right);
    for (int i = sortLayout_.numPrefixKeys_; i < sortLayout_.numSortKeys_;
         i++) {
      int result = rowContainer_->compare(
          leftAddress, rightAddress, i, sortLayout_.keyCompareFlags_[i]);
      if (result != 0) {
        return result;
      }
    }
  }
  return 0;
}

void PrefixSort::sort(std::vector<char*>& rows) {
  auto start = PrefixSortIterator(prefixes_, sortLayout_.entrySize);
  auto end = start + numInputRows_;
  auto prefixSortContext = PrefixSortContext(sortLayout_.entrySize, *end);
  PrefixQuickSort(
      prefixSortContext,
      start,
      end,
      [&](const PrefixSortIterator& a, const PrefixSortIterator& b) {
        return compare(a, b);
      });
  // copy address from prefix tail to returnRows
  for (int i = 0; i < end - start; i++) {
    rows[i] = getAddressFromPrefix(start + i);
  }
}

} // namespace facebook::velox::exec