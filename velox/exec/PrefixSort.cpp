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
}

void PrefixSort::sort(std::vector<char*>& rows) {
  RowContainerIterator iter;
  rowContainer_->listRows(&iter, numInputRows_, rows.data());
  for (uint64_t i = 0; i < rows.size(); ++i) {
    extractRowToPrefix(rows[i], prefixes_ + sortLayout_.entrySize * i);
  }
  auto swapBuffer = AlignedBuffer::allocate<char>(
      sortLayout_.entrySize, rowContainer_->pool());
  PrefixSortRunner sortRunner(
      sortLayout_.entrySize, swapBuffer->asMutable<char>());
  auto start = prefixes_;
  auto end = prefixes_ + numInputRows_ * sortLayout_.entrySize;

  sortRunner.quickSort(
      start, end, [&](char* a, char* b) { return compare(a, b); });

  for (int i = 0; i < rows.size(); i++) {
    rows[i] = *reinterpret_cast<char**>(
        prefixes_ + i * sortLayout_.entrySize + sortLayout_.keySize);
  }
}

} // namespace facebook::velox::exec