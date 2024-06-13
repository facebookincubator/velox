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
#pragma once

#include "velox/common/memory/AllocationCompactor.h"

namespace facebook::velox {

class HashStringAllocatorCompactor {
 public:
  explicit HashStringAllocatorCompactor(HashStringAllocator* hsa)
      : hsa_(hsa), pool_(&hsa->allocationPool()) {
    VELOX_CHECK_NOT_NULL(hsa_);
  }

  /// Estimates reclaimale memory size by identifying reclaimable allocations.
  /// Builds 'compactors_' for each allocation except the last one. Returns 0 if
  /// no allocation is reclaimable or if contiguous memory has been explicitly
  /// requested to 'hsa_' and its 'allowSplittingContiguous_' is false.
  int64_t estimateReclaimableSize();

  /// Compacts owned memory by squeezing unreclaimable allocations and
  /// relocating data from reclaimable allocations to unreclaimable allocations.
  /// Reclaimable allocations are then freed to AllocationPool. Returns the
  /// bytes freed and the mapping from original pointer of the moved blocks'
  /// header to their new pointer.
  std::pair<int64_t, AllocationCompactor::HeaderMap> compact();

 private:
  void clearFreeList();

  size_t moveBlocksInReclaimableAllocations(
      std::queue<HashStringAllocator::Header*> destBlocks);

  void checkFreeListInCurrentRange() const;

  void addFreeBlocksToFreeList();

 private:
  HashStringAllocator* hsa_;
  const memory::AllocationPool* pool_;
  std::vector<AllocationCompactor> compactors_;
  AllocationCompactor::HeaderMap movedBlocks_;
  // Maps from the header of succeeding block to the header of preceeding block
  // of all the multipart blocks.
  AllocationCompactor::HeaderMap multipartMap_;
};

} // namespace facebook::velox
