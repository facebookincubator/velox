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
#include "velox/common/memory/HashStringAllocatorCompactor.h"

namespace facebook::velox::detail {

struct AllocationState {
  int64_t size;
  int64_t nonFreeBlockSize;
  int32_t allocationIndex;

  // Priority is given to allocations with larger sizes. When allocations have
  // identical sizes, those with less stored data take precedence.
  friend bool operator<(
      const AllocationState& lhs,
      const AllocationState& rhs) {
    if (lhs.size == rhs.size) {
      return lhs.nonFreeBlockSize < rhs.nonFreeBlockSize;
    }
    return lhs.size > rhs.size;
  }
};

} // namespace facebook::velox::detail

namespace facebook::velox {

using Header = HashStringAllocator::Header;

int64_t HashStringAllocatorCompactor::estimateReclaimableSize() {
  if (hsa_->requestedContiguous_ && !hsa_->allowSplittingContiguous_) {
    return 0;
  }

  // The last allocation is excluded from the compaction since it's currently in
  // use for serving allocations.
  const auto allocations{hsa_->pool_.numRanges() - 1};
  if (allocations == 0) {
    return 0;
  }
  compactors_.clear();
  compactors_.reserve(allocations);

  std::vector<detail::AllocationState> entries;
  entries.reserve(allocations);

  int64_t remainingUsableSize{0};
  int64_t totalNonFreeBlockSize{0};
  for (auto i = 0; i < allocations; ++i) {
    compactors_.emplace_back(hsa_->pool_.rangeAt(i));
    const auto& compactor{compactors_.back()};
    remainingUsableSize += compactor.usableSize();
    totalNonFreeBlockSize += compactor.nonFreeBlockSize();
    entries.push_back(detail::AllocationState{
        compactor.size(), compactor.nonFreeBlockSize(), i});
  }

  std::sort(entries.begin(), entries.end());

  int64_t reclaimableSize{0};
  for (const auto& entry : entries) {
    auto& compactor{compactors_[entry.allocationIndex]};
    if (remainingUsableSize - compactor.usableSize() >= totalNonFreeBlockSize) {
      remainingUsableSize -= compactor.usableSize();
      compactor.setReclaimable();
      reclaimableSize += compactor.size();
    }
  }
  return reclaimableSize;
}

// TODO: Update cumulativeBytes_.
std::pair<int64_t, AllocationCompactor::HeaderMap>
HashStringAllocatorCompactor::compact() {
  // Redo the estimation to ensure 'compactors_' are up-to-date.
  if (estimateReclaimableSize() == 0) {
    return {0, {}};
  }

  for (const auto& compactor : compactors_) {
    compactor.accumulateMultipartMap(multipartMap_);
  }

  // Remove all the free blocks in candidate allocations from free list since:
  // 1. Reclaimable allocations will be freed to AllocationPool;
  // 2. Unreclaimable allocations's free blocks will be squeezed, at the
  // end of the compaction the remaining free blocks will be added back to
  // free list.
  clearFreeList();
  // Check free list has only block in the last allocation, which was excluded
  // from compaction.
  checkFreeListInCurrentRange();

  // Compact unreclaimable allocations and collect free blocks.
  std::queue<Header*> destBlocks;
  for (auto& compactor : compactors_) {
    // Squeeze unreclaimable allocations, whose remaining free blocks will be
    // the destination blocks for the compaction.
    if (compactor.isReclaimable()) {
      continue;
    }
    auto freeBlocks = compactor.squeeze(multipartMap_, movedBlocks_);
    for (auto* freeBlock : freeBlocks) {
      destBlocks.push(freeBlock);
    }
  }

  const auto compactedSize =
      moveBlocksInReclaimableAllocations(std::move(destBlocks));

  // Add free blocks in the unreclaimable allocations to free list after moving
  // the blocks.
  addFreeBlocksToFreeList();

  // Free empty allocations.
  for (int32_t i = compactors_.size() - 1; i >= 0; --i) {
    if (compactors_[i].isReclaimable()) {
      LOG(INFO) << "Freeing allocation " << i << " with size of "
                << compactors_[i].size();
      hsa_->pool_.freeRangeAt(i);
    }
  }
  compactors_.clear();
  multipartMap_.clear();

  hsa_->checkConsistency();
  return {compactedSize, std::move(movedBlocks_)};
};

// Remove all the free blocks in candidate allocations from free list.
void HashStringAllocatorCompactor::clearFreeList() {
  for (auto& compactor : compactors_) {
    compactor.foreachBlock([&](Header* header) {
      if (header->isFree()) {
        hsa_->removeFromFreeList(header);
        header->setFree();
        --hsa_->numFree_;
        hsa_->freeBytes_ -= sizeof(Header) + header->size();
      }
    });
  }
}

size_t HashStringAllocatorCompactor::moveBlocksInReclaimableAllocations(
    std::queue<Header*> destBlocks) {
  size_t compactedSize{0};
  Header* destBlock{nullptr};
  for (auto& compactor : compactors_) {
    if (!compactor.isReclaimable()) {
      continue;
    }
    auto srcBlock = compactor.nextBlock(false /* isFree */);
    int64_t srcOffset{0};
    Header** prevContPtr{nullptr};
    while (srcBlock != nullptr) {
      if (destBlock == nullptr) {
        VELOX_CHECK(!destBlocks.empty());
        destBlock = destBlocks.front();
        destBlocks.pop();
      }

      VELOX_CHECK_EQ((srcOffset == 0), (prevContPtr == nullptr));
      auto moveResult = AllocationCompactor::moveBlock(
          srcBlock,
          srcOffset,
          prevContPtr,
          destBlock,
          multipartMap_,
          movedBlocks_);
      srcOffset += moveResult.srcMovedSize;
      if (moveResult.prevContPtr != nullptr) {
        VELOX_CHECK_GT(moveResult.srcMovedSize, 0);
        VELOX_CHECK_LT(srcOffset, srcBlock->size());
        prevContPtr = moveResult.prevContPtr;
      }
      if (srcOffset == srcBlock->size()) {
        srcBlock = compactor.nextBlock(false, srcBlock);
        srcOffset = 0;
        prevContPtr = nullptr;
      }
      destBlock = moveResult.remainingDestBlock;
    }
    compactedSize += compactor.size();
  }
  return compactedSize;
}

void HashStringAllocatorCompactor::checkFreeListInCurrentRange() const {
  for (auto i = 0; i < HashStringAllocator::kNumFreeLists; ++i) {
    auto* item = hsa_->free_[i].next();
    while (item != &hsa_->free_[i]) {
      auto header = HashStringAllocator::headerOf(item);
      VELOX_CHECK(hsa_->pool_.isInCurrentRange(header));
      item = item->next();
    }
  }
}

void HashStringAllocatorCompactor::addFreeBlocksToFreeList() {
  for (auto& compactor : compactors_) {
    if (compactor.isReclaimable()) {
      continue;
    }
    compactor.foreachBlock([&](Header* header) {
      if (!header->isFree()) {
        return;
      }
      header->clearFree();
      hsa_->free(header);
    });
  }
}

} // namespace facebook::velox