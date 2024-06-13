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

#include <folly/container/F14Map.h>
#include "velox/common/memory/HashStringAllocator.h"

namespace facebook::velox {

// Provides some helper functionalities for compacting an allocation.
class AllocationCompactor {
 public:
  using AllocationRange = folly::Range<char*>;
  using Header = HashStringAllocator::Header;
  using HeaderMap = folly::F14FastMap<Header*, Header*>;

  static constexpr auto kHugePageSize = memory::AllocationTraits::kHugePageSize;
  static constexpr auto kMinBlockSize =
      sizeof(Header) + HashStringAllocator::kMinAlloc;

  /// Reservation used for estimating if an arena can accommodate certain size
  /// of data. There's certain case that the free space of destination free
  /// block cannot accommodate src block: That is when the src block is a
  /// continued block and:
  /// 1. destSize-kMinBlockSize < srcSize < destSize, and
  /// min(destSize-kMinBlockSize-kContinuedPtrSize, srcSize-kMinAlloc) <
  /// kMinAlloc-kContinuedPtrSize.
  /// 2. destSize < srcSize < destSize+kMinAlloc-kContinuedPtrSize, and
  /// destSize < kMinBlockSize+kMinAlloc.
  /// In the above cases, src block cannot fit into dest block directly, and
  /// when splitting it, the result cannot forms the valid blocks. So we can
  /// only skip the dest block. The largest size of the dest block satisfies the
  /// above cases are of the size of 'kReservationPerArena'.
  static constexpr int64_t kReservationPerArena =
      sizeof(Header) + 2 * HashStringAllocator::kMinAlloc + kMinBlockSize;

 public:
  explicit AllocationCompactor(AllocationRange allocationRange);

  int64_t size() const {
    return range_.size();
  }

  /// Lower bound of size that can be used for accommodating data within this
  /// allocation and data from reclaimable allocation.
  int64_t usableSize() const {
    return size() - simd::kPadding - kReservationPerArena;
  }

  int64_t nonFreeBlockSize() const {
    return nonFreeBlockSize_;
  }

  void setReclaimable() {
    reclaimable_ = true;
  }

  bool isReclaimable() const {
    return reclaimable_;
  }

  // 'multipartMap' is the mapping from the header of succeeding block to the
  // header of preceeding block of all the multipart blocks. This method adds
  // the multipart mapping of this allocation to 'multipartMap'.
  void accumulateMultipartMap(HeaderMap& multipartMap) const;

  // For each arena within this allocation, shift the non-free blocks towards
  // the beginning of the arena, make the arena a sequence of non-free blocks
  // followed by one free block(If there is free space left). Update
  // 'multipartMap' and 'movedBlocks' during the squeezing. Returns a vector
  // of header of free blocks, for an allocation with N arenas, at most N
  // headers are returned.
  std::vector<Header*> squeeze(HeaderMap& multipartMap, HeaderMap& movedBlocks);

  // Get next free/non-free block starting from 'header'. If 'header' is
  // nullptr, returns the first free/non-free block in the allocation. Returns
  // nullptr if there is no such block.
  Header* nextBlock(bool isFree, Header* header = nullptr) const;

  struct MoveResult {
    int64_t srcMovedSize;
    Header** prevContPtr;
    // nullptr if no remaining or discarded.
    Header* remainingDestBlock;
    int64_t destDiscardedSize;
  };

  // Move non-free block 'srcBlock' to free block 'destBlock'. If 'destBlock'
  // cannot accommodate the whole 'srcBlock', it might split the 'srcBlock' to
  // make part of it fit into 'destBlock', or fail the moving. If 'srcOffset'
  // is not 0, moves part of 'srcBlock' [srcOffset, srcBlockSize). In this
  // case, the previous part of 'srcBlock' should have continued pointer that
  // points to this part, which is 'prevContPtr'. 'multipartMap' and
  // 'movedBlocks' are updated during moving.
  //
  // Returns:
  // 1. 'srcMovedSize': Equals to the size of 'srcBlock' is the whole block is
  // moved. If 'destBlock' cannot accommodate the whole 'srcBlock', 'srcBlock'
  // might be splitted and 'srcMovedSize' is the size of splitted block that
  // fits into 'destBlock'. Equals to 0 if this dest block cannot fit any of
  // src block.
  // 2. 'prevContPtr': If 'srcBlock' is splitted, 'prevContPtr' is the first
  // splitted block's continued ptr. The caller is responsible for filling
  // 'prevContPtr' when the second splitted block gets moved. If no splitting
  // happened, 'prevContPtr' is nullptr.
  // 3. 'remainingDestBlock': In case that 'destBlock' still has free space
  // after accommodating 'srcBlock', a new free block is created and this is
  // 'remainingDestBlock'.
  // 4. 'destDiscardedSize': If 'destBlock' is slightly larger or smaller than
  // 'srcBlock' and 'srcBlock' needs to be splitted so the remaining
  // 'destBlock' can be valid(not less than kMinAlloc), the 'srcBlock' is
  // splitted, and 'destBlock' is also splitted, with the later one being of
  // size kMinAlloc, this block is discarded.
  static MoveResult moveBlock(
      Header* srcBlock,
      int64_t srcOffset,
      Header** prevContPtr,
      Header* destBlock,
      HeaderMap& multipartMap,
      HeaderMap& movedBlocks);

  // Update 'multipartMap' and 'movedBlocks' when moving block 'from' to block
  // 'to'.
  static void updateMap(
      Header* from,
      Header* to,
      HeaderMap& multipartMap,
      HeaderMap& movedBlocks);

  // Updates maps as potential "next" block in multipart. If 'from' has the
  // previous block links to it, updates previous block's next continued
  // pointer to 'to'. otherwise, inserts {'from', 'to'} to 'movedBlocks'.
  static void updateMapAsNext(
      Header* from,
      Header* to,
      HeaderMap& multipartMap,
      HeaderMap& movedBlocks);

  static void
  updateMapAsPrevious(Header* from, Header* to, HeaderMap& multipartMap);

  static void foreachBlock(
      folly::Range<char*> range,
      const std::function<void(Header*)>& func);

  void foreachBlock(const std::function<void(Header*)>& func) const {
    foreachBlock(range_, func);
  }

 private:
  Header* squeezeArena(
      AllocationRange arena,
      HeaderMap& multipartMap,
      HeaderMap& movedBlocks);

  AllocationRange range_;
  bool reclaimable_{false};
  // Sum of Non-free block size in this allocation(header included).
  int64_t nonFreeBlockSize_{0};
};

} // namespace facebook::velox
