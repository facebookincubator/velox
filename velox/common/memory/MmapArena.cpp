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

#include "velox/common/memory/MmapArena.h"

#include <sys/mman.h>
#include "velox/common/base/BitUtil.h"
#include "velox/common/memory/Memory.h"

namespace facebook::velox::memory {
uint64_t MmapArena::roundBytes(uint64_t bytes) {
  return bits::nextPowerOfTwo(bytes);
}

MmapArena::MmapArena(size_t capacityBytes) : byteSize_(capacityBytes) {
  VELOX_CHECK_EQ(
      byteSize_ % kMinGrainSizeBytes,
      0,
      "Arena must have a multiple of {} bytes capacity.",
      kMinGrainSizeBytes);
  void* ptr = mmap(
      nullptr,
      capacityBytes,
      PROT_READ | PROT_WRITE,
      MAP_PRIVATE | MAP_ANONYMOUS,
      -1,
      0);
  if (ptr == MAP_FAILED || ptr == nullptr) {
    VELOX_FAIL(
        "Could not allocate working memory"
        "mmap failed with errno {} with capacity bytes {}",
        folly::errnoStr(errno),
        capacityBytes);
  }
  address_ = reinterpret_cast<uint8_t*>(ptr);
  addFreeBlock(reinterpret_cast<uint64_t>(address_), byteSize_);
  freeBytes_ = byteSize_;
}

MmapArena::~MmapArena() {
  munmap(address_, byteSize_);
}

void* MmapArena::allocate(uint64_t bytes) {
  if (bytes == 0) {
    return nullptr;
  }
  bytes = roundBytes(bytes);

  // First match in the list that can give this many bytes
  auto lookupItr = freeLookup_.lower_bound(bytes);
  if (lookupItr == freeLookup_.end()) {
    VELOX_MEM_LOG_EVERY_MS(WARNING, 1000)
        << "Cannot find a free block that is large enough to allocate " << bytes
        << " bytes. Current arena freeBytes " << freeBytes_ << " & lookup table"
        << freeLookupStr();
    return nullptr;
  }

  freeBytes_ -= bytes;
  auto address = *(lookupItr->second.begin());
  auto curFreeBytes = lookupItr->first;
  void* result = reinterpret_cast<void*>(address);
  if (curFreeBytes == bytes) {
    removeFreeBlock(address, curFreeBytes);
    return result;
  }
  addFreeBlock(address + bytes, curFreeBytes - bytes);
  removeFreeBlock(address, curFreeBytes);
  return result;
}

void MmapArena::free(void* FOLLY_NONNULL address, uint64_t bytes) {
  if (address == nullptr || bytes == 0) {
    return;
  }
  bytes = roundBytes(bytes);

  ::madvise(address, bytes, MADV_DONTNEED);
  freeBytes_ += bytes;

  const auto curAddr = reinterpret_cast<uint64_t>(address);
  auto curIter = addFreeBlock(curAddr, bytes);
  auto prevIter = freeList_.end();
  uint64_t prevAddr;
  uint64_t prevBytes;
  bool mergePrev = false;
  if (curIter != freeList_.begin()) {
    prevIter = std::prev(curIter);
    prevAddr = prevIter->first;
    prevBytes = prevIter->second;
    auto prevEndAddr = prevAddr + prevBytes;
    VELOX_CHECK_LE(
        prevEndAddr,
        curAddr,
        "New free node (addr:{} size:{}) overlaps with previous free node (addr:{} size:{}) in free list",
        curAddr,
        bytes,
        prevAddr,
        prevBytes);
    mergePrev = prevEndAddr == curAddr;
  }

  auto nextItr = std::next(curIter);
  uint64_t nextAddr;
  uint64_t nextBytes;
  bool mergeNext = false;
  if (nextItr != freeList_.end()) {
    nextAddr = nextItr->first;
    nextBytes = nextItr->second;
    auto curEndAddr = curAddr + bytes;
    VELOX_CHECK_LE(
        curEndAddr,
        nextAddr,
        "New free node (addr:{} size:{}) overlaps with next free node (addr:{} size:{}) in free list",
        curAddr,
        bytes,
        nextAddr,
        nextBytes);
    mergeNext = (curEndAddr == nextAddr);
  }

  if (!mergePrev && !mergeNext) {
    return;
  }

  if (mergePrev) {
    removeFreeBlock(curIter);
    removeFromLookup(prevAddr, prevBytes);
    auto newFreeSize = curAddr - prevAddr + bytes;
    if (mergeNext) {
      removeFreeBlock(nextItr);
      newFreeSize = nextAddr - prevAddr + nextBytes;
    }
    freeList_[prevIter->first] = newFreeSize;
    freeLookup_[newFreeSize].emplace(prevAddr);
    return;
  }

  if (mergeNext) {
    VELOX_DCHECK(!mergePrev);
    removeFreeBlock(nextItr);
    removeFromLookup(curAddr, bytes);
    const auto newFreeSize = nextAddr - curAddr + nextBytes;
    freeList_[curIter->first] = newFreeSize;
    freeLookup_[newFreeSize].emplace(curAddr);
  }
}

void MmapArena::removeFromLookup(uint64_t addr, uint64_t bytes) {
  freeLookup_[bytes].erase(addr);
  if (freeLookup_[bytes].empty()) {
    freeLookup_.erase(bytes);
  }
}

std::map<uint64_t, uint64_t>::iterator MmapArena::addFreeBlock(
    uint64_t address,
    uint64_t bytes) {
  auto insertResult = freeList_.emplace(address, bytes);
  VELOX_CHECK(
      insertResult.second,
      "Trying to free a memory space that is already freed. Already in free list address {} size {}. Attempted to free address {} size {}",
      address,
      freeList_[address],
      address,
      bytes);
  freeLookup_[bytes].emplace(address);
  return insertResult.first;
}

void MmapArena::removeFreeBlock(uint64_t addr, uint64_t bytes) {
  freeList_.erase(addr);
  removeFromLookup(addr, bytes);
}

void MmapArena::removeFreeBlock(std::map<uint64_t, uint64_t>::iterator& iter) {
  removeFromLookup(iter->first, iter->second);
  freeList_.erase(iter);
}

bool MmapArena::checkConsistency() const {
  uint64_t numErrors = 0;
  uint64_t bytes = 0;
  auto arenaEndAddress = reinterpret_cast<uint64_t>(address_) + byteSize_;
  auto iter = freeList_.begin();
  auto end = freeList_.end();
  uint8_t* current = reinterpret_cast<uint8_t*>(address_);
  int64_t freeListTotalBytes = 0;
  while (iter != end) {
    // Lookup list should contain the address
    auto freeLookupIter = freeLookup_.find(iter->second);
    if (freeLookupIter == freeLookup_.end() ||
        freeLookupIter->second.find(iter->first) ==
            freeLookupIter->second.end()) {
      LOG(WARNING)
          << "MmapArena::checkConsistency(): freeLookup_ out of sync: Not "
             "containing item from freeList_ {addr:"
          << iter->first << ", size:" << iter->second << "}";
      numErrors++;
    }

    // Verify current free block end
    auto blockEndAddress = iter->first + iter->second;
    if (blockEndAddress > arenaEndAddress) {
      LOG(WARNING)
          << "MmapArena::checkConsistency(): freeList_ out of sync: Block "
             "extruding arena boundary {addr:"
          << iter->first << ", size:" << iter->second << "}";
      numErrors++;
    }

    // Verify next free block not overlapping
    auto next = std::next(iter);
    if (next != end && blockEndAddress > next->first) {
      LOG(WARNING)
          << "MmapArena::checkConsistency(): freeList_ out of sync: Overlapping"
             " blocks {addr:"
          << iter->first << ", size:" << iter->second
          << "} {addr:" << next->first << ", size:" << next->second << "}";
      numErrors++;
    }

    freeListTotalBytes += iter->second;
    iter++;
  }

  // Check consistency of lookup list
  int64_t freeLookupTotalBytes = 0;
  for (auto iter = freeLookup_.begin(); iter != freeLookup_.end(); iter++) {
    if (iter->second.empty()) {
      LOG(WARNING)
          << "MmapArena::checkConsistency(): freeLookup_ out of sync: Empty "
             "address list for size "
          << iter->first;
      numErrors++;
    }
    freeLookupTotalBytes += (iter->first * iter->second.size());
  }

  // Check consistency of freeList_ and freeLookup_ in terms of bytes
  if (freeListTotalBytes != freeLookupTotalBytes ||
      freeListTotalBytes != freeBytes_) {
    LOG(WARNING)
        << "MmapArena::checkConsistency(): free bytes out of sync: freeListTotalBytes "
        << freeListTotalBytes << " freeLookupTotalBytes "
        << freeLookupTotalBytes << " freeBytes_ " << freeBytes_;
    numErrors++;
  }

  if (numErrors) {
    LOG(ERROR) << "MmapArena::checkConsistency(): " << numErrors << " errors";
  }
  return numErrors == 0;
}

ManagedMmapArenas::ManagedMmapArenas(uint64_t singleArenaCapacity)
    : singleArenaCapacity_(singleArenaCapacity) {
  auto arena = std::make_shared<MmapArena>(singleArenaCapacity);
  arenas_.emplace(reinterpret_cast<uint64_t>(arena->address()), arena);
  currentArena_ = arena;
}

void* FOLLY_NULLABLE ManagedMmapArenas::allocate(uint64_t bytes) {
  auto* result = currentArena_->allocate(bytes);
  if (result != nullptr) {
    return result;
  }

  // If first allocation fails we create a new MmapArena for another attempt. If
  // it ever fails again then it means requested bytes is larger than a single
  // MmapArena's capacity. No further attempts will happen.
  auto newArena = std::make_shared<MmapArena>(singleArenaCapacity_);
  arenas_.emplace(reinterpret_cast<uint64_t>(newArena->address()), newArena);
  currentArena_ = newArena;
  return currentArena_->allocate(bytes);
}

void ManagedMmapArenas::free(void* FOLLY_NONNULL address, uint64_t bytes) {
  const uint64_t addressU64 = reinterpret_cast<uint64_t>(address);
  auto iter = arenas_.lower_bound(addressU64);
  if (iter == arenas_.end()) {
    return;
  }
  if (iter->first == addressU64) {
    iter->second->free(address, bytes);
  } else if (iter == arenas_.begin()) {
    return;
  } else {
    --iter;
    iter->second->free(address, bytes);
  }

  if (iter->second->empty() && iter->second != currentArena_) {
    arenas_.erase(iter);
  }
}

} // namespace facebook::velox::memory
