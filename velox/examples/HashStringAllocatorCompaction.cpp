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

/// Example program to demonstrate the usage of 'HashStringAllocatorCompactor'.
/// The program firstly fills random generated blocks into memory allocated via
/// HSA, then removes some of them to make room for compaction, then uses
/// HashStringAllocatorCompactor to compact the memory. After the compaction,
/// verify if all the filled data is still valid.
/// NOTE: This program is not intended to be merged into main.
#include <algorithm>
#include <chrono>
#include <random>

#include "velox/common/memory/HashStringAllocatorCompactor.h"

using namespace facebook::velox;

using Header = HashStringAllocator::Header;

folly::Random::DefaultGenerator rng;
std::random_device rd;

/// Generates at least 'bytes' random data with size in range [16,3072] and
/// fills into block allocated from 'hsa'. Stores the header of the block and
/// random data pair into corresponding element in 'payloads', indexed by the
/// allocation id that the block belongs to. Accumulates the data size of each
/// allocation in 'allocationsDataSize'.
void fillAllocation(
    HashStringAllocator* hsa,
    int64_t bytes,
    std::vector<std::vector<std::pair<Header*, std::string>>>& payloads,
    std::vector<size_t>& allocationsDataSize);

/// Frees at least 'bytesToFree' of data in allocation 'allocation_data' from
/// 'hsa'. Removes corresponding pairs in 'allocation_data'.
int64_t freeDataAtAllocation(
    HashStringAllocator* hsa,
    size_t bytesToFree,
    std::vector<std::pair<Header*, std::string>>& allocation_data);

/// Verifies if all the payload in 'payloads' match the content stored in the
/// block that their header point to. If 'movedBlocks' doesn't contain the
/// header, than the block was not moved, otherwise, it's moved during
/// compaction and should use the corresponding value in the map as the new
/// header to do the verification.
void verifyPayload(
    const std::vector<std::vector<std::pair<Header*, std::string>>>& payloads,
    folly::F14FastMap<Header*, Header*>& movedBlocks);

std::string randomString(int32_t size = 0);

int main(int argc, char* argv[]) {
  if (argc != 2) {
    LOG(ERROR)
        << "Usage: " << argv[0] << " [data size to fill in MB].\nFor example, '"
        << argv[0]
        << " 4096' will fill 4GB of data, then free some and try compaction.";
    return 1;
  }
  const size_t payloadSize = (1LL << 20) * std::atoi(argv[1]);

  // TODO: use velox_check
  memory::MemoryManager::initialize({});
  auto rootPool = memory::memoryManager()->addRootPool(
      "", memory::kMaxMemory, memory::MemoryReclaimer::create());
  auto pool =
      rootPool->addLeafChild("", false, memory::MemoryReclaimer::create());

  auto hsa = std::make_unique<HashStringAllocator>(pool.get());
  rng.seed(1);

  // Stores all the data(Header and payload) created in HSA indexed by
  // allocation index.
  std::vector<std::vector<std::pair<Header*, std::string>>> payloads;
  // Total data size (header size + payload size) of each allocation.
  std::vector<size_t> allocationsDataSize;

  // Fills at least 'payloadSize' bytes of random generated data into memory
  // allocated via HSA.
  LOG(INFO) << "Filling not less than " << payloadSize << " bytes of data...";
  fillAllocation(hsa.get(), payloadSize, payloads, allocationsDataSize);
  if (allocationsDataSize.size() <= 1) {
    LOG(ERROR)
        << "HashStringAllocator only allocated 1 allocation, cannot be compacted.";
    return 0;
  }
  LOG(INFO) << "Allocated " << payloads.size() << " allocations.";

  LOG(INFO) << "Randomly free data...";
  size_t totalFreed{0};
  for (size_t i = 0; i < allocationsDataSize.size() - 1; ++i) {
    const auto freeRatio = folly::Random::randDouble01(rng);
    const auto freed = freeDataAtAllocation(
        hsa.get(), allocationsDataSize[i] * freeRatio, payloads[i]);
    LOG(INFO) << "Allocation " << i << ": Freed " << freed << " out of "
              << allocationsDataSize[i] << " bytes (" << freeRatio * 100
              << "%).";
    totalFreed += freed;
  }
  LOG(INFO) << "Freed " << totalFreed << " bytes in total.";

  LOG(INFO) << "Memory usage before compaction:\n"
            << pool->treeMemoryUsage(true);

  HashStringAllocatorCompactor compactor(hsa.get());
  hsa->allowSplittingContiguous();
  const auto estimatedReclaimable = compactor.estimateReclaimableSize();
  LOG(INFO) << "Estimated reclaimable data size:" << estimatedReclaimable;

  LOG(INFO) << "Starting compaction...";
  const auto startTime = std::chrono::steady_clock::now();
  auto [bytesFreed, updatedBlocks] = compactor.compact();
  VELOX_CHECK_EQ(bytesFreed, estimatedReclaimable);
  const auto elapsed = std::chrono::steady_clock::now() - startTime;
  // TODO: Make elapsed millisecond.
  LOG(INFO) << "Compaction done, elapsed time: "
            << std::chrono::duration_cast<
                   std::chrono::duration<double, std::ratio<1, 1000>>>(elapsed)
                   .count()
            << " milliseconds.";

  LOG(INFO) << "Starting to verify payload...";
  verifyPayload(payloads, updatedBlocks);
  VELOX_CHECK_EQ(compactor.estimateReclaimableSize(), 0);
  LOG(INFO) << "Verification done.";

  LOG(INFO) << "Memory usage after compaction:\n"
            << pool->treeMemoryUsage(true);

  return 0;
}

void fillAllocation(
    HashStringAllocator* hsa,
    int64_t bytes,
    std::vector<std::vector<std::pair<Header*, std::string>>>& payloads,
    std::vector<size_t>& allocationsDataSize) {
  std::vector<Header*> headers;

  int64_t bytesAllocated{0};
  while (bytesAllocated < bytes) {
    const auto size = 16 +
        (folly::Random::rand32(rng) % (HashStringAllocator::kMaxAlloc - 16));
    auto header = hsa->allocate(size);
    VELOX_CHECK(!header->isFree());
    VELOX_CHECK(!header->isContinued());
    bytesAllocated += header->size();
    headers.push_back(header);
  }

  auto& pool = hsa->allocationPool();
  const auto allocations = pool.numRanges();
  std::map<char*, int32_t> rangeStartToAllocId;
  for (int32_t i = 0; i < allocations; ++i) {
    rangeStartToAllocId[pool.rangeAt(i).data()] = i;
  }

  payloads.clear();
  payloads.resize(allocations);
  allocationsDataSize.resize(allocations, 0);
  for (const auto header : headers) {
    auto it = rangeStartToAllocId.upper_bound(reinterpret_cast<char*>(header));
    --it;
    const auto allocationId{it->second};

    auto payload = randomString(header->size());
    memcpy(header->begin(), payload.data(), header->size());
    payloads[allocationId].emplace_back(header, std::move(payload));

    allocationsDataSize[allocationId] += header->size() + sizeof(Header);
  }
}

int64_t freeDataAtAllocation(
    HashStringAllocator* hsa,
    size_t bytesToFree,
    std::vector<std::pair<Header*, std::string>>& allocation_data) {
  std::shuffle(
      allocation_data.begin(), allocation_data.end(), std::mt19937(rd()));

  int64_t freedBytes{0};
  while (freedBytes < bytesToFree) {
    if (allocation_data.empty()) {
      break;
    }
    auto& [header, payload] = allocation_data.back();
    freedBytes += header->size() + sizeof(Header);
    hsa->free(header);
    allocation_data.pop_back();
  }
  return freedBytes;
}

// Verify all the content that's not been freed are still valid. Takes an map
// of moved blocks' header, which is the product of HSA::compact.
void verifyPayload(
    const std::vector<std::vector<std::pair<Header*, std::string>>>& payloads,
    folly::F14FastMap<Header*, Header*>& movedBlocks) {
  for (const auto& allocationData : payloads) {
    for (const auto& [header, str] : allocationData) {
      if (str.empty()) {
        continue;
      }
      auto currentHeader = header;
      int32_t offset{0};
      if (movedBlocks.contains(currentHeader)) {
        currentHeader = movedBlocks[header];
      }
      while (true) {
        VELOX_CHECK(!currentHeader->isFree());
        const auto sizeToCompare =
            std::min<int32_t>(currentHeader->usableSize(), str.size() - offset);
        VELOX_CHECK_GT(sizeToCompare, 0);
        VELOX_CHECK(!memcmp(
            currentHeader->begin(), str.data() + offset, sizeToCompare));
        offset += sizeToCompare;
        if (offset == str.size()) {
          break;
        }
        VELOX_CHECK(currentHeader->isContinued());
        currentHeader = currentHeader->nextContinued();
      }
    }
  }
}

// Stolen from HashStringAllocatorTest.cpp
std::string randomString(int32_t size) {
  std::string result;
  result.resize(
      size != 0 ? size
                : 20 +
              (folly::Random::rand32(rng) % 10 > 8
                   ? folly::Random::rand32(rng) % 200
                   : 1000 + folly::Random::rand32(rng) % 1000));
  for (auto i = 0; i < result.size(); ++i) {
    result[i] = 32 + (folly::Random::rand32(rng) % 96);
  }
  return result;
}
