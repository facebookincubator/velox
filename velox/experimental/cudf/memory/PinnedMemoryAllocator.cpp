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

#include "velox/experimental/cudf/memory/PinnedMemoryAllocator.h"

#include "velox/common/testutil/TestValue.h"

#include <cuda/memory_resource>
#include <cuda_runtime_api.h>

#include <folly/ScopeGuard.h>

#include <mutex>
#include <unordered_map>

namespace facebook::velox::cudf_velox {
namespace {
using memory::Allocation;
using memory::AllocationTraits;
using memory::ContiguousAllocation;
using memory::MachinePageCount;
constexpr uint64_t kPage = AllocationTraits::kPageSize;

cuda::pinned_memory_pool makePool(
    const PinnedMemoryAllocator::Options& options) {
  VELOX_USER_CHECK_GT(options.capacity, 0);
  VELOX_USER_CHECK_EQ(options.capacity % kPage, 0);
  VELOX_USER_CHECK_LE(options.capacity, std::numeric_limits<int64_t>::max());
  // Initialize the runtime before using CCCL's driver-backed pool.
  int device;
  auto status = cudaGetDevice(&device);
  VELOX_CHECK(status == cudaSuccess, "{}", cudaGetErrorString(status));
  cuda::memory_pool_properties properties;
  properties.max_pool_size = options.capacity;
  properties.release_threshold = std::numeric_limits<size_t>::max();
  if (options.numaNode.has_value()) {
    VELOX_USER_CHECK_GE(*options.numaNode, 0);
    return cuda::pinned_memory_pool(*options.numaNode, properties);
  }
#if CUDART_VERSION >= 13000
  return cuda::pinned_memory_pool(properties);
#else
  VELOX_USER_FAIL(
      "PinnedMemoryAllocator requires a launcher-supplied NUMA node with CUDA < 13");
#endif
}
} // namespace

struct PinnedMemoryAllocator::Impl {
  struct Block {
    void* base;
    uint64_t backingBytes;
    uint64_t requestedBytes;
  };
  explicit Impl(const Options& options)
      : capacity(options.capacity), pool(makePool(options)) {}
  const uint64_t capacity;
  cuda::pinned_memory_pool pool;
  mutable std::mutex mutex;
  std::unordered_map<void*, Block> blocks;
  uint64_t chargedBytes{0};
  uint64_t requestedBytes{0};
};

PinnedMemoryAllocator::PinnedMemoryAllocator(const Options& options)
    : impl_(std::make_unique<Impl>(options)) {}

PinnedMemoryAllocator::~PinnedMemoryAllocator() {
  cache_.reset();
  VELOX_CHECK(
      impl_->blocks.empty(), "Pinned allocations outlived their allocator");
}

void PinnedMemoryAllocator::registerCache(
    const std::shared_ptr<memory::Cache>& cache) {
  VELOX_CHECK_NULL(cache_);
  VELOX_CHECK_NOT_NULL(cache);
  VELOX_CHECK(cache->allocator() == this);
  cache_ = cache;
}

size_t PinnedMemoryAllocator::capacity() const {
  return impl_->capacity;
}
size_t PinnedMemoryAllocator::totalUsedBytes() const {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  return impl_->requestedBytes;
}
MachinePageCount PinnedMemoryAllocator::numAllocated() const {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  return impl_->chargedBytes / kPage;
}
uint64_t PinnedMemoryAllocator::retainedBytes() const {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  return cuda::memory_pool_attributes::reserved_mem_current(impl_->pool.get());
}
bool PinnedMemoryAllocator::checkConsistency() const {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  uint64_t charged = 0, requested = 0;
  for (const auto& [ptr, block] : impl_->blocks) {
    charged += block.backingBytes;
    requested += block.requestedBytes;
  }
  return charged == impl_->chargedBytes && requested == impl_->requestedBytes &&
      charged <= impl_->capacity;
}
std::string PinnedMemoryAllocator::toString() const {
  return fmt::format(
      "PinnedMemoryAllocator[capacity:{}, used:{}, chargedPages:{}, retained:{}]",
      capacity(),
      totalUsedBytes(),
      numAllocated(),
      retainedBytes());
}

void* PinnedMemoryAllocator::allocateBytesWithoutRetry(
    uint64_t bytes,
    uint16_t alignment) {
  VELOX_USER_CHECK(
      alignment >= kMinAlignment && (alignment & (alignment - 1)) == 0,
      "Invalid pinned allocation alignment: {}",
      alignment);
  if (bytes == 0) {
    return nullptr;
  }
  constexpr uint64_t kCudaAlignment = cuda::mr::default_cuda_malloc_alignment;
  const uint64_t padding = alignment > kCudaAlignment ? alignment - 1 : 0;
  // Validate before adding padding/rounding, avoiding overflow for oversized
  // requests.
  if (bytes > impl_->capacity || padding > impl_->capacity - bytes) {
    setAllocatorFailureMessage("Pinned allocation exceeds capacity");
    return nullptr;
  }
  const auto charged = AllocationTraits::roundUpPageBytes(bytes + padding);
  std::unique_lock<std::mutex> lock(impl_->mutex);
  uint64_t reclaimBytes = std::max<uint64_t>(charged, 32ULL << 20);
  uint64_t reclaimedBytes = 0;
  void* base;
  for (;;) {
    // Recheck after eviction: another allocator thread may consume the space.
    if (charged > impl_->capacity - impl_->chargedBytes ||
        testingHasInjectedFailure(InjectedFailure::kCap)) {
      setAllocatorFailureMessage("Pinned allocator capacity exhausted");
      return nullptr;
    }
    bool fail = testingHasInjectedFailure(InjectedFailure::kAllocate);
    common::testutil::TestValue::adjust(
        "PinnedMemoryAllocator::allocate", &fail);
    if (fail) {
      setAllocatorFailureMessage("Injected pinned allocation failure");
      return nullptr;
    }
    try {
      common::testutil::TestValue::adjust(
          "PinnedMemoryAllocator::poolAllocate", &impl_->chargedBytes);
      stats_.recordAllocate(
          charged, 1, [&] { base = impl_->pool.allocate_sync(charged); });
      break;
    } catch (const cuda::cuda_error& error) {
      if (error.status() != cudaErrorMemoryAllocation) {
        throw;
      }
      setAllocatorFailureMessage(error.what());
      if (!cache_ || reclaimedBytes >= impl_->capacity) {
        return nullptr;
      }
      // Host-pool fragmentation can require much more eviction than the byte
      // request. The base makeSpace() loop has bounded, request-sized retries.
      // Reclaim progressively larger batches without changing AsyncDataCache.
      // shrink() calls freeBytes(), so never hold the allocator lock here.
      const auto target =
          std::min(reclaimBytes, impl_->capacity - reclaimedBytes);
      lock.unlock();
      const auto reclaimed = cache_->shrink(target);
      lock.lock();
      if (reclaimed == 0) {
        return nullptr;
      }
      reclaimedBytes += reclaimed;
      reclaimBytes = reclaimBytes >= impl_->capacity / 2 ? impl_->capacity
                                                         : reclaimBytes * 2;
    }
  }
  auto guard =
      folly::makeGuard([&] { impl_->pool.deallocate_sync(base, charged); });
  const auto address = reinterpret_cast<uintptr_t>(base);
  auto* result = reinterpret_cast<void*>(
      (address + alignment - 1) & ~(uintptr_t(alignment) - 1));
  impl_->blocks.emplace(result, Impl::Block{base, charged, bytes});
  impl_->chargedBytes += charged;
  impl_->requestedBytes += bytes;
  guard.dismiss();
  return result;
}

void PinnedMemoryAllocator::freeBytes(void* ptr, uint64_t bytes) noexcept {
  if (ptr == nullptr) {
    return;
  }
  std::lock_guard<std::mutex> lock(impl_->mutex);
  const auto it = impl_->blocks.find(ptr);
  VELOX_CHECK(it != impl_->blocks.end(), "Unknown pinned allocation");
  const auto block = it->second;
  VELOX_CHECK_EQ(bytes, block.requestedBytes);
  // The caller has completed GPU uses. CCCL orders this free before subsequent
  // allocate_sync calls on its allocation stream; no application stream is
  // exposed.
  stats_.recordFree(block.backingBytes, [&] {
    impl_->pool.deallocate_sync(block.base, block.backingBytes);
  });
  impl_->chargedBytes -= block.backingBytes;
  impl_->requestedBytes -= block.requestedBytes;
  impl_->blocks.erase(it);
}

bool PinnedMemoryAllocator::allocateNonContiguousWithoutRetry(
    const SizeMix& mix,
    Allocation& out) {
  freeNonContiguous(out);
  auto rollback = folly::makeGuard([&] { freeNonContiguous(out); });
  for (int32_t i = 0; i < mix.numSizes; ++i) {
    const auto pages = sizeClassSizes_[mix.sizeIndices[i]] * mix.sizeCounts[i];
    const auto bytes = AllocationTraits::pageBytes(pages);
    auto* ptr = static_cast<uint8_t*>(allocateBytesWithoutRetry(bytes, kPage));
    if (ptr == nullptr) {
      return false;
    }
    auto freeBlock = folly::makeGuard([&] { freeBytes(ptr, bytes); });
    out.append(ptr, pages);
    freeBlock.dismiss();
  }
  rollback.dismiss();
  return true;
}
int64_t PinnedMemoryAllocator::freeNonContiguous(Allocation& allocation) {
  const auto bytes = allocation.byteSize();
  for (uint32_t i = 0; i < allocation.numRuns(); ++i) {
    const auto run = allocation.runAt(i);
    freeBytes(run.data(), run.numBytes());
  }
  allocation.clear();
  return bytes;
}
bool PinnedMemoryAllocator::allocateContiguousWithoutRetry(
    MachinePageCount,
    Allocation* collateral,
    ContiguousAllocation& allocation,
    MachinePageCount) {
  if (collateral) {
    freeNonContiguous(*collateral);
  }
  VELOX_CHECK(allocation.empty());
  setAllocatorFailureMessage(
      "PinnedMemoryAllocator does not support contiguous page allocations; use allocateBytes");
  return false;
}
bool PinnedMemoryAllocator::growContiguousWithoutRetry(
    MachinePageCount,
    ContiguousAllocation&) {
  setAllocatorFailureMessage(
      "PinnedMemoryAllocator does not support contiguous growth");
  return false;
}
void PinnedMemoryAllocator::freeContiguous(ContiguousAllocation& allocation) {
  VELOX_CHECK(
      allocation.empty(),
      "PinnedMemoryAllocator does not own contiguous page allocations");
}

} // namespace facebook::velox::cudf_velox
