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

#include "velox/experimental/cudf/CudfDefaultStreamOverload.h"
#include "velox/experimental/cudf/exec/GpuResources.h"

#include <cudf/detail/utilities/stream_pool.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/prefetch.hpp>

#include <rmm/cuda_device.hpp>
#include <rmm/error.hpp>
#include <rmm/mr/arena_memory_resource.hpp>
#include <rmm/mr/cuda_async_managed_memory_resource.hpp>
#include <rmm/mr/cuda_async_memory_resource.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/managed_memory_resource.hpp>
#include <rmm/mr/pool_memory_resource.hpp>
#include <rmm/mr/prefetch_resource_adaptor.hpp>

#include <cuda_runtime_api.h>

#include <common/base/Exceptions.h>
#include <glog/logging.h>

#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <optional>
#include <string_view>
#include <unordered_map>

namespace facebook::velox::cudf_velox {

namespace {
constexpr std::size_t kOverflowLogEvery = 64;
} // namespace

class TieredMemoryResource::Impl {
 public:
  explicit Impl(std::size_t deviceCapBytes)
      : device_{std::nullopt, deviceCapBytes}, capBytes_{deviceCapBytes} {
    // The overflow tier is plain CUDA managed (unified) memory: the simplest
    // resource that can oversubscribe the device. Pooling or prefetching it is
    // an optimization we deliberately leave out of the default -- correctness
    // first -- so a query that spills completes rather than failing.
  }

  Impl(Impl const&) = delete;
  Impl& operator=(Impl const&) = delete;

  void*
  allocate(cuda::stream_ref stream, std::size_t bytes, std::size_t alignment) {
    if (deviceBytes_.load(std::memory_order_relaxed) + bytes <= capBytes_) {
      try {
        void* ptr = device_.allocate(stream, bytes, alignment);
        auto const outstanding =
            deviceBytes_.fetch_add(bytes, std::memory_order_relaxed) + bytes;
        updatePeak(deviceBytesPeak_, outstanding);
        return ptr;
      } catch (rmm::out_of_memory const&) {
        // The pool could not satisfy this even though we are under the cap;
        // fall through to the overflow tier rather than failing the query.
      }
    }
    return allocateOverflow(stream, bytes, alignment);
  }

  void deallocate(
      cuda::stream_ref stream,
      void* ptr,
      std::size_t bytes,
      std::size_t alignment) noexcept {
    // Fast path: when nothing is live in the overflow tier, `ptr` cannot be an
    // overflow pointer, so skip the lock and the map lookup. An overflow
    // allocation keeps overflowLive_ > 0 for its whole lifetime, so a zero
    // reading here is a definitive "device tier". This keeps the common case
    // -- queries that fit and never spill -- lock-free on every free.
    if (overflowLive_.load(std::memory_order_acquire) != 0) {
      std::lock_guard<std::mutex> lock(mu_);
      auto it = overflowPtrs_.find(ptr);
      if (it != overflowPtrs_.end()) {
        overflowPtrs_.erase(it);
        overflowBytes_.fetch_sub(bytes, std::memory_order_relaxed);
        overflowLive_.fetch_sub(1, std::memory_order_release);
        overflow_.deallocate(stream, ptr, bytes, alignment);
        return;
      }
    }
    device_.deallocate(stream, ptr, bytes, alignment);
    deviceBytes_.fetch_sub(bytes, std::memory_order_relaxed);
  }

  TieredMemoryResourceStats stats() const {
    return TieredMemoryResourceStats{
        deviceBytes_.load(std::memory_order_relaxed),
        deviceBytesPeak_.load(std::memory_order_relaxed),
        overflowBytes_.load(std::memory_order_relaxed),
        overflowBytesPeak_.load(std::memory_order_relaxed),
        overflowAllocations_.load(std::memory_order_relaxed),
        capBytes_};
  }

 private:
  static void updatePeak(
      std::atomic<std::size_t>& peak,
      std::size_t outstanding) {
    auto current = peak.load(std::memory_order_relaxed);
    while (current < outstanding &&
           !peak.compare_exchange_weak(
               current, outstanding, std::memory_order_relaxed)) {
    }
  }

  void* allocateOverflow(
      cuda::stream_ref stream,
      std::size_t bytes,
      std::size_t alignment) {
    void* ptr = overflow_.allocate(stream, bytes, alignment);
    {
      std::lock_guard<std::mutex> lock(mu_);
      overflowPtrs_.emplace(ptr, bytes);
    }
    // Publish the live count before returning so a concurrent free of this
    // pointer takes the locked path and finds it in the map.
    overflowLive_.fetch_add(1, std::memory_order_release);
    auto const outstanding =
        overflowBytes_.fetch_add(bytes, std::memory_order_relaxed) + bytes;
    updatePeak(overflowBytesPeak_, outstanding);
    auto const count =
        overflowAllocations_.fetch_add(1, std::memory_order_relaxed) + 1;
    if (count == 1 || count % kOverflowLogEvery == 0) {
      VLOG(1) << "TieredMemoryResource: overflow allocation #" << count
              << " of " << bytes << " bytes; device tier "
              << deviceBytes_.load(std::memory_order_relaxed) << "/"
              << capBytes_ << " bytes, overflow tier " << outstanding
              << " bytes";
    }
    return ptr;
  }

  rmm::mr::cuda_async_memory_resource device_;
  rmm::mr::managed_memory_resource overflow_;
  std::size_t capBytes_;
  std::atomic<std::size_t> deviceBytes_{0};
  std::atomic<std::size_t> deviceBytesPeak_{0};
  std::atomic<std::size_t> overflowBytes_{0};
  std::atomic<std::size_t> overflowBytesPeak_{0};
  std::atomic<std::size_t> overflowAllocations_{0};
  // Number of live overflow allocations; guards the lock-free deallocate path.
  std::atomic<std::size_t> overflowLive_{0};
  mutable std::mutex mu_;
  std::unordered_map<void*, std::size_t> overflowPtrs_;
};

namespace {
std::mutex& tieredHandleMutex() {
  static std::mutex mutex;
  return mutex;
}

std::weak_ptr<TieredMemoryResource::Impl>& tieredHandle() {
  static std::weak_ptr<TieredMemoryResource::Impl> handle;
  return handle;
}
} // namespace

TieredMemoryResource::TieredMemoryResource(std::size_t deviceCapBytes)
    : impl_{std::make_shared<Impl>(deviceCapBytes)} {}

void* TieredMemoryResource::allocate(
    cuda::stream_ref stream,
    std::size_t bytes,
    std::size_t alignment) {
  return impl_->allocate(stream, bytes, alignment);
}

void TieredMemoryResource::deallocate(
    cuda::stream_ref stream,
    void* ptr,
    std::size_t bytes,
    std::size_t alignment) noexcept {
  impl_->deallocate(stream, ptr, bytes, alignment);
}

void* TieredMemoryResource::allocate_sync(
    std::size_t bytes,
    std::size_t alignment) {
  auto const stream = cuda::stream_ref{cudaStream_t{nullptr}};
  void* ptr = impl_->allocate(stream, bytes, alignment);
  VELOX_CHECK_EQ(
      static_cast<int>(cudaStreamSynchronize(cudaStream_t{nullptr})),
      static_cast<int>(cudaSuccess),
      "TieredMemoryResource: cudaStreamSynchronize failed");
  return ptr;
}

void TieredMemoryResource::deallocate_sync(
    void* ptr,
    std::size_t bytes,
    std::size_t alignment) noexcept {
  auto const stream = cuda::stream_ref{cudaStream_t{nullptr}};
  impl_->deallocate(stream, ptr, bytes, alignment);
  static_cast<void>(cudaStreamSynchronize(cudaStream_t{nullptr}));
}

TieredMemoryResourceStats TieredMemoryResource::stats() const {
  return impl_->stats();
}

void TieredMemoryResource::registerGlobal() const {
  std::lock_guard<std::mutex> lock(tieredHandleMutex());
  tieredHandle() = impl_;
}

std::optional<TieredMemoryResourceStats> tieredMemoryResourceStats() {
  std::shared_ptr<TieredMemoryResource::Impl> impl;
  {
    std::lock_guard<std::mutex> lock(tieredHandleMutex());
    impl = tieredHandle().lock();
  }
  if (impl == nullptr) {
    return std::nullopt;
  }
  return impl->stats();
}

void logTieredMemoryResourceStats() {
  auto const stats = tieredMemoryResourceStats();
  if (!stats.has_value()) {
    LOG(INFO) << "TieredMemoryResource: not active";
    return;
  }
  LOG(INFO) << "TieredMemoryResource: device " << stats->deviceBytes
            << " bytes (peak " << stats->deviceBytesPeak << ", cap "
            << stats->capBytes << "), overflow " << stats->overflowBytes
            << " bytes (peak " << stats->overflowBytesPeak << ") across "
            << stats->overflowAllocations << " allocations";
}

cuda::mr::any_resource<cuda::mr::device_accessible> createMemoryResource(
    std::string_view mode,
    int percent) {
  if (mode == "cuda") {
    return rmm::mr::cuda_memory_resource{};
  } else if (mode == "pool") {
    return rmm::mr::pool_memory_resource(
        rmm::mr::cuda_memory_resource{},
        rmm::percent_of_free_device_memory(percent));
  } else if (mode == "async") {
    return rmm::mr::cuda_async_memory_resource{};
  } else if (mode == "arena") {
    return rmm::mr::arena_memory_resource(
        rmm::mr::cuda_memory_resource{},
        rmm::percent_of_free_device_memory(percent));
  } else if (mode == "managed") {
    return rmm::mr::managed_memory_resource{};
  } else if (mode == "managed_pool") {
    return rmm::mr::pool_memory_resource(
        rmm::mr::managed_memory_resource{},
        rmm::percent_of_free_device_memory(percent));
  } else if (mode == "managed_async") {
    return rmm::mr::cuda_async_managed_memory_resource{};
  } else if (mode == "prefetch_managed") {
    cudf::prefetch::enable();
    return rmm::mr::prefetch_resource_adaptor(
        rmm::mr::managed_memory_resource{});
  } else if (mode == "prefetch_managed_pool") {
    cudf::prefetch::enable();
    return rmm::mr::prefetch_resource_adaptor(
        rmm::mr::pool_memory_resource(
            rmm::mr::managed_memory_resource{},
            rmm::percent_of_free_device_memory(percent)));
  } else if (mode == "prefetch_managed_async") {
    cudf::prefetch::enable();
    return rmm::mr::prefetch_resource_adaptor(
        rmm::mr::cuda_async_managed_memory_resource{});
  } else if (mode == "tiered") {
    // Unlike the pooling modes, `percent` here is a fraction of *total* device
    // memory and bounds the fast device tier; allocations past it spill to
    // managed memory instead of failing.
    auto const totalBytes = rmm::available_device_memory().second;
    auto const capBytes = static_cast<std::size_t>(
        (static_cast<double>(totalBytes) * static_cast<double>(percent)) /
        100.0);
    TieredMemoryResource resource{capBytes};
    resource.registerGlobal();
    return resource;
  }
  VELOX_FAIL(
      "Unknown memory resource mode: " + std::string(mode) +
      "\nExpecting: cuda, pool, async, arena, managed, prefetch_managed, " +
      "managed_pool, prefetch_managed_pool, managed_async, prefetch_managed_async, " +
      "tiered");
}

cudf::detail::cuda_stream_pool& cudfGlobalStreamPool() {
  return cudf::detail::global_cuda_stream_pool();
};

std::optional<cuda::mr::any_resource<cuda::mr::device_accessible>> mr_;
std::optional<cuda::mr::any_resource<cuda::mr::device_accessible>> output_mr_;

rmm::device_async_resource_ref get_output_mr() {
  return output_mr_.value();
}

} // namespace facebook::velox::cudf_velox

// This must NOT be in a file that includes CudfNoDefaults.h, because
// CudfNoDefaults.h redeclares cudf::get_default_stream() with
// __attribute__((error)). The overload below calls the real function.
namespace cudf {

rmm::cuda_stream_view const get_default_stream(allow_default_stream_t) {
  return cudf::get_default_stream();
}

} // namespace cudf
