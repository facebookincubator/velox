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

#include "velox/common/base/RuntimeMetrics.h"
#include "velox/exec/Operator.h"

#include <cudf/detail/utilities/stream_pool.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/prefetch.hpp>

#include <rmm/mr/arena_memory_resource.hpp>
#include <rmm/mr/cuda_async_managed_memory_resource.hpp>
#include <rmm/mr/cuda_async_memory_resource.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/device_memory_resource.hpp>
#include <rmm/mr/managed_memory_resource.hpp>
#include <rmm/mr/owning_wrapper.hpp>
#include <rmm/mr/per_device_resource.hpp>
#include <rmm/mr/pool_memory_resource.hpp>
#include <rmm/mr/prefetch_resource_adaptor.hpp>

#include <common/base/Exceptions.h>

#include <cstdlib>
#include <memory>
#include <string_view>
#include <thread>

namespace facebook::velox::cudf_velox {

namespace {
/// \brief Makes a cuda resource
[[nodiscard]] auto makeCudaMr() {
  return std::make_shared<rmm::mr::cuda_memory_resource>();
}

/// \brief Makes a pool<cuda> resource
[[nodiscard]] auto makePoolMr(int percent) {
  return rmm::mr::make_owning_wrapper<rmm::mr::pool_memory_resource>(
      makeCudaMr(), rmm::percent_of_free_device_memory(percent));
}

/// \brief Makes an async resource
[[nodiscard]] auto makeAsyncMr() {
  return std::make_shared<rmm::mr::cuda_async_memory_resource>();
}

/// \brief Makes a managed resource
[[nodiscard]] auto makeManagedMr() {
  return std::make_shared<rmm::mr::managed_memory_resource>();
}

/// \brief Makes a prefetch<managed> resource
[[nodiscard]] auto makePrefetchManagedMr() {
  return rmm::mr::make_owning_wrapper<rmm::mr::prefetch_resource_adaptor>(
      makeManagedMr());
}

/// \brief Makes an arena<cuda> resource
[[nodiscard]] auto makeArenaMr(int percent) {
  return rmm::mr::make_owning_wrapper<rmm::mr::arena_memory_resource>(
      makeCudaMr(), rmm::percent_of_free_device_memory(percent));
}

/// \brief Makes a pool<managed> resource
[[nodiscard]] auto makeManagedPoolMr(int percent) {
  return rmm::mr::make_owning_wrapper<rmm::mr::pool_memory_resource>(
      makeManagedMr(), rmm::percent_of_free_device_memory(percent));
}

/// \brief Makes a prefetch<pool<managed>> resource
[[nodiscard]] auto makePrefetchManagedPoolMr(int percent) {
  return rmm::mr::make_owning_wrapper<rmm::mr::prefetch_resource_adaptor>(
      makeManagedPoolMr(percent));
}

/// \brief Makes a managed_async resource (experimental, requires CUDA 13+)
[[nodiscard]] auto makeManagedAsyncMr() {
  return std::make_shared<rmm::mr::cuda_async_managed_memory_resource>();
}

/// \brief Makes a prefetch<managed_async> resource (experimental, requires CUDA
/// 13+)
[[nodiscard]] auto makePrefetchManagedAsyncMr() {
  return rmm::mr::make_owning_wrapper<rmm::mr::prefetch_resource_adaptor>(
      makeManagedAsyncMr());
}

void enablePrefetching() {
  cudf::prefetch::enable();
}

} // namespace

std::shared_ptr<rmm::mr::device_memory_resource> createMemoryResource(
    std::string_view mode,
    int percent) {
  if (mode == "cuda")
    return makeCudaMr();
  if (mode == "pool")
    return makePoolMr(percent);
  if (mode == "async")
    return makeAsyncMr();
  if (mode == "arena")
    return makeArenaMr(percent);
  if (mode == "managed")
    return makeManagedMr();
  if (mode == "managed_pool")
    return makeManagedPoolMr(percent);
  if (mode == "managed_async")
    return makeManagedAsyncMr();
  if (mode == "prefetch_managed") {
    enablePrefetching();
    return makePrefetchManagedMr();
  }
  if (mode == "prefetch_managed_pool") {
    enablePrefetching();
    return makePrefetchManagedPoolMr(percent);
  }
  if (mode == "prefetch_managed_async") {
    enablePrefetching();
    return makePrefetchManagedAsyncMr();
  }
  VELOX_FAIL(
      "Unknown memory resource mode: " + std::string(mode) +
      "\nExpecting: cuda, pool, async, arena, managed, prefetch_managed, " +
      "managed_pool, prefetch_managed_pool, managed_async, prefetch_managed_async");
}

cudf::detail::cuda_stream_pool& cudfGlobalStreamPool() {
  return cudf::detail::global_cuda_stream_pool();
};

std::shared_ptr<rmm::mr::device_memory_resource> mr_;
std::shared_ptr<rmm::mr::device_memory_resource> output_mr_;

namespace {
StatisticsResourceAdaptor* stats_mr_ptr_{nullptr};
StatisticsResourceAdaptor* output_stats_mr_ptr_{nullptr};
} // namespace

rmm::device_async_resource_ref get_output_mr() {
  return output_mr_.get();
}

StatisticsResourceAdaptor* getStatsMr() {
  return stats_mr_ptr_;
}

StatisticsResourceAdaptor* getOutputStatsMr() {
  return output_stats_mr_ptr_;
}

void setStatsMr(StatisticsResourceAdaptor* ptr) {
  stats_mr_ptr_ = ptr;
}

void setOutputStatsMr(StatisticsResourceAdaptor* ptr) {
  output_stats_mr_ptr_ = ptr;
}

namespace {
StatsTrackingMemoryResource* statsTrackingMrPtr_{nullptr};
} // namespace

StatsTrackingMemoryResource* getStatsTrackingMr() {
  return statsTrackingMrPtr_;
}

void setStatsTrackingMr(StatsTrackingMemoryResource* ptr) {
  statsTrackingMrPtr_ = ptr;
}

void* StatsTrackingMemoryResource::do_allocate(
    std::size_t bytes,
    rmm::cuda_stream_view stream) {
  auto* writer = getThreadLocalRunTimeStatWriter();
  if (writer) {
    writer->addRuntimeStat(
        "gpuAllocBytes",
        RuntimeCounter(
            static_cast<int64_t>(bytes), RuntimeCounter::Unit::kBytes));
  } else {
    orphanAllocBytes_.fetch_add(
        static_cast<int64_t>(bytes), std::memory_order_relaxed);
    orphanAllocCount_.fetch_add(1, std::memory_order_relaxed);
    VLOG_EVERY_N(1, 100) << "GPU allocation of " << bytes
                         << " bytes with no active operator on thread "
                         << std::this_thread::get_id();
  }
  return upstream_->allocate(stream, bytes);
}

void StatsTrackingMemoryResource::do_deallocate(
    void* ptr,
    std::size_t bytes,
    rmm::cuda_stream_view stream) noexcept {
  auto* writer = getThreadLocalRunTimeStatWriter();
  if (writer) {
    writer->addRuntimeStat(
        "gpuFreeBytes",
        RuntimeCounter(
            static_cast<int64_t>(bytes), RuntimeCounter::Unit::kBytes));
  } else {
    orphanFreeBytes_.fetch_add(
        static_cast<int64_t>(bytes), std::memory_order_relaxed);
    orphanFreeCount_.fetch_add(1, std::memory_order_relaxed);
  }
  upstream_->deallocate(stream, ptr, bytes);
}

bool StatsTrackingMemoryResource::do_is_equal(
    device_memory_resource const& other) const noexcept {
  if (auto const* o =
          dynamic_cast<StatsTrackingMemoryResource const*>(&other)) {
    return upstream_->is_equal(*o->upstream_);
  }
  return upstream_->is_equal(other);
}

StatsTrackingMemoryResource::OrphanStats
StatsTrackingMemoryResource::getOrphanStats() const {
  return {
      orphanAllocBytes_.load(std::memory_order_relaxed),
      orphanAllocCount_.load(std::memory_order_relaxed),
      orphanFreeBytes_.load(std::memory_order_relaxed),
      orphanFreeCount_.load(std::memory_order_relaxed)};
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
