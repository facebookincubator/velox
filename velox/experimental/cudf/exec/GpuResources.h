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

#include <cudf/detail/utilities/stream_pool.hpp>
#include <cudf/table/table.hpp>

#include <rmm/mr/device_memory_resource.hpp>
#include <rmm/mr/statistics_resource_adaptor.hpp>
#include <rmm/resource_ref.hpp>

#include <atomic>
#include <memory>
#include <string_view>

namespace facebook::velox::exec {
class Operator;
} // namespace facebook::velox::exec

namespace facebook::velox::cudf_velox {

using StatisticsResourceAdaptor =
    rmm::mr::statistics_resource_adaptor<rmm::mr::device_memory_resource>;

extern std::shared_ptr<rmm::mr::device_memory_resource> mr_;
extern std::shared_ptr<rmm::mr::device_memory_resource> output_mr_;

/// Returns the memory resource designated for output vector allocations.
rmm::device_async_resource_ref get_output_mr();

/// Returns the statistics resource adaptor for the main memory resource.
/// Returns nullptr if statistics tracking is not active.
StatisticsResourceAdaptor* getStatsMr();

/// Returns the statistics resource adaptor for the output memory resource.
/// Returns nullptr if statistics tracking is not active.
StatisticsResourceAdaptor* getOutputStatsMr();

/// Sets the statistics resource adaptor pointer for the main memory resource.
void setStatsMr(StatisticsResourceAdaptor* ptr);

/// Sets the statistics resource adaptor pointer for the output memory resource.
void setOutputStatsMr(StatisticsResourceAdaptor* ptr);

/// GPU memory resource that routes per-allocation stats to the Velox operator
/// currently active on the calling thread via the thread-local
/// RuntimeStatWriter. Allocations without an active operator are tracked
/// under global "orphan" counters and periodically logged.
///
/// TODO: Output MR distinction.
/// When mr_ != output_mr_, create two StatsTrackingMemoryResource instances
/// wrapping each. The one wrapping output_mr_ writes "gpuOutputAllocBytes"
/// stats; the one wrapping mr_ writes "gpuTempAllocBytes" stats.
/// Operators that explicitly pass output_mr to cudf APIs would route output
/// allocations through the output stats MR. Both MRs write to the same
/// thread-local operator (the active one), but with different stat names.
class StatsTrackingMemoryResource : public rmm::mr::device_memory_resource {
 public:
  struct OrphanStats {
    int64_t allocBytes;
    int64_t allocCount;
    int64_t freeBytes;
    int64_t freeCount;
  };

  explicit StatsTrackingMemoryResource(
      rmm::mr::device_memory_resource* upstream)
      : upstream_(upstream) {}

  rmm::mr::device_memory_resource* upstream() const {
    return upstream_;
  }

  OrphanStats getOrphanStats() const;

 private:
  void* do_allocate(std::size_t bytes, rmm::cuda_stream_view stream) override;
  void do_deallocate(
      void* ptr,
      std::size_t bytes,
      rmm::cuda_stream_view stream) noexcept override;
  [[nodiscard]] bool do_is_equal(
      device_memory_resource const& other) const noexcept override;

  rmm::mr::device_memory_resource* upstream_;
  std::atomic<int64_t> orphanAllocBytes_{0};
  std::atomic<int64_t> orphanAllocCount_{0};
  std::atomic<int64_t> orphanFreeBytes_{0};
  std::atomic<int64_t> orphanFreeCount_{0};
};

/// Returns the global StatsTrackingMemoryResource, or nullptr if not active.
StatsTrackingMemoryResource* getStatsTrackingMr();

/// Sets the global StatsTrackingMemoryResource pointer.
void setStatsTrackingMr(StatsTrackingMemoryResource* ptr);

/**
 * @brief Creates a memory resource based on the given mode.
 *
 * @param mode rmm::mr::pool_memory_resource mode.
 * @param percent The initial percent of GPU memory to allocate for memory
 * resource.
 */
[[nodiscard]] std::shared_ptr<rmm::mr::device_memory_resource>
createMemoryResource(std::string_view mode, int percent);

/**
 * @brief Returns the global CUDA stream pool used by cudf.
 */
[[nodiscard]] cudf::detail::cuda_stream_pool& cudfGlobalStreamPool();

} // namespace facebook::velox::cudf_velox
