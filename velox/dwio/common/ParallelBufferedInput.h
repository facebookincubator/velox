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

#include "velox/dwio/common/BufferedInput.h"

namespace facebook::velox::dwio::common {

class ParallelBufferedInput : public BufferedInput {
 public:
  ParallelBufferedInput(
      std::shared_ptr<ReadFile> readFile,
      memory::MemoryPool& pool,
      const MetricsLogPtr& metricsLog = MetricsLog::voidLog(),
      std::shared_ptr<IoStatistics> ioStats = nullptr,
      folly::Executor* executor = nullptr,
      int32_t loadQuantum = ReaderOptions::kDefaultLoadQuantum,
      int32_t mergeDistance = ReaderOptions::kMaxMergeDistance,
      bool parallelLoad = false)
      : BufferedInput(
            std::move(readFile),
            pool,
            metricsLog,
            nullptr,
            mergeDistance),
        ioStats_(std::move(ioStats)),
        executor_(executor),
        loadQuantum_(loadQuantum),
        parallelLoad_(parallelLoad) {}

  ParallelBufferedInput(
      std::shared_ptr<ReadFileInputStream> input,
      memory::MemoryPool& pool,
      std::shared_ptr<IoStatistics> ioStats = nullptr,
      folly::Executor* executor = nullptr,
      int32_t loadQuantum = ReaderOptions::kDefaultLoadQuantum,
      int32_t mergeDistance = ReaderOptions::kMaxMergeDistance,
      bool parallelLoad = false)
      : BufferedInput(std::move(input), pool, mergeDistance),
        ioStats_(std::move(ioStats)),
        executor_(executor),
        loadQuantum_(loadQuantum),
        parallelLoad_(parallelLoad) {}

  ParallelBufferedInput(ParallelBufferedInput&&) = default;
  virtual ~ParallelBufferedInput() = default;

  ParallelBufferedInput(const ParallelBufferedInput&) = delete;
  ParallelBufferedInput& operator=(const ParallelBufferedInput&) = delete;
  ParallelBufferedInput& operator=(ParallelBufferedInput&&) = delete;

  // load all regions to be read in an optimized way (IO efficiency)
  void load(const LogType) override;

  virtual folly::Executor* executor() const {
    return executor_;
  }

 private:
  std::shared_ptr<IoStatistics> ioStats_;
  folly::Executor* const executor_;
  const int32_t loadQuantum_;

  // Enable parallel load regions by executor.
  const bool parallelLoad_;

  // try split large region into several small regions by load quantum
  void splitRegion(
      const uint64_t length,
      const int32_t loadQuantum,
      std::vector<std::tuple<uint64_t, uint64_t>>& range);

  // load all regions parallelly
  void loadParallel(
      const std::vector<void*>& buffers,
      const std::vector<Region>& regions,
      const LogType purpose);
};

} // namespace facebook::velox::dwio::common
