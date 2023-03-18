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
#include "velox/dwio/common/Options.h"

namespace facebook::velox::dwio::common {

class ParallelBufferedInput : public BufferedInput {
 public:
  ParallelBufferedInput(
      std::shared_ptr<ReadFile> readFile,
      memory::MemoryPool& pool,
      const MetricsLogPtr& metricsLog = MetricsLog::voidLog(),
      std::shared_ptr<IoStatistics> ioStats = nullptr,
      folly::Executor* executor = nullptr,
      int32_t loadQuantum = ReaderOptions::kDefaultLoadQuantum)
      : BufferedInput(std::move(readFile), pool, metricsLog, nullptr),
        ioStats_(std::move(ioStats)),
        executor_(executor),
        loadQuantum_(loadQuantum) {}

  ParallelBufferedInput(
      std::shared_ptr<ReadFileInputStream> input,
      memory::MemoryPool& pool,
      std::shared_ptr<IoStatistics> ioStats = nullptr,
      folly::Executor* executor = nullptr,
      int32_t loadQuantum = ReaderOptions::kDefaultLoadQuantum)
      : BufferedInput(std::move(input), pool),
        ioStats_(std::move(ioStats)),
        executor_(executor),
        loadQuantum_(loadQuantum) {}

  ParallelBufferedInput(ParallelBufferedInput&&) = default;
  virtual ~ParallelBufferedInput() = default;

  ParallelBufferedInput(const ParallelBufferedInput&) = delete;
  ParallelBufferedInput& operator=(const ParallelBufferedInput&) = delete;
  ParallelBufferedInput& operator=(ParallelBufferedInput&&) = delete;

  // Load all regions to be read in an optimized way (IO efficiency).
  void load(const LogType) override;

  folly::Executor* executor() const override {
    return executor_;
  }

  std::unique_ptr<BufferedInput> clone() const override {
    return std::make_unique<ParallelBufferedInput>(
        input_, pool_, ioStats_, executor_, loadQuantum_);
  }

 private:
  std::shared_ptr<IoStatistics> ioStats_;
  folly::Executor* const executor_;

  // Used to split large region.
  const int32_t loadQuantum_;

  // Try split large region into several small regions by load quantum.
  void splitRegion(
      const uint64_t length,
      const int32_t loadQuantum,
      std::vector<std::tuple<uint64_t, uint64_t>>& range);

  // Load all regions parallelly.
  void loadParallel(
      const std::vector<void*>& buffers,
      const std::vector<Region>& regions,
      const LogType purpose);
};

} // namespace facebook::velox::dwio::common
