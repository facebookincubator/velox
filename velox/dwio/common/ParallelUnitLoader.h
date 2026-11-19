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

#include <folly/Executor.h>
#include <chrono>
#include <functional>
#include <vector>
#include "velox/dwio/common/UnitLoader.h"

namespace facebook::velox::dwio::common {

class BoundedPrefetchBudget;

/// Configures shared admission and optional observability for unit loads.
struct ParallelUnitLoaderOptions {
  /// Limits speculative loads across loaders created by the same factory.
  /// Set both limits to zero to disable shared budget admission.
  size_t maxPrefetchLoads{0};
  /// Limits retained speculative IO bytes across the factory's loaders.
  uint64_t maxPrefetchBytes{0};
  /// Reports load activity from reader and executor threads.
  std::function<void(UnitLoader::Event, uint64_t)> metricCallback{};
  /// Reports IO waits across all loaders created by this factory.
  std::function<void(std::chrono::high_resolution_clock::duration)>
      blockedOnIoCallback{};
};

/// Creates loaders with parallel lookahead and optional shared admission
/// limits. Destruction cancels queued loads and waits for active loads before
/// releasing their resources.
class ParallelUnitLoaderFactory : public UnitLoaderFactory {
 public:
  ParallelUnitLoaderFactory(
      folly::Executor* ioExecutor,
      size_t maxConcurrentLoads,
      ParallelUnitLoaderOptions options = {});

  std::unique_ptr<UnitLoader> create(
      std::vector<std::unique_ptr<LoadUnit>> loadUnits,
      uint64_t rowsToSkip) override;

 private:
  folly::Executor* ioExecutor_;
  size_t maxConcurrentLoads_;
  ParallelUnitLoaderOptions options_;
  // Shares admission accounting across all loaders created by this factory.
  std::shared_ptr<BoundedPrefetchBudget> prefetchBudget_;
};

} // namespace facebook::velox::dwio::common
