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

#include <folly/experimental/ThreadedRepeatingFunctionRunner.h>
#include "velox/common/memory/MemoryArbitrator.h"

namespace folly {
class CPUThreadPoolExecutor;
}

namespace facebook::velox {

namespace memory {
class MemoryAllocator;
}

namespace cache {
class AsyncDataCache;
}

/// Manages a background daemon thread to report stats through 'StatsReporter'.
class PeriodicStatsReporter {
 public:
  struct Options {
    Options() {}

    uint64_t arbitratorStatsIntervalMs{60'000};

    std::string toString() const {
      return fmt::format(
          "arbitratorStatsIntervalMs:{}", arbitratorStatsIntervalMs);
    }
  };

  PeriodicStatsReporter(
      const velox::memory::MemoryArbitrator* arbitrator,
      const Options& options = Options());

  /// Invoked to start the report daemon in background.
  void start();

  /// Invoked to stop the report daemon in background.
  void stop();

 private:
  // Add a task to run periodically.
  template <typename TFunc>
  void addTask(const std::string& taskName, TFunc&& func, size_t intervalMs) {
    scheduler_.add(
        taskName,
        [taskName,
         intervalMs,
         func = std::forward<TFunc>(func)]() mutable noexcept {
          try {
            func();
          } catch (const std::exception& e) {
            LOG(ERROR) << "Error running periodic task " << taskName << ": "
                       << e.what();
          }
          return std::chrono::milliseconds(intervalMs);
        });
  }

  void reportArbitratorStats();

  const velox::memory::MemoryArbitrator* const arbitrator_{nullptr};
  const Options options_;

  folly::ThreadedRepeatingFunctionRunner scheduler_;
};
} // namespace facebook::velox
