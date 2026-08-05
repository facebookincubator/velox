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

#include "velox/common/base/TrackedExecutor.h"

#include <folly/ScopeGuard.h>
#include <folly/synchronization/SanitizeThread.h>
#include "velox/common/time/CpuWallTimer.h"

namespace facebook::velox {

TrackedExecutor::Metrics::Metrics() {
  folly::annotate_benign_race_sized(
      this,
      sizeof(*this),
      "velox::TrackedExecutor::Metrics is deliberately not thread safe",
      __FILE__,
      __LINE__);
}

TrackedExecutor::Func TrackedExecutor::wrapFunc(Func func) {
  auto enqueueTime = std::chrono::steady_clock::now();
  return [func = std::move(func), enqueueTime, metrics = metrics_]() mutable {
    metrics->waitTime.addValue(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - enqueueTime)
            .count());
    CpuWallTiming timing;
    // Record on scope exit so a throwing callback still samples each metric.
    SCOPE_EXIT {
      metrics->executionWallTime.addValue(
          static_cast<int64_t>(timing.wallNanos));
      metrics->executionCpuTime.addValue(static_cast<int64_t>(timing.cpuNanos));
    };
    DeltaCpuWallTimer timer(
        [&timing](const CpuWallTiming& delta) { timing = delta; });
    func();
  };
}

void TrackedExecutor::reportTo(
    BaseRuntimeStatWriter& writer,
    std::string_view prefix) const {
  writer.setRuntimeStat(
      fmt::format("{}-{}", prefix, kExecutorWaitNanos), metrics_->waitTime);
  writer.setRuntimeStat(
      fmt::format("{}-{}", prefix, kExecutorExecutionWallNanos),
      metrics_->executionWallTime);
  writer.setRuntimeStat(
      fmt::format("{}-{}", prefix, kExecutorExecutionCpuNanos),
      metrics_->executionCpuTime);
}

} // namespace facebook::velox
