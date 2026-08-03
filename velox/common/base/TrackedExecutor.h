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
#include <folly/Function.h>
#include "velox/common/base/Exceptions.h"
#include "velox/common/base/RuntimeMetrics.h"

namespace facebook::velox {

/// Wraps an executor to measure callback wait time and execution time. All
/// tasks submitted via add() are delegated to the underlying executor after
/// being instrumented.
class TrackedExecutor final : public folly::Executor {
  using Func = folly::Function<void()>;

 public:
  /// Metric name for the time a callback spent enqueued before it started
  /// running, in nanoseconds.
  static constexpr std::string_view kExecutorWaitNanos{"executorWaitNanos"};

  /// Metric name for the wall time a callback spent running, in nanoseconds.
  static constexpr std::string_view kExecutorExecutionWallNanos{
      "executorExecutionWallNanos"};

  /// Metric name for the cpu time a callback spent running, in nanoseconds.
  static constexpr std::string_view kExecutorExecutionCpuNanos{
      "executorExecutionCpuNanos"};

  /// Wraps 'executor', which must be non-null. All submitted callbacks run on
  /// 'executor' after instrumentation.
  explicit TrackedExecutor(folly::Executor::KeepAlive<> executor)
      : executor_{std::move(executor)} {
    VELOX_CHECK(executor_);
  }

  TrackedExecutor(const TrackedExecutor&) = delete;
  TrackedExecutor& operator=(const TrackedExecutor&) = delete;
  TrackedExecutor(TrackedExecutor&&) = delete;
  TrackedExecutor& operator=(TrackedExecutor&&) = delete;
  ~TrackedExecutor() override = default;

  void add(Func func) override {
    executor_->add(wrapFunc(std::move(func)));
  }

  void addWithPriority(Func func, int8_t priority) override {
    executor_->addWithPriority(wrapFunc(std::move(func)), priority);
  }

  uint8_t getNumPriorities() const override {
    return executor_->getNumPriorities();
  }

  /// Reports accumulated metrics to 'writer', naming each 'prefix-<metric>'.
  void reportTo(BaseRuntimeStatWriter& writer, std::string_view prefix) const;

 private:
  // Instruments 'func' to record its enqueue-wait, wall, and cpu time into
  // metrics_ when it runs.
  Func wrapFunc(Func func);

  folly::Executor::KeepAlive<> executor_;

  struct Metrics {
    // Not thread safe. Prefer potential inaccuracy over synchronization
    // overhead. Construction annotates the whole struct as benignly racy so
    // ThreadSanitizer does not flag concurrent updates from the wrapped tasks
    // or reads from reportTo().
    Metrics();

    RuntimeMetric waitTime{RuntimeCounter::Unit::kNanos};
    RuntimeMetric executionWallTime{RuntimeCounter::Unit::kNanos};
    RuntimeMetric executionCpuTime{RuntimeCounter::Unit::kNanos};
  };

  const std::shared_ptr<Metrics> metrics_{std::make_shared<Metrics>()};
};

} // namespace facebook::velox
