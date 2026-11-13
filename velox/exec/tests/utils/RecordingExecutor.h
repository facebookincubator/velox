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

#include <cstdint>
#include <mutex>
#include <optional>
#include <utility>
#include <vector>

#include <folly/Executor.h>
#include <folly/executors/CPUThreadPoolExecutor.h>

namespace facebook::velox::exec::test {

/// Records the priority of every submitted task: the priority passed to
/// 'addWithPriority', or empty for work submitted with 'add'. Execution
/// delegates to an inner thread pool, so the recorded priorities are the
/// ones Velox actually submits through the query executor: drivers from
/// Driver::enqueue and parallel hash join build steps from
/// prepareJoinTable. A query without a hash join sees only driver
/// submissions.
class RecordingExecutor : public folly::Executor {
 public:
  RecordingExecutor(size_t numThreads, int8_t numPriorities)
      : executor_(numThreads, numPriorities) {}

  void add(folly::Func func) override {
    record(std::nullopt);
    executor_.add(std::move(func));
  }

  void addWithPriority(folly::Func func, int8_t priority) override {
    record(priority);
    executor_.addWithPriority(std::move(func), priority);
  }

  uint8_t getNumPriorities() const override {
    return executor_.getNumPriorities();
  }

  /// Returns priorities recorded so far, in submission order.
  std::vector<std::optional<int8_t>> recordedPriorities() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return recorded_;
  }

 private:
  void record(std::optional<int8_t> priority) {
    std::lock_guard<std::mutex> lock(mutex_);
    recorded_.push_back(priority);
  }

  folly::CPUThreadPoolExecutor executor_;
  mutable std::mutex mutex_;
  std::vector<std::optional<int8_t>> recorded_;
};

} // namespace facebook::velox::exec::test
