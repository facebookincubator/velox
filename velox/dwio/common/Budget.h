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

#include <condition_variable>
#include <functional>
#include <limits>
#include <mutex>
#include "velox/common/base/Exceptions.h"
#include "velox/dwio/common/MeasureTime.h"
#include "velox/dwio/common/UnitLoaderTools.h"

namespace facebook::velox::dwio::common {

// This class allows threads to block until enough memory budget for their
// request is available.
class Budget {
 public:
  explicit Budget(std::function<uint64_t()> budgetCallback)
      : budget_{budgetCallback()},
        availableBudget_{asSigned(budget_)},
        budgetCallback_{std::move(budgetCallback)} {}

  std::optional<std::chrono::high_resolution_clock::duration> waitForBudget(
      uint64_t requiredBudget) {
    const auto reqBudget = asSigned(requiredBudget);
    auto lock = std::unique_lock(mutex_);
    if (hasBudget(reqBudget)) {
      useBudget(reqBudget);
      return std::nullopt;
    }
    const std::chrono::time_point<std::chrono::high_resolution_clock>
        startTime = std::chrono::high_resolution_clock::now();
    // Wait until budget is available because another thread released it. Also
    // check every second if the total budget didn't change (so there may be
    // budget available).
    while (!hasBudget(reqBudget)) {
      cv_.wait_for(lock, std::chrono::seconds(1));
    }
    useBudget(reqBudget);
    return std::chrono::high_resolution_clock::now() - startTime;
  }

  void releaseBudget(uint64_t releasedBudget) {
    const auto relBudget = asSigned(releasedBudget);
    auto lock = std::unique_lock(mutex_);
    availableBudget_ += relBudget;
    cv_.notify_all();
  }

 private:
  int64_t asSigned(uint64_t value) const {
    VELOX_CHECK(value <= std::numeric_limits<int64_t>::max());
    return static_cast<int64_t>(value);
  }

  void useBudget(int64_t requiredBudget) {
    availableBudget_ -= requiredBudget;
  }

  bool hasBudget(int64_t requiredBudget) {
    return updateBudget() >= requiredBudget;
  }

  int64_t updateBudget() {
    const auto newBudget = asSigned(budgetCallback_());
    if (newBudget != budget_) {
      availableBudget_ += newBudget - budget_;
      budget_ = newBudget;
    }
    return availableBudget_;
  }

  // The budget can only be > 0, but when we set a new budget, availableBudget_
  // can go negative, if the budget is already being used and there isn't enough
  // left.
  uint64_t budget_;
  int64_t availableBudget_;
  std::function<uint64_t()> budgetCallback_;
  std::condition_variable cv_;
  std::mutex mutex_;
  std::chrono::high_resolution_clock::duration lastDuration_;
};

} // namespace facebook::velox::dwio::common
