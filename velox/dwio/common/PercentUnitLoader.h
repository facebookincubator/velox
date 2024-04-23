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

#include "folly/Executor.h"
#include "velox/dwio/common/Budget.h"
#include "velox/dwio/common/UnitLoader.h"
#include "velox/dwio/common/UnitLoaderTools.h"

namespace facebook::velox::dwio::common {

class PercentUnitLoaderFactory : public velox::dwio::common::UnitLoaderFactory {
 public:
  PercentUnitLoaderFactory(
      uint32_t percent,
      std::shared_ptr<velox::dwio::common::Budget> memoryBudget,
      folly::Executor& firstLoadExecutor,
      folly::Executor& loadExecutor,
      std::function<void()> noMoreIoCallback,
      std::function<void(std::chrono::high_resolution_clock::duration)>
          blockedOnIoCallback,
      std::function<void(std::chrono::high_resolution_clock::duration)>
          blockedOnBudgetCallback);

  ~PercentUnitLoaderFactory() override = default;

  std::unique_ptr<velox::dwio::common::UnitLoader> create(
      std::vector<std::unique_ptr<velox::dwio::common::LoadUnit>> loadUnits)
      override;

 private:
  uint32_t percent_;
  std::shared_ptr<velox::dwio::common::Budget> memoryBudget_;
  folly::Executor& firstLoadExecutor_;
  folly::Executor& loadExecutor_;
  std::function<void(std::chrono::high_resolution_clock::duration)>
      blockedOnIoCallback_;
  std::function<void(std::chrono::high_resolution_clock::duration)>
      blockedOnBudgetCallback_;
  std::shared_ptr<std::atomic_size_t> noMoreIoCallbacksLeftToTrigger_;
  facebook::velox::dwio::common::unit_loader_tools::CallbackOnLastSignal
      callbackOnLastSignal_;
};

} // namespace facebook::velox::dwio::common
