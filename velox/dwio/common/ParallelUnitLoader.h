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
#include <folly/futures/Future.h>

#include <vector>
#include "velox/dwio/common/UnitLoader.h"

namespace facebook::velox::dwio::common {
class ParallelUnitLoaderFactory : public UnitLoaderFactory {
 public:
  ParallelUnitLoaderFactory(
      folly::Executor* ioExecutor,
      size_t maxConcurrentLoads)
      : ioExecutor_(ioExecutor), maxConcurrentLoads_(maxConcurrentLoads) {}

  std::unique_ptr<UnitLoader> create(
      std::vector<std::unique_ptr<LoadUnit>> loadUnits,
      uint64_t rowsToSkip) override;

 private:
  folly::Executor* ioExecutor_;
  size_t maxConcurrentLoads_;
};

} // namespace facebook::velox::dwio::common
