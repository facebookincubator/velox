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

#include <folly/experimental/coro/BlockingWait.h>
#include <folly/experimental/coro/Task.h>
#include <iostream>

#include "velox/expression/VectorFunction.h"
#include "velox/type/Type.h"

namespace facebook::velox::exec {
class AsyncVectorFunction : public VectorFunction {
 public:
  virtual folly::coro::Task<void> applyAsync(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& outputType,
      EvalCtx* context,
      VectorPtr* result) const = 0;

  // Calls `applyAsyn` and block.
  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& outputType,
      EvalCtx* context,
      VectorPtr* result) const override {
    folly::coro::blockingWait(
        applyAsync(rows, args, outputType, context, result));
  }

  virtual bool threadSafe() {
    return false;
  }
};

} // namespace facebook::velox::exec
