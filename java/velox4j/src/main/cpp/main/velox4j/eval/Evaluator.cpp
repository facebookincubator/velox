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
#include "velox4j/eval/Evaluator.h"

#include <fmt/core.h>
#include <folly/json/json.h>
#include <stdint.h>
#include <velox/common/base/Exceptions.h>
#include <velox/common/caching/AsyncDataCache.h>
#include <velox/common/memory/MemoryPool.h>
#include <velox/common/serialization/Serializable.h>
#include <velox/core/QueryConfig.h>
#include <velox/core/QueryCtx.h>
#include <velox/expression/Expr.h>
#include <velox/vector/BaseVector.h>
#include <atomic>
#include <utility>

#include "Evaluation.h"
#include "velox4j/conf/Config.h"
#include "velox4j/memory/MemoryManager.h"

namespace facebook::velox4j {
using namespace facebook::velox;

Evaluator::Evaluator(
    MemoryManager* memoryManager,
    const std::shared_ptr<const Evaluation>& evaluation)
    : evaluation_(evaluation) {
  static std::atomic<uint32_t> nextExecutionId{0};
  const uint32_t executionId = nextExecutionId++;
  queryCtx_ = core::QueryCtx::create(
      nullptr,
      core::QueryConfig{evaluation_->queryConfig()->toMap()},
      evaluation_->connectorConfig()->toMap(),
      cache::AsyncDataCache::getInstance(),
      memoryManager
          ->getVeloxPool(
              fmt::format(
                  "Evaluator Memory Pool - Execution ID {}",
                  std::to_string(executionId)),
              memory::MemoryPool::Kind::kAggregate)
          ->shared_from_this(),
      nullptr,
      fmt::format(
          "Evaluator Context - Execution ID {}", std::to_string(executionId)));
  expressionEvaluator_ = std::make_unique<exec::SimpleExpressionEvaluator>(
      queryCtx_.get(),
      memoryManager->getVeloxPool(
          fmt::format(
              "Evaluator Leaf Memory Pool - Execution ID {}",
              std::to_string(executionId)),
          memory::MemoryPool::Kind::kLeaf));
  exprSet_ = expressionEvaluator_->compile(evaluation_->expr());
}

VectorPtr Evaluator::eval(
    const SelectivityVector& rows,
    const RowVector& input) {
  VectorPtr vector{};
  expressionEvaluator_->evaluate(exprSet_.get(), rows, input, vector);
  VELOX_CHECK_NOT_NULL(vector, "Failed to evaluate expression");
  return vector;
}

} // namespace facebook::velox4j
