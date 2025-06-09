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
#include "Evaluator.h"

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

Evaluator::Evaluator(MemoryManager* memoryManager, const std::string& exprJson)
    : exprJson_(exprJson) {
  static std::atomic<uint32_t> executionId{0};
  const uint32_t eid = executionId++;
  auto evaluatorSerdePool = memoryManager->getVeloxPool(
      fmt::format("Evaluator Serde Memory Pool - EID {}", std::to_string(eid)),
      memory::MemoryPool::Kind::kLeaf);
  auto exprDynamic = folly::parseJson(exprJson_);
  auto expr =
      ISerializable::deserialize<Evaluation>(exprDynamic, evaluatorSerdePool);
  queryCtx_ = core::QueryCtx::create(
      nullptr,
      core::QueryConfig{expr->queryConfig()->toMap()},
      expr->connectorConfig()->toMap(),
      cache::AsyncDataCache::getInstance(),
      memoryManager
          ->getVeloxPool(
              fmt::format(
                  "Evaluator Memory Pool - EID {}", std::to_string(eid)),
              memory::MemoryPool::Kind::kAggregate)
          ->shared_from_this(),
      nullptr,
      fmt::format("Evaluator Context - EID {}", std::to_string(eid)));
  expressionEvaluator_ = std::make_unique<exec::SimpleExpressionEvaluator>(
      queryCtx_.get(),
      memoryManager->getVeloxPool(
          fmt::format(
              "Evaluator Leaf Memory Pool - EID {}", std::to_string(eid)),
          memory::MemoryPool::Kind::kLeaf));
  exprSet_ = expressionEvaluator_->compile(expr->expr());
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
