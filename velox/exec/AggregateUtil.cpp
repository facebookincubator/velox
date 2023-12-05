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

#include "velox/exec/AggregateUtil.h"
#include "velox/exec/Aggregate.h"
#include "velox/expression/Expr.h"

namespace facebook::velox::exec {

AggregateInfo AggregateUtil::toAggregateInfo(
    const core::AggregationNode::Aggregate& aggregate,
    const RowTypePtr& inputType,
    const RowTypePtr& outputType,
    core::AggregationNode::Step step,
    const std::unique_ptr<OperatorCtx>& operatorCtx,
    memory::MemoryPool* pool,
    uint32_t index,
    std::shared_ptr<core::ExpressionEvaluator>& expressionEvaluator,
    bool supportLambda) {
  AggregateInfo info;
  // Populate input.
  auto& channels = info.inputs;
  auto& constants = info.constantInputs;
  for (const auto& arg : aggregate.call->inputs()) {
    if (auto field =
            dynamic_cast<const core::FieldAccessTypedExpr*>(arg.get())) {
      channels.push_back(inputType->getChildIdx(field->name()));
      constants.push_back(nullptr);
    } else if (
        auto constant =
            dynamic_cast<const core::ConstantTypedExpr*>(arg.get())) {
      channels.push_back(kConstantChannel);
      constants.push_back(constant->toConstantVector(pool));
    } else if (
        auto lambda = dynamic_cast<const core::LambdaTypedExpr*>(arg.get())) {
      VELOX_USER_CHECK(
          supportLambda,
          "Streaming aggregation doesn't support lambda functions yet");
      for (const auto& name : lambda->signature()->names()) {
        if (auto captureIndex = inputType->getChildIdxIfExists(name)) {
          channels.push_back(captureIndex.value());
          constants.push_back(nullptr);
        }
      }
    } else {
      VELOX_FAIL(
          "Expression must be field access, constant, or lambda: {}",
          arg->toString());
    }
  }

  info.distinct = aggregate.distinct;
  info.intermediateType = Aggregate::intermediateType(
      aggregate.call->name(), aggregate.rawInputTypes);

  // Setup aggregation mask: convert the Variable Reference name to the
  // channel (projection) index, if there is a mask.
  if (const auto& mask = aggregate.mask) {
    info.mask = inputType->asRow().getChildIdx(mask->name());
  } else {
    info.mask = std::nullopt;
  }

  const auto& aggResultType = outputType->childAt(index);
  info.function = Aggregate::create(
      aggregate.call->name(),
      isPartialOutput(step) ? core::AggregationNode::Step::kPartial
                            : core::AggregationNode::Step::kSingle,
      aggregate.rawInputTypes,
      aggResultType,
      operatorCtx->driverCtx()->queryConfig());

  if (supportLambda) {
    auto lambdas = extractLambdaInputs(aggregate);
    if (!lambdas.empty()) {
      if (expressionEvaluator == nullptr) {
        expressionEvaluator = std::make_shared<SimpleExpressionEvaluator>(
            operatorCtx->execCtx()->queryCtx(), operatorCtx->execCtx()->pool());
      }
      info.function->setLambdaExpressions(lambdas, expressionEvaluator);
    }
  }

  // Sorting keys and orders.
  const auto numSortingKeys = aggregate.sortingKeys.size();
  VELOX_CHECK_EQ(numSortingKeys, aggregate.sortingOrders.size());
  info.sortingOrders = aggregate.sortingOrders;
  info.sortingKeys.reserve(numSortingKeys);
  for (const auto& key : aggregate.sortingKeys) {
    info.sortingKeys.push_back(exprToChannel(key.get(), inputType));
  }

  info.output = index;

  return info;
}

std::vector<core::LambdaTypedExprPtr> AggregateUtil::extractLambdaInputs(
    const core::AggregationNode::Aggregate& aggregate) {
  std::vector<core::LambdaTypedExprPtr> lambdas;
  for (const auto& arg : aggregate.call->inputs()) {
    if (auto lambda =
            std::dynamic_pointer_cast<const core::LambdaTypedExpr>(arg)) {
      lambdas.push_back(lambda);
    }
  }

  return lambdas;
}

} // namespace facebook::velox::exec
