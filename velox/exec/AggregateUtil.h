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

#include "velox/exec/AggregateInfo.h"
#include "velox/exec/Operator.h"

namespace facebook::velox::exec {

// The result of aggregation function registration.
struct AggregateRegistrationResult {
  bool mainFunction{false};
  bool partialFunction{false};
  bool mergeFunction{false};
  bool extractFunction{false};
  bool mergeExtractFunction{false};

  bool operator==(const AggregateRegistrationResult& other) const {
    return mainFunction == other.mainFunction &&
        partialFunction == other.partialFunction &&
        mergeFunction == other.mergeFunction &&
        extractFunction == other.extractFunction &&
        mergeExtractFunction == other.mergeExtractFunction;
  }
};

class AggregateUtil {
 public:
  static AggregateInfo toAggregateInfo(
      const core::AggregationNode::Aggregate& aggregate,
      const RowTypePtr& inputType,
      const RowTypePtr& outputType,
      core::AggregationNode::Step step,
      const std::unique_ptr<OperatorCtx>& operatorCtx,
      memory::MemoryPool* pool,
      uint32_t index,
      std::shared_ptr<core::ExpressionEvaluator>& expressionEvaluator,
      bool supportLambda = true);

  static std::vector<core::LambdaTypedExprPtr> extractLambdaInputs(
      const core::AggregationNode::Aggregate& aggregate);
};

} // namespace facebook::velox::exec
