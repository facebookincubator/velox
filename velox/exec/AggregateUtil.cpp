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

namespace facebook::velox::exec {

void AggregateUtil::populateAggregateMask(
    const core::AggregationNode::Aggregate& aggregate,
    const RowTypePtr& inputType,
    AggregateInfo& info) {
  if (const auto& mask = aggregate.mask) {
    info.mask = inputType->asRow().getChildIdx(mask->name());
  } else {
    info.mask = std::nullopt;
  }
}

void AggregateUtil::populateAggregateKeysOrders(
    const core::AggregationNode::Aggregate& aggregate,
    const RowTypePtr& inputType,
    AggregateInfo& info) {
  const auto numSortingKeys = aggregate.sortingKeys.size();
  VELOX_CHECK_EQ(numSortingKeys, aggregate.sortingOrders.size());
  info.sortingOrders = aggregate.sortingOrders;

  info.sortingKeys.reserve(numSortingKeys);
  for (const auto& key : aggregate.sortingKeys) {
    info.sortingKeys.push_back(exprToChannel(key.get(), inputType));
  }
}

void AggregateUtil::populateAggregateFunction(
    const core::AggregationNode::Aggregate& aggregate,
    const RowTypePtr& outputType,
    core::AggregationNode::Step step,
    AggregateInfo& info,
    const std::unique_ptr<OperatorCtx>& operatorCtx,
    uint32_t index) {
  const auto& aggResultType = outputType->childAt(index);
  info.function = Aggregate::create(
      aggregate.call->name(),
      isPartialOutput(step) ? core::AggregationNode::Step::kPartial
                            : core::AggregationNode::Step::kSingle,
      aggregate.rawInputTypes,
      aggResultType,
      operatorCtx->driverCtx()->queryConfig());
}

} // namespace facebook::velox::exec
