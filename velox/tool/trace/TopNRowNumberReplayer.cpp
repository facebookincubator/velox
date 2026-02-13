/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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

#include "velox/tool/trace/TopNRowNumberReplayer.h"
#include "velox/common/Casts.h"
#include "velox/exec/OperatorTraceReader.h"
#include "velox/exec/tests/utils/PlanBuilder.h"

using namespace facebook::velox;
using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;

namespace facebook::velox::tool::trace {
core::PlanNodePtr TopNRowNumberReplayer::createPlanNode(
    const core::PlanNode* node,
    const core::PlanNodeId& nodeId,
    const core::PlanNodePtr& source) const {
  const auto* topNRowNumberNode =
      checkedPointerCast<const core::TopNRowNumberNode>(node);
  const auto generateRowNumber = topNRowNumberNode->generateRowNumber();
  return std::make_shared<core::TopNRowNumberNode>(
      nodeId,
      topNRowNumberNode->rankFunction(),
      topNRowNumberNode->partitionKeys(),
      topNRowNumberNode->sortingKeys(),
      topNRowNumberNode->sortingOrders(),
      generateRowNumber
          ? std::make_optional(topNRowNumberNode->outputType()->names().back())
          : std::nullopt,
      topNRowNumberNode->limit(),
      source);
}
} // namespace facebook::velox::tool::trace
