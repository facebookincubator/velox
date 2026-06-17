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

#include "velox/experimental/cxl/CxlDriverAdapter.h"

#include <memory>
#include <vector>

#include "velox/core/QueryConfig.h"
#include "velox/core/QueryCtx.h"
#include "velox/exec/Aggregate.h"
#include "velox/exec/Driver.h"
#include "velox/exec/HashAggregation.h"
#include "velox/exec/Task.h"
#include "velox/experimental/cxl/CxlHashAggregation.h"
#include "velox/experimental/cxl/CxlMemoryResource.h"

namespace facebook::velox::cxl {
namespace {

std::shared_ptr<const core::AggregationNode> findAggregationNode(
    const exec::DriverFactory& factory,
    const core::PlanNodeId& planNodeId) {
  for (const auto& node : factory.planNodes) {
    if (node->id() == planNodeId) {
      return std::dynamic_pointer_cast<const core::AggregationNode>(node);
    }
  }
  return nullptr;
}

// True if every grouping key is fixed-width and every accumulator is fixed-size
// without external (HashStringAllocator-backed) memory, so a row can be
// relocated DRAM -> CXL by a plain byte copy.
bool relocationIsSafe(
    const core::AggregationNode& node,
    const core::QueryConfig& config) {
  for (const auto& key : node.groupingKeys()) {
    if (!key->type()->isFixedWidth()) {
      return false;
    }
  }
  const auto numKeys = node.groupingKeys().size();
  const auto& outputType = node.outputType();
  const auto& aggregates = node.aggregates();
  for (auto i = 0; i < aggregates.size(); ++i) {
    auto function = exec::Aggregate::create(
        aggregates[i].call->name(),
        exec::isPartialOutput(node.step())
            ? core::AggregationNode::Step::kPartial
            : core::AggregationNode::Step::kSingle,
        aggregates[i].rawInputTypes,
        outputType->childAt(numKeys + i),
        config);
    if (function->accumulatorUsesExternalMemory()) {
      return false;
    }
  }
  return true;
}

// Replaces a HashAggregation with a CxlHashAggregation when its rows can be
// byte-copied to CXL; returns null for any other operator or unsupported
// aggregation.
std::unique_ptr<exec::Operator> replaceHashAggregation(
    exec::Operator* op,
    const exec::DriverFactory& factory,
    exec::DriverCtx* driverCtx) {
  auto* aggregation = dynamic_cast<exec::HashAggregation*>(op);
  if (aggregation == nullptr) {
    return nullptr;
  }
  auto node = findAggregationNode(factory, aggregation->planNodeId());
  if (node == nullptr) {
    return nullptr;
  }
  // CxlHashAggregation handles a grouped, complete (non-partial) aggregation
  // with no ignore-null-keys and byte-relocatable rows.
  if (node->groupingKeys().empty() || node->ignoreNullKeys() ||
      exec::isPartialOutput(node->step()) ||
      !relocationIsSafe(*node, driverCtx->queryConfig())) {
    return nullptr;
  }
  return std::make_unique<CxlHashAggregation>(
      aggregation->operatorId(), driverCtx, node);
}

bool adaptDriver(const exec::DriverFactory& factory, exec::Driver& driver) {
  auto* driverCtx = driver.driverCtx();
  // No CXL tier configured for this query: nothing to gain from replacing.
  if (driverCtx->task->queryCtx()->customPool(std::string{kCxlResourceTag}) ==
      nullptr) {
    return false;
  }
  const auto operators = driver.operators();
  bool replaced = false;
  for (size_t i = 0; i < operators.size(); ++i) {
    auto replacement = replaceHashAggregation(operators[i], factory, driverCtx);
    if (replacement == nullptr) {
      continue;
    }
    std::vector<std::unique_ptr<exec::Operator>> replacements;
    replacements.push_back(std::move(replacement));
    factory.replaceOperators(
        driver,
        static_cast<int32_t>(i),
        static_cast<int32_t>(i + 1),
        std::move(replacements));
    replaced = true;
  }
  return replaced;
}

} // namespace

void registerCxlDriverAdapter() {
  exec::DriverAdapter adapter{
      std::string{kCxlResourceTag}, /*inspect=*/{}, adaptDriver};
  exec::DriverFactory::registerAdapter(std::move(adapter));
}

} // namespace facebook::velox::cxl
