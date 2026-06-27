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

#include "velox/experimental/cudf/benchmarks/TpchPlanRewriter.h"

#include "velox/connectors/hive/TableHandle.h"
#include "velox/core/Expressions.h"
#include "velox/core/PlanNode.h"
#include "velox/parse/PlanNodeIdGenerator.h"

namespace facebook::velox {

namespace {

int maxPlanNodeId(const core::PlanNodePtr& node) {
  int maxId = 0;
  try {
    maxId = std::stoi(node->id());
  } catch (...) {
  }
  for (const auto& src : node->sources()) {
    maxId = std::max(maxId, maxPlanNodeId(src));
  }
  return maxId;
}

core::PlanNodePtr withNewSources(
    const core::PlanNode& node,
    const std::vector<core::PlanNodePtr>& newSources) {
  if (auto* f = dynamic_cast<const core::FilterNode*>(&node)) {
    return core::FilterNode::Builder(*f).source(newSources[0]).build();
  }
  if (auto* p = dynamic_cast<const core::ProjectNode*>(&node)) {
    return core::ProjectNode::Builder(*p).source(newSources[0]).build();
  }
  if (auto* a = dynamic_cast<const core::AggregationNode*>(&node)) {
    return core::AggregationNode::Builder(*a).source(newSources[0]).build();
  }
  if (auto* lp = dynamic_cast<const core::LocalPartitionNode*>(&node)) {
    return core::LocalPartitionNode::Builder(*lp).sources(newSources).build();
  }
  if (auto* o = dynamic_cast<const core::OrderByNode*>(&node)) {
    return core::OrderByNode::Builder(*o).source(newSources[0]).build();
  }
  if (auto* t = dynamic_cast<const core::TopNNode*>(&node)) {
    return core::TopNNode::Builder(*t).source(newSources[0]).build();
  }
  if (auto* l = dynamic_cast<const core::LimitNode*>(&node)) {
    return core::LimitNode::Builder(*l).source(newSources[0]).build();
  }
  if (auto* h = dynamic_cast<const core::HashJoinNode*>(&node)) {
    return core::HashJoinNode::Builder(*h)
        .left(newSources[0])
        .right(newSources[1])
        .build();
  }
  if (auto* m = dynamic_cast<const core::MergeJoinNode*>(&node)) {
    return core::MergeJoinNode::Builder(*m)
        .left(newSources[0])
        .right(newSources[1])
        .build();
  }
  if (auto* e = dynamic_cast<const core::EnforceSingleRowNode*>(&node)) {
    return core::EnforceSingleRowNode::Builder(*e).source(newSources[0]).build();
  }
  if (auto* lm = dynamic_cast<const core::LocalMergeNode*>(&node)) {
    return core::LocalMergeNode::Builder(*lm).sources(newSources).build();
  }
  VELOX_FAIL(
      "Unhandled plan node type in TpchPlanRewriter: {}",
      node.name());
}

core::PlanNodePtr replaceTableScansWithValuesRecursive(
    const core::PlanNodePtr& node,
    const std::unordered_map<std::string, std::vector<RowVectorPtr>>&
        preloadedTables,
    std::shared_ptr<core::PlanNodeIdGenerator> idGen) {
  if (auto* scan = dynamic_cast<const core::TableScanNode*>(node.get())) {
    auto* hiveHandle =
        dynamic_cast<const connector::hive::HiveTableHandle*>(scan->tableHandle().get());
    VELOX_CHECK_NOT_NULL(
        hiveHandle,
        "TableScanNode must use HiveTableHandle for TPC-H plans");
    const auto& tableName = hiveHandle->tableName();

    auto it = preloadedTables.find(tableName);
    VELOX_CHECK(
        it != preloadedTables.end(),
        "Preloaded data not found for table: {}",
        tableName);

    const auto& vectors = it->second;
    VELOX_CHECK(!vectors.empty(), "Preloaded vectors empty for table: {}", tableName);

    auto valuesId = idGen->next();
    auto valuesNode = std::make_shared<core::ValuesNode>(
        valuesId, vectors, false, 1);

    const auto& outputType = scan->outputType();
    if (outputType->size() == 0) {
      return valuesNode;
    }

    std::vector<std::string> names;
    std::vector<core::TypedExprPtr> projections;
    for (size_t i = 0; i < outputType->size(); ++i) {
      names.push_back(outputType->nameOf(i));
      projections.push_back(std::make_shared<core::FieldAccessTypedExpr>(
          outputType->childAt(i), outputType->nameOf(i)));
    }

    auto projectId = idGen->next();
    return std::make_shared<core::ProjectNode>(
        projectId, names, projections, std::move(valuesNode));
  }

  std::vector<core::PlanNodePtr> newSources;
  bool changed = false;
  for (const auto& src : node->sources()) {
    auto newSrc =
        replaceTableScansWithValuesRecursive(src, preloadedTables, idGen);
    changed = changed || (newSrc.get() != src.get());
    newSources.push_back(std::move(newSrc));
  }

  if (!changed) {
    return node;
  }
  return withNewSources(*node, newSources);
}

} // namespace

core::PlanNodePtr replaceTableScansWithValues(
    const core::PlanNodePtr& plan,
    const std::unordered_map<std::string, std::vector<RowVectorPtr>>&
        preloadedTables) {
  auto idGen = std::make_shared<core::PlanNodeIdGenerator>(
      maxPlanNodeId(plan) + 1);
  return replaceTableScansWithValuesRecursive(plan, preloadedTables, idGen);
}

} // namespace facebook::velox
