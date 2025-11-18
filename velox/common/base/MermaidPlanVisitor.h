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

#include <string>
#include <string_view>
#include <unordered_map>

#include "velox/core/PlanNode.h"
#include "velox/exec/PlanNodeStats.h"
#include "velox/exec/TaskStats.h"

namespace facebook::velox::common::base {

/// Velox query plan visitor to generate Mermaid diagrams
/// (https://mermaid.js.org/).
///
/// This visitor renders Velox query plans with optional task execution
/// statistics included. It uses the PlanNodeVisitor interface to properly
/// traverse the plan node tree.
///
/// Usage:
///   MermaidPlanVisitor visitor;
///   std::string planOnly = visitor.build(plan);
///   std::string planAndStats = visitor.build(plan, stats);
///
/// Mermaid renders charts as SVG and uses foreignObject to embed HTML
/// inside boxes. Details are rendered using HTML tables with inline styles.
class MermaidPlanVisitor : public core::PlanNodeVisitor {
 public:
  using TaskStats = std::unordered_map<core::PlanNodeId, exec::PlanNodeStats>;

  /// Builds the Mermaid diagram for the given Velox
  /// query plan.
  ///
  /// @param plan The Velox query plan to visualize.
  /// @return Mermaid diagram corresponding with the query plan.
  [[nodiscard]] std::string build(const core::PlanNodePtr& plan);

  /// Builds the Mermaid diagram for the given Velox query plan and task stats.
  ///
  /// @param plan The Velox query plan to visualize.
  /// @param stats The Velox task stats to include in the visualization.
  /// @return Mermaid diagram corresponding with the query plan.
  [[nodiscard]] std::string build(
      const core::PlanNodePtr& plan,
      const exec::TaskStats& stats);

  // PlanNodeVisitor implementation
  void visit(
      const core::AggregationNode& node,
      core::PlanNodeVisitorContext& ctx) const override;
  void visit(
      const core::ArrowStreamNode& node,
      core::PlanNodeVisitorContext& ctx) const override;
  void visit(
      const core::AssignUniqueIdNode& node,
      core::PlanNodeVisitorContext& ctx) const override;
  void visit(
      const core::EnforceSingleRowNode& node,
      core::PlanNodeVisitorContext& ctx) const override;
  void visit(const core::ExchangeNode& node, core::PlanNodeVisitorContext& ctx)
      const override;
  void visit(const core::ExpandNode& node, core::PlanNodeVisitorContext& ctx)
      const override;
  void visit(const core::FilterNode& node, core::PlanNodeVisitorContext& ctx)
      const override;
  void visit(const core::GroupIdNode& node, core::PlanNodeVisitorContext& ctx)
      const override;
  void visit(const core::HashJoinNode& node, core::PlanNodeVisitorContext& ctx)
      const override;
  void visit(
      const core::IndexLookupJoinNode& node,
      core::PlanNodeVisitorContext& ctx) const override;
  void visit(const core::LimitNode& node, core::PlanNodeVisitorContext& ctx)
      const override;
  void visit(
      const core::LocalMergeNode& node,
      core::PlanNodeVisitorContext& ctx) const override;
  void visit(
      const core::LocalPartitionNode& node,
      core::PlanNodeVisitorContext& ctx) const override;
  void visit(
      const core::MarkDistinctNode& node,
      core::PlanNodeVisitorContext& ctx) const override;
  void visit(
      const core::MergeExchangeNode& node,
      core::PlanNodeVisitorContext& ctx) const override;
  void visit(const core::MergeJoinNode& node, core::PlanNodeVisitorContext& ctx)
      const override;
  void visit(
      const core::NestedLoopJoinNode& node,
      core::PlanNodeVisitorContext& ctx) const override;
  void visit(
      const core::SpatialJoinNode& node,
      core::PlanNodeVisitorContext& ctx) const override;
  void visit(const core::OrderByNode& node, core::PlanNodeVisitorContext& ctx)
      const override;
  void visit(
      const core::PartitionedOutputNode& node,
      core::PlanNodeVisitorContext& ctx) const override;
  void visit(const core::ProjectNode& node, core::PlanNodeVisitorContext& ctx)
      const override;
  void visit(
      const core::ParallelProjectNode& node,
      core::PlanNodeVisitorContext& ctx) const override;
  void visit(const core::RowNumberNode& node, core::PlanNodeVisitorContext& ctx)
      const override;
  void visit(const core::TableScanNode& node, core::PlanNodeVisitorContext& ctx)
      const override;
  void visit(
      const core::TableWriteNode& node,
      core::PlanNodeVisitorContext& ctx) const override;
  void visit(
      const core::TableWriteMergeNode& node,
      core::PlanNodeVisitorContext& ctx) const override;
  void visit(const core::TopNNode& node, core::PlanNodeVisitorContext& ctx)
      const override;
  void visit(
      const core::TopNRowNumberNode& node,
      core::PlanNodeVisitorContext& ctx) const override;
  void visit(const core::TraceScanNode& node, core::PlanNodeVisitorContext& ctx)
      const override;
  void visit(const core::UnnestNode& node, core::PlanNodeVisitorContext& ctx)
      const override;
  void visit(const core::ValuesNode& node, core::PlanNodeVisitorContext& ctx)
      const override;
  void visit(const core::WindowNode& node, core::PlanNodeVisitorContext& ctx)
      const override;
  void visit(const core::PlanNode& node, core::PlanNodeVisitorContext& ctx)
      const override;

 private:
  /// Context class for maintaining visitor state during traversal
  class MermaidContext : public core::PlanNodeVisitorContext {
   public:
    MermaidContext(
        std::string& diagram,
        std::size_t& nodeId,
        const std::string& parentId)
        : diagram_(diagram), nodeId_(nodeId), parentId_(parentId) {}

    std::string& diagram_;
    std::size_t& nodeId_;
    std::string parentId_;
  };

  /// Initializes the Mermaid diagram with default settings.
  void initDiagram();

  /// Generates a new unique node id for the diagram.
  ///
  /// @return Previously unused node id.
  std::string nextId() const;

  /// Appends a new node to the Mermaid diagram.
  ///
  /// @param parentId The ID of the parent node to which this node will be
  /// connected.
  /// @param nodeId The unique ID for the new node in the diagram.
  /// @param value The label or content to display inside the node's box.
  /// @param className The CSS class name used for styling the node.
  void appendMermaidNode(
      const std::string& parentId,
      const std::string& nodeId,
      const std::string& value,
      std::string_view className) const;

  /// Default visitation logic for node types which we have not yet provided
  /// specific implementation for.
  template <typename T>
  [[nodiscard]] std::string visitNode(
      const T& node,
      const std::string& parentId) const;

  // Currently available node id.
  mutable std::size_t nodeId_{0};

  // The overall mermaid diagram being built.
  mutable std::string diagram_;

  // Task stats for the current query plan.
  TaskStats taskStats_;
};

} // namespace facebook::velox::common::base
