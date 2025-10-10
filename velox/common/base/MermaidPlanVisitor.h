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
#include <type_traits>
#include <unordered_map>

#include "velox/core/PlanNode.h"
#include "velox/exec/PlanNodeStats.h"
#include "velox/exec/TaskStats.h"

namespace facebook::velox::common::base {

/// Velox query plan visitor to generate Mermaid diagrams.
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
  // Determines the direction of the mermaid diagram.
  enum class Direction {
    TopToBottom,
    BottomToTop,
    LeftToRight,
    RightToLeft,
  };

  enum class NodeShape {
    BOX,
    ROUND,
    STADIUM,
    SUBROUTINE,
  };

  using TaskStats = std::unordered_map<core::PlanNodeId, exec::PlanNodeStats>;

  /// Sets the diagram rendering direction.
  ///
  /// @param dir The rendering direction.
  /// @return The current instance (to allow chaining).
  [[maybe_unused]] MermaidPlanVisitor& setDirection(Direction dir);

  /// Sets the flag to enable (disable) rendering of Velox execution statistics.
  ///
  /// Note that this flag is relevant only if the diagram is rendered with the
  /// task statistics included.
  ///
  /// @param flag Flag to enable (disable) rendering of Velox execution
  /// statistics.
  /// @return The current instance (to allow chaining).
  [[maybe_unused]] MermaidPlanVisitor& includeTaskStats(bool flag);

  /// Builds the Mermaid diagram for the given Velox query plan
  ///
  /// @param plan The Velox query plan to visualize.
  /// @return Mermaid diagram corresponding with the query plan.
  [[nodiscard]] std::string build(const core::PlanNodePtr& plan);

  /// Builds the Mermaid diagram for the given Velox query plan and task stats.
  ///
  /// Note that including the task statistics in the Mermaid diagram may
  /// increase its size considerably and cause it to exceed the URL length
  /// limits making it impossible to share directly with a link.
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
  static constexpr std::string_view kColorOrange{"orange"};
  static constexpr std::string_view kColorYellow{"yellow"};
  static constexpr std::string_view kColorGreen{"green"};
  static constexpr std::string_view kColorGrey{"grey"};
  static constexpr std::string_view kColorRed{"red"};
  static constexpr std::string_view kColorPurple{"purple"};
  static constexpr std::string_view kColorCyan{"cyan"};
  static constexpr std::string_view kColorBlue{"blue"};

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
  ///
  /// @param dir The direction of the flowchart.
  void initDiagram();

  /// Generates a new unique node id for the diagram.
  ///
  /// @return Previously unused node id.
  std::string nextId() const;

  /// Appends a new node in the Mermaid diagram
  ///
  /// @param value The node value to render inside the box.
  /// @param className The class name to use for styling.
  void appendMermaidNode(
      const std::string& parentId,
      const std::string& nodeId,
      const std::string& value,
      std::string_view className,
      NodeShape shape = NodeShape::BOX) const;

  /// Checks if the given type parameter `T` is any of the given types `Ts` with
  /// cvref qualifiers removed.
  ///
  /// @tparam T The type to check.
  /// @tparam Ts The types to check against.
  template <typename T, typename... Ts>
  static constexpr bool is_any_of_v =
      (std::is_same_v<std::remove_cvref_t<T>, Ts> || ...);

  /// Determines a suitable class name for the given node type.
  ///
  /// The class name is used to style the Mermaid nodes and use colors to
  /// tell apart the different groups for easier reading of the diagram.
  ///
  /// @tparam T The Velox plan node type.
  /// @return Corresponding class name.
  template <typename T>
  [[nodiscard]] static std::string_view className() {
    if constexpr (is_any_of_v<T, core::TableScanNode, core::ValuesNode>) {
      return kColorGreen;
    } else if constexpr (is_any_of_v<T, core::AggregationNode>) {
      return kColorBlue;
    } else if constexpr (is_any_of_v<T, core::ProjectNode>) {
      return kColorYellow;
    } else if constexpr (is_any_of_v<
                             T,
                             core::LocalPartitionNode,
                             core::LocalMergeNode,
                             core::HashJoinNode,
                             core::MergeJoinNode,
                             core::NestedLoopJoinNode>) {
      return kColorPurple;
    } else if constexpr (is_any_of_v<T, core::FilterNode>) {
      return kColorOrange;
    } else if constexpr (is_any_of_v<T, core::WindowNode>) {
      return kColorRed;
    } else if constexpr (is_any_of_v<
                             T,
                             core::TopNRowNumberNode,
                             core::OrderByNode,
                             core::TopNNode>) {
      return kColorCyan;
    } else {
      return kColorGrey;
    }
  }

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

  // Flowchart direction
  Direction direction_{Direction::BottomToTop};

  // Flag to determine if task stats (if available) should be included.
  bool includeTaskStats_{true};

  // Task stats for the current query plan.
  TaskStats taskStats_;
};

} // namespace facebook::velox::common::base

// Formatter for Direction enum to support fmt::format
template <>
struct fmt::formatter<
    facebook::velox::common::base::MermaidPlanVisitor::Direction>
    : fmt::formatter<std::string_view> {
  template <typename FormatContext>
  auto format(
      facebook::velox::common::base::MermaidPlanVisitor::Direction dir,
      FormatContext& ctx) const {
    std::string_view name = "LR";
    switch (dir) {
      case facebook::velox::common::base::MermaidPlanVisitor::Direction::
          LeftToRight:
        name = "LR";
        break;
      case facebook::velox::common::base::MermaidPlanVisitor::Direction::
          RightToLeft:
        name = "RL";
        break;
      case facebook::velox::common::base::MermaidPlanVisitor::Direction::
          TopToBottom:
        name = "TD";
        break;
      case facebook::velox::common::base::MermaidPlanVisitor::Direction::
          BottomToTop:
        name = "BT";
        break;
    }
    return fmt::formatter<std::string_view>::format(name, ctx);
  }
};
