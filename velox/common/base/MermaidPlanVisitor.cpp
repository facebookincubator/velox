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

#include "velox/common/base/MermaidPlanVisitor.h"

#include <fmt/chrono.h>
#include <fmt/core.h>
#include <fmt/format.h>
#include <fmt/ranges.h>
#include <folly/base64.h>
#include <folly/logging/xlog.h>
#include <range/v3/range/conversion.hpp>
#include <range/v3/view/enumerate.hpp>
#include <range/v3/view/iota.hpp>
#include <range/v3/view/transform.hpp>
#include <range/v3/view/zip.hpp>
#include <re2/re2.h>
#include <string>
#include <string_view>
#include <unordered_map>

#include "velox/common/base/RuntimeMetrics.h"
#include "velox/common/base/SuccinctPrinter.h"
#include "velox/core/PlanNode.h"
#include "velox/exec/PlanNodeStats.h"
#include "velox/exec/TaskStats.h"

using facebook::velox::succinctBytes;
using facebook::velox::succinctNanos;

namespace facebook::velox::common::base {

// Import core types for convenience
using namespace facebook::velox::core;

namespace {

/// Converts a string to uppercase.
///
/// @param str The string to convert.
/// @return Uppercased string.
std::string toUpper(std::string_view str) {
  return str //
      |
      ranges::views::transform([](const char c) { return std::toupper(c); }) //
      | ranges::to<std::string>;
}

/// Escapes HTML entities for Mermaid diagram embedding.
std::string escapeEntities(std::string&& str) {
  static const re2::RE2 lessThanRe{R"(<)"};
  static const re2::RE2 greaterThanRe{R"(>)"};

  re2::RE2::GlobalReplace(&str, lessThanRe, "#lt;");
  re2::RE2::GlobalReplace(&str, greaterThanRe, "#gt;");

  return str;
}

/// Creates a table <caption> element.
std::string caption(
    std::string_view name,
    std::string_view subtitle = std::string_view{}) {
  return fmt::format(
      "<caption style='font-size:1.2em;font-weight:bold'>{}{}</caption>",
      // toUpper(name),
      name,
      subtitle.empty() ? "" : fmt::format(" ({})", subtitle));
}

std::string section(std::string_view title, size_t colspan = 1) {
  if (title.empty()) {
    return "";
  }

  return fmt::format(
      "<tr><td colspan={} style='font-weight:bold;font-style:italic;color:#000000'>{}</td></tr>",
      colspan,
      toUpper(title));
}

std::string outputTypes(const RowTypePtr& rowType) {
  return fmt::format(
      "<table>"
      "{}"
      "{}"
      "</table>",
      section("Output types", 2),
      fmt::join(
          ranges::views::iota(
              static_cast<decltype(rowType->size())>(0), rowType->size()) //
              | ranges::views::transform([&rowType](auto index) {
                  return fmt::format(
                      "<tr><td>{}</td><td>{}</td></tr>",
                      rowType->nameOf(index),
                      escapeEntities(rowType->childAt(index)->toString()));
                }),
          ""));
}

template <typename T>
std::string sortingKeys(const T& node) {
  return fmt::format(
      "{}"
      "{}",
      section("Sorting keys", 2),
      fmt::join(
          ranges::views::zip(node.sortingKeys(), node.sortingOrders()) //
              | ranges::views::transform([](const auto& pair) {
                  const auto& [key, order] = pair;
                  return fmt::format(
                      "<tr><td>{}</td><td>{}</td></tr>",
                      key->name(),
                      order.toString());
                }),
          ""));
}

std::string windowFunctions(
    const std::vector<WindowNode::Function>& windowFunctions) {
  return fmt::format(
      "{}"
      "{}",
      section("Window functions", 2),
      fmt::join(
          windowFunctions | ranges::views::transform([&](const auto& function) {
            return fmt::format(
                "<tr>"
                "<td>Function call</td>"
                "<td>{}</td>"
                "</tr>"
                "<tr>"
                "<td>Frame type</td>"
                "<td>{}</td>"
                "</tr>"
                "<tr>"
                "<td>Frame start type</td>"
                "<td>{}</td>"
                "</tr>"
                "<td>Frame start value</td>"
                "<td>{}</td>"
                "</tr>"
                "<tr>"
                "<td>Frame end type</td>"
                "<td>{}</td>"
                "</tr>"
                "<tr>"
                "<td>Frame end value</td>"
                "<td>{}</td>"
                "</tr>",
                function.functionCall ? function.functionCall->toString() : "",
                WindowNode::toName(function.frame.type),
                WindowNode::toName(function.frame.startType),
                function.frame.startValue
                    ? function.frame.startValue->toString()
                    : "",
                WindowNode::toName(function.frame.endType),
                function.frame.endValue ? function.frame.endValue->toString()
                                        : "");
          }),
          ""));
}

std::string fieldsAsRows(
    const std::vector<FieldAccessTypedExprPtr>& fields,
    const std::string& heading,
    size_t colspan = 1) {
  if (fields.empty()) {
    return "";
  }

  return fmt::format(
      "{}"
      "{}",
      section(heading, colspan),
      fmt::join(
          fields | ranges::views::transform([&](const auto& field) {
            return fmt::format(
                "<tr><td colspan={}>{}</td></tr>", colspan, field->name());
          }),
          ""));
}

std::string fullRow(std::string_view contents, size_t colspan = 1) {
  if (contents.empty()) {
    return "";
  }

  return colspan > 1
      ? fmt::format("<tr><td colspan={}>{}</td></tr>", colspan, contents)
      : fmt::format("<tr><td>{}</td></tr>", contents);
}

std::string customStats(
    const std::unordered_map<std::string, velox::RuntimeMetric>& stats) {
  if (stats.empty()) {
    return "";
  }

  const auto fmtValue = [](int64_t value,
                           velox::RuntimeCounter::Unit unit) -> std::string {
    switch (unit) {
      case velox::RuntimeCounter::Unit::kBytes:
        return succinctBytes(value);
      case velox::RuntimeCounter::Unit::kNanos:
        return succinctNanos(value);
      default:
        return fmt::format("{}", value);
    }
  };

  return fmt::format(
      "<table style='width:100%'>"
      "<tr><td></td><th>count</th><th>sum</th><th>min</th><th>max</th></tr>"
      "{}"
      "</table>",
      fmt::join(
          stats | ranges::views::transform([&](const auto& pair) {
            const auto& [op, metric] = pair;
            return fmt::format(
                "<tr>"
                "<td>{}</td>"
                "<td>{}</td>"
                "<td>{}</td>"
                "<td>{}</td>"
                "<td>{}</td>"
                "</tr>",
                op,
                metric.count,
                fmtValue(metric.sum, metric.unit),
                fmtValue(metric.min, metric.unit),
                fmtValue(metric.max, metric.unit));
          }),
          ""));
}

std::string planNodeStats(const exec::PlanNodeStats& stats) {
  return fmt::format(
      "<table style='width:100%'>"
      "<caption style='font-weight:bold'>Task statistics</caption>"
      "{}"
      "<tr>"
      "<td>Input</td>"
      "<td>{} rows ({}, {} batches)</td>"
      "<td>Blocked wall time</td>"
      "<td>{}</td>"
      "</tr>"
      "<tr>"
      "<td>Raw input</td>"
      "<td>{} rows ({})</td>"
      "<td>CPU time</td>"
      "<td>{}</td>"
      "</tr>"
      "<tr>"
      "<td>Output</td>"
      "<td>{} rows ({}, {} batches)</td>"
      "<td>Wall time</td>"
      "<td>{}</td>"
      "</tr>"
      "<tr>"
      "<td>Peak memory</td>"
      "<td>{}</td>"
      "<td>Mem allocations</td>"
      "<td>{}</td>"
      "</tr>"
      "<tr>"
      "<td>Drivers</td>"
      "<td>{}</td>"
      "<td>Splits</td>"
      "<td>{}</td>"
      "</tr>"
      "<tr>"
      "<td>Physical written</td>"
      "<td>{}</td>"
      "<td>Scheduled time</td>"
      "<td>{}</td>"
      "</tr>"
      "{}"
      "<tr>"
      "<td>Input spilled</td>"
      "<td>{}</td>"
      "<td>Spilled bytes</td>"
      "<td>{}</td>"
      "</tr>"
      "<tr>"
      "<td>Spilled rows</td>"
      "<td>{}</td>"
      "<td>Spilled files</td>"
      "<td>{}</td>"
      "</tr>"
      "<tr>"
      "<td>Spilled partitions</td>"
      "<td>{}</td>"
      "<td>Background time</td>"
      "<td>{}</td>"
      "</tr>"
      "{}"
      "{}"
      "{}"
      "{}"
      "{}"
      "{}"
      "</table>",
      section("Node stats", 4),
      // Input row
      stats.inputRows,
      succinctBytes(stats.inputBytes),
      stats.inputVectors,
      succinctNanos(stats.blockedWallNanos),
      // Raw input row
      stats.rawInputRows,
      succinctBytes(stats.rawInputBytes),
      succinctNanos(stats.cpuWallTiming.cpuNanos),
      // Output row
      stats.outputRows,
      succinctBytes(stats.outputBytes),
      stats.outputVectors,
      succinctNanos(stats.cpuWallTiming.wallNanos),
      // Memory row
      succinctBytes(stats.peakMemoryBytes),
      stats.numMemoryAllocations,
      // Drivers row
      stats.numDrivers,
      stats.numSplits,
      // Physical written row
      succinctBytes(stats.physicalWrittenBytes),
      succinctNanos(
          stats.cpuWallTiming.wallNanos - stats.cpuWallTiming.cpuNanos),
      // Spill section
      section("Spill stats", 4),
      // Spilled input bytes row
      succinctBytes(stats.spilledInputBytes),
      succinctBytes(stats.spilledBytes),
      // Spilled rows row
      stats.spilledRows,
      stats.spilledFiles,
      // Spilled partitions row
      stats.spilledPartitions,
      succinctNanos(stats.backgroundTiming.cpuNanos),
      // Timing section
      section("Timing breakdown", 4),
      fullRow(
          fmt::format(
              "Add input: {} | Get output: {} | Finish: {} | Is blocked: {}",
              succinctNanos(stats.addInputTiming.cpuNanos),
              succinctNanos(stats.getOutputTiming.cpuNanos),
              succinctNanos(stats.finishTiming.cpuNanos),
              succinctNanos(stats.isBlockedTiming.cpuNanos)),
          4),
      // Dynamic filter stats
      stats.dynamicFilterStats.empty()
          ? ""
          : fullRow(
                fmt::format(
                    "Dynamic filters: {} producers",
                    stats.dynamicFilterStats.producerNodeIds.size()),
                4),
      // Expression stats
      stats.expressionStats.empty() ? "" : section("Expression stats", 4),
      // Operator stats section
      section("Operator stats", 4),
      fullRow(fmt::format("{}", customStats(stats.customStats)), 4));
}

} // namespace

MermaidPlanVisitor& MermaidPlanVisitor::setDirection(Direction dir) {
  direction_ = dir;
  return *this;
}

MermaidPlanVisitor& MermaidPlanVisitor::includeTaskStats(bool flag) {
  includeTaskStats_ = flag;
  return *this;
}

std::string MermaidPlanVisitor::build(const PlanNodePtr& plan) {
  initDiagram();
  MermaidContext ctx(diagram_, nodeId_, "");
  plan->accept(*this, ctx);
  return diagram_;
}

std::string MermaidPlanVisitor::build(
    const PlanNodePtr& plan,
    const exec::TaskStats& stats) {
  taskStats_ = exec::toPlanStats(stats);

  initDiagram();
  MermaidContext ctx(diagram_, nodeId_, "");
  plan->accept(*this, ctx);
  return diagram_;
}

std::string MermaidPlanVisitor::nextId() const {
  return fmt::format("node{}", nodeId_++);
}

void MermaidPlanVisitor::initDiagram() {
  if (diagram_.empty()) {
    diagram_ = fmt::format(
        "%%{{init: {{\"flowchart\": {{\"nodeSpacing\": 25}}}}}}%%\n\n"
        "flowchart {};\n"
        "classDef {} fill:#F5DBBF,stroke:#953D24,color:#000000;\n"
        "classDef {} fill:#FDF7E0,stroke:#A8601D,color:#000000;\n"
        "classDef {} fill:#CCE1CD,stroke:#2A5E4B,color:#000000;\n"
        "classDef {} fill:#F3F5F8,stroke:#526170,color:#000000;\n"
        "classDef {} fill:#F4D6CF,stroke:#701F18,color:#000000;\n"
        "classDef {} fill:#D9CDE2,stroke:#4D1A56,color:#000000;\n"
        "classDef {} fill:#C8E1DF,stroke:#2C575E,color:#000000;\n"
        "classDef {} fill:#D2E7F9,stroke:#2D6091,color:#000000;\n"
        "linkStyle default stroke:#737F90,stroke-width:1px;\n"
        "\n",
        fmt::format("{}", direction_),
        kColorOrange,
        kColorYellow,
        kColorGreen,
        kColorGrey,
        kColorRed,
        kColorPurple,
        kColorCyan,
        kColorBlue);
  }
}

void MermaidPlanVisitor::appendMermaidNode(
    const std::string& parentId,
    const std::string& nodeId,
    const std::string& value,
    std::string_view className,
    NodeShape shape) const {
  static const re2::RE2 dblQuoteRe{R"(")"};

  std::string node;
  std::string stringValue = fmt::format("{}", value);
  re2::RE2::GlobalReplace(&stringValue, dblQuoteRe, "#quot;");

  switch (shape) {
    case NodeShape::STADIUM:
      node =
          fmt::format("{}([\"{}\"]):::{};\n", nodeId, stringValue, className);
      break;

    case NodeShape::ROUND:
      node = fmt::format("{}(\"{}\"):::{};\n", nodeId, stringValue, className);
      break;

    case NodeShape::SUBROUTINE:
      node =
          fmt::format("{}[[\"{}\"]]:::{};\n", nodeId, stringValue, className);
      break;

    case NodeShape::BOX:
    default:
      node = fmt::format("{}[\"{}\"]:::{};\n", nodeId, stringValue, className);
      break;
  }

  std::string_view arrow = "---";
  if (parentId.empty()) {
    // Root node
    diagram_ += node;
  } else {
    diagram_ += fmt::format("{} {} {}", parentId, arrow, node);
  }
}

template <typename T>
std::string MermaidPlanVisitor::visitNode(
    const T& node,
    const std::string& parentId) const {
  std::string currentNodeId = nextId();
  appendMermaidNode(
      parentId, currentNodeId, toUpper(node.name()), className<T>());
  return currentNodeId;
}

// PlanNodeVisitor implementations
void MermaidPlanVisitor::visit(
    const AggregationNode& node,
    PlanNodeVisitorContext& ctx) const {
  auto& mctx = static_cast<MermaidContext&>(ctx);
  std::string nodeId = nextId();
  std::string contents = fmt::format(
      "<table>"
      "{}"
      "{}"
      "{}"
      "{}"
      "{}"
      "{}"
      "</table>",
      caption(node.name(), AggregationNode::toName(node.step())),
      fieldsAsRows(node.groupingKeys(), "Grouping keys", 2),
      section("Aggregates", 2),
      fmt::join(
          ranges::views::zip(node.aggregateNames(), node.aggregates()) //
              | ranges::views::transform([](const auto& pair) {
                  const auto& [name, aggregate] = pair;
                  return fmt::format(
                      "<tr><td>{}</td><td>{}</td></tr>",
                      name,
                      aggregate.call->toString());
                }),
          ""),
      fullRow(outputTypes(node.outputType()), 2),
      fullRow(
          includeTaskStats_ && taskStats_.count(node.id())
              ? planNodeStats(taskStats_.at(node.id()))
              : "",
          2));

  appendMermaidNode(
      mctx.parentId_, nodeId, contents, className<AggregationNode>());

  // Visit sources
  for (const auto& source : node.sources()) {
    MermaidContext childCtx(mctx.diagram_, mctx.nodeId_, nodeId);
    source->accept(*this, childCtx);
  }
}

void MermaidPlanVisitor::visit(
    const ArrowStreamNode& node,
    PlanNodeVisitorContext& ctx) const {
  auto& mctx = static_cast<MermaidContext&>(ctx);
  std::string nodeId = visitNode(node, mctx.parentId_);

  // Visit sources
  for (const auto& source : node.sources()) {
    MermaidContext childCtx(mctx.diagram_, mctx.nodeId_, nodeId);
    source->accept(*this, childCtx);
  }
}

void MermaidPlanVisitor::visit(
    const AssignUniqueIdNode& node,
    PlanNodeVisitorContext& ctx) const {
  auto& mctx = static_cast<MermaidContext&>(ctx);
  std::string nodeId = visitNode(node, mctx.parentId_);

  // Visit sources
  for (const auto& source : node.sources()) {
    MermaidContext childCtx(mctx.diagram_, mctx.nodeId_, nodeId);
    source->accept(*this, childCtx);
  }
}

void MermaidPlanVisitor::visit(
    const EnforceSingleRowNode& node,
    PlanNodeVisitorContext& ctx) const {
  auto& mctx = static_cast<MermaidContext&>(ctx);
  std::string nodeId = visitNode(node, mctx.parentId_);

  // Visit sources
  for (const auto& source : node.sources()) {
    MermaidContext childCtx(mctx.diagram_, mctx.nodeId_, nodeId);
    source->accept(*this, childCtx);
  }
}

void MermaidPlanVisitor::visit(
    const ExchangeNode& node,
    PlanNodeVisitorContext& ctx) const {
  auto& mctx = static_cast<MermaidContext&>(ctx);
  std::string nodeId = visitNode(node, mctx.parentId_);

  // Visit sources
  for (const auto& source : node.sources()) {
    MermaidContext childCtx(mctx.diagram_, mctx.nodeId_, nodeId);
    source->accept(*this, childCtx);
  }
}

void MermaidPlanVisitor::visit(
    const ExpandNode& node,
    PlanNodeVisitorContext& ctx) const {
  auto& mctx = static_cast<MermaidContext&>(ctx);
  std::string nodeId = visitNode(node, mctx.parentId_);

  // Visit sources
  for (const auto& source : node.sources()) {
    MermaidContext childCtx(mctx.diagram_, mctx.nodeId_, nodeId);
    source->accept(*this, childCtx);
  }
}

void MermaidPlanVisitor::visit(
    const FilterNode& node,
    PlanNodeVisitorContext& ctx) const {
  auto& mctx = static_cast<MermaidContext&>(ctx);
  std::string nodeId = nextId();
  std::string contents = fmt::format(
      "<table>"
      "{}"
      "{}"
      "{}"
      "{}"
      "{}"
      "</table>",
      caption(node.name()),
      section("Expression"),
      fullRow(node.filter()->toString()),
      fullRow(outputTypes(node.outputType())),
      fullRow(
          includeTaskStats_ && taskStats_.count(node.id())
              ? planNodeStats(taskStats_.at(node.id()))
              : ""));

  appendMermaidNode(mctx.parentId_, nodeId, contents, className<FilterNode>());

  // Visit sources
  for (const auto& source : node.sources()) {
    MermaidContext childCtx(mctx.diagram_, mctx.nodeId_, nodeId);
    source->accept(*this, childCtx);
  }
}

void MermaidPlanVisitor::visit(
    const GroupIdNode& node,
    PlanNodeVisitorContext& ctx) const {
  auto& mctx = static_cast<MermaidContext&>(ctx);
  std::string nodeId = visitNode(node, mctx.parentId_);

  // Visit sources
  for (const auto& source : node.sources()) {
    MermaidContext childCtx(mctx.diagram_, mctx.nodeId_, nodeId);
    source->accept(*this, childCtx);
  }
}

void MermaidPlanVisitor::visit(
    const HashJoinNode& node,
    PlanNodeVisitorContext& ctx) const {
  auto& mctx = static_cast<MermaidContext&>(ctx);
  std::string nodeId = nextId();
  std::string contents = fmt::format(
      "<table>"
      "{}"
      "{}"
      "{}"
      "{}"
      "{}"
      "</table>",
      caption(node.name(), JoinTypeName::toName(node.joinType())),
      fmt::join(
          ranges::views::zip(node.leftKeys(), node.rightKeys()) //
              | ranges::views::transform([](const auto& join) {
                  const auto& [left, right] = join;
                  return fmt::format(
                      "<tr><td>{}</td><td>= {}</td></tr>",
                      left->name(),
                      right->name());
                }),
          ""),
      fullRow(node.filter() ? node.filter()->toString() : "", 2),
      fullRow(outputTypes(node.outputType()), 2),
      fullRow(
          includeTaskStats_ && taskStats_.count(node.id())
              ? planNodeStats(taskStats_.at(node.id()))
              : "",
          2));

  appendMermaidNode(
      mctx.parentId_, nodeId, contents, className<HashJoinNode>());

  // Visit sources
  for (const auto& source : node.sources()) {
    MermaidContext childCtx(mctx.diagram_, mctx.nodeId_, nodeId);
    source->accept(*this, childCtx);
  }
}

void MermaidPlanVisitor::visit(
    const IndexLookupJoinNode& node,
    PlanNodeVisitorContext& ctx) const {
  auto& mctx = static_cast<MermaidContext&>(ctx);
  std::string nodeId = visitNode(node, mctx.parentId_);

  // Visit sources
  for (const auto& source : node.sources()) {
    MermaidContext childCtx(mctx.diagram_, mctx.nodeId_, nodeId);
    source->accept(*this, childCtx);
  }
}

void MermaidPlanVisitor::visit(
    const LimitNode& node,
    PlanNodeVisitorContext& ctx) const {
  auto& mctx = static_cast<MermaidContext&>(ctx);
  std::string nodeId = visitNode(node, mctx.parentId_);

  // Visit sources
  for (const auto& source : node.sources()) {
    MermaidContext childCtx(mctx.diagram_, mctx.nodeId_, nodeId);
    source->accept(*this, childCtx);
  }
}

void MermaidPlanVisitor::visit(
    const LocalMergeNode& node,
    PlanNodeVisitorContext& ctx) const {
  auto& mctx = static_cast<MermaidContext&>(ctx);
  std::string nodeId = nextId();
  std::string contents = fmt::format(
      "<table>"
      "{}"
      "{}"
      "{}"
      "{}"
      "</table>",
      caption(node.name()),
      sortingKeys(node),
      fullRow(outputTypes(node.outputType()), 2),
      fullRow(
          includeTaskStats_ && taskStats_.count(node.id())
              ? planNodeStats(taskStats_.at(node.id()))
              : "",
          2));

  appendMermaidNode(
      mctx.parentId_, nodeId, contents, className<LocalMergeNode>());

  // Visit sources
  for (const auto& source : node.sources()) {
    MermaidContext childCtx(mctx.diagram_, mctx.nodeId_, nodeId);
    source->accept(*this, childCtx);
  }
}

void MermaidPlanVisitor::visit(
    const LocalPartitionNode& node,
    PlanNodeVisitorContext& ctx) const {
  auto& mctx = static_cast<MermaidContext&>(ctx);
  std::string nodeId = nextId();
  std::string contents = fmt::format(
      "<table>"
      "{}"
      "{}"
      "{}"
      "{}"
      "{}"
      "</table>",
      caption(node.name()),
      LocalPartitionNode::toName(node.type()),
      fullRow(node.partitionFunctionSpec().toString()),
      fullRow(outputTypes(node.outputType()), 2),
      fullRow(
          includeTaskStats_ && taskStats_.count(node.id())
              ? planNodeStats(taskStats_.at(node.id()))
              : "",
          2));

  appendMermaidNode(
      mctx.parentId_, nodeId, contents, className<LocalPartitionNode>());

  // Visit sources
  for (const auto& source : node.sources()) {
    MermaidContext childCtx(mctx.diagram_, mctx.nodeId_, nodeId);
    source->accept(*this, childCtx);
  }
}

void MermaidPlanVisitor::visit(
    const MarkDistinctNode& node,
    PlanNodeVisitorContext& ctx) const {
  auto& mctx = static_cast<MermaidContext&>(ctx);
  std::string nodeId = visitNode(node, mctx.parentId_);

  // Visit sources
  for (const auto& source : node.sources()) {
    MermaidContext childCtx(mctx.diagram_, mctx.nodeId_, nodeId);
    source->accept(*this, childCtx);
  }
}

void MermaidPlanVisitor::visit(
    const MergeExchangeNode& node,
    PlanNodeVisitorContext& ctx) const {
  auto& mctx = static_cast<MermaidContext&>(ctx);
  std::string nodeId = visitNode(node, mctx.parentId_);

  // Visit sources
  for (const auto& source : node.sources()) {
    MermaidContext childCtx(mctx.diagram_, mctx.nodeId_, nodeId);
    source->accept(*this, childCtx);
  }
}

void MermaidPlanVisitor::visit(
    const MergeJoinNode& node,
    PlanNodeVisitorContext& ctx) const {
  auto& mctx = static_cast<MermaidContext&>(ctx);
  std::string nodeId = visitNode(node, mctx.parentId_);

  // Visit sources
  for (const auto& source : node.sources()) {
    MermaidContext childCtx(mctx.diagram_, mctx.nodeId_, nodeId);
    source->accept(*this, childCtx);
  }
}

void MermaidPlanVisitor::visit(
    const NestedLoopJoinNode& node,
    PlanNodeVisitorContext& ctx) const {
  auto& mctx = static_cast<MermaidContext&>(ctx);
  std::string nodeId = visitNode(node, mctx.parentId_);

  // Visit sources
  for (const auto& source : node.sources()) {
    MermaidContext childCtx(mctx.diagram_, mctx.nodeId_, nodeId);
    source->accept(*this, childCtx);
  }
}

void MermaidPlanVisitor::visit(
    const SpatialJoinNode& node,
    PlanNodeVisitorContext& ctx) const {
  auto& mctx = static_cast<MermaidContext&>(ctx);
  std::string nodeId = visitNode(node, mctx.parentId_);

  // Visit sources
  for (const auto& source : node.sources()) {
    MermaidContext childCtx(mctx.diagram_, mctx.nodeId_, nodeId);
    source->accept(*this, childCtx);
  }
}

void MermaidPlanVisitor::visit(
    const OrderByNode& node,
    PlanNodeVisitorContext& ctx) const {
  auto& mctx = static_cast<MermaidContext&>(ctx);
  std::string nodeId = nextId();
  std::string contents = fmt::format(
      "<table>"
      "{}"
      "{}"
      "{}"
      "{}"
      "</table>",
      caption(node.name(), node.isPartial() ? "partial" : ""),
      sortingKeys(node),
      fullRow(outputTypes(node.outputType()), 2),
      fullRow(
          includeTaskStats_ && taskStats_.count(node.id())
              ? planNodeStats(taskStats_.at(node.id()))
              : "",
          2));

  appendMermaidNode(mctx.parentId_, nodeId, contents, className<OrderByNode>());

  // Visit sources
  for (const auto& source : node.sources()) {
    MermaidContext childCtx(mctx.diagram_, mctx.nodeId_, nodeId);
    source->accept(*this, childCtx);
  }
}

void MermaidPlanVisitor::visit(
    const PartitionedOutputNode& node,
    PlanNodeVisitorContext& ctx) const {
  auto& mctx = static_cast<MermaidContext&>(ctx);
  std::string nodeId = visitNode(node, mctx.parentId_);

  // Visit sources
  for (const auto& source : node.sources()) {
    MermaidContext childCtx(mctx.diagram_, mctx.nodeId_, nodeId);
    source->accept(*this, childCtx);
  }
}

void MermaidPlanVisitor::visit(
    const ProjectNode& node,
    PlanNodeVisitorContext& ctx) const {
  auto& mctx = static_cast<MermaidContext&>(ctx);
  std::string nodeId = nextId();
  std::string contents = fmt::format(
      "<table>"
      "{}"
      "{}"
      "{}"
      "{}"
      "{}"
      "</table>",
      caption(node.name()),
      section("Projections", 2),
      fmt::join(
          ranges::views::zip(node.names(), node.projections()) //
              |
              ranges::views::transform([](const auto& pair) {
                const auto& [name, proj] = pair;
                return fmt::format(
                    "<tr><td>{}</td><td>{}</td></tr>", name, proj->toString());
              }),
          ""),
      fullRow(outputTypes(node.outputType()), 2),
      fullRow(
          includeTaskStats_ && taskStats_.count(node.id())
              ? planNodeStats(taskStats_.at(node.id()))
              : "",
          2));

  appendMermaidNode(mctx.parentId_, nodeId, contents, className<ProjectNode>());

  // Visit sources
  for (const auto& source : node.sources()) {
    MermaidContext childCtx(mctx.diagram_, mctx.nodeId_, nodeId);
    source->accept(*this, childCtx);
  }
}

void MermaidPlanVisitor::visit(
    const ParallelProjectNode& node,
    PlanNodeVisitorContext& ctx) const {
  auto& mctx = static_cast<MermaidContext&>(ctx);
  std::string nodeId = visitNode(node, mctx.parentId_);

  // Visit sources
  for (const auto& source : node.sources()) {
    MermaidContext childCtx(mctx.diagram_, mctx.nodeId_, nodeId);
    source->accept(*this, childCtx);
  }
}

void MermaidPlanVisitor::visit(
    const RowNumberNode& node,
    PlanNodeVisitorContext& ctx) const {
  auto& mctx = static_cast<MermaidContext&>(ctx);
  std::string nodeId = visitNode(node, mctx.parentId_);

  // Visit sources
  for (const auto& source : node.sources()) {
    MermaidContext childCtx(mctx.diagram_, mctx.nodeId_, nodeId);
    source->accept(*this, childCtx);
  }
}

void MermaidPlanVisitor::visit(
    const TableScanNode& node,
    PlanNodeVisitorContext& ctx) const {
  auto& mctx = static_cast<MermaidContext&>(ctx);
  std::string nodeId = nextId();
  std::string contents = fmt::format(
      "<table>"
      "{}"
      "{}"
      "{}"
      "</table>",
      caption(node.name()),
      fullRow(outputTypes(node.outputType())),
      fullRow(
          includeTaskStats_ && taskStats_.count(node.id())
              ? planNodeStats(taskStats_.at(node.id()))
              : ""));
  appendMermaidNode(
      mctx.parentId_, nodeId, contents, className<TableScanNode>());

  // Visit sources
  for (const auto& source : node.sources()) {
    MermaidContext childCtx(mctx.diagram_, mctx.nodeId_, nodeId);
    source->accept(*this, childCtx);
  }
}

void MermaidPlanVisitor::visit(
    const TableWriteNode& node,
    PlanNodeVisitorContext& ctx) const {
  auto& mctx = static_cast<MermaidContext&>(ctx);
  std::string nodeId = visitNode(node, mctx.parentId_);

  // Visit sources
  for (const auto& source : node.sources()) {
    MermaidContext childCtx(mctx.diagram_, mctx.nodeId_, nodeId);
    source->accept(*this, childCtx);
  }
}

void MermaidPlanVisitor::visit(
    const TableWriteMergeNode& node,
    PlanNodeVisitorContext& ctx) const {
  auto& mctx = static_cast<MermaidContext&>(ctx);
  std::string nodeId = visitNode(node, mctx.parentId_);

  // Visit sources
  for (const auto& source : node.sources()) {
    MermaidContext childCtx(mctx.diagram_, mctx.nodeId_, nodeId);
    source->accept(*this, childCtx);
  }
}

void MermaidPlanVisitor::visit(
    const TopNNode& node,
    PlanNodeVisitorContext& ctx) const {
  auto& mctx = static_cast<MermaidContext&>(ctx);
  std::string nodeId = visitNode(node, mctx.parentId_);

  // Visit sources
  for (const auto& source : node.sources()) {
    MermaidContext childCtx(mctx.diagram_, mctx.nodeId_, nodeId);
    source->accept(*this, childCtx);
  }
}

void MermaidPlanVisitor::visit(
    const TopNRowNumberNode& node,
    PlanNodeVisitorContext& ctx) const {
  auto& mctx = static_cast<MermaidContext&>(ctx);
  std::string nodeId = nextId();
  std::string contents = fmt::format(
      "<table>"
      "{}"
      "{}"
      "{}"
      "{}"
      "{}"
      "</table>",
      caption(node.name()),
      fieldsAsRows(node.partitionKeys(), "partition keys", 2),
      sortingKeys(node),
      fullRow(outputTypes(node.outputType()), 2),
      fullRow(
          includeTaskStats_ && taskStats_.count(node.id())
              ? planNodeStats(taskStats_.at(node.id()))
              : "",
          2));

  appendMermaidNode(
      mctx.parentId_, nodeId, contents, className<TopNRowNumberNode>());

  // Visit sources
  for (const auto& source : node.sources()) {
    MermaidContext childCtx(mctx.diagram_, mctx.nodeId_, nodeId);
    source->accept(*this, childCtx);
  }
}

void MermaidPlanVisitor::visit(
    const TraceScanNode& node,
    PlanNodeVisitorContext& ctx) const {
  auto& mctx = static_cast<MermaidContext&>(ctx);
  std::string nodeId = visitNode(node, mctx.parentId_);

  // Visit sources
  for (const auto& source : node.sources()) {
    MermaidContext childCtx(mctx.diagram_, mctx.nodeId_, nodeId);
    source->accept(*this, childCtx);
  }
}

void MermaidPlanVisitor::visit(
    const UnnestNode& node,
    PlanNodeVisitorContext& ctx) const {
  auto& mctx = static_cast<MermaidContext&>(ctx);
  std::string nodeId = visitNode(node, mctx.parentId_);

  // Visit sources
  for (const auto& source : node.sources()) {
    MermaidContext childCtx(mctx.diagram_, mctx.nodeId_, nodeId);
    source->accept(*this, childCtx);
  }
}

void MermaidPlanVisitor::visit(
    const ValuesNode& node,
    PlanNodeVisitorContext& ctx) const {
  auto& mctx = static_cast<MermaidContext&>(ctx);
  std::string nodeId = visitNode(node, mctx.parentId_);

  // Visit sources
  for (const auto& source : node.sources()) {
    MermaidContext childCtx(mctx.diagram_, mctx.nodeId_, nodeId);
    source->accept(*this, childCtx);
  }
}

void MermaidPlanVisitor::visit(
    const WindowNode& node,
    PlanNodeVisitorContext& ctx) const {
  auto& mctx = static_cast<MermaidContext&>(ctx);
  std::string nodeId = nextId();
  std::string contents = fmt::format(
      "<table>"
      "{}"
      "{}"
      "{}"
      "{}"
      "{}"
      "{}"
      "</table>",
      caption(node.name()),
      fieldsAsRows(node.partitionKeys(), "partition keys", 2),
      sortingKeys(node),
      windowFunctions(node.windowFunctions()),
      fullRow(outputTypes(node.outputType()), 2),
      fullRow(
          includeTaskStats_ && taskStats_.count(node.id())
              ? planNodeStats(taskStats_.at(node.id()))
              : "",
          2));

  appendMermaidNode(mctx.parentId_, nodeId, contents, className<WindowNode>());

  // Visit sources
  for (const auto& source : node.sources()) {
    MermaidContext childCtx(mctx.diagram_, mctx.nodeId_, nodeId);
    source->accept(*this, childCtx);
  }
}

void MermaidPlanVisitor::visit(
    const PlanNode& node,
    PlanNodeVisitorContext& ctx) const {
  auto& mctx = static_cast<MermaidContext&>(ctx);
  std::string nodeId = nextId();
  std::string contents = fmt::format(
      "<table>"
      "{}"
      "</table>",
      caption(node.name(), "Unrecognized"));

  appendMermaidNode(mctx.parentId_, nodeId, contents, kColorGrey);

  // Visit sources
  for (const auto& source : node.sources()) {
    MermaidContext childCtx(mctx.diagram_, mctx.nodeId_, nodeId);
    source->accept(*this, childCtx);
  }
}

} // namespace facebook::velox::common::base
