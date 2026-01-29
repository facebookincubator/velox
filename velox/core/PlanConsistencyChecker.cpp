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

#include "velox/core/PlanConsistencyChecker.h"

namespace facebook::velox::core {

namespace {

class Checker : public PlanNodeVisitor {
 public:
  void visit(const AggregationNode& node, PlanNodeVisitorContext& ctx)
      const override {
    const auto& rowType = node.sources().at(0)->outputType();
    for (const auto& expr : node.groupingKeys()) {
      checkInputs(expr, rowType);
    }

    for (const auto& expr : node.preGroupedKeys()) {
      checkInputs(expr, rowType);
    }

    for (const auto& aggregate : node.aggregates()) {
      checkInputs(aggregate.call, rowType);

      for (const auto& expr : aggregate.sortingKeys) {
        checkInputs(expr, rowType);
      }

      if (aggregate.mask) {
        checkInputs(aggregate.mask, rowType);
      }
    }

    verifyOutputNames(node);

    visitSources(&node, ctx);
  }

  void visit(const ArrowStreamNode& node, PlanNodeVisitorContext& ctx)
      const override {
    visitSources(&node, ctx);
  }

  void visit(const AssignUniqueIdNode& node, PlanNodeVisitorContext& ctx)
      const override {
    visitSources(&node, ctx);
  }

  void visit(const EnforceSingleRowNode& node, PlanNodeVisitorContext& ctx)
      const override {
    visitSources(&node, ctx);
  }

  void visit(const ExchangeNode& node, PlanNodeVisitorContext& ctx)
      const override {
    visitSources(&node, ctx);
  }

  void visit(const ExpandNode& node, PlanNodeVisitorContext& ctx)
      const override {
    visitSources(&node, ctx);
  }

  void visit(const FilterNode& node, PlanNodeVisitorContext& ctx)
      const override {
    checkInputs(node.filter(), node.sources().at(0)->outputType());

    visitSources(&node, ctx);
  }

  void visit(const GroupIdNode& node, PlanNodeVisitorContext& ctx)
      const override {
    visitSources(&node, ctx);
  }

  void visit(const HashJoinNode& node, PlanNodeVisitorContext& ctx)
      const override {
    std::unordered_set<std::pair<std::string, std::string>> keyNames;
    for (auto i = 0; i < node.leftKeys().size(); ++i) {
      const auto& leftKey = node.leftKeys().at(i);
      const auto& rightKey = node.rightKeys().at(i);

      bool unique = keyNames.emplace(leftKey->name(), rightKey->name()).second;
      VELOX_CHECK(
          unique,
          "Duplicate join condition: {} = {}",
          leftKey->toString(),
          rightKey->toString());
    }

    if (node.filter() != nullptr) {
      const auto& leftRowType = node.sources().at(0)->outputType();
      const auto& rightRowType = node.sources().at(1)->outputType();
      auto rowType = leftRowType->unionWith(rightRowType);
      checkInputs(node.filter(), rowType);
    }

    visitSources(&node, ctx);
  }

  void visit(const IndexLookupJoinNode& node, PlanNodeVisitorContext& ctx)
      const override {
    visitSources(&node, ctx);
  }

  void visit(const LimitNode& node, PlanNodeVisitorContext& ctx)
      const override {
    visitSources(&node, ctx);
  }

  void visit(const LocalMergeNode& node, PlanNodeVisitorContext& ctx)
      const override {
    visitSources(&node, ctx);
  }

  void visit(const LocalPartitionNode& node, PlanNodeVisitorContext& ctx)
      const override {
    visitSources(&node, ctx);
  }

  void visit(const MarkDistinctNode& node, PlanNodeVisitorContext& ctx)
      const override {
    visitSources(&node, ctx);
  }

  void visit(const MergeExchangeNode& node, PlanNodeVisitorContext& ctx)
      const override {
    visitSources(&node, ctx);
  }

  void visit(const MergeJoinNode& node, PlanNodeVisitorContext& ctx)
      const override {
    visitSources(&node, ctx);
  }

  void visit(const NestedLoopJoinNode& node, PlanNodeVisitorContext& ctx)
      const override {
    if (node.joinCondition() != nullptr) {
      const auto& leftRowType = node.sources().at(0)->outputType();
      const auto& rightRowType = node.sources().at(1)->outputType();
      auto rowType = leftRowType->unionWith(rightRowType);
      checkInputs(node.joinCondition(), rowType);
    }

    visitSources(&node, ctx);
  }

  void visit(const SpatialJoinNode& node, PlanNodeVisitorContext& ctx)
      const override {
    visitSources(&node, ctx);
  }

  void visit(const OrderByNode& node, PlanNodeVisitorContext& ctx)
      const override {
    visitSources(&node, ctx);
  }

  void visit(const PartitionedOutputNode& node, PlanNodeVisitorContext& ctx)
      const override {
    visitSources(&node, ctx);
  }

  void visit(const ProjectNode& node, PlanNodeVisitorContext& ctx)
      const override {
    const auto& rowType = node.sources().at(0)->outputType();
    for (const auto& expr : node.projections()) {
      checkInputs(expr, rowType);
    }

    verifyOutputNames(node);

    visitSources(&node, ctx);
  }

  void visit(const ParallelProjectNode& node, PlanNodeVisitorContext& ctx)
      const override {
    const auto& rowType = node.sources().at(0)->outputType();
    for (const auto& group : node.exprGroups()) {
      for (const auto& expr : group) {
        checkInputs(expr, rowType);
      }
    }
    visitSources(&node, ctx);
  }

  void visit(const RowNumberNode& node, PlanNodeVisitorContext& ctx)
      const override {
    visitSources(&node, ctx);
  }

  void visit(const TableScanNode& node, PlanNodeVisitorContext& ctx)
      const override {
    verifyOutputNames(node);

    // Verify assignments match outputType 1:1.
    const auto& names = node.outputType()->names();
    VELOX_USER_CHECK_EQ(
        names.size(),
        node.assignments().size(),
        "Column assignments must match output type 1:1.");

    for (const auto& name : names) {
      VELOX_USER_CHECK(
          node.assignments().contains(name),
          "Column assignment is missing for {}",
          name);
    }
  }

  void visit(const TableWriteNode& node, PlanNodeVisitorContext& ctx)
      const override {
    visitSources(&node, ctx);
  }

  void visit(const TableWriteMergeNode& node, PlanNodeVisitorContext& ctx)
      const override {
    visitSources(&node, ctx);
  }

  void visit(const TopNNode& node, PlanNodeVisitorContext& ctx) const override {
    visitSources(&node, ctx);
  }

  void visit(const TopNRowNumberNode& node, PlanNodeVisitorContext& ctx)
      const override {
    visitSources(&node, ctx);
  }

  void visit(const TraceScanNode& node, PlanNodeVisitorContext& ctx)
      const override {
    visitSources(&node, ctx);
  }

  void visit(const UnnestNode& node, PlanNodeVisitorContext& ctx)
      const override {
    visitSources(&node, ctx);
  }

  void visit(const ValuesNode& node, PlanNodeVisitorContext& ctx)
      const override {
    visitSources(&node, ctx);
  }

  void visit(const WindowNode& node, PlanNodeVisitorContext& ctx)
      const override {
    visitSources(&node, ctx);
  }

  void visit(const PlanNode& node, PlanNodeVisitorContext& ctx) const override {
    visitSources(&node, ctx);
  }

 private:
  void visitSources(const PlanNode* node, PlanNodeVisitorContext& ctx) const {
    for (auto& source : node->sources()) {
      source->accept(*this, ctx);
    }
  }

  // Verify that output column names are not empty and unique.
  static void verifyOutputNames(const PlanNode& node) {
    folly::F14FastSet<std::string_view> names;
    for (const auto& name : node.outputType()->names()) {
      VELOX_USER_CHECK(!name.empty(), "Output column name cannot be empty");
      VELOX_USER_CHECK(
          names.emplace(name).second, "Duplicate output column: {}", name);
    }
  }

  static void checkInputs(
      const core::TypedExprPtr& expr,
      const RowTypePtr& rowType) {
    if (expr->isFieldAccessKind()) {
      auto fieldAccess = expr->asUnchecked<core::FieldAccessTypedExpr>();
      if (fieldAccess->isInputColumn()) {
        // Verify that field name points to an existing column in the input and
        // the type matches.
        const auto& name = fieldAccess->name();
        const auto& type = fieldAccess->type();
        const auto& expectedType = rowType->findChild(fieldAccess->name());
        VELOX_USER_CHECK(
            *type == *expectedType,
            "Wrong type of input column: {}, {} vs. {}",
            name,
            type->toString(),
            expectedType->toString());
      }
    }

    if (expr->isLambdaKind()) {
      const auto& lambda = expr->asUnchecked<core::LambdaTypedExpr>();
      checkInputs(lambda->body(), lambda->signature()->unionWith(rowType));
    }

    for (const auto& input : expr->inputs()) {
      checkInputs(input, rowType);
    }
  }
};
} // namespace

void PlanConsistencyChecker::check(const core::PlanNodePtr& plan) {
  PlanNodeVisitorContext ctx;
  Checker checker;
  plan->accept(checker, ctx);
}
}; // namespace facebook::velox::core
