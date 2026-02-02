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
#include <gtest/gtest.h>

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/core/PlanConsistencyChecker.h"
#include "velox/parse/PlanNodeIdGenerator.h"

namespace facebook::velox::core {

namespace {
class PlanConsistencyCheckerTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});
  }

  void SetUp() override {
    idGenerator_.reset();
  }

  std::string nextId() {
    return idGenerator_.next();
  }

  PlanNodeIdGenerator idGenerator_;
};

TypedExprPtr Lit(Variant value) {
  auto type = value.inferType();
  return std::make_shared<ConstantTypedExpr>(std::move(type), std::move(value));
}

FieldAccessTypedExprPtr Col(TypePtr type, std::string name) {
  return std::make_shared<FieldAccessTypedExpr>(
      std::move(type), std::move(name));
}

TEST_F(PlanConsistencyCheckerTest, filter) {
  auto valuesNode =
      std::make_shared<ValuesNode>(nextId(), std::vector<RowVectorPtr>{});

  auto projectNode = std::make_shared<ProjectNode>(
      nextId(),
      std::vector<std::string>{"a", "b", "c"},
      std::vector<TypedExprPtr>{Lit(true), Lit(1), Lit(0.1)},
      valuesNode);

  auto filterNode =
      std::make_shared<FilterNode>(nextId(), Col(BOOLEAN(), "a"), projectNode);
  ASSERT_NO_THROW(PlanConsistencyChecker::check(filterNode));

  // Wrong type.
  filterNode =
      std::make_shared<FilterNode>(nextId(), Col(BOOLEAN(), "b"), projectNode);

  VELOX_ASSERT_THROW(
      PlanConsistencyChecker::check(filterNode),
      "Wrong type of input column: b, BOOLEAN vs. INTEGER");

  // Wrong name.
  filterNode =
      std::make_shared<FilterNode>(nextId(), Col(BOOLEAN(), "x"), projectNode);

  VELOX_ASSERT_THROW(
      PlanConsistencyChecker::check(filterNode), "Field not found: x");

  // Non-existent column referenced in a lambda expression.
  filterNode = std::make_shared<FilterNode>(
      nextId(),
      std::make_shared<CallTypedExpr>(
          BOOLEAN(),
          "any_match",
          Lit(Variant::array({1, 2, 3})),
          std::make_shared<LambdaTypedExpr>(
              ROW("x", INTEGER()),
              std::make_shared<CallTypedExpr>(
                  BOOLEAN(),
                  "lt",
                  Col(INTEGER(), "x"),
                  Col(INTEGER(), "blah")))),
      projectNode);

  VELOX_ASSERT_THROW(
      PlanConsistencyChecker::check(filterNode), "Field not found: blah");
}

TEST_F(PlanConsistencyCheckerTest, project) {
  auto valuesNode =
      std::make_shared<ValuesNode>(nextId(), std::vector<RowVectorPtr>{});

  auto projectNode = std::make_shared<ProjectNode>(
      nextId(),
      std::vector<std::string>{"a", "b", "c"},
      std::vector<TypedExprPtr>{Lit(true), Lit(1), Lit(0.1)},
      valuesNode);
  ASSERT_NO_THROW(PlanConsistencyChecker::check(projectNode));

  // Duplicate output name.
  projectNode = std::make_shared<ProjectNode>(
      nextId(),
      std::vector<std::string>{"a", "a", "c"},
      std::vector<TypedExprPtr>{Lit(true), Lit(1), Lit(0.1)},
      valuesNode);

  VELOX_ASSERT_THROW(
      PlanConsistencyChecker::check(projectNode), "Duplicate output column: a");

  // Wrong column name.
  projectNode = std::make_shared<ProjectNode>(
      nextId(),
      std::vector<std::string>{"a", "a", "c"},
      std::vector<TypedExprPtr>{Lit(true), Col(REAL(), "x"), Lit(0.1)},
      valuesNode);

  VELOX_ASSERT_THROW(
      PlanConsistencyChecker::check(projectNode), "Field not found: x");
}

TEST_F(PlanConsistencyCheckerTest, aggregation) {
  auto valuesNode =
      std::make_shared<ValuesNode>(nextId(), std::vector<RowVectorPtr>{});

  auto projectNode = std::make_shared<ProjectNode>(
      nextId(),
      std::vector<std::string>{"a", "b", "c"},
      std::vector<TypedExprPtr>{Lit(true), Lit(1), Lit(0.1)},
      valuesNode);
  ASSERT_NO_THROW(PlanConsistencyChecker::check(projectNode));

  {
    auto aggregationNode = std::make_shared<AggregationNode>(
        nextId(),
        AggregationNode::Step::kPartial,
        std::vector<FieldAccessTypedExprPtr>{},
        std::vector<FieldAccessTypedExprPtr>{},
        std::vector<std::string>{"sum", "cnt"},
        std::vector<AggregationNode::Aggregate>{
            {
                .call = std::make_shared<CallTypedExpr>(
                    BIGINT(), "sum", Col(INTEGER(), "x")),
                .rawInputTypes = {BIGINT()},
            },
            {
                .call = std::make_shared<CallTypedExpr>(BIGINT(), "count"),
                .rawInputTypes = {},
            },
        },
        /*ignoreNullKeys=*/false,
        /*noGroupsSpanBatches=*/false,
        projectNode);
    VELOX_ASSERT_THROW(
        PlanConsistencyChecker::check(aggregationNode), "Field not found: x");
  }

  {
    auto aggregationNode = std::make_shared<AggregationNode>(
        nextId(),
        AggregationNode::Step::kPartial,
        std::vector<FieldAccessTypedExprPtr>{Col(INTEGER(), "y")},
        std::vector<FieldAccessTypedExprPtr>{},
        std::vector<std::string>{"sum", "cnt"},
        std::vector<AggregationNode::Aggregate>{
            {
                .call = std::make_shared<CallTypedExpr>(
                    BIGINT(), "sum", Col(INTEGER(), "b")),
                .rawInputTypes = {BIGINT()},
            },
            {
                .call = std::make_shared<CallTypedExpr>(BIGINT(), "count"),
                .rawInputTypes = {},
            },
        },
        /*ignoreNullKeys=*/false,
        /*noGroupsSpanBatches=*/false,
        projectNode);
    VELOX_ASSERT_THROW(
        PlanConsistencyChecker::check(aggregationNode), "Field not found: y");
  }

  {
    auto aggregationNode = std::make_shared<AggregationNode>(
        nextId(),
        AggregationNode::Step::kPartial,
        std::vector<FieldAccessTypedExprPtr>{},
        std::vector<FieldAccessTypedExprPtr>{},
        std::vector<std::string>{"sum", "cnt"},
        std::vector<AggregationNode::Aggregate>{
            {
                .call = std::make_shared<CallTypedExpr>(
                    BIGINT(), "sum", Col(INTEGER(), "b")),
                .rawInputTypes = {BIGINT()},
                .mask = Col(BOOLEAN(), "z"),
            },
            {
                .call = std::make_shared<CallTypedExpr>(BIGINT(), "count"),
                .rawInputTypes = {},
            },
        },
        /*ignoreNullKeys=*/false,
        /*noGroupsSpanBatches=*/false,
        projectNode);
    VELOX_ASSERT_THROW(
        PlanConsistencyChecker::check(aggregationNode), "Field not found: z");
  }

  {
    auto aggregationNode = std::make_shared<AggregationNode>(
        nextId(),
        AggregationNode::Step::kPartial,
        std::vector<FieldAccessTypedExprPtr>{},
        std::vector<FieldAccessTypedExprPtr>{},
        std::vector<std::string>{"sum", "sum"},
        std::vector<AggregationNode::Aggregate>{
            {
                .call = std::make_shared<CallTypedExpr>(
                    BIGINT(), "sum", Col(INTEGER(), "b")),
                .rawInputTypes = {BIGINT()},
            },
            {
                .call = std::make_shared<CallTypedExpr>(BIGINT(), "count"),
                .rawInputTypes = {},
            },
        },
        /*ignoreNullKeys=*/false,
        /*noGroupsSpanBatches=*/false,
        projectNode);
    VELOX_ASSERT_THROW(
        PlanConsistencyChecker::check(aggregationNode),
        "Duplicate output column: sum");
  }
}

TEST_F(PlanConsistencyCheckerTest, hashJoin) {
  auto leftValuesNode =
      std::make_shared<ValuesNode>(nextId(), std::vector<RowVectorPtr>{});

  auto leftProjectNode = std::make_shared<ProjectNode>(
      nextId(),
      std::vector<std::string>{"a", "b"},
      std::vector<TypedExprPtr>{Lit(1), Lit(2)},
      leftValuesNode);
  ASSERT_NO_THROW(PlanConsistencyChecker::check(leftValuesNode));

  auto rightValuesNode =
      std::make_shared<ValuesNode>(nextId(), std::vector<RowVectorPtr>{});

  auto rightProjectNode = std::make_shared<ProjectNode>(
      nextId(),
      std::vector<std::string>{"c", "d"},
      std::vector<TypedExprPtr>{Lit(1), Lit(2)},
      leftValuesNode);
  ASSERT_NO_THROW(PlanConsistencyChecker::check(rightProjectNode));

  // Invalid reference in the filter.
  {
    auto joinNode = std::make_shared<HashJoinNode>(
        nextId(),
        JoinType::kLeft,
        /*nullAware=*/false,
        std::vector<FieldAccessTypedExprPtr>{Col(INTEGER(), "a")},
        std::vector<FieldAccessTypedExprPtr>{Col(INTEGER(), "c")},
        std::make_shared<CallTypedExpr>(
            BOOLEAN(), "lt", Col(INTEGER(), "b"), Col(INTEGER(), "blah")),
        leftProjectNode,
        rightProjectNode,
        ROW({}));
    VELOX_ASSERT_THROW(
        PlanConsistencyChecker::check(joinNode),
        "Field not found: blah. Available fields are: a, b, c, d.");
  }

  // Duplicate join condition.
  {
    auto joinNode = std::make_shared<HashJoinNode>(
        nextId(),
        JoinType::kLeft,
        /*nullAware=*/false,
        std::vector<FieldAccessTypedExprPtr>{
            Col(INTEGER(), "a"), Col(INTEGER(), "a")},
        std::vector<FieldAccessTypedExprPtr>{
            Col(INTEGER(), "c"), Col(INTEGER(), "c")},
        /*filter=*/nullptr,
        leftProjectNode,
        rightProjectNode,
        ROW({}));
    VELOX_ASSERT_THROW(
        PlanConsistencyChecker::check(joinNode),
        "Duplicate join condition: \"a\" = \"c\"");
  }
}

TEST_F(PlanConsistencyCheckerTest, nestedLoopJoin) {
  auto leftValuesNode =
      std::make_shared<ValuesNode>(nextId(), std::vector<RowVectorPtr>{});

  auto leftProjectNode = std::make_shared<ProjectNode>(
      nextId(),
      std::vector<std::string>{"a", "b"},
      std::vector<TypedExprPtr>{Lit(1), Lit(2)},
      leftValuesNode);
  ASSERT_NO_THROW(PlanConsistencyChecker::check(leftValuesNode));

  auto rightValuesNode =
      std::make_shared<ValuesNode>(nextId(), std::vector<RowVectorPtr>{});

  auto rightProjectNode = std::make_shared<ProjectNode>(
      nextId(),
      std::vector<std::string>{"c", "d"},
      std::vector<TypedExprPtr>{Lit(1), Lit(2)},
      leftValuesNode);
  ASSERT_NO_THROW(PlanConsistencyChecker::check(rightProjectNode));

  // Invalid reference in the filter.
  {
    auto joinNode = std::make_shared<NestedLoopJoinNode>(
        nextId(),
        JoinType::kLeft,
        std::make_shared<CallTypedExpr>(
            BOOLEAN(), "lt", Col(INTEGER(), "b"), Col(INTEGER(), "blah")),
        leftProjectNode,
        rightProjectNode,
        ROW({}));
    VELOX_ASSERT_THROW(
        PlanConsistencyChecker::check(joinNode),
        "Field not found: blah. Available fields are: a, b, c, d.");
  }

  // Duplicate output name.
  {
    auto joinNode = std::make_shared<NestedLoopJoinNode>(
        nextId(),
        leftProjectNode,
        rightProjectNode,
        ROW({"a", "c", "a"}, INTEGER()));
    VELOX_ASSERT_THROW(
        PlanConsistencyChecker::check(joinNode), "Duplicate output column: a");
  }
}

namespace {
class TestTableHandle : public connector::ConnectorTableHandle {
 public:
  explicit TestTableHandle(std::string connectorId, std::string name)
      : connector::ConnectorTableHandle(std::move(connectorId)),
        name_{std::move(name)} {}

  const std::string& name() const override {
    return name_;
  }

 private:
  const std::string name_;
};

class TestColumnHandle : public connector::ColumnHandle {
 public:
  explicit TestColumnHandle(std::string name) : name_{std::move(name)} {}

  const std::string& name() const override {
    return name_;
  }

 private:
  const std::string name_;
};
} // namespace

TEST_F(PlanConsistencyCheckerTest, tableScan) {
  // Empty output column name.
  {
    auto scanNode = std::make_shared<TableScanNode>(
        nextId(),
        ROW({"", "b"}, INTEGER()),
        std::make_shared<TestTableHandle>("test", "t"),
        connector::ColumnHandleMap{});

    VELOX_ASSERT_THROW(
        PlanConsistencyChecker::check(scanNode),
        "Output column name cannot be empty");
  }

  // Duplicate output column name.
  {
    auto scanNode = std::make_shared<TableScanNode>(
        nextId(),
        ROW({"a", "b", "a"}, INTEGER()),
        std::make_shared<TestTableHandle>("test", "t"),
        connector::ColumnHandleMap{});

    VELOX_ASSERT_THROW(
        PlanConsistencyChecker::check(scanNode), "Duplicate output column: a");
  }

  // Missing assignments.
  {
    auto scanNode = std::make_shared<TableScanNode>(
        nextId(),
        ROW({"a", "b", "c"}, INTEGER()),
        std::make_shared<TestTableHandle>("test", "t"),
        connector::ColumnHandleMap{});

    VELOX_ASSERT_THROW(
        PlanConsistencyChecker::check(scanNode),
        "Column assignments must match output type");
  }

  {
    connector::ColumnHandleMap assignments{
        {"a", std::make_shared<TestColumnHandle>("x")},
        {"b", std::make_shared<TestColumnHandle>("y")},
        {"blah", std::make_shared<TestColumnHandle>("z")},
    };

    auto scanNode = std::make_shared<TableScanNode>(
        nextId(),
        ROW({"a", "b", "c"}, INTEGER()),
        std::make_shared<TestTableHandle>("test", "t"),
        assignments);

    VELOX_ASSERT_THROW(
        PlanConsistencyChecker::check(scanNode),
        "Column assignment is missing for c");
  }

  // No issues.
  {
    connector::ColumnHandleMap assignments{
        {"a", std::make_shared<TestColumnHandle>("x")},
        {"b", std::make_shared<TestColumnHandle>("y")},
        {"c", std::make_shared<TestColumnHandle>("z")},
    };

    auto scanNode = std::make_shared<TableScanNode>(
        nextId(),
        ROW({"a", "b", "c"}, INTEGER()),
        std::make_shared<TestTableHandle>("test", "t"),
        assignments);

    ASSERT_NO_THROW(PlanConsistencyChecker::check(scanNode));
  }
}

} // namespace
} // namespace facebook::velox::core
