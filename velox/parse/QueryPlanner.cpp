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

#include "velox/parse/QueryPlanner.h"
#include "velox/duckdb/conversion/DuckConversion.h"
#include "velox/parse/DuckLogicalOperator.h"

#include <duckdb.hpp> // @manual
#include <duckdb/main/connection.hpp> // @manual
#include <duckdb/planner/expression/bound_aggregate_expression.hpp> // @manual
#include <duckdb/planner/expression/bound_cast_expression.hpp> // @manual
#include <duckdb/planner/expression/bound_columnref_expression.hpp> // @manual
#include <duckdb/planner/expression/bound_comparison_expression.hpp> // @manual
#include <duckdb/planner/expression/bound_conjunction_expression.hpp> // @manual
#include <duckdb/planner/expression/bound_constant_expression.hpp> // @manual
#include <duckdb/planner/expression/bound_function_expression.hpp> // @manual
#include <duckdb/planner/expression/bound_reference_expression.hpp> // @manual
#include <duckdb/planner/operator/logical_dummy_scan.hpp> // @manual

namespace facebook::velox::core {

namespace {

using DuckColumnBindings = std::vector<::duckdb::ColumnBinding>;

class ColumnNameGenerator {
 public:
  std::string next(const std::string& prefix = "_c") {
    if (names_.count(prefix)) {
      auto name = fmt::format("{}{}", prefix, nextId_++);
      names_.insert(name);
      return name;
    }

    names_.insert(prefix);
    return prefix;
  }

 private:
  std::unordered_set<std::string> names_;
  int nextId_{0};
};

struct QueryContext {
  PlanNodeIdGenerator planNodeIdGenerator;
  ColumnNameGenerator columnNameGenerator;
  const std::unordered_map<std::string, std::vector<RowVectorPtr>>&
      inMemoryTables;
  MakeTableScan makeTableScan;

  struct DelimJoinContext {
    ::duckdb::LogicalDelimJoin* join;
  };
  std::vector<DelimJoinContext> delimJoinStack;

  QueryContext(
      const std::unordered_map<std::string, std::vector<RowVectorPtr>>&
          _inMemoryTables)
      : inMemoryTables{_inMemoryTables} {}

  std::string nextNodeId() {
    return planNodeIdGenerator.next();
  }

  std::string nextColumnName() {
    return columnNameGenerator.next();
  }

  std::string nextColumnName(const std::string& prefix) {
    return columnNameGenerator.next(prefix);
  }
};
std::string mapScalarFunctionName(const std::string& name) {
  static const std::unordered_map<std::string, std::string> kMapping = {
      {"+", "plus"},
      {"-", "minus"},
      {"*", "multiply"},
      {"/", "divide"},
      {"%", "mod"},
      {"~~", "like"},
      {"!~~", "not_like"},
      {"list_value", "array_constructor"},
  };

  auto it = kMapping.find(name);
  if (it != kMapping.end()) {
    return it->second;
  }

  return name;
}

std::string mapAggregateFunctionName(const std::string& name) {
  static const std::unordered_map<std::string, std::string> kMapping = {
      {"count_star", "count"},
  };

  auto it = kMapping.find(name);
  if (it != kMapping.end()) {
    return it->second;
  }

  return name;
}

PlanNodePtr toVeloxPlan(
    ::duckdb::LogicalOperator& plan,
    memory::MemoryPool* pool,
    QueryContext& queryContext);

PlanNodePtr toVeloxPlan(
    ::duckdb::LogicalDummyScan& logicalDummyScan,
    memory::MemoryPool* pool,
    QueryContext& queryContext) {
  std::vector<std::string> names;
  std::vector<TypePtr> types;
  for (auto i = 0; i < logicalDummyScan.types.size(); ++i) {
    names.push_back(queryContext.nextColumnName());
    types.push_back(duckdb::toVeloxType(logicalDummyScan.types[i]));
  }

  auto rowType = ROW(std::move(names), std::move(types));

  std::vector<RowVectorPtr> vectors = {std::make_shared<RowVector>(
      pool, rowType, nullptr, 1, std::vector<VectorPtr>{})};
  return std::make_shared<ValuesNode>(queryContext.nextNodeId(), vectors);
}

PlanNodePtr toVeloxPlan(
    ::duckdb::LogicalGet& logicalGet,
    memory::MemoryPool* pool,
    std::vector<PlanNodePtr> sources,
    QueryContext& queryContext) {
  if (logicalGet.function.name == "unnest") {
    VELOX_CHECK_EQ(1, sources.size());
    return std::make_shared<UnnestNode>(
        queryContext.nextNodeId(),
        std::vector<FieldAccessTypedExprPtr>{}, // replicateVariables
        std::vector<FieldAccessTypedExprPtr>{
            std::make_shared<FieldAccessTypedExpr>(
                sources[0]->outputType()->childAt(0),
                sources[0]->outputType()->asRow().nameOf(0))},
        std::vector<std::string>{"a"},
        /*ordinalityName=*/std::nullopt,
        /*emptyUnnestValueName=*/std::nullopt,
        std::move(sources[0]));
  }

  VELOX_CHECK_EQ(logicalGet.function.name, "seq_scan");
  VELOX_CHECK_EQ(0, sources.size());

  std::vector<std::string> columnNames;
  const auto& columnIds = logicalGet.column_ids;
  std::vector<std::string> names;
  std::vector<TypePtr> types;
  constexpr uint64_t kNone = ~0UL;
  for (auto i = 0; i < columnIds.size(); ++i) {
    if (columnIds[i] == kNone) {
      continue;
    }
    names.push_back(
        queryContext.nextColumnName(logicalGet.names[columnIds[i]]));
    types.push_back(
        duckdb::toVeloxType(logicalGet.returned_types[columnIds[i]]));
    columnNames.push_back(logicalGet.names[columnIds[i]]);
  }

  auto rowType = ROW(std::move(names), std::move(types));

  auto tableName = logicalGet.function.to_string(logicalGet.bind_data.get());
  auto it = queryContext.inMemoryTables.find(tableName);

  if (it == queryContext.inMemoryTables.end()) {
    return queryContext.makeTableScan(
        queryContext.nextNodeId(), tableName, rowType, columnNames);
  }

  std::vector<RowVectorPtr> data;
  for (auto& rowVector : it->second) {
    std::vector<VectorPtr> children;
    if (rowVector->size() > 0) {
      for (auto i = 0; i < columnIds.size(); ++i) {
        children.push_back(rowVector->childAt(columnIds[i]));
      }
    }
    data.push_back(
        std::make_shared<RowVector>(
            pool, rowType, nullptr, rowVector->size(), children));
  }

  return std::make_shared<ValuesNode>(queryContext.nextNodeId(), data);
}

TypedExprPtr toVeloxExpression(
    ::duckdb::Expression& expression,
    const TypePtr& inputType,
    const DuckColumnBindings* inputBindings = nullptr);

TypedExprPtr toVeloxComparisonExpression(
    const std::string& name,
    ::duckdb::Expression& expression,
    const TypePtr& inputType,
    const DuckColumnBindings* inputBindings = nullptr) {
  auto* comparison =
      dynamic_cast<::duckdb::BoundComparisonExpression*>(&expression);
  std::vector<TypedExprPtr> children{
      toVeloxExpression(*comparison->left, inputType, inputBindings),
      toVeloxExpression(*comparison->right, inputType, inputBindings)};

  return std::make_shared<CallTypedExpr>(BOOLEAN(), std::move(children), name);
}

namespace {
struct VeloxColumnProjections {
  VeloxColumnProjections(QueryContext& context) : context(context) {}

  core::FieldAccessTypedExprPtr toFieldAccess(
      ::duckdb::Expression& expression,
      const TypePtr& inputType,
      const DuckColumnBindings* inputBindings = nullptr) {
    auto expr = toVeloxExpression(expression, inputType, inputBindings);
    auto column = std::dynamic_pointer_cast<const FieldAccessTypedExpr>(expr);
    if (column) {
      columns.push_back(column);
      exprs.push_back(expr);
      return column;
    }

    allIdentity = false;
    exprs.push_back(expr);
    const auto name = context.nextColumnName("_p");
    auto projected =
        std::make_shared<FieldAccessTypedExpr>(exprs.back()->type(), name);
    columns.push_back(projected);
    return projected;
  }

  void addColumn(const FieldAccessTypedExprPtr& column) {
    for (auto& existingColumn : columns) {
      if (column->name() == existingColumn->name()) {
        return;
      }
    }
    exprs.push_back(column);
    columns.push_back(column);
  }

  /// Returns 'input' wrapped in the projections of 'this'. May only be called
  /// once.
  PlanNodePtr source(PlanNodePtr input) {
    if (allIdentity) {
      return input;
    }

    std::vector<std::string> names;
    names.reserve(columns.size());
    for (auto& column : columns) {
      names.push_back(column->name());
    }
    return std::make_shared<ProjectNode>(
        context.nextNodeId(), std::move(names), std::move(exprs), input);
  }

  QueryContext& context;
  bool allIdentity{true};
  std::vector<core::TypedExprPtr> exprs;
  std::vector<core::FieldAccessTypedExprPtr> columns;
};
} // namespace

std::string comparisonFunctionName(::duckdb::ExpressionType type) {
  switch (type) {
    case ::duckdb::ExpressionType::COMPARE_EQUAL:
      return "eq";
    case ::duckdb::ExpressionType::COMPARE_NOTEQUAL:
      return "neq";
    case ::duckdb::ExpressionType::COMPARE_GREATERTHAN:
      return "gt";
    case ::duckdb::ExpressionType::COMPARE_GREATERTHANOREQUALTO:
      return "gte";
    case ::duckdb::ExpressionType::COMPARE_LESSTHAN:
      return "lt";
    case ::duckdb::ExpressionType::COMPARE_LESSTHANOREQUALTO:
      return "lte";
    case ::duckdb::ExpressionType::COMPARE_DISTINCT_FROM:
    case ::duckdb::ExpressionType::COMPARE_NOT_DISTINCT_FROM:
      return "distinct_from";
    default:
      VELOX_NYI(
          "Comparison type {} is not supported yet",
          ::duckdb::ExpressionTypeToString(type));
  }
}

TypedExprPtr maybeNegateNotDistinct(
    ::duckdb::ExpressionType type,
    TypedExprPtr expr) {
  if (type == ::duckdb::ExpressionType::COMPARE_NOT_DISTINCT_FROM) {
    return std::make_shared<CallTypedExpr>(BOOLEAN(), "not", std::move(expr));
  }
  return expr;
}

std::string conjunctionFunctionName(::duckdb::ExpressionType type) {
  switch (type) {
    case ::duckdb::ExpressionType::CONJUNCTION_AND:
      return "and";
    case ::duckdb::ExpressionType::CONJUNCTION_OR:
      return "or";
    default:
      VELOX_NYI(
          "Conjunction type {} is not supported yet",
          ::duckdb::ExpressionTypeToString(type));
  }
}

int32_t findColumnBindingIndex(
    const ::duckdb::ColumnBinding& binding,
    const DuckColumnBindings& inputBindings) {
  for (auto i = 0; i < inputBindings.size(); ++i) {
    if (inputBindings[i].table_index == binding.table_index &&
        inputBindings[i].column_index == binding.column_index) {
      return i;
    }
  }
  return -1;
}

TypedExprPtr toVeloxExpression(
    ::duckdb::Expression& expression,
    const TypePtr& inputType,
    const DuckColumnBindings* inputBindings) {
  switch (expression.type) {
    case ::duckdb::ExpressionType::VALUE_CONSTANT: {
      auto* constant =
          dynamic_cast<::duckdb::BoundConstantExpression*>(&expression);
      return std::make_shared<ConstantTypedExpr>(
          duckdb::toVeloxType(constant->return_type),
          duckdb::duckValueToVariant(constant->value));
    }
    case ::duckdb::ExpressionType::COMPARE_EQUAL:
    case ::duckdb::ExpressionType::COMPARE_NOTEQUAL:
    case ::duckdb::ExpressionType::COMPARE_GREATERTHAN:
    case ::duckdb::ExpressionType::COMPARE_GREATERTHANOREQUALTO:
    case ::duckdb::ExpressionType::COMPARE_LESSTHAN:
    case ::duckdb::ExpressionType::COMPARE_LESSTHANOREQUALTO:
    case ::duckdb::ExpressionType::COMPARE_DISTINCT_FROM:
    case ::duckdb::ExpressionType::COMPARE_NOT_DISTINCT_FROM: {
      auto call = toVeloxComparisonExpression(
          comparisonFunctionName(expression.type),
          expression,
          inputType,
          inputBindings);
      return maybeNegateNotDistinct(expression.type, std::move(call));
    }
    case ::duckdb::ExpressionType::CONJUNCTION_AND:
    case ::duckdb::ExpressionType::CONJUNCTION_OR: {
      auto* conjunction =
          dynamic_cast<::duckdb::BoundConjunctionExpression*>(&expression);
      VELOX_CHECK_GT(conjunction->children.size(), 0);
      TypedExprPtr result;
      for (auto& child : conjunction->children) {
        auto conjunct = toVeloxExpression(*child, inputType, inputBindings);
        if (!result) {
          result = std::move(conjunct);
        } else {
          result = std::make_shared<CallTypedExpr>(
              BOOLEAN(),
              conjunctionFunctionName(expression.type),
              std::move(result),
              std::move(conjunct));
        }
      }
      return result;
    }

    case ::duckdb::ExpressionType::OPERATOR_CAST: {
      auto* cast = dynamic_cast<::duckdb::BoundCastExpression*>(&expression);
      return std::make_shared<CastTypedExpr>(
          duckdb::toVeloxType(cast->return_type),
          std::vector<TypedExprPtr>{
              toVeloxExpression(*cast->child, inputType, inputBindings)},
          cast->try_cast);
    }
    case ::duckdb::ExpressionType::BOUND_FUNCTION: {
      auto* func =
          dynamic_cast<::duckdb::BoundFunctionExpression*>(&expression);

      std::vector<TypedExprPtr> children;
      for (auto& child : func->children) {
        children.push_back(toVeloxExpression(*child, inputType, inputBindings));
      }

      auto name = mapScalarFunctionName(func->function.name);
      bool negate = false;
      if (name == "not_like") {
        name = "like";
        negate = true;
      }
      auto call = std::make_shared<CallTypedExpr>(
          duckdb::toVeloxType(func->function.return_type),
          std::move(children),
          name);
      if (negate) {
        return std::make_shared<CallTypedExpr>(BOOLEAN(), "not", call);
      }
      return call;
    }
    case ::duckdb::ExpressionType::BOUND_REF: {
      auto* ref =
          dynamic_cast<::duckdb::BoundReferenceExpression*>(&expression);
      return std::make_shared<FieldAccessTypedExpr>(
          duckdb::toVeloxType(ref->return_type),
          inputType->asRow().nameOf(ref->index));
    }
    case ::duckdb::ExpressionType::BOUND_COLUMN_REF: {
      auto* ref =
          dynamic_cast<::duckdb::BoundColumnRefExpression*>(&expression);
      VELOX_CHECK_NOT_NULL(
          inputBindings,
          "Column binding expression requires input bindings: {}",
          expression.ToString());
      auto index = findColumnBindingIndex(ref->binding, *inputBindings);
      VELOX_CHECK_GE(
          index,
          0,
          "Column binding not found while planning expression: {}",
          expression.ToString());
      return std::make_shared<FieldAccessTypedExpr>(
          duckdb::toVeloxType(ref->return_type),
          inputType->asRow().nameOf(index));
    }
    case ::duckdb::ExpressionType::BOUND_AGGREGATE: {
      auto* agg =
          dynamic_cast<::duckdb::BoundAggregateExpression*>(&expression);

      std::vector<TypedExprPtr> children;
      for (auto& child : agg->children) {
        children.push_back(toVeloxExpression(*child, inputType, inputBindings));
      }

      return std::make_shared<CallTypedExpr>(
          duckdb::toVeloxType(agg->return_type),
          std::move(children),
          mapAggregateFunctionName(agg->function.name));
    }
    default:
      VELOX_NYI(
          "Expression type {} is not supported yet: {}",
          ::duckdb::ExpressionTypeToString(expression.type),
          expression.ToString());
  }
}

PlanNodePtr toVeloxPlan(
    ::duckdb::LogicalFilter& logicalFilter,
    memory::MemoryPool* pool,
    std::vector<PlanNodePtr> sources,
    QueryContext& queryContext) {
  TypedExprPtr veloxFilter;
  auto inputBindings = logicalFilter.children[0]->GetColumnBindings();
  for (auto& expr : logicalFilter.expressions) {
    auto conjunct =
        toVeloxExpression(*expr, sources[0]->outputType(), &inputBindings);
    if (!veloxFilter) {
      veloxFilter = conjunct;
    } else {
      veloxFilter = std::make_shared<CallTypedExpr>(
          BOOLEAN(), "and", veloxFilter, conjunct);
    }
  }
  return std::make_shared<FilterNode>(
      queryContext.nextNodeId(), std::move(veloxFilter), std::move(sources[0]));
}

PlanNodePtr toVeloxPlan(
    ::duckdb::LogicalProjection& logicalProjection,
    memory::MemoryPool* pool,
    std::vector<PlanNodePtr> sources,
    QueryContext& queryContext) {
  std::vector<TypedExprPtr> projections;
  auto inputBindings = logicalProjection.children[0]->GetColumnBindings();
  for (auto& expression : logicalProjection.expressions) {
    projections.push_back(
        toVeloxExpression(
            *expression, sources[0]->outputType(), &inputBindings));
  }

  // TODO Figure out how to use these.
  auto columnBindings = logicalProjection.GetColumnBindings();

  std::vector<std::string> names;
  names.reserve(projections.size());
  for (auto i = 0; i < projections.size(); ++i) {
    names.push_back(queryContext.nextColumnName("_p"));
  }
  return std::make_shared<ProjectNode>(
      queryContext.nextNodeId(),
      std::move(names),
      std::move(projections),
      std::move(sources[0]));
}

namespace {
std::string translateAggregateName(const std::string& name) {
  // first(x) is used to get one element of a set. The closes Velox counterpart
  // is arbitrary, which usually returns the first value it sees.
  if (name == "first") {
    return "arbitrary";
  }
  return name;
}
} // namespace

PlanNodePtr toVeloxPlan(
    ::duckdb::LogicalAggregate& logicalAggregate,
    memory::MemoryPool* pool,
    std::vector<PlanNodePtr> sources,
    QueryContext& queryContext) {
  std::vector<AggregationNode::Aggregate> aggregates;
  auto inputBindings = logicalAggregate.children[0]->GetColumnBindings();

  std::vector<std::string> projectNames;
  std::vector<TypedExprPtr> projections;

  bool identityProjection = true;
  for (auto& expression : logicalAggregate.expressions) {
    auto call = std::dynamic_pointer_cast<const CallTypedExpr>(
        toVeloxExpression(
            *expression, sources[0]->outputType(), &inputBindings));
    if (expression->return_type.InternalType() ==
        ::duckdb::PhysicalType::INT128) {
      call = std::make_shared<CallTypedExpr>(
          BIGINT(), call->inputs(), call->name());
    }
    std::vector<TypedExprPtr> fieldInputs;
    std::vector<TypePtr> rawInputTypes;

    for (auto& input : call->inputs()) {
      projections.push_back(input);
      rawInputTypes.push_back(input->type());

      if (auto field =
              std::dynamic_pointer_cast<const FieldAccessTypedExpr>(input)) {
        projectNames.push_back(field->name());
        fieldInputs.push_back(field);
      } else {
        identityProjection = false;
        projectNames.push_back(queryContext.nextColumnName("_p"));
        fieldInputs.push_back(
            std::make_shared<FieldAccessTypedExpr>(
                input->type(), projectNames.back()));
      }
    }

    auto aggName = translateAggregateName(call->name());
    aggregates.push_back({
        std::make_shared<CallTypedExpr>(call->type(), fieldInputs, aggName),
        rawInputTypes,
        nullptr, // mask
        {}, // sortingKeys
        {} // sortingOrders
    });
  }

  std::vector<FieldAccessTypedExprPtr> groupingKeys;
  for (auto& expression : logicalAggregate.groups) {
    auto groupingExpr =
        toVeloxExpression(*expression, sources[0]->outputType(), &inputBindings);
    projections.push_back(groupingExpr);
    if (auto field = std::dynamic_pointer_cast<const FieldAccessTypedExpr>(
            groupingExpr)) {
      projectNames.push_back(field->name());
      groupingKeys.push_back(field);
    } else {
      identityProjection = false;
      projectNames.push_back(queryContext.nextColumnName("_p"));
      groupingKeys.push_back(
          std::make_shared<FieldAccessTypedExpr>(
              groupingExpr->type(), projectNames.back()));
    }
  }

  auto source = sources[0];

  if (!identityProjection) {
    source = std::make_shared<ProjectNode>(
        queryContext.nextNodeId(),
        std::move(projectNames),
        std::move(projections),
        std::move(sources[0]));
  }

  std::vector<std::string> names;
  names.reserve(aggregates.size());
  for (auto i = 0; i < aggregates.size(); ++i) {
    names.push_back(queryContext.nextColumnName("_a"));
  }

  return std::make_shared<AggregationNode>(
      queryContext.nextNodeId(),
      AggregationNode::Step::kSingle,
      groupingKeys,
      std::vector<FieldAccessTypedExprPtr>{}, // preGroupedKeys
      names,
      std::move(aggregates),
      /*ignoreNullKeys=*/false,
      /*noGroupsSpanBatches=*/false,
      source);
}

PlanNodePtr toVeloxPlan(
    ::duckdb::LogicalOrder& logicalOrder,
    memory::MemoryPool* pool,
    std::vector<PlanNodePtr> sources,
    QueryContext& queryContext) {
  VeloxColumnProjections projections(queryContext);
  std::vector<FieldAccessTypedExprPtr> keys;
  std::vector<SortOrder> sortOrder;
  const auto& source = sources[0];
  auto inputBindings = logicalOrder.children[0]->GetColumnBindings();
  for (auto& order : logicalOrder.orders) {
    keys.push_back(projections.toFieldAccess(
        *order.expression, source->outputType(), &inputBindings));
    sortOrder.push_back(SortOrder(
        order.type == ::duckdb::OrderType::ASCENDING ||
            order.type == ::duckdb::OrderType::ORDER_DEFAULT,
        order.null_order == ::duckdb::OrderByNullType::NULLS_FIRST ||
            order.null_order == ::duckdb::OrderByNullType::ORDER_DEFAULT));
  }

  return std::make_shared<OrderByNode>(
      queryContext.nextNodeId(),
      keys,
      sortOrder,
      /*isPartial=*/false,
      projections.source(source));
}

PlanNodePtr toVeloxPlan(
    ::duckdb::LogicalCrossProduct& logicalCrossProduct,
    memory::MemoryPool* pool,
    std::vector<PlanNodePtr> sources,
    QueryContext& queryContext) {
  VELOX_CHECK_EQ(2, sources.size());

  const auto& leftInputType = sources[0]->outputType()->asRow();
  const auto& rightInputType = sources[1]->outputType()->asRow();

  std::vector<std::string> names;
  std::vector<TypePtr> types;
  for (auto i = 0; i < leftInputType.size(); ++i) {
    names.push_back(leftInputType.nameOf(i));
    types.push_back(leftInputType.childAt(i));
  }
  for (auto i = 0; i < rightInputType.size(); ++i) {
    names.push_back(rightInputType.nameOf(i));
    types.push_back(rightInputType.childAt(i));
  }

  return std::make_shared<NestedLoopJoinNode>(
      queryContext.nextNodeId(),
      std::move(sources[0]),
      std::move(sources[1]),
      ROW(std::move(names), std::move(types)));
}

namespace {
std::vector<idx_t> columnIndices(std::vector<idx_t> map, int32_t size) {
  if (size > 0 && map.empty()) {
    std::vector<idx_t> result(size);
    std::iota(result.begin(), result.end(), 0);
    return result;
  }
  return map;
}

JoinType toVeloxJoinType(::duckdb::JoinType joinType) {
  switch (joinType) {
    case ::duckdb::JoinType::INNER:
    case ::duckdb::JoinType::SINGLE:
      return JoinType::kInner;
    case ::duckdb::JoinType::LEFT:
      return JoinType::kLeft;
    case ::duckdb::JoinType::RIGHT:
      return JoinType::kRight;
    case ::duckdb::JoinType::OUTER:
      return JoinType::kFull;
    case ::duckdb::JoinType::SEMI:
      return JoinType::kLeftSemiFilter;
    case ::duckdb::JoinType::ANTI:
      return JoinType::kAnti;
    case ::duckdb::JoinType::MARK:
      return JoinType::kLeftSemiProject;
    default:
      VELOX_NYI("Bad Duck join type {}", static_cast<int32_t>(joinType));
  }
}

RowTypePtr concatRowTypes(
    const RowTypePtr& leftType,
    const RowTypePtr& rightType) {
  std::vector<std::string> names;
  std::vector<TypePtr> types;
  names.reserve(leftType->size() + rightType->size());
  types.reserve(leftType->size() + rightType->size());
  for (auto i = 0; i < leftType->size(); ++i) {
    names.push_back(leftType->nameOf(i));
    types.push_back(leftType->childAt(i));
  }
  for (auto i = 0; i < rightType->size(); ++i) {
    names.push_back(rightType->nameOf(i));
    types.push_back(rightType->childAt(i));
  }
  return ROW(std::move(names), std::move(types));
}

DuckColumnBindings concatBindings(
    DuckColumnBindings leftBindings,
    const DuckColumnBindings& rightBindings) {
  leftBindings.insert(
      leftBindings.end(), rightBindings.begin(), rightBindings.end());
  return leftBindings;
}

RowTypePtr joinOutputType(
    ::duckdb::LogicalJoin& join,
    const RowTypePtr& leftInputType,
    const RowTypePtr& rightInputType,
    JoinType joinType,
    const PlanNodeId& joinNodeId) {
  std::vector<std::string> names;
  std::vector<TypePtr> types;
  for (auto i :
       columnIndices(join.left_projection_map, leftInputType->size())) {
    names.push_back(leftInputType->nameOf(i));
    types.push_back(leftInputType->childAt(i));
  }

  switch (joinType) {
    case JoinType::kLeftSemiFilter:
    case JoinType::kAnti:
      break;
    case JoinType::kLeftSemiProject:
      names.push_back(fmt::format("exists{}", joinNodeId));
      types.push_back(BOOLEAN());
      break;
    default:
      for (auto i :
           columnIndices(join.right_projection_map, rightInputType->size())) {
        names.push_back(rightInputType->nameOf(i));
        types.push_back(rightInputType->childAt(i));
      }
  }
  return ROW(std::move(names), std::move(types));
}

TypedExprPtr andExpr(TypedExprPtr left, TypedExprPtr right) {
  if (!left) {
    return right;
  }
  return std::make_shared<CallTypedExpr>(
      BOOLEAN(), "and", std::move(left), std::move(right));
}

TypedExprPtr toVeloxJoinCondition(
    const ::duckdb::JoinCondition& condition,
    const RowTypePtr& leftInputType,
    const RowTypePtr& rightInputType,
    const DuckColumnBindings& leftBindings,
    const DuckColumnBindings& rightBindings) {
  std::vector<TypedExprPtr> inputs{
      toVeloxExpression(*condition.left, leftInputType, &leftBindings),
      toVeloxExpression(*condition.right, rightInputType, &rightBindings)};
  auto call = std::make_shared<CallTypedExpr>(
      BOOLEAN(),
      std::move(inputs),
      comparisonFunctionName(condition.comparison));
  return maybeNegateNotDistinct(condition.comparison, std::move(call));
}

TypedExprPtr toVeloxJoinCondition(
    ::duckdb::LogicalComparisonJoin& join,
    const RowTypePtr& leftInputType,
    const RowTypePtr& rightInputType) {
  auto leftBindings = join.children[0]->GetColumnBindings();
  auto rightBindings = join.children[1]->GetColumnBindings();
  auto joinInputType = concatRowTypes(leftInputType, rightInputType);
  auto joinInputBindings = concatBindings(leftBindings, rightBindings);

  TypedExprPtr condition;
  for (auto& joinCondition : join.conditions) {
    condition = andExpr(
        std::move(condition),
        toVeloxJoinCondition(
            joinCondition,
            leftInputType,
            rightInputType,
            leftBindings,
            rightBindings));
  }
  for (auto& expression : join.expressions) {
    condition = andExpr(
        std::move(condition),
        toVeloxExpression(*expression, joinInputType, &joinInputBindings));
  }
  return condition;
}

bool canUseHashJoin(const ::duckdb::LogicalComparisonJoin& join) {
  if (join.conditions.empty()) {
    return false;
  }
  for (auto& condition : join.conditions) {
    if (condition.comparison != ::duckdb::ExpressionType::COMPARE_EQUAL) {
      return false;
    }
  }
  return true;
}
} // namespace

PlanNodePtr toVeloxPlan(
    ::duckdb::LogicalComparisonJoin& join,
    memory::MemoryPool* pool,
    std::vector<PlanNodePtr> sources,
    QueryContext& queryContext) {
  VELOX_CHECK_EQ(2, sources.size());

  auto leftInputType = sources[0]->outputType();
  auto rightInputType = sources[1]->outputType();
  auto joinNodeId = queryContext.nextNodeId();
  auto joinType = toVeloxJoinType(join.join_type);
  auto outputType =
      joinOutputType(join, leftInputType, rightInputType, joinType, joinNodeId);

  if (!canUseHashJoin(join)) {
    VELOX_USER_CHECK(
        NestedLoopJoinNode::isSupported(joinType),
        "The join type is not supported by nested loop join: {}",
        JoinTypeName::toName(joinType));
    return std::make_shared<NestedLoopJoinNode>(
        joinNodeId,
        joinType,
        toVeloxJoinCondition(join, leftInputType, rightInputType),
        std::move(sources[0]),
        std::move(sources[1]),
        outputType);
  }

  VeloxColumnProjections leftProjection(queryContext);
  VeloxColumnProjections rightProjection(queryContext);
  auto leftBindings = join.children[0]->GetColumnBindings();
  auto rightBindings = join.children[1]->GetColumnBindings();

  for (auto i :
       columnIndices(join.left_projection_map, leftInputType->size())) {
    auto source = std::make_shared<core::FieldAccessTypedExpr>(
        leftInputType->childAt(i), leftInputType->nameOf(i));
    leftProjection.addColumn(source);
  }
  if (!isProbeOnlyJoin(joinType)) {
    for (auto i :
         columnIndices(join.right_projection_map, rightInputType->size())) {
      auto source = std::make_shared<core::FieldAccessTypedExpr>(
          rightInputType->childAt(i), rightInputType->nameOf(i));
      rightProjection.addColumn(source);
    }
  }

  TypedExprPtr filter;
  auto outputBindings = join.GetColumnBindings();
  for (auto& expression : join.expressions) {
    filter = andExpr(
        std::move(filter),
        toVeloxExpression(*expression, outputType, &outputBindings));
  }

  std::vector<FieldAccessTypedExprPtr> leftKeys;
  std::vector<FieldAccessTypedExprPtr> rightKeys;
  for (auto& condition : join.conditions) {
    leftKeys.push_back(leftProjection.toFieldAccess(
        *condition.left, leftInputType, &leftBindings));
    rightKeys.push_back(
        rightProjection.toFieldAccess(
            *condition.right, rightInputType, &rightBindings));
  }

  return std::make_shared<HashJoinNode>(
      joinNodeId,
      joinType,
      false,
      std::move(leftKeys),
      std::move(rightKeys),
      filter,
      leftProjection.source(sources[0]),
      rightProjection.source(sources[1]),
      outputType);
}

PlanNodePtr toVeloxPlan(
    ::duckdb::LogicalAnyJoin& join,
    memory::MemoryPool* pool,
    std::vector<PlanNodePtr> sources,
    QueryContext& queryContext) {
  VELOX_CHECK_EQ(2, sources.size());
  auto leftInputType = sources[0]->outputType();
  auto rightInputType = sources[1]->outputType();
  auto joinNodeId = queryContext.nextNodeId();
  auto joinType = toVeloxJoinType(join.join_type);
  VELOX_USER_CHECK(
      NestedLoopJoinNode::isSupported(joinType),
      "The join type is not supported by nested loop join: {}",
      JoinTypeName::toName(joinType));

  auto leftBindings = join.children[0]->GetColumnBindings();
  auto rightBindings = join.children[1]->GetColumnBindings();
  auto joinInputType = concatRowTypes(leftInputType, rightInputType);
  auto joinInputBindings = concatBindings(leftBindings, rightBindings);
  return std::make_shared<NestedLoopJoinNode>(
      joinNodeId,
      joinType,
      toVeloxExpression(*join.condition, joinInputType, &joinInputBindings),
      std::move(sources[0]),
      std::move(sources[1]),
      joinOutputType(join, leftInputType, rightInputType, joinType, joinNodeId));
}

PlanNodePtr toVeloxPlan(
    ::duckdb::LogicalDelimGet& logicalDelimGet,
    memory::MemoryPool* pool,
    QueryContext& queryContext) {
  VELOX_CHECK(
      !queryContext.delimJoinStack.empty(),
      "LOGICAL_DELIM_GET must be planned under LOGICAL_DELIM_JOIN");
  auto& delimJoin = *queryContext.delimJoinStack.back().join;
  VELOX_CHECK_EQ(
      delimJoin.duplicate_eliminated_columns.size(),
      logicalDelimGet.chunk_types.size());

  auto left = toVeloxPlan(*delimJoin.children[0], pool, queryContext);
  auto leftBindings = delimJoin.children[0]->GetColumnBindings();

  std::vector<std::string> names;
  std::vector<TypedExprPtr> exprs;
  std::vector<FieldAccessTypedExprPtr> groupingKeys;
  names.reserve(delimJoin.duplicate_eliminated_columns.size());
  exprs.reserve(delimJoin.duplicate_eliminated_columns.size());
  groupingKeys.reserve(delimJoin.duplicate_eliminated_columns.size());
  for (auto& expression : delimJoin.duplicate_eliminated_columns) {
    auto veloxExpr =
        toVeloxExpression(*expression, left->outputType(), &leftBindings);
    auto name = queryContext.nextColumnName("_delim");
    groupingKeys.push_back(
        std::make_shared<FieldAccessTypedExpr>(veloxExpr->type(), name));
    names.push_back(std::move(name));
    exprs.push_back(std::move(veloxExpr));
  }

  auto project = std::make_shared<ProjectNode>(
      queryContext.nextNodeId(), std::move(names), std::move(exprs), left);
  return std::make_shared<AggregationNode>(
      queryContext.nextNodeId(),
      AggregationNode::Step::kSingle,
      groupingKeys,
      std::vector<FieldAccessTypedExprPtr>{}, // preGroupedKeys
      std::vector<std::string>{}, // aggregateNames
      std::vector<AggregationNode::Aggregate>{}, // aggregates
      /*ignoreNullKeys=*/false,
      /*noGroupsSpanBatches=*/false,
      project);
}

PlanNodePtr toVeloxPlan(
    ::duckdb::LogicalDelimJoin& join,
    memory::MemoryPool* pool,
    QueryContext& queryContext) {
  VELOX_CHECK_EQ(2, join.children.size());
  std::vector<PlanNodePtr> sources;
  sources.push_back(toVeloxPlan(*join.children[0], pool, queryContext));
  queryContext.delimJoinStack.push_back(QueryContext::DelimJoinContext{&join});
  sources.push_back(toVeloxPlan(*join.children[1], pool, queryContext));
  queryContext.delimJoinStack.pop_back();
  return toVeloxPlan(
      static_cast<::duckdb::LogicalComparisonJoin&>(join),
      pool,
      std::move(sources),
      queryContext);
}

PlanNodePtr toVeloxPlan(
    ::duckdb::LogicalOperator& plan,
    memory::MemoryPool* pool,
    QueryContext& queryContext) {
  std::vector<PlanNodePtr> sources;

  if (plan.type == ::duckdb::LogicalOperatorType::LOGICAL_DELIM_JOIN) {
    return toVeloxPlan(
        dynamic_cast<::duckdb::LogicalDelimJoin&>(plan), pool, queryContext);
  }
  if (plan.type == ::duckdb::LogicalOperatorType::LOGICAL_DELIM_GET) {
    return toVeloxPlan(
        dynamic_cast<::duckdb::LogicalDelimGet&>(plan), pool, queryContext);
  }
  for (auto& child : plan.children) {
    sources.push_back(toVeloxPlan(*child, pool, queryContext));
    if (sources.back() == nullptr) {
      VELOX_FAIL("null plan for: {}", child->ToString());
    }
  }

  switch (plan.type) {
    case ::duckdb::LogicalOperatorType::LOGICAL_DUMMY_SCAN:
      return toVeloxPlan(
          dynamic_cast<::duckdb::LogicalDummyScan&>(plan), pool, queryContext);
    case ::duckdb::LogicalOperatorType::LOGICAL_GET:
      return toVeloxPlan(
          dynamic_cast<::duckdb::LogicalGet&>(plan),
          pool,
          std::move(sources),
          queryContext);
    case ::duckdb::LogicalOperatorType::LOGICAL_FILTER:
      return toVeloxPlan(
          dynamic_cast<::duckdb::LogicalFilter&>(plan),
          pool,
          std::move(sources),
          queryContext);
    case ::duckdb::LogicalOperatorType::LOGICAL_PROJECTION:
      return toVeloxPlan(
          dynamic_cast<::duckdb::LogicalProjection&>(plan),
          pool,
          std::move(sources),
          queryContext);
    case ::duckdb::LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY:
      return toVeloxPlan(
          dynamic_cast<::duckdb::LogicalAggregate&>(plan),
          pool,
          std::move(sources),
          queryContext);
    case ::duckdb::LogicalOperatorType::LOGICAL_CROSS_PRODUCT:
      return toVeloxPlan(
          dynamic_cast<::duckdb::LogicalCrossProduct&>(plan),
          pool,
          std::move(sources),
          queryContext);
    case ::duckdb::LogicalOperatorType::LOGICAL_ORDER_BY: {
      return toVeloxPlan(
          dynamic_cast<::duckdb::LogicalOrder&>(plan),
          pool,
          std::move(sources),
          queryContext);
    }
    case ::duckdb::LogicalOperatorType::LOGICAL_LIMIT: {
      auto& limit = dynamic_cast<const ::duckdb::LogicalLimit&>(plan);
      return std::make_shared<core::LimitNode>(
          queryContext.nextNodeId(),
          limit.offset_val,
          limit.limit_val,
          false,
          sources[0]);
    }
    case ::duckdb::LogicalOperatorType::LOGICAL_COMPARISON_JOIN:
      return toVeloxPlan(
          dynamic_cast<::duckdb::LogicalComparisonJoin&>(plan),
          pool,
          std::move(sources),
          queryContext);
    case ::duckdb::LogicalOperatorType::LOGICAL_ANY_JOIN:
      return toVeloxPlan(
          dynamic_cast<::duckdb::LogicalAnyJoin&>(plan),
          pool,
          std::move(sources),
          queryContext);
    default:
      VELOX_NYI(
          "Plan node is not supported yet: {}",
          ::duckdb::LogicalOperatorToString(plan.type));
  }
}

static void customScalarFunction(
    ::duckdb::DataChunk& args,
    ::duckdb::ExpressionState& state,
    ::duckdb::Vector& result) {
  VELOX_UNREACHABLE();
}

static ::duckdb::idx_t customAggregateState() {
  VELOX_UNREACHABLE();
}

static void customAggregateInitialize(::duckdb::data_ptr_t) {
  VELOX_UNREACHABLE();
}

static void customAggregateUpdate(
    ::duckdb::Vector inputs[],
    ::duckdb::AggregateInputData& aggr_input_data,
    ::duckdb::idx_t input_count,
    ::duckdb::Vector& state,
    ::duckdb::idx_t count) {
  VELOX_UNREACHABLE();
}

static void customAggregateCombine(
    ::duckdb::Vector& state,
    ::duckdb::Vector& combined,
    ::duckdb::AggregateInputData& aggr_input_data,
    ::duckdb::idx_t count) {
  VELOX_UNREACHABLE();
}

static void customAggregateFinalize(
    ::duckdb::Vector& state,
    ::duckdb::AggregateInputData& aggr_input_data,
    ::duckdb::Vector& result,
    ::duckdb::idx_t count,
    ::duckdb::idx_t offset) {
  VELOX_UNREACHABLE();
}

} // namespace

PlanNodePtr parseQuery(
    const std::string& sql,
    memory::MemoryPool* pool,
    const std::unordered_map<std::string, std::vector<RowVectorPtr>>&
        inMemoryTables) {
  DuckDbQueryPlanner planner(pool);

  for (auto& [name, data] : inMemoryTables) {
    planner.registerTable(name, data);
  }

  return planner.plan(sql);
}

void DuckDbQueryPlanner::registerTable(
    const std::string& name,
    const std::vector<RowVectorPtr>& data) {
  VELOX_CHECK_EQ(
      tables_.count(name), 0, "Table is already registered: {}", name);

  auto createTableSql =
      duckdb::makeCreateTableSql(name, *asRowType(data[0]->type()));
  auto res = conn_.Query(createTableSql);
  VELOX_CHECK(
      !res->HasError(), "Failed to create DuckDB table: {}", res->GetError());

  tables_.insert({name, data});
}

void DuckDbQueryPlanner::registerTable(
    const std::string& name,
    const RowTypePtr& type) {
  VELOX_CHECK_EQ(
      tables_.count(name), 0, "Table is already registered: {}", name);

  auto createTableSql = duckdb::makeCreateTableSql(name, *type);
  auto res = conn_.Query(createTableSql);
}

void DuckDbQueryPlanner::registerScalarFunction(
    const std::string& name,
    const std::vector<TypePtr>& argTypes,
    const TypePtr& returnType) {
  ::duckdb::vector<::duckdb::LogicalType> argDuckTypes;
  for (auto& type : argTypes) {
    argDuckTypes.push_back(duckdb::fromVeloxType(type));
  }

  conn_.CreateVectorizedFunction(
      name,
      argDuckTypes,
      duckdb::fromVeloxType(returnType),
      customScalarFunction);
}

void DuckDbQueryPlanner::registerAggregateFunction(
    const std::string& name,
    const std::vector<TypePtr>& argTypes,
    const TypePtr& returnType) {
  ::duckdb::vector<::duckdb::LogicalType> argDuckTypes;
  for (auto& type : argTypes) {
    argDuckTypes.push_back(duckdb::fromVeloxType(type));
  }

  conn_.CreateAggregateFunction(
      name,
      argDuckTypes,
      duckdb::fromVeloxType(returnType),
      customAggregateState,
      customAggregateInitialize,
      customAggregateUpdate,
      customAggregateCombine,
      customAggregateFinalize);
}

PlanNodePtr DuckDbQueryPlanner::plan(const std::string& sql) {
  // Disable the optimizer. Otherwise, the filter over table scan gets pushdown
  // as a callback that is impossible to recover.
  conn_.Query("PRAGMA disable_optimizer");

  auto plan = conn_.ExtractPlan(sql);

  QueryContext queryContext{tables_};
  queryContext.makeTableScan = makeTableScan_;
  return toVeloxPlan(*plan, pool_, queryContext);
}

} // namespace facebook::velox::core
