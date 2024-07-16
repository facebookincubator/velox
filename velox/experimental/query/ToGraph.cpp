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

#include "velox/connectors/hive/HiveConnector.h"
#include "velox/connectors/hive/TableHandle.h"
#include "velox/exec/Aggregate.h"
#include "velox/experimental/query/Plan.h"
#include "velox/experimental/query/PlanUtils.h"
#include "velox/expression/ConstantExpr.h"

namespace facebook::verax {

using namespace facebook::velox;

using velox::connector::hive::HiveColumnHandle;
using velox::connector::hive::HiveTableHandle;
std::string veloxToString(core::PlanNode* plan) {
  return plan->toString(true, true);
}

void Optimization::setDerivedTableOutput(
    DerivedTablePtr dt,
    const velox::core::PlanNode& planNode) {
  auto& outputType = planNode.outputType();
  for (auto i = 0; i < outputType->size(); ++i) {
    auto fieldType = outputType->childAt(i);
    registerType(fieldType);
    auto fieldName = outputType->nameOf(i);
    auto expr = translateColumn(fieldName);
    Value value(fieldType.get(), 0);
    Declare(Column, column, toName(fieldName), dt, value);
    dt->columns.push_back(column);
    dt->exprs.push_back(expr);
    renames_[fieldName] = column;
  }
}

DerivedTablePtr Optimization::makeQueryGraph() {
  Declare(DerivedTable, root);
  root_ = root;
  currentSelect_ = root_;
  root->cname = toName(fmt::format("dt{}", ++nameCounter_));
  makeQueryGraph(inputPlan_, kAllAllowedInDt);
  return root_;
}

const std::string* FOLLY_NULLABLE columnName(const core::TypedExprPtr& expr) {
  if (auto column =
          dynamic_cast<const core::FieldAccessTypedExpr*>(expr.get())) {
    if (column->inputs().empty() ||
        dynamic_cast<const core::InputTypedExpr*>(column->inputs()[0].get())) {
      return &column->name();
    }
  }
  return nullptr;
}

bool isCall(const core::TypedExprPtr& expr, const std::string& name) {
  if (auto call = std::dynamic_pointer_cast<const core::CallTypedExpr>(expr)) {
    return call->name() == name;
  }
  return false;
}

void Optimization::translateConjuncts(
    const core::TypedExprPtr& input,
    ExprVector& flat) {
  if (!input) {
    return;
  }
  if (isCall(input, "and")) {
    for (auto& child : input->inputs()) {
      translateConjuncts(child, flat);
    }
  } else {
    flat.push_back(translateExpr(input));
  }
}

void Optimization::registerType(const TypePtr& type) {
  if (toTypePtr_.find(type.get()) != toTypePtr_.end()) {
    return;
  }
  toTypePtr_[type.get()] = type;
  for (auto i = 0; i < type->size(); ++i) {
    registerType(type->childAt(i));
  }
}

TypePtr Optimization::toTypePtr(const Type* type) {
  auto it = toTypePtr_.find(type);
  if (it != toTypePtr_.end()) {
    return it->second;
  }
  VELOX_FAIL("Cannot translate {} back to TypePtr", type->toString());
}

template <TypeKind kind>
variant toVariant(BaseVector& constantVector) {
  using T = typename TypeTraits<kind>::NativeType;
  if (auto typed = dynamic_cast<ConstantVector<T>*>(&constantVector)) {
    return variant(typed->valueAt(0));
  }
  VELOX_FAIL("Literal not of foldable type");
}

ExprPtr Optimization::tryFoldConstant(
    const core::CallTypedExpr* call,
    const core::CastTypedExpr* cast,
    const ExprVector& literals) {
  try {
    Value value(call ? call->type().get() : cast->type().get(), 1);
    Declare(
        Call,
        veraxExpr,
        PlanType::kCall,
        cast ? toName("cast") : toName(call->name()),
        value,
        literals,
        FunctionSet());
    auto typedExpr = toTypedExpr(veraxExpr);
    auto exprSet = evaluator_.compile(typedExpr);
    auto first = exprSet->exprs().front().get();
    if (auto constantExpr = dynamic_cast<const exec::ConstantExpr*>(first)) {
      auto variantLiteral = VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH(
          toVariant, constantExpr->value()->typeKind(), *constantExpr->value());
      Value value(constantExpr->value()->type().get(), 1);
      // Copy the variant from value to allocated in arena.
      Declare(variant, copy, variantLiteral);
      Declare(Literal, literal, value, copy);
      return literal;
    }
    return nullptr;
  } catch (const std::exception& e) {
    return nullptr;
  }
}

ExprPtr Optimization::translateExpr(const core::TypedExprPtr& expr) {
  registerType(expr->type());
  if (auto name = columnName(expr)) {
    return translateColumn(*name);
  }
  if (auto constant =
          dynamic_cast<const core::ConstantTypedExpr*>(expr.get())) {
    Declare(
        Literal, literal, Value(constant->type().get(), 1), &constant->value());
    return literal;
  }
  auto it = exprDedup_.find(expr.get());
  if (it != exprDedup_.end()) {
    return it->second;
  }
  ExprVector args{expr->inputs().size()};
  PlanObjectSet columns;
  FunctionSet funcs;
  auto& inputs = expr->inputs();
  float cardinality = 1;
  bool allConstant = true;
  for (auto i = 0; i < inputs.size(); ++i) {
    args[i] = translateExpr(inputs[i]);
    allConstant &= args[i]->type() == PlanType::kLiteral;
    cardinality = std::max(cardinality, args[i]->value().cardinality);
    if (args[i]->type() == PlanType::kCall) {
      funcs = funcs | args[i]->as<Call>()->functions();
    }
  }
  auto call = dynamic_cast<const core::CallTypedExpr*>(expr.get());
  auto cast = dynamic_cast<const core::CastTypedExpr*>(expr.get());
  if (allConstant && (call || cast)) {
    auto literal = tryFoldConstant(call, cast, args);
    if (literal) {
      return literal;
    }
  }

  if (call) {
    auto name = toName(call->name());
    funcs = funcs | functionBits(name);

    Declare(
        Call,
        callExpr,
        name,
        Value(call->type().get(), cardinality),
        args,
        funcs);
    exprDedup_[expr.get()] = callExpr;
    return callExpr;
  }
  if (cast) {
    auto name = toName("cast");
    funcs = funcs | functionBits(name);

    Declare(
        Call,
        callExpr,
        name,
        Value(cast->type().get(), cardinality),
        args,
        funcs);
    exprDedup_[expr.get()] = callExpr;
    return callExpr;
  }

  VELOX_NYI();
  return nullptr;
}

ExprPtr Optimization::translateColumn(const std::string& name) {
  auto column = renames_.find(name);
  if (column != renames_.end()) {
    return column->second;
  }
  VELOX_FAIL("could not resolve name {}", name);
}

ExprVector Optimization::translateColumns(
    const std::vector<core::FieldAccessTypedExprPtr>& source) {
  ExprVector result{source.size()};
  for (auto i = 0; i < source.size(); ++i) {
    registerType(source[i]->type());
    result[i] = translateColumn(source[i]->name()); // NOLINT
  }
  return result;
}

TypePtr intermediateType(const core::CallTypedExprPtr& call) {
  std::vector<TypePtr> types;
  for (auto& arg : call->inputs()) {
    types.push_back(arg->type());
  }
  return exec::Aggregate::intermediateType(call->name(), types);
}

AggregationPtr FOLLY_NULLABLE
Optimization::translateAggregation(const core::AggregationNode& source) {
  using velox::core::AggregationNode;

  if (source.step() == AggregationNode::Step::kPartial ||
      source.step() == AggregationNode::Step::kSingle) {
    Declare(
        Aggregation,
        aggregation,
        nullptr,
        translateColumns(source.groupingKeys()));
    for (auto i = 0; i < source.groupingKeys().size(); ++i) {
      if (aggregation->grouping[i]->type() == PlanType::kColumn) {
        aggregation->mutableColumns().push_back(
            aggregation->grouping[i]->as<Column>());
      } else {
        auto name = toName(source.outputType()->nameOf(i));
        auto type = source.outputType()->childAt(i);
        registerType(type);
        Declare(
            Column,
            column,
            name,
            currentSelect_,
            aggregation->grouping[i]->value());
        aggregation->mutableColumns().push_back(column);
      }
    }
    // The keys for intermediate are the same as for final.
    aggregation->intermediateColumns = aggregation->columns();
    for (auto i = 0; i < source.aggregateNames().size(); ++i) {
      auto rawFunc = translateExpr(source.aggregates()[i].call)->as<Call>();
      ExprPtr condition = nullptr;
      if (source.aggregates()[i].mask) {
        condition = translateExpr(source.aggregates()[i].mask);
      }
      VELOX_CHECK(source.aggregates()[i].sortingKeys.empty());
      auto accumulatorType = intermediateType(source.aggregates()[i].call);
      registerType(accumulatorType);
      Declare(
          Aggregate,
          agg,
          rawFunc->name(),
          rawFunc->value(),
          rawFunc->args(),
          rawFunc->functions(),
          false,
          condition,
          false,
          accumulatorType.get());
      auto name = toName(source.aggregateNames()[i]);
      Declare(Column, column, name, currentSelect_, agg->value());
      aggregation->mutableColumns().push_back(column);
      auto intermediateValue = agg->value();
      intermediateValue.type = accumulatorType.get();
      Declare(
          Column, intermediateColumn, name, currentSelect_, intermediateValue);
      aggregation->intermediateColumns.push_back(intermediateColumn);
      auto dedupped = queryCtx()->dedup(agg);
      aggregation->aggregates.push_back(dedupped->as<Aggregate>());
      auto resultName = toName(source.aggregateNames()[i]);
      renames_[resultName] = aggregation->columns().back();
    }
    return aggregation;
  }
  return nullptr;
}

OrderByPtr Optimization::translateOrderBy(const core::OrderByNode& order) {
  OrderTypeVector orderType;
  for (auto& sort : order.sortingOrders()) {
    orderType.push_back(
        sort.isAscending() ? (sort.isNullsFirst() ? OrderType::kAscNullsFirst
                                                  : OrderType::kAscNullsLast)
                           : (sort.isNullsFirst() ? OrderType::kDescNullsFirst
                                                  : OrderType::kDescNullsLast));
  }
  auto keys = translateColumns(order.sortingKeys());
  Declare(OrderBy, orderBy, nullptr, keys, orderType, {});
  return orderBy;
}

ColumnPtr Optimization::makeMark(const core::AbstractJoinNode& join) {
  auto type = join.outputType();
  auto name = toName(type->nameOf(type->size() - 1));
  Value value(type->childAt(type->size() - 1).get(), 2);
  registerType(type->childAt(type->size() - 1));
  Declare(Column, column, name, currentSelect_, value);
  return column;
}

void Optimization::translateJoin(const core::AbstractJoinNode& join) {
  bool isInner = join.isInnerJoin();
  makeQueryGraph(*join.sources()[0], allow(PlanType::kJoin));
  auto leftKeys = translateColumns(join.leftKeys());
  // For an inner join a join tree on the right can be flattened, for all other
  // kinds it must be kept together in its own dt.
  makeQueryGraph(*join.sources()[1], isInner ? allow(PlanType::kJoin) : 0);
  auto rightKeys = translateColumns(join.rightKeys());
  ExprVector conjuncts;
  translateConjuncts(join.filter(), conjuncts);
  if (isInner) {
    // Every column to column equality adds to an equivalence class and is an
    // independent bidirectional join edge.
    for (auto i = 0; i < leftKeys.size(); ++i) {
      auto l = leftKeys[i];
      auto r = rightKeys.at(i);
      if (l->type() == PlanType::kColumn && r->type() == PlanType::kColumn) {
        l->as<Column>()->equals(r->as<Column>());
        currentSelect_->addJoinEquality(l, r, {}, false, false, false, false);
      } else {
        currentSelect_->addJoinEquality(l, r, {}, false, false, false, false);
      }
    }
    currentSelect_->conjuncts.insert(
        currentSelect_->conjuncts.end(), conjuncts.begin(), conjuncts.end());
  } else {
    auto joinType = join.joinType();
    bool leftOptional =
        joinType == core::JoinType::kRight || joinType == core::JoinType::kFull;
    bool rightOptional =
        joinType == core::JoinType::kLeft || joinType == core::JoinType::kFull;
    bool rightExists = joinType == core::JoinType::kLeftSemiFilter;
    bool rightNotExists = joinType == core::JoinType::kAnti;
    ColumnPtr markColumn =
        joinType == core::JoinType::kLeftSemiProject ? makeMark(join) : nullptr;
    ;

    PlanObjectSet leftTables;
    PlanObjectConstPtr rightTable = nullptr;

    for (auto i = 0; i < leftKeys.size(); ++i) {
      auto l = leftKeys[i];
      leftTables.unionSet(l->allTables());
      auto r = rightKeys.at(i);
      auto rightKeyTable = r->singleTable();
      if (rightTable) {
        VELOX_CHECK(rightKeyTable == rightTable);
      } else {
        rightTable = rightKeyTable;
      }
    }
    VELOX_CHECK(rightTable, "No right side in join");
    std::vector<PlanObjectConstPtr> leftTableVector;
    leftTables.forEach(
        [&](PlanObjectConstPtr table) { leftTableVector.push_back(table); });
    Declare(
        JoinEdge,
        edge,
        leftTableVector.size() == 1 ? leftTableVector[0] : nullptr,
        rightTable,
        conjuncts,
        leftOptional,
        rightOptional,
        rightExists,
        rightNotExists,
        markColumn);
    if (markColumn) {
      renames_[markColumn->name()] = markColumn;
    }
    currentSelect_->joins.push_back(edge);
    for (auto i = 0; i < leftKeys.size(); ++i) {
      edge->addEquality(leftKeys[i], rightKeys[i]);
    }
  }
}

void Optimization::translateNonEqualityJoin(
    const core::NestedLoopJoinNode& join) {
  auto joinType = join.joinType();
  bool isInner = joinType == core::JoinType::kInner;
  makeQueryGraph(*join.sources()[0], allow(PlanType::kJoin));
  // For an inner join a join tree on the right can be flattened, for all other
  // kinds it must be kept together in its own dt.
  makeQueryGraph(*join.sources()[1], isInner ? allow(PlanType::kJoin) : 0);
  ExprVector conjuncts;
  translateConjuncts(join.joinCondition(), conjuncts);
  if (conjuncts.empty()) {
    // Inner cross product. Join conditions may be added from
    // conjuncts of the enclosing DerivedTable.
    return;
  }
  PlanObjectSet tables;
  for (auto& conjunct : conjuncts) {
    tables.unionColumns(conjunct);
  }
  std::vector<PlanObjectConstPtr> tableVector;
  tables.forEach(
      [&](PlanObjectConstPtr table) { tableVector.push_back(table); });
  if (tableVector.size() == 2) {
    Declare(
        JoinEdge,
        edge,
        tableVector[0],
        tableVector[1],
        conjuncts,
        false,
        false,
        false,
        false);
    edge->guessFanout();
    currentSelect_->joins.push_back(edge);

  } else {
    VELOX_NYI("Multiway non-equality join not supported");
    currentSelect_->conjuncts.insert(
        currentSelect_->conjuncts.end(), conjuncts.begin(), conjuncts.end());
  }
}

bool isJoin(const core::PlanNode& node) {
  auto name = node.name();
  if (name == "HashJoin" || name == "MergeJoin" || name == "NestedLoopJoin") {
    return true;
  }
  if (name == "Project" || name == "Filter") {
    return isJoin(*node.sources()[0]);
  }
  return false;
}

bool isDirectOver(const core::PlanNode& node, const std::string& name) {
  auto source = node.sources()[0];
  if (source && source->name() == name) {
    return true;
  }
  return false;
}

PlanObjectPtr Optimization::wrapInDt(const core::PlanNode& node) {
  DerivedTablePtr previousDt = currentSelect_;
  Declare(DerivedTable, newDt);
  auto cname = toName(fmt::format("dt{}", ++nameCounter_));
  newDt->cname = cname;
  currentSelect_ = newDt;
  makeQueryGraph(node, kAllAllowedInDt);

  currentSelect_ = previousDt;
  velox::RowTypePtr type = node.outputType();
  // node.name() == "Aggregation" ? aggFinalType_ : node.outputType();
  for (auto i = 0; i < type->size(); ++i) {
    registerType(type->childAt(i));
    ExprPtr inner = translateColumn(type->nameOf(i));
    newDt->exprs.push_back(inner);
    Declare(Column, outer, toName(type->nameOf(i)), newDt, inner->value());
    newDt->columns.push_back(outer);
    renames_[type->nameOf(i)] = outer;
  }
  currentSelect_->tables.push_back(newDt);
  currentSelect_->tableSet.add(newDt);
  newDt->makeInitialPlan();

  return newDt;
}

PlanObjectPtr Optimization::makeQueryGraph(
    const core::PlanNode& node,
    uint64_t allowedInDt) {
  auto name = node.name();
  if (isJoin(node) && !contains(allowedInDt, PlanType::kJoin)) {
    return wrapInDt(node);
  }
  if (name == "TableScan") {
    auto tableScan = reinterpret_cast<const core::TableScanNode*>(&node);
    auto tableHandle =
        dynamic_cast<const HiveTableHandle*>(tableScan->tableHandle().get());
    VELOX_CHECK(tableHandle);
    auto assignments = tableScan->assignments();
    auto schemaTable = schema_.findTable(tableHandle->tableName());
    auto cname = fmt::format("t{}", ++nameCounter_);

    Declare(BaseTable, baseTable);
    baseTable->cname = toName(cname);
    baseTable->schemaTable = schemaTable;
    ColumnVector columns;
    ColumnVector schemaColumns;
    for (auto& pair : assignments) {
      auto handle =
          reinterpret_cast<const HiveColumnHandle*>(pair.second.get());
      auto schemaColumn = schemaTable->findColumn(handle->name());
      schemaColumns.push_back(schemaColumn);
      auto value = schemaColumn->value();
      Declare(Column, column, toName(handle->name()), baseTable, value);
      columns.push_back(column);
      renames_[pair.first] = column;
    }
    baseTable->columns = columns;

    setLeafHandle(baseTable->id(), tableScan->tableHandle());
    setLeafSelectivity(*baseTable);
    currentSelect_->tables.push_back(baseTable);
    currentSelect_->tableSet.add(baseTable);
    return baseTable;
  }
  if (name == "Project") {
    makeQueryGraph(*node.sources()[0], allowedInDt);
    auto project = reinterpret_cast<const core::ProjectNode*>(&node);
    auto names = project->names();
    auto exprs = project->projections();
    for (auto i = 0; i < names.size(); ++i) {
      if (auto field = dynamic_cast<const core::FieldAccessTypedExpr*>(
              exprs.at(i).get())) {
        // A variable projected to itself adds no renames. Inputs contain this
        // all the time.
        if (field->name() == names[i]) {
          continue;
        }
      }
      auto expr = translateExpr(exprs.at(i));
      renames_[names[i]] = expr;
    }
    return currentSelect_;
  }
  if (name == "Filter") {
    makeQueryGraph(*node.sources()[0], allowedInDt);
    auto filter = reinterpret_cast<const core::FilterNode*>(&node);
    ExprVector flat;
    translateConjuncts(filter->filter(), flat);
    if (isDirectOver(node, "Aggregation")) {
      VELOX_CHECK(
          currentSelect_->having.empty(),
          "Must have aall of HAVING in one filter");
      currentSelect_->having = flat;
    } else {
      currentSelect_->conjuncts.insert(
          currentSelect_->conjuncts.end(), flat.begin(), flat.end());
    }
    return currentSelect_;
  }
  if (name == "HashJoin" || name == "MergeJoin") {
    if (!contains(allowedInDt, PlanType::kJoin)) {
      return wrapInDt(node);
    }
    translateJoin(*reinterpret_cast<const core::AbstractJoinNode*>(&node));
    return currentSelect_;
  }
  if (name == "NestedLoopJoin") {
    if (!contains(allowedInDt, PlanType::kJoin)) {
      return wrapInDt(node);
    }
    translateNonEqualityJoin(
        *reinterpret_cast<const core::NestedLoopJoinNode*>(&node));
    return currentSelect_;
  }
  if (name == "LocalPartition") {
    makeQueryGraph(*node.sources()[0], allowedInDt);
    return currentSelect_;
  }
  if (name == "Aggregation") {
    using AggregationNode = velox::core::AggregationNode;
    auto& aggNode = *reinterpret_cast<const core::AggregationNode*>(&node);
    if (aggNode.step() == AggregationNode::Step::kPartial ||
        aggNode.step() == AggregationNode::Step::kSingle) {
      if (!contains(allowedInDt, PlanType::kAggregation)) {
        return wrapInDt(node);
      }
      if (aggNode.step() == AggregationNode::Step::kSingle) {
        aggFinalType_ = aggNode.outputType();
      }
      makeQueryGraph(
          *node.sources()[0], makeDtIf(allowedInDt, PlanType::kAggregation));
      auto agg = translateAggregation(aggNode);
      if (agg) {
        Declare(AggregationPlan, aggPlan, agg);
        currentSelect_->aggregation = aggPlan;
      }
    } else {
      if (aggNode.step() == AggregationNode::Step::kFinal) {
        aggFinalType_ = aggNode.outputType();
      }
      makeQueryGraph(*aggNode.sources()[0], allowedInDt);
    }
    return currentSelect_;
  }
  if (name == "OrderBy") {
    if (!contains(allowedInDt, PlanType::kOrderBy)) {
      return wrapInDt(node);
    }
    makeQueryGraph(
        *node.sources()[0], makeDtIf(allowedInDt, PlanType::kOrderBy));
    currentSelect_->orderBy =
        translateOrderBy(*reinterpret_cast<const core::OrderByNode*>(&node));
    return currentSelect_;
  }
  if (name == "Limit") {
    if (!contains(allowedInDt, PlanType::kLimit)) {
      return wrapInDt(node);
    }
    makeQueryGraph(*node.sources()[0], makeDtIf(allowedInDt, PlanType::kLimit));
    auto limit = reinterpret_cast<const core::LimitNode*>(&node);
    currentSelect_->limit = limit->count();
    currentSelect_->offset = limit->offset();
  } else {
    VELOX_NYI("Unsupported PlanNode {}", name);
  }
  return currentSelect_;
}

} // namespace facebook::verax
