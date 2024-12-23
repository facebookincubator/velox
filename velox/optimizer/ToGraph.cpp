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

#include "velox/exec/Aggregate.h"
#include "velox/expression/ConstantExpr.h"
#include "velox/optimizer/Plan.h"
#include "velox/optimizer/PlanUtils.h"

namespace facebook::velox::optimizer {

using namespace facebook::velox;

std::string veloxToString(core::PlanNode* plan) {
  return plan->toString(true, true);
}

void Optimization::setDerivedTableOutput(
    DerivedTableP dt,
    const velox::core::PlanNode& planNode) {
  auto& outputType = planNode.outputType();
  for (auto i = 0; i < outputType->size(); ++i) {
    auto fieldType = outputType->childAt(i);
    auto fieldName = outputType->nameOf(i);
    auto expr = translateColumn(fieldName);
    Value value(toType(fieldType), 0);
    auto* column = make<Column>(toName(fieldName), dt, value);
    dt->columns.push_back(column);
    dt->exprs.push_back(expr);
    renames_[fieldName] = column;
  }
}

DerivedTableP Optimization::makeQueryGraph() {
  auto* root = make<DerivedTable>();
  root_ = root;
  currentSelect_ = root_;
  root->cname = toName(fmt::format("dt{}", ++nameCounter_));
  makeQueryGraph(inputPlan_, kAllAllowedInDt);
  return root_;
}

const std::string* columnName(const core::TypedExprPtr& expr) {
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

template <TypeKind kind>
variant toVariant(BaseVector& constantVector) {
  using T = typename TypeTraits<kind>::NativeType;
  if (auto typed = dynamic_cast<ConstantVector<T>*>(&constantVector)) {
    return variant(typed->valueAt(0));
  }
  VELOX_FAIL("Literal not of foldable type");
}

ExprCP Optimization::tryFoldConstant(
    const core::CallTypedExpr* call,
    const core::CastTypedExpr* cast,
    const ExprVector& literals) {
  try {
    Value value(call ? toType(call->type()) : toType(cast->type()), 1);
    auto* veraxExpr = make<Call>(
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
      Value value(toType(constantExpr->value()->type()), 1);
      // Copy the variant from value to allocated in arena.
      auto* copy = make<variant>(variantLiteral);
      auto* literal = make<Literal>(value, copy);
      return literal;
    }
    return nullptr;
  } catch (const std::exception& e) {
    return nullptr;
  }
}

ExprCP Optimization::translateExpr(const core::TypedExprPtr& expr) {
  if (auto name = columnName(expr)) {
    return translateColumn(*name);
  }
  if (auto constant =
          dynamic_cast<const core::ConstantTypedExpr*>(expr.get())) {
    auto* literal =
        make<Literal>(Value(toType(constant->type()), 1), &constant->value());
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

    auto* callExpr =
        make<Call>(name, Value(toType(call->type()), cardinality), args, funcs);
    exprDedup_[expr.get()] = callExpr;
    return callExpr;
  }
  if (cast) {
    auto name = toName("cast");
    funcs = funcs | functionBits(name);

    auto* callExpr =
        make<Call>(name, Value(toType(cast->type()), cardinality), args, funcs);
    exprDedup_[expr.get()] = callExpr;
    return callExpr;
  }

  VELOX_NYI();
  return nullptr;
}

ExprCP Optimization::translateColumn(const std::string& name) {
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

AggregationP Optimization::translateAggregation(
    const core::AggregationNode& source) {
  using velox::core::AggregationNode;

  if (source.step() == AggregationNode::Step::kPartial ||
      source.step() == AggregationNode::Step::kSingle) {
    auto* aggregation =
        make<Aggregation>(nullptr, translateColumns(source.groupingKeys()));
    for (auto i = 0; i < source.groupingKeys().size(); ++i) {
      if (aggregation->grouping[i]->type() == PlanType::kColumn) {
        aggregation->mutableColumns().push_back(
            aggregation->grouping[i]->as<Column>());
      } else {
        auto name = toName(source.outputType()->nameOf(i));
        auto type = toType(source.outputType()->childAt(i));

        auto* column = make<Column>(
            name, currentSelect_, aggregation->grouping[i]->value());
        aggregation->mutableColumns().push_back(column);
      }
    }
    // The keys for intermediate are the same as for final.
    aggregation->intermediateColumns = aggregation->columns();
    for (auto i = 0; i < source.aggregateNames().size(); ++i) {
      auto rawFunc = translateExpr(source.aggregates()[i].call)->as<Call>();
      ExprCP condition = nullptr;
      if (source.aggregates()[i].mask) {
        condition = translateExpr(source.aggregates()[i].mask);
      }
      VELOX_CHECK(source.aggregates()[i].sortingKeys.empty());
      auto accumulatorType =
          toType(intermediateType(source.aggregates()[i].call));
      auto* agg = make<Aggregate>(
          rawFunc->name(),
          rawFunc->value(),
          rawFunc->args(),
          rawFunc->functions(),
          false,
          condition,
          false,
          accumulatorType);
      auto name = toName(source.aggregateNames()[i]);
      auto* column = make<Column>(name, currentSelect_, agg->value());
      aggregation->mutableColumns().push_back(column);
      auto intermediateValue = agg->value();
      intermediateValue.type = accumulatorType;
      auto* intermediateColumn =
          make<Column>(name, currentSelect_, intermediateValue);
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

OrderByP Optimization::translateOrderBy(const core::OrderByNode& order) {
  OrderTypeVector orderType;
  for (auto& sort : order.sortingOrders()) {
    orderType.push_back(
        sort.isAscending() ? (sort.isNullsFirst() ? OrderType::kAscNullsFirst
                                                  : OrderType::kAscNullsLast)
                           : (sort.isNullsFirst() ? OrderType::kDescNullsFirst
                                                  : OrderType::kDescNullsLast));
  }
  auto keys = translateColumns(order.sortingKeys());
  auto* orderBy = QGC_MAKE_IN_ARENA(OrderBy)(nullptr, keys, orderType, {});
  return orderBy;
}

ColumnCP Optimization::makeMark(const core::AbstractJoinNode& join) {
  auto type = join.outputType();
  auto name = toName(type->nameOf(type->size() - 1));
  Value value(toType(type->childAt(type->size() - 1)), 2);
  auto* column = make<Column>(name, currentSelect_, value);
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
    ColumnCP markColumn =
        joinType == core::JoinType::kLeftSemiProject ? makeMark(join) : nullptr;
    ;

    PlanObjectSet leftTables;
    PlanObjectCP rightTable = nullptr;

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
    std::vector<PlanObjectCP> leftTableVector;
    leftTables.forEach(
        [&](PlanObjectCP table) { leftTableVector.push_back(table); });
    auto* edge = make<JoinEdge>(
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
  std::vector<PlanObjectCP> tableVector;
  tables.forEach([&](PlanObjectCP table) { tableVector.push_back(table); });
  if (tableVector.size() == 2) {
    auto* edge = make<JoinEdge>(
        tableVector[0], tableVector[1], conjuncts, false, false, false, false);
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

PlanObjectP Optimization::wrapInDt(const core::PlanNode& node) {
  DerivedTableP previousDt = currentSelect_;
  auto* newDt = make<DerivedTable>();
  auto cname = toName(fmt::format("dt{}", ++nameCounter_));
  newDt->cname = cname;
  currentSelect_ = newDt;
  makeQueryGraph(node, kAllAllowedInDt);

  currentSelect_ = previousDt;
  velox::RowTypePtr type = node.outputType();
  // node.name() == "Aggregation" ? aggFinalType_ : node.outputType();
  for (auto i = 0; i < type->size(); ++i) {
    ExprCP inner = translateColumn(type->nameOf(i));
    newDt->exprs.push_back(inner);
    auto* outer = make<Column>(toName(type->nameOf(i)), newDt, inner->value());
    newDt->columns.push_back(outer);
    renames_[type->nameOf(i)] = outer;
  }
  currentSelect_->tables.push_back(newDt);
  currentSelect_->tableSet.add(newDt);
  newDt->makeInitialPlan();

  return newDt;
}

PlanObjectP Optimization::makeBaseTable(const core::TableScanNode* tableScan) {
  auto tableHandle = tableScan->tableHandle().get();
  auto assignments = tableScan->assignments();
  auto schemaTable = schema_.findTable(tableHandle->tableName());
  auto cname = fmt::format("t{}", ++nameCounter_);

  auto* baseTable = make<BaseTable>();
  baseTable->cname = toName(cname);
  baseTable->schemaTable = schemaTable;
  ColumnVector columns;
  ColumnVector schemaColumns;
  for (auto& pair : assignments) {
    auto schemaColumn = schemaTable->findColumn(pair.second->name());
    schemaColumns.push_back(schemaColumn);
    auto value = schemaColumn->value();
    auto* column = make<Column>(toName(pair.second->name()), baseTable, value);
    columns.push_back(column);
    renames_[pair.first] = column;
  }
  baseTable->columns = columns;

  setLeafHandle(baseTable->id(), tableScan->tableHandle(), {});
  setLeafSelectivity(*baseTable);
  currentSelect_->tables.push_back(baseTable);
  currentSelect_->tableSet.add(baseTable);
  return baseTable;
}

void Optimization::addProjection(const core::ProjectNode* project) {
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
}

void Optimization::addFilter(const core::FilterNode* filter) {
  ExprVector flat;
  translateConjuncts(filter->filter(), flat);
  if (isDirectOver(*filter, "Aggregation")) {
    VELOX_CHECK(
        currentSelect_->having.empty(),
        "Must have aall of HAVING in one filter");
    currentSelect_->having = flat;
  } else {
    currentSelect_->conjuncts.insert(
        currentSelect_->conjuncts.end(), flat.begin(), flat.end());
  }
}

PlanObjectP Optimization::addAggregation(
    const core::AggregationNode& aggNode,
    uint64_t allowedInDt) {
  using AggregationNode = velox::core::AggregationNode;
  if (aggNode.step() == AggregationNode::Step::kPartial ||
      aggNode.step() == AggregationNode::Step::kSingle) {
    if (!contains(allowedInDt, PlanType::kAggregation)) {
      return wrapInDt(aggNode);
    }
    if (aggNode.step() == AggregationNode::Step::kSingle) {
      aggFinalType_ = aggNode.outputType();
    }
    makeQueryGraph(
        *aggNode.sources()[0], makeDtIf(allowedInDt, PlanType::kAggregation));
    auto agg = translateAggregation(aggNode);
    if (agg) {
      auto* aggPlan = make<AggregationPlan>(agg);
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

PlanObjectP Optimization::makeQueryGraph(
    const core::PlanNode& node,
    uint64_t allowedInDt) {
  auto name = node.name();
  if (isJoin(node) && !contains(allowedInDt, PlanType::kJoin)) {
    return wrapInDt(node);
  }
  if (name == "TableScan") {
    return makeBaseTable(reinterpret_cast<const core::TableScanNode*>(&node));
  }
  if (name == "Project") {
    makeQueryGraph(*node.sources()[0], allowedInDt);
    addProjection(reinterpret_cast<const core::ProjectNode*>(&node));
    return currentSelect_;
  }
  if (name == "Filter") {
    makeQueryGraph(*node.sources()[0], allowedInDt);
    addFilter(reinterpret_cast<const core::FilterNode*>(&node));
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
    return addAggregation(
        *reinterpret_cast<const core::AggregationNode*>(&node), allowedInDt);
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

} // namespace facebook::velox::optimizer
