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

#include "velox/common/base/SuccinctPrinter.h"
#include "velox/optimizer/Plan.h"
#include "velox/optimizer/PlanUtils.h"
#include "velox/optimizer/QueryGraph.h"

namespace facebook::velox::optimizer {

const Value& RelationOp::value(ExprCP expr) const {
  // Compute new Value by applying restrictions from operators
  // between the place Expr is first defined and the output of
  // 'this'. Memoize the result in 'this'.
  return expr->value();
}

std::string RelationOp::toString(bool recursive, bool detail) const {
  if (input_ && recursive) {
    return input_->toString(true, detail);
  }
  return "";
}

// static
Distribution TableScan::outputDistribution(
    const BaseTable* baseTable,
    ColumnGroupP index,
    const ColumnVector& columns) {
  auto schemaColumns = transform<ColumnVector>(
      columns, [](auto& c) { return c->schemaColumn(); });

  ExprVector partition;
  ExprVector order;
  OrderTypeVector orderType;
  // if all partitioning columns are projected, the output is partitioned.
  if (isSubset(index->distribution().partition, schemaColumns)) {
    partition = index->distribution().partition;
    replace(partition, schemaColumns, columns.data());
  }
  auto numPrefix = prefixSize(index->distribution().order, schemaColumns);
  if (numPrefix > 0) {
    order = index->distribution().order;
    order.resize(numPrefix);
    orderType = index->distribution().orderType;
    orderType.resize(numPrefix);
    replace(order, schemaColumns, columns.data());
  }
  return Distribution(
      index->distribution().distributionType,
      index->distribution().cardinality * baseTable->filterSelectivity,
      std::move(partition),
      std::move(order),
      std::move(orderType),
      index->distribution().numKeysUnique <= numPrefix
          ? index->distribution().numKeysUnique
          : 0,
      1.0 / baseTable->filterSelectivity);
}

// static
PlanObjectSet TableScan::availableColumns(
    const BaseTable* baseTable,
    ColumnGroupP index) {
  // The columns of base table that exist in 'index'.
  PlanObjectSet result;
  for (auto column : index->columns()) {
    for (auto baseColumn : baseTable->columns) {
      if (baseColumn->name() == column->name()) {
        result.add(baseColumn);
        break;
      }
    }
  }
  return result;
}

std::string Cost::toString(bool /*detail*/, bool isUnit) const {
  std::stringstream out;
  float multiplier = isUnit ? 1 : inputCardinality;
  out << succinctNumber(fanout * multiplier) << " rows "
      << succinctNumber(unitCost * multiplier) << "CU";
  if (setupCost > 0) {
    out << ", setup " << succinctNumber(setupCost) << "CU";
  }
  if (static_cast<bool>(totalBytes)) {
    out << " " << velox::succinctBytes(totalBytes);
  }
  return out.str();
}

void RelationOp::printCost(bool detail, std::stringstream& out) const {
  auto ctx = queryCtx();
  if (ctx && ctx->contextPlan()) {
    auto plan = ctx->contextPlan();
    auto totalCost = plan->cost.unitCost + plan->cost.setupCost;
    auto pct = 100 * cost_.inputCardinality * cost_.unitCost / totalCost;
    out << " " << std::fixed << std::setprecision(2) << pct << "% ";
  }
  if (detail) {
    out << " " << cost_.toString(detail, false) << std::endl;
  }
}

const char* joinTypeLabel(velox::core::JoinType type) {
  switch (type) {
    case velox::core::JoinType::kLeft:
      return "left";
    case velox::core::JoinType::kRight:
      return "right";
    case velox::core::JoinType::kRightSemiFilter:
      return "right exists";
    case velox::core::JoinType::kRightSemiProject:
      return "right exists-flag";
    case velox::core::JoinType::kLeftSemiFilter:
      return "exists";
    case velox::core::JoinType::kLeftSemiProject:
      return "exists-flag";
    case velox::core::JoinType::kAnti:
      return "not exists";
    default:
      return "";
  }
}

std::string TableScan::toString(bool /*recursive*/, bool detail) const {
  std::stringstream out;
  if (input()) {
    out << input()->toString(true, detail);
    out << " *I " << joinTypeLabel(joinType);
  }
  out << baseTable->schemaTable->name << " " << baseTable->cname;
  if (detail) {
    printCost(detail, out);
    if (!input()) {
      out << distribution_.toString() << std::endl;
    }
  }
  return out.str();
}

std::string Join::toString(bool recursive, bool detail) const {
  std::stringstream out;
  if (recursive) {
    out << input()->toString(true, detail);
  }
  out << "*" << (method == JoinMethod::kHash ? "H" : "M") << " "
      << joinTypeLabel(joinType);
  printCost(detail, out);
  if (recursive) {
    out << " (" << right->toString(true, detail) << ")";
    if (detail) {
      out << std::endl;
    }
  }
  return out.str();
}

std::string Repartition::toString(bool recursive, bool detail) const {
  std::stringstream out;
  if (recursive) {
    out << input()->toString(true, detail) << " ";
  }
  out << (distribution().isBroadcast ? "broadcast" : "shuffle") << " ";
  if (detail && !distribution().isBroadcast) {
    out << distribution().toString();
    printCost(detail, out);
  } else if (detail) {
    printCost(detail, out);
  }
  return out.str();
}

Aggregation::Aggregation(
    const Aggregation& other,
    RelationOpPtr input,
    velox::core::AggregationNode::Step _step)
    : Aggregation(other) {
  *const_cast<Distribution*>(&distribution_) = input->distribution();
  input_ = std::move(input);
  step = _step;
  using velox::core::AggregationNode;
  if (step == AggregationNode::Step::kPartial ||
      step == AggregationNode::Step::kIntermediate) {
    *const_cast<ColumnVector*>(&columns_) = intermediateColumns;
  } else if (step == AggregationNode::Step::kFinal) {
    for (auto i = 0; i < grouping.size(); ++i) {
      grouping[i] = intermediateColumns[i];
    }
  }
}

std::string Aggregation::toString(bool recursive, bool detail) const {
  std::stringstream out;
  if (recursive) {
    out << input()->toString(true, detail) << " ";
  }
  out << velox::core::AggregationNode::stepName(step) << " agg";
  printCost(detail, out);
  return out.str();
}

std::string HashBuild::toString(bool recursive, bool detail) const {
  std::stringstream out;
  if (recursive) {
    out << input()->toString(true, detail) << " ";
  }
  out << " Build ";
  printCost(detail, out);
  return out.str();
}

std::string Filter::toString(bool recursive, bool detail) const {
  std::stringstream out;
  if (recursive) {
    out << input()->toString(true, detail) << " ";
  }
  if (detail) {
    out << "Filter (";
    for (auto i = 0; i < exprs_.size(); ++i) {
      out << exprs_[i]->toString();
      if (i < exprs_.size() - 1) {
        out << " and ";
      }
    }
    out << ")\n";
  } else {
    out << "filter " << exprs_.size() << " exprs ";
  }
  return out.str();
}

std::string Project::toString(bool recursive, bool detail) const {
  std::stringstream out;
  if (recursive) {
    out << input()->toString(true, detail) << " ";
  }
  if (detail) {
    out << "Project (";
    for (auto i = 0; i < exprs_.size(); ++i) {
      out << columns_[i]->toString() << " = " << exprs_[i]->toString();
      if (i < exprs_.size() - 1) {
        out << ", ";
      }
    }
    out << ")\n";
  } else {
    out << "project " << exprs_.size() << " columns ";
  }
  return out.str();
}

} // namespace facebook::velox::optimizer
