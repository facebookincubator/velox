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

#include "velox/optimizer/QueryGraph.h"
#include "velox/optimizer/Schema.h"

/// Plan candidates.
/// A candidate plan is constructed based  on the join graph/derived table
/// tree.

namespace facebook::velox::optimizer {

struct PlanState;

// Represents the cost and cardinality of a RelationOp or Plan. A Cost has a
// per-row cost, a per-row fanout and a one-time setup cost. For example, a hash
// join probe has a fanout of 0.3 if 3 of 10 input rows are expected to hit, a
// constant small per-row cost that is fixed and a setup cost that is
// the one time cost of the build side subplan. The inputCardinality
// is a precalculated product of the left deep inputs for the hash
// probe. For a leaf table scan, input cardinality is 1 and the fanout
// is the estimated cardinality after filters, the unitCost is the
// cost of the scan and all filters. For an index lookup, the unit
// cost is a function of the index size and the input spacing and
// input cardinality. A lookup that hits densely is cheaper than one
// that hits sparsely. An index lookup has no setup cost.
struct Cost {
  // Cardinality of the output of the left deep input tree. 1 for a leaf
  // scan.
  float inputCardinality{1};

  // Cost of processing one input tuple. Complete cost of the operation for a
  // leaf.
  float unitCost{0};

  // 'fanout * inputCardinality' is the number of result rows. For a leaf scan,
  // this is the number of rows.
  float fanout{1};

  // One time setup cost. Cost of build subplan for the first use of a hash
  // build side. 0 for the second use of a hash build side. 0 for table scan
  // or index access.
  float setupCost{0};

  // Estimate of total data volume  for a hash join build or group/order
  // by/distinct / repartition. The memory footprint may not be this if the
  // operation is streaming or spills.
  float totalBytes{0};

  // Maximum memory occupancy. If the operation is blocking, e.g. group by, the
  // amount of spill is 'totalBytes' - 'peakResidentBytes'.
  float peakResidentBytes{0};

  /// If 'isUnit' shows the cost/cardinality for one row, else for
  /// 'inputCardinality' rows.
  std::string toString(bool detail, bool isUnit = false) const;
};

/// Physical relational operator. This is the common base class of all elements
/// of plan candidates. The immutable Exprs, Columns and BaseTables in the query
/// graph are referenced from these. RelationOp instances are also arena
/// allocated but are reference counted so that no longer interesting
/// candidate plans can be freed, since a very large number of these
/// could be generated.
class RelationOp : public Relation {
 public:
  RelationOp(
      RelType type,
      boost::intrusive_ptr<RelationOp> input,
      Distribution distribution,
      ColumnVector columns = {})
      : Relation(type, std::move(distribution), std::move(columns)),
        input_(std::move(input)) {}

  virtual ~RelationOp() = default;

  void operator delete(void* ptr) {
    queryCtx()->free(ptr);
  }

  const boost::intrusive_ptr<class RelationOp>& input() const {
    return input_;
  }

  const Cost& cost() const {
    return cost_;
  }

  /// Returns the number of output rows.
  float resultCardinality() const {
    return cost_.inputCardinality * cost_.fanout;
  }

  /// Returns the value constraints of 'expr' at the output of
  /// 'this'. For example, a filter or join may limit values. An Expr
  /// will for example have no more distinct values than the number of
  /// rows. This is computed on first use.
  const Value& value(ExprCP expr) const;

  /// Fills in 'cost_' after construction. Depends on 'input' and is defined for
  /// each subclass.
  virtual void setCost(const PlanState& input);

  /// Returns human redable string for 'this' and inputs if 'recursive' is true.
  /// If 'detail' is true, includes cost and other details.
  virtual std::string toString(bool recursive, bool detail) const;

 protected:
  // adds a line of cost information to 'out'
  void printCost(bool detail, std::stringstream& out) const;

  // Input of filter/project/group by etc., Left side of join, nullptr for a
  // leaf table scan.
  boost::intrusive_ptr<class RelationOp> input_;

  Cost cost_;

 private:
  // thread local reference count. PlanObjects are freed when the
  // QueryGraphContext arena is freed, candidate plans are freed when no longer
  // referenced.
  mutable int32_t refCount_{0};

  friend void intrusive_ptr_add_ref(RelationOp* op);
  friend void intrusive_ptr_release(RelationOp* op);
};

using RelationOpPtr = boost::intrusive_ptr<RelationOp>;

inline void intrusive_ptr_add_ref(RelationOp* op) {
  ++op->refCount_;
}

inline void intrusive_ptr_release(RelationOp* op) {
  if (0 == --op->refCount_) {
    delete op;
  }
}

/// Represents a full table scan or an index lookup.
struct TableScan : public RelationOp {
  TableScan(
      RelationOpPtr input,
      Distribution _distribution,
      const BaseTable* table,
      ColumnGroupP _index,
      float fanout,
      ColumnVector columns,
      ExprVector lookupKeys = {},
      velox::core::JoinType joinType = velox::core::JoinType::kInner,
      ExprVector joinFilter = {})
      : RelationOp(
            RelType::kTableScan,
            input,
            std::move(_distribution),
            std::move(columns)),
        baseTable(table),
        index(_index),
        keys(std::move(lookupKeys)),
        joinType(joinType),
        joinFilter(std::move(joinFilter)) {
    cost_.fanout = fanout;
  }

  /// Columns of base table available in 'index'.
  static PlanObjectSet availableColumns(
      const BaseTable* baseTable,
      ColumnGroupP index);

  /// Returns the distribution given the table, index and columns. If
  /// partitioning/ordering columns are in the output columns, the
  /// distribution reflects the distribution of the index.
  static Distribution outputDistribution(
      const BaseTable* baseTable,
      ColumnGroupP index,
      const ColumnVector& columns);

  void setCost(const PlanState& input) override;

  std::string toString(bool recursive, bool detail) const override;

  // The base table reference. May occur in multiple scans if the base
  // table decomposes into access via secondary index joined to pk or
  // if doing another pass for late materialization.
  const BaseTable* baseTable;

  // Index (or other materialization of table) used for the physical data
  // access.
  ColumnGroupP index;

  // Columns read from 'baseTable'. Can be more than 'columns' if
  // there are filters that need columns that are not projected out to
  // next op.
  PlanObjectSet extractedColumns;

  // Lookup keys, empty if full table scan.
  ExprVector keys;

  // If this is a lookup, 'joinType' can  be inner, left or anti.
  velox::core::JoinType joinType{velox::core::JoinType::kInner};

  // If this is a non-inner join,  extra filter for the join.
  const ExprVector joinFilter;
};

/// Represents a repartition, i.e. query fragment boundary. The distribution of
/// the output is '_distribution'.
class Repartition : public RelationOp {
 public:
  Repartition(
      RelationOpPtr input,
      Distribution distribution,
      ColumnVector columns)
      : RelationOp(
            RelType::kRepartition,
            std::move(input),
            std::move(distribution),
            std::move(columns)) {}

  void setCost(const PlanState& input) override;
  std::string toString(bool recursive, bool detail) const override;
};

using RepartitionPtr = const Repartition*;

/// Represents a usually multitable filter not associated with any non-inner
/// join. Non-equality constraints over inner joins become Filters.
class Filter : public RelationOp {
 public:
  Filter(RelationOpPtr input, ExprVector exprs)
      : RelationOp(
            RelType::kFilter,
            input,
            input->distribution(),
            input->columns()),
        exprs_(std::move(exprs)) {}
  const ExprVector& exprs() const {
    return exprs_;
  }

  void setCost(const PlanState& input) override;
  std::string toString(bool recursive, bool detail) const override;

 private:
  const ExprVector exprs_;
};

/// Assigns names to expressions. Used to rename output from a derived table.
class Project : public RelationOp {
 public:
  Project(RelationOpPtr input, ExprVector exprs, ColumnVector columns)
      : RelationOp(
            RelType::kProject,
            input,
            input->distribution().rename(exprs, columns),
            columns),
        exprs_(std::move(exprs)),
        columns_(std::move(columns)) {}

  const ExprVector& exprs() const {
    return exprs_;
  }

  const ColumnVector& columns() const {
    return columns_;
  }

  std::string toString(bool recursive, bool detail) const override;

 private:
  const ExprVector exprs_;
  const ColumnVector columns_;
};

enum class JoinMethod { kHash, kMerge, kCross };

/// Represents a hash or merge join.
struct Join : public RelationOp {
  Join(
      JoinMethod _method,
      velox::core::JoinType _joinType,
      RelationOpPtr input,
      RelationOpPtr right,
      ExprVector leftKeys,
      ExprVector rightKeys,
      ExprVector filter,
      float fanout,
      ColumnVector columns)
      : RelationOp(RelType::kJoin, input, input->distribution(), columns),
        method(_method),
        joinType(_joinType),
        right(std::move(right)),
        leftKeys(std::move(leftKeys)),
        rightKeys(std::move(rightKeys)),
        filter(std::move(filter)) {
    cost_.fanout = fanout;
  }

  JoinMethod method;
  velox::core::JoinType joinType;
  RelationOpPtr right;
  ExprVector leftKeys;
  ExprVector rightKeys;
  ExprVector filter;

  void setCost(const PlanState& input) override;
  std::string toString(bool recursive, bool detail) const override;
};

using JoinPtr = Join*;

/// Occurs as right input of JoinOp with type kHash. Contains the
/// cost and memory specific to building the table. Can be
/// referenced from multiple JoinOps. The unit cost * input
/// cardinality of this is counted as setup cost in the first
/// referencing join and not counted in subsequent ones.
struct HashBuild : public RelationOp {
  HashBuild(RelationOpPtr input, int32_t id, ExprVector _keys, PlanPtr plan)
      : RelationOp(
            RelType::kHashBuild,
            input,
            input->distribution(),
            input->columns()),
        buildId(id),
        keys(std::move(_keys)),
        plan(plan) {}

  int32_t buildId{0};
  ExprVector keys;
  // The plan producing the build data. Used for deduplicating joins.
  PlanPtr plan;

  void setCost(const PlanState& input) override;

  std::string toString(bool recursive, bool detail) const override;
};

using HashBuildPtr = HashBuild*;

/// Represents aggregation with or without grouping.
struct Aggregation : public RelationOp {
  Aggregation(
      const Aggregation& other,
      RelationOpPtr input,
      velox::core::AggregationNode::Step _step);

  Aggregation(RelationOpPtr input, ExprVector _grouping)
      : RelationOp(
            RelType::kAggregation,
            input,
            input ? input->distribution() : Distribution()),
        grouping(std::move(_grouping)) {}

  // Grouping keys
  ExprVector grouping;

  // Keys where the key expression is functionally dependent on
  // another key or keys. These can be late materialized or converted
  // to any() aggregates.
  PlanObjectSet dependentKeys;

  std::vector<AggregateCP, QGAllocator<AggregateCP>> aggregates;

  velox::core::AggregationNode::Step step{
      velox::core::AggregationNode::Step::kSingle};

  // 'columns' of RelationOp is the final columns. 'intermediateColumns is the
  // output of the corresponding partial aggregation.
  ColumnVector intermediateColumns;

  void setCost(const PlanState& input) override;
  std::string toString(bool recursive, bool detail) const override;
};

/// Represents an order by. The order is given by the distribution.
struct OrderBy : public RelationOp {
  OrderBy(
      RelationOpPtr input,
      ExprVector keys,
      OrderTypeVector orderType,
      PlanObjectSet dependentKeys = {})
      : RelationOp(
            RelType::kOrderBy,
            input,
            input ? input->distribution().copyWithOrder(keys, orderType)
                  : Distribution(DistributionType(), 1, {}, keys, orderType)),
        dependentKeys(dependentKeys) {}

  // Keys where the key expression is functionally dependent on
  // another key or keys. These can be late materialized or converted
  // to payload.
  PlanObjectSet dependentKeys;
};

} // namespace facebook::velox::optimizer
