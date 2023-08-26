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

#include "velox/experimental/query/PlanObject.h"
#include "velox/experimental/query/SchemaSource.h"

/// Schema representation for use in query planning. All objects are
/// arena allocated for the duration of planning the query. We do
/// not expect to keep a full schema in memory, rather we expect to
/// instantiate the relevant schema objects based on the query. The
/// arena for these can be different from that for the PlanObjects,
/// though, so that a schema cache can have its own lifetime.
namespace facebook::verax {

template <typename T>
using NameMap = std::unordered_map<
    Name,
    T,
    std::hash<Name>,
    std::equal_to<Name>,
    QGAllocator<std::pair<const Name, T>>>;

/// Represents constraints on a column value or intermediate result.
struct Value {
  Value(const velox::Type* _type, float _cardinality)
      : type(_type), cardinality(_cardinality) {}

  /// Returns the average byte size of a value when it occurs as an intermediate
  /// result without dictionary or other encoding.
  float byteSize() const;

  const velox::Type* FOLLY_NONNULL type;
  const velox::variant* min{nullptr};
  const velox::variant* max{nullptr};

  // Count of distinct values. Is not exact and is used for estimating
  // cardinalities of group bys or joins.
  const float cardinality{1};

  // Estimate of true fraction for booleans. 0 means always
  // false. This is an estimate and 1 or 0 do not allow pruning
  // dependent code paths.
  float trueFraction{1};

  // 0 means no nulls, 0.5 means half are null.
  float nullFraction{0};

  // True if nulls may occur. 'false' means that plans that allow no nulls may
  // be generated.
  bool nullable{true};
};

/// Describes order in an order by or index.
enum class OrderType {
  kAscNullsFirst,
  kAscNullsLast,
  kDescNullsFirst,
  kDescNullsLast
};

using OrderTypeVector = std::vector<OrderType, QGAllocator<OrderType>>;

class RelationOp;

/// Represents a system that contains or produces data.For cases of federation
/// where data is only accessible via a specific instance of a specific type of
/// system, the locus represents the instance and the subclass of Locus
/// represents the type of system for a schema object. For a
/// RelationOp, the  locus of its distribution means that the op is performed by
/// the corresponding system. Distributions can be copartitioned only
/// if their locus is equal (==) to the other locus. A Locus is referenced by
/// raw pointer and may be allocated from outside the optimization arena. It is
/// immutable and lives past the optimizer arena.
class Locus {
 public:
  explicit Locus(Name name) : name_(name) {}

  virtual ~Locus() = default;

  Name name() const {
    // Make sure the name is in the current optimization
    // arena. 'this' may live across several arenas.
    return toName(name_);
  }

  /// Sets the cardinality in op. Returns true if set. If false, default
  /// cardinality determination.
  virtual bool setCardinality(RelationOp& /*op*/) const {
    return false;
  }

  /// Sets the cost. Returns true if set. If false, the default cost is set with
  /// RelationOp::setCost.
  virtual bool setCost(RelationOp& /*op*/) const {
    return false;
  }

  std::string toString() const {
    return name_;
  }

 private:
  Name name_;
};

using LocusPtr = const Locus*;

/// Method for determining a partition given an ordered list of partitioning
/// keys. Hive hash is an example, range partitioning is another. Add values
/// here for more types.
enum class ShuffleMode { kNone, kHive };

/// Distribution of data. 'numPartitions' is 1 if the data is not partitioned.
/// There is copartitioning if the DistributionType is the same on both sides
/// and both sides have an equal number of 1:1 type matched partitioning keys.
struct DistributionType {
  bool operator==(const DistributionType& other) const {
    return mode == other.mode && numPartitions == other.numPartitions &&
        locus == other.locus && isGather == other.isGather;
  }

  ShuffleMode mode{ShuffleMode::kNone};
  int32_t numPartitions{1};
  LocusPtr locus{nullptr};
  bool isGather{false};
};

// Describes output of relational operator. If base table, cardinality is
// after filtering.
struct Distribution {
  Distribution() = default;
  Distribution(
      DistributionType type,
      float cardinality,
      ExprVector _partition,
      ExprVector _order = {},
      OrderTypeVector _orderType = {},
      int32_t uniquePrefix = 0,
      float _spacing = 0)
      : distributionType(std::move(type)),
        cardinality(cardinality),
        partition(std::move(_partition)),
        order(std::move(_order)),
        orderType(std::move(_orderType)),
        numKeysUnique(uniquePrefix),
        spacing(_spacing) {}

  /// Returns a Distribution for use in a broadcast shuffle.
  static Distribution broadcast(DistributionType type, float cardinality) {
    Distribution result(type, cardinality, {});
    result.isBroadcast = true;
    return result;
  }

  /// Returns a distribution for an end of query gather from last stage
  /// fragments. Specifying order will create a merging exchange when the
  /// Distribution occurs in a Repartition.
  static Distribution gather(
      DistributionType type,
      const ExprVector& order = {},
      const OrderTypeVector& orderType = {}) {
    auto singleType = type;
    singleType.numPartitions = 1;
    singleType.isGather = true;
    return Distribution(singleType, 1, {}, order, orderType);
  }

  /// Returns a copy of 'this' with 'order' and 'orderType' set from
  /// arguments.
  Distribution copyWithOrder(ExprVector order, OrderTypeVector orderType)
      const {
    Distribution copy = *this;
    copy.order = order;
    copy.orderType = orderType;
    return copy;
  }

  /// True if 'this' and 'other' have the same number/type of keys and same
  /// distribution type. Data is copartitioned if both sides have a 1:1
  /// equality on all partitioning key columns.
  bool isSamePartition(const Distribution& other) const;

  Distribution rename(const ExprVector& exprs, const ColumnVector& names) const;

  std::string toString() const;

  DistributionType distributionType;

  // Number of rows 'this' applies to. This is the size in rows if 'this'
  // occurs in a table or index.
  float cardinality;

  // Partitioning columns. The values of these columns determine which of
  // 'numPartitions' contains any given row. This does not specify the
  // partition function (e.g. Hive bucket or range partition).
  ExprVector partition;

  // Ordering columns. Each partition is ordered by these. Specifies that
  // streaming group by or merge join are possible.
  ExprVector order;

  // Corresponds 1:1 to 'order'. The size of this gives the number of leading
  // columns of 'order' on which the data is sorted.
  OrderTypeVector orderType;

  // Number of leading elements of 'order' such that these uniquely
  // identify a row. 0 if there is no uniqueness. This can be non-0 also if
  // data is not sorted. This indicates a uniqueness for joining.
  int32_t numKeysUnique{0};

  // Specifies the selectivity between the source of the ordered data
  // and 'this'. For example, if orders join lineitem and both are
  // ordered on orderkey and there is a 1/1000 selection on orders,
  // the distribution after the filter would have a spacing of 1000,
  // meaning that lineitem is hit every 1000 orders, meaning that an
  // index join with lineitem would skip 4000 rows between hits
  // because lineitem has an average of 4 repeats of orderkey.
  float spacing{-1};

  // True if the data is replicated to 'numPartitions'.
  bool isBroadcast{false};
};

/// Identifies a base table or the operator type producing the relation. Base
/// data as in Index has type kBase. The result of a table scan is kTableScan.
enum class RelType {
  kBase,
  kTableScan,
  kRepartition,
  kFilter,
  kProject,
  kJoin,
  kHashBuild,
  kAggregation,
  kOrderBy
};

/// Represents a relation (table) that is either physically stored or is the
/// streaming output of a query operator. This has a distribution describing
/// partitioning and data order and a set of columns describing the payload.
class Relation {
 public:
  Relation(
      RelType relType,
      Distribution distribution,
      const ColumnVector& columns)
      : relType_(relType),
        distribution_(std::move(distribution)),
        columns_(columns) {}

  RelType relType() const {
    return relType_;
  }

  const Distribution& distribution() const {
    return distribution_;
  }

  const ColumnVector& columns() const {
    return columns_;
  }

  ColumnVector& mutableColumns() {
    return columns_;
  }

  template <typename T>
  const T* as() const {
    return static_cast<const T*>(this);
  }

  template <typename T>
  T* as() {
    return static_cast<T*>(this);
  }

 protected:
  const RelType relType_;
  const Distribution distribution_;
  ColumnVector columns_;
};

struct SchemaTable;
using SchemaTablePtr = const SchemaTable*;

/// Represents a stored collection of rows. An index may have a uniqueness
/// constraint over a set of columns, a partitioning and an ordering plus a
/// set of payload columns.
struct Index : public Relation {
  Index(
      Name _name,
      SchemaTablePtr _table,
      Distribution distribution,
      const ColumnVector& _columns)
      : Relation(RelType::kBase, distribution, _columns),
        name(_name),
        table(_table) {}

  Name name;
  SchemaTablePtr table;

  /// Returns cost of next lookup when the hit is within 'range' rows
  /// of the previous hit. If lookups are not batched or not ordered,
  /// then 'range' should be the cardinality of the index.
  float lookupCost(float range) const;
};

using IndexPtr = Index*;

// Describes the number of rows to look at and the number of expected matches
// given equality constraints for a set of columns. See
// SchemaTable::indexInfo().
struct IndexInfo {
  // Index chosen based on columns.
  IndexPtr index;

  // True if the column combination is unique. This can be true even if there
  // is no key order in 'index'.
  bool unique{false};

  // The number of rows selected after index lookup based on 'lookupKeys'. For
  // empty 'lookupKeys', this is the cardinality of 'index'.
  float scanCardinality;

  // The expected number of hits for an equality match of lookup keys. This is
  // the expected number of rows given the lookup column combination
  // regardless of whether an index order can be used.
  float joinCardinality;

  // The lookup columns that match 'index'. These match 1:1 the leading keys
  // of 'index'. If 'index' has no ordering columns or if the lookup columns
  // are not a prefix of these, this is empty.
  std::vector<ColumnPtr> lookupKeys;

  // The columns that were considered in 'scanCardinality' and
  // 'joinCardinality'. This may be fewer columns than given to
  // indexInfo() if the index does not cover some columns.
  PlanObjectSet coveredColumns;

  /// Returns the schema column for the BaseTable column 'column' or nullptr
  /// if not in the index.
  ColumnPtr schemaColumn(ColumnPtr keyValue) const;
};

/// A table in a schema. The table may have multiple differently ordered and
/// partitioned physical representations (indices). Not all indices need to
/// contain all columns.

struct SchemaTable {
  SchemaTable(Name _name, const velox::RowTypePtr& _type)
      : name(_name), type(_type) {}

  /// Adds an index. The arguments set the corresponding members of a
  /// Distribution.
  void addIndex(
      Name name,
      float cardinality,
      int32_t numKeysUnique,
      int32_t numOrdering,
      const ColumnVector& keys,
      DistributionType distType,
      const ColumnVector& partition,
      const ColumnVector& columns);

  /// Finds or adds a column with 'name' and 'value'.
  ColumnPtr column(const std::string& name, const Value& value);

  ColumnPtr findColumn(const std::string& name) const;

  /// True if 'columns' match no more than one row.
  bool isUnique(PtrSpan<Column> columns) const;

  /// Returns   uniqueness and cardinality information for a lookup on 'index'
  /// where 'columns' have an equality constraint.
  IndexInfo indexInfo(IndexPtr index, PtrSpan<Column> columns) const;

  /// Returns the best index to use for lookup where 'columns' have an
  /// equality constraint.
  IndexInfo indexByColumns(PtrSpan<Column> columns) const;

  std::vector<ColumnPtr> toColumns(const std::vector<std::string>& names);
  Name name;
  const velox::RowTypePtr& type;

  // Lookup from name to column.
  NameMap<ColumnPtr> columns;

  // All indices. Must contain at least one.
  std::vector<IndexPtr, QGAllocator<IndexPtr>> indices;
};

/// Represents a collection of tables. Normally filled in ad hoc given the set
/// of tables referenced by a query.
class Schema {
 public:
  Schema(Name _name, std::vector<SchemaTablePtr> tables);
  Schema(Name _name, SchemaSource* source);

  /// Returns the table with 'name' or nullptr if not found.
  SchemaTablePtr findTable(const std::string& name) const;

  Name name() const {
    return name_;
  }

  void addTable(SchemaTablePtr table) const;

 private:
  Name name_;
  mutable NameMap<SchemaTablePtr> tables_;
  SchemaSource* source_ = nullptr;
};

using SchemaPtr = Schema*;

} // namespace facebook::verax
