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

#include "velox/connectors/Connector.h"
#include "velox/core/Expressions.h"
#include "velox/core/QueryConfig.h"

#include "velox/vector/arrow/Abi.h"
#include "velox/vector/arrow/Bridge.h"

namespace facebook::velox::core {

typedef std::string PlanNodeId;

/**
 * Generic representation of InsertTable
 */
struct InsertTableHandle {
 public:
  InsertTableHandle(
      const std::string& connectorId,
      const std::shared_ptr<connector::ConnectorInsertTableHandle>&
          connectorInsertTableHandle)
      : connectorId_(connectorId),
        connectorInsertTableHandle_(connectorInsertTableHandle) {}

  const std::string& connectorId() const {
    return connectorId_;
  }

  const std::shared_ptr<connector::ConnectorInsertTableHandle>&
  connectorInsertTableHandle() const {
    return connectorInsertTableHandle_;
  }

 private:
  // Connector ID
  const std::string connectorId_;

  // Write request to a DataSink of that connector type
  const std::shared_ptr<connector::ConnectorInsertTableHandle>
      connectorInsertTableHandle_;
};

class SortOrder {
 public:
  SortOrder(bool ascending, bool nullsFirst)
      : ascending_(ascending), nullsFirst_(nullsFirst) {}

  bool isAscending() const {
    return ascending_;
  }

  bool isNullsFirst() const {
    return nullsFirst_;
  }

  std::string toString() const {
    return fmt::format(
        "{} NULLS {}",
        (ascending_ ? "ASC" : "DESC"),
        (nullsFirst_ ? "FIRST" : "LAST"));
  }

  folly::dynamic serialize() const;

  static SortOrder deserialize(const folly::dynamic& obj);

 private:
  bool ascending_;
  bool nullsFirst_;
};

extern const SortOrder kAscNullsFirst;
extern const SortOrder kAscNullsLast;
extern const SortOrder kDescNullsFirst;
extern const SortOrder kDescNullsLast;

class PlanNode : public ISerializable {
 public:
  explicit PlanNode(const PlanNodeId& id) : id_{id} {}

  virtual ~PlanNode() {}

  const PlanNodeId& id() const {
    return id_;
  }

  folly::dynamic serialize() const override;

  static void registerSerDe();

  virtual const RowTypePtr& outputType() const = 0;

  virtual const std::vector<std::shared_ptr<const PlanNode>>& sources()
      const = 0;

  /// Returns true if this is a leaf plan node and corresponding operator
  /// requires an ExchangeClient to retrieve data. For instance, TableScanNode
  /// is a leaf node that doesn't require an ExchangeClient. But ExchangeNode is
  /// a leaf node that requires an ExchangeClient.
  virtual bool requiresExchangeClient() const {
    return false;
  }

  /// Returns true if this is a leaf plan node and corresponding operator
  /// requires splits to make progress. ValueNode is a leaf node that doesn't
  /// require splits, but TableScanNode and ExchangeNode are leaf nodes that
  /// require splits.
  virtual bool requiresSplits() const {
    return false;
  }

  /// Returns true if this plan node operator is spillable and 'queryConfig' has
  /// enabled it.
  virtual bool canSpill(const QueryConfig& queryConfig) const {
    return false;
  }

  /// Returns a set of leaf plan node IDs.
  std::unordered_set<core::PlanNodeId> leafPlanNodeIds() const;

  /// Returns human-friendly representation of the plan. By default, returns the
  /// plan node name. Includes plan node details such as join keys and aggregate
  /// function names if 'detailed' is true. Returns the whole sub-tree if
  /// 'recursive' is true. Includes additional context for each plan node if
  /// 'addContext' is not null.
  ///
  /// @param addContext Optional lambda to add context for a given plan node.
  /// Receives plan node ID, indentation and std::stringstring where to append
  /// the context. Use indentation for second and subsequent lines of a
  /// mult-line context. Do not use indentation for single-line context. Do not
  /// add trailing new-line character for the last or only line of context.
  std::string toString(
      bool detailed = false,
      bool recursive = false,
      std::function<void(
          const PlanNodeId& planNodeId,
          const std::string& indentation,
          std::stringstream& stream)> addContext = nullptr) const {
    std::stringstream stream;
    toString(stream, detailed, recursive, 0, addContext);
    return stream.str();
  }

  /// The name of the plan node, used in toString.
  virtual std::string_view name() const = 0;

  /// Recursively checks the node tree for a first node that satisfy a given
  /// condition. Returns pointer to the node if found, nullptr if not.
  static const PlanNode* findFirstNode(
      const PlanNode* node,
      const std::function<bool(const PlanNode* node)>& predicate) {
    if (predicate(node)) {
      return node;
    }

    // Recursively go further through the sources.
    for (const auto& source : node->sources()) {
      const auto* ret = PlanNode::findFirstNode(source.get(), predicate);
      if (ret != nullptr) {
        return ret;
      }
    }
    return nullptr;
  }

 private:
  /// The details of the plan node in textual format.
  virtual void addDetails(std::stringstream& stream) const = 0;

  // Format when detailed and recursive are enabled is:
  //  -> name[details]
  //      -> child1Name [details]
  //         ...
  //      -> child2Name [details]
  //         ...
  void toString(
      std::stringstream& stream,
      bool detailed,
      bool recursive,
      size_t indentationSize,
      std::function<void(
          const PlanNodeId& planNodeId,
          const std::string& indentation,
          std::stringstream& stream)> addContext) const;

  const std::string id_;
};

using PlanNodePtr = std::shared_ptr<const PlanNode>;

class ValuesNode : public PlanNode {
 public:
  ValuesNode(
      const PlanNodeId& id,
      std::vector<RowVectorPtr> values,
      bool parallelizable = false,
      size_t repeatTimes = 1)
      : PlanNode(id),
        values_(std::move(values)),
        outputType_(
            values_.empty()
                ? ROW({})
                : std::dynamic_pointer_cast<const RowType>(values_[0]->type())),
        parallelizable_(parallelizable),
        repeatTimes_(repeatTimes) {}

  const RowTypePtr& outputType() const override {
    return outputType_;
  }

  const std::vector<PlanNodePtr>& sources() const override;

  const std::vector<RowVectorPtr>& values() const {
    return values_;
  }

  // For testing only.
  bool isParallelizable() const {
    return parallelizable_;
  }

  // Controls how many times each input buffer will be produced as input.
  // For example, if `values_` contains 3 rowVectors {v1, v2, v3}
  // and repeatTimes = 2, the following input will be produced:
  //   v1, v2, v3, v1, v2, v3
  size_t repeatTimes() const {
    return repeatTimes_;
  }

  std::string_view name() const override {
    return "Values";
  }

  folly::dynamic serialize() const override;

  static PlanNodePtr create(const folly::dynamic& obj, void* context);

 private:
  void addDetails(std::stringstream& stream) const override;

  const std::vector<RowVectorPtr> values_;
  const RowTypePtr outputType_;
  const bool parallelizable_;
  const size_t repeatTimes_;
};

class ArrowStreamNode : public PlanNode {
 public:
  ArrowStreamNode(
      const PlanNodeId& id,
      const RowTypePtr& outputType,
      std::shared_ptr<ArrowArrayStream> arrowStream)
      : PlanNode(id),
        outputType_(outputType),
        arrowStream_(std::move(arrowStream)) {
    VELOX_CHECK_NOT_NULL(arrowStream_);
  }

  const RowTypePtr& outputType() const override {
    return outputType_;
  }

  const std::vector<PlanNodePtr>& sources() const override;

  const std::shared_ptr<ArrowArrayStream>& arrowStream() const {
    return arrowStream_;
  }

  std::string_view name() const override {
    return "ArrowStream";
  }

  folly::dynamic serialize() const override {
    VELOX_UNSUPPORTED("ArrowStream plan node is not serializable");
  }

 private:
  void addDetails(std::stringstream& stream) const override;

  const RowTypePtr outputType_;
  std::shared_ptr<ArrowArrayStream> arrowStream_;
};

class FilterNode : public PlanNode {
 public:
  FilterNode(const PlanNodeId& id, TypedExprPtr filter, PlanNodePtr source)
      : PlanNode(id), sources_{std::move(source)}, filter_(std::move(filter)) {
    VELOX_CHECK(
        filter_->type()->isBoolean(),
        "Filter expression must be of type BOOLEAN. Got {}.",
        filter_->type()->toString());
  }

  const RowTypePtr& outputType() const override {
    return sources_[0]->outputType();
  }

  const std::vector<PlanNodePtr>& sources() const override {
    return sources_;
  }

  const TypedExprPtr& filter() const {
    return filter_;
  }

  std::string_view name() const override {
    return "Filter";
  }

  folly::dynamic serialize() const override;

  static PlanNodePtr create(const folly::dynamic& obj, void* context);

 private:
  void addDetails(std::stringstream& stream) const override {
    stream << "expression: " << filter_->toString();
  }

  const std::vector<PlanNodePtr> sources_;
  const TypedExprPtr filter_;
};

class ProjectNode : public PlanNode {
 public:
  ProjectNode(
      const PlanNodeId& id,
      std::vector<std::string>&& names,
      std::vector<TypedExprPtr>&& projections,
      PlanNodePtr source)
      : PlanNode(id),
        sources_{source},
        names_(std::move(names)),
        projections_(std::move(projections)),
        outputType_(makeOutputType(names_, projections_)) {}

  ProjectNode(
      const PlanNodeId& id,
      const std::vector<std::string>& names,
      const std::vector<TypedExprPtr>& projections,
      PlanNodePtr source)
      : PlanNode(id),
        sources_{source},
        names_(names),
        projections_(projections),
        outputType_(makeOutputType(names_, projections_)) {}

  const RowTypePtr& outputType() const override {
    return outputType_;
  }

  const std::vector<PlanNodePtr>& sources() const override {
    return sources_;
  }

  const std::vector<std::string>& names() const {
    return names_;
  }

  const std::vector<TypedExprPtr>& projections() const {
    return projections_;
  }

  // This function is virtual to allow customized projections to inherit from
  // this class without re-implementing the other functions.
  virtual std::string_view name() const override {
    return "Project";
  }

  folly::dynamic serialize() const override;

  static PlanNodePtr create(const folly::dynamic& obj, void* context);

 private:
  void addDetails(std::stringstream& stream) const override;

  static RowTypePtr makeOutputType(
      const std::vector<std::string>& names,
      const std::vector<TypedExprPtr>& projections) {
    std::vector<TypePtr> types;
    for (auto& projection : projections) {
      types.push_back(projection->type());
    }

    auto namesCopy = names;
    return std::make_shared<RowType>(std::move(namesCopy), std::move(types));
  }

  const std::vector<PlanNodePtr> sources_;
  const std::vector<std::string> names_;
  const std::vector<TypedExprPtr> projections_;
  const RowTypePtr outputType_;
};

class TableScanNode : public PlanNode {
 public:
  TableScanNode(
      const PlanNodeId& id,
      const RowTypePtr& outputType,
      const std::shared_ptr<connector::ConnectorTableHandle>& tableHandle,
      const std::unordered_map<
          std::string,
          std::shared_ptr<connector::ColumnHandle>>& assignments)
      : PlanNode(id),
        outputType_(outputType),
        tableHandle_(tableHandle),
        assignments_(assignments) {}

  const std::vector<PlanNodePtr>& sources() const override;

  const RowTypePtr& outputType() const override {
    return outputType_;
  }

  bool requiresSplits() const override {
    return true;
  }

  const std::shared_ptr<connector::ConnectorTableHandle>& tableHandle() const {
    return tableHandle_;
  }

  const std::
      unordered_map<std::string, std::shared_ptr<connector::ColumnHandle>>&
      assignments() const {
    return assignments_;
  }

  std::string_view name() const override {
    return "TableScan";
  }

  folly::dynamic serialize() const override;

  static PlanNodePtr create(const folly::dynamic& obj, void* context);

 private:
  void addDetails(std::stringstream& stream) const override;

  const RowTypePtr outputType_;
  const std::shared_ptr<connector::ConnectorTableHandle> tableHandle_;
  const std::
      unordered_map<std::string, std::shared_ptr<connector::ColumnHandle>>
          assignments_;
};

class TableWriteNode : public PlanNode {
 public:
  TableWriteNode(
      const PlanNodeId& id,
      const RowTypePtr& columns,
      const std::vector<std::string>& columnNames,
      const std::shared_ptr<InsertTableHandle>& insertTableHandle,
      const RowTypePtr& outputType,
      connector::CommitStrategy commitStrategy,
      const PlanNodePtr& source)
      : PlanNode(id),
        sources_{source},
        columns_{columns},
        columnNames_{columnNames},
        insertTableHandle_(insertTableHandle),
        outputType_(outputType),
        commitStrategy_(commitStrategy) {
    VELOX_CHECK_EQ(columns->size(), columnNames.size());
    for (const auto& column : columns->names()) {
      VELOX_CHECK(source->outputType()->containsChild(column));
    }
  }

  const std::vector<PlanNodePtr>& sources() const override {
    return sources_;
  }

  const RowTypePtr& outputType() const override {
    return outputType_;
  }

  // The subset of columns in the output of the source node, potentially in
  // different order, to write to the table.
  const RowTypePtr& columns() const {
    return columns_;
  }

  // Column names to use when writing the table. This vector is aligned with
  // 'columns' vector.
  const std::vector<std::string>& columnNames() const {
    return columnNames_;
  }

  const std::shared_ptr<InsertTableHandle>& insertTableHandle() const {
    return insertTableHandle_;
  }

  connector::CommitStrategy commitStrategy() const {
    return commitStrategy_;
  }

  std::string_view name() const override {
    return "TableWrite";
  }

  folly::dynamic serialize() const override;

  static PlanNodePtr create(const folly::dynamic& obj, void* context);

 private:
  void addDetails(std::stringstream& stream) const override;

  const std::vector<PlanNodePtr> sources_;
  const RowTypePtr columns_;
  const std::vector<std::string> columnNames_;
  const std::shared_ptr<InsertTableHandle> insertTableHandle_;
  const RowTypePtr outputType_;
  const connector::CommitStrategy commitStrategy_;
};

class AggregationNode : public PlanNode {
 public:
  enum class Step {
    // raw input in - partial result out
    kPartial,
    // partial result in - final result out
    kFinal,
    // partial result in - partial result out
    kIntermediate,
    // raw input in - final result out
    kSingle
  };

  static const char* stepName(Step step);

  static Step stepFromName(const std::string& name);

  /**
   * @param preGroupedKeys A subset of the 'groupingKeys' on which the input is
   * clustered, i.e. identical sets of values for these keys always appear next
   * to each other. Can be empty. If contains all the 'groupingKeys', the
   * aggregation will run in streaming mode.
   * @param ignoreNullKeys True if rows with at least one null key should be
   * ignored. Used when group by is a source of a join build side and grouping
   * keys are join keys.
   */
  AggregationNode(
      const PlanNodeId& id,
      Step step,
      const std::vector<FieldAccessTypedExprPtr>& groupingKeys,
      const std::vector<FieldAccessTypedExprPtr>& preGroupedKeys,
      const std::vector<std::string>& aggregateNames,
      const std::vector<CallTypedExprPtr>& aggregates,
      const std::vector<FieldAccessTypedExprPtr>& aggregateMasks,
      bool ignoreNullKeys,
      PlanNodePtr source);

  const std::vector<PlanNodePtr>& sources() const override {
    return sources_;
  }

  const RowTypePtr& outputType() const override {
    return outputType_;
  }

  Step step() const {
    return step_;
  }

  const std::vector<FieldAccessTypedExprPtr>& groupingKeys() const {
    return groupingKeys_;
  }

  const std::vector<FieldAccessTypedExprPtr>& preGroupedKeys() const {
    return preGroupedKeys_;
  }

  const std::vector<std::string>& aggregateNames() const {
    return aggregateNames_;
  }

  const std::vector<CallTypedExprPtr>& aggregates() const {
    return aggregates_;
  }

  const std::vector<FieldAccessTypedExprPtr>& aggregateMasks() const {
    return aggregateMasks_;
  }

  bool ignoreNullKeys() const {
    return ignoreNullKeys_;
  }

  std::string_view name() const override {
    return "Aggregation";
  }

  bool canSpill(const QueryConfig& queryConfig) const override {
    // NOTE: as for now, we don't allow spilling for distinct aggregation
    // (https://github.com/facebookincubator/velox/issues/3263) and pre-grouped
    // aggregation (https://github.com/facebookincubator/velox/issues/3264). We
    // will add support later to re-enable.
    return (isFinal() || isSingle()) && !(aggregates().empty()) &&
        preGroupedKeys().empty() && queryConfig.aggregationSpillEnabled();
  }

  bool isFinal() const {
    return step_ == Step::kFinal;
  }

  bool isSingle() const {
    return step_ == Step::kSingle;
  }

  folly::dynamic serialize() const override;

  static PlanNodePtr create(const folly::dynamic& obj, void* context);

 private:
  void addDetails(std::stringstream& stream) const override;

  const Step step_;
  const std::vector<FieldAccessTypedExprPtr> groupingKeys_;
  const std::vector<FieldAccessTypedExprPtr> preGroupedKeys_;
  const std::vector<std::string> aggregateNames_;
  const std::vector<CallTypedExprPtr> aggregates_;
  // Keeps mask/'no mask' for every aggregation. Mask, if given, is a reference
  // to a boolean projection column, used to mask out rows for the aggregation.
  const std::vector<FieldAccessTypedExprPtr> aggregateMasks_;
  const bool ignoreNullKeys_;
  const std::vector<PlanNodePtr> sources_;
  const RowTypePtr outputType_;
};

inline std::ostream& operator<<(
    std::ostream& out,
    const AggregationNode::Step& step) {
  switch (step) {
    case AggregationNode::Step::kFinal:
      return out << "FINAL";
    case AggregationNode::Step::kIntermediate:
      return out << "INTERMEDIATE";
    case AggregationNode::Step::kPartial:
      return out << "PARTIAL";
    case AggregationNode::Step::kSingle:
      return out << "SINGLE";
  }
  VELOX_UNREACHABLE();
}

inline std::string mapAggregationStepToName(const AggregationNode::Step& step) {
  std::stringstream ss;
  ss << step;
  return ss.str();
}

/// Plan node used to implement aggregations over grouping sets. Duplicates the
/// aggregation input for each set of grouping keys. The output contains one
/// column for each grouping key, followed by aggregation inputs, followed by a
/// column containing grouping set ID. For a given grouping set, a subset
/// of the grouping key columns present in the set are populated with values.
/// The rest of the grouping key columns are filled in with nulls.
class GroupIdNode : public PlanNode {
 public:
  struct GroupingKeyInfo {
    // The name to use in the output.
    std::string output;
    // The input field.
    FieldAccessTypedExprPtr input;

    folly::dynamic serialize() const;
  };

  /// @param id Plan node ID.
  /// @param groupingSets A list of grouping key sets. Grouping keys within the
  /// set must be unique, but grouping keys across sets may repeat.
  /// @param groupingKeyInfos The names and order of the grouping keys in the
  /// output.
  /// @param aggregationInputs Columns that contain inputs to the aggregate
  /// functions.
  /// @param groupIdName Name of the column that will contain the grouping set
  /// ID (a zero based integer).
  /// @param source Input plan node.
  GroupIdNode(
      PlanNodeId id,
      std::vector<std::vector<FieldAccessTypedExprPtr>> groupingSets,
      std::vector<GroupingKeyInfo> groupingKeyInfos,
      std::vector<FieldAccessTypedExprPtr> aggregationInputs,
      std::string groupIdName,
      PlanNodePtr source);

  const RowTypePtr& outputType() const override {
    return outputType_;
  }

  const std::vector<PlanNodePtr>& sources() const override {
    return sources_;
  }

  const std::vector<std::vector<FieldAccessTypedExprPtr>>& groupingSets()
      const {
    return groupingSets_;
  }

  const std::vector<GroupingKeyInfo>& groupingKeyInfos() const {
    return groupingKeyInfos_;
  }

  const std::vector<FieldAccessTypedExprPtr>& aggregationInputs() const {
    return aggregationInputs_;
  }

  const std::string& groupIdName() {
    return groupIdName_;
  }

  int32_t numGroupingKeys() const {
    return outputType_->size() - aggregationInputs_.size() - 1;
  }

  std::string_view name() const override {
    return "GroupId";
  }

  folly::dynamic serialize() const override;

  static PlanNodePtr create(const folly::dynamic& obj, void* context);

 private:
  void addDetails(std::stringstream& stream) const override;

  const std::vector<PlanNodePtr> sources_;
  const RowTypePtr outputType_;
  const std::vector<std::vector<FieldAccessTypedExprPtr>> groupingSets_;
  const std::vector<GroupingKeyInfo> groupingKeyInfos_;
  const std::vector<FieldAccessTypedExprPtr> aggregationInputs_;
  const std::string groupIdName_;
};

class ExchangeNode : public PlanNode {
 public:
  ExchangeNode(const PlanNodeId& id, RowTypePtr type)
      : PlanNode(id), outputType_(type) {}

  const RowTypePtr& outputType() const override {
    return outputType_;
  }

  const std::vector<PlanNodePtr>& sources() const override;

  bool requiresExchangeClient() const override {
    return true;
  }

  bool requiresSplits() const override {
    return true;
  }

  std::string_view name() const override {
    return "Exchange";
  }

  folly::dynamic serialize() const override;

  static PlanNodePtr create(const folly::dynamic& obj, void* context);

 private:
  void addDetails(std::stringstream& stream) const override;

  RowTypePtr outputType_;
};

class MergeExchangeNode : public ExchangeNode {
 public:
  explicit MergeExchangeNode(
      const PlanNodeId& id,
      const RowTypePtr& type,
      const std::vector<FieldAccessTypedExprPtr>& sortingKeys,
      const std::vector<SortOrder>& sortingOrders)
      : ExchangeNode(id, type),
        sortingKeys_(sortingKeys),
        sortingOrders_(sortingOrders) {}

  const std::vector<FieldAccessTypedExprPtr>& sortingKeys() const {
    return sortingKeys_;
  }

  const std::vector<SortOrder>& sortingOrders() const {
    return sortingOrders_;
  }

  std::string_view name() const override {
    return "MergeExchange";
  }

  folly::dynamic serialize() const override;

  static PlanNodePtr create(const folly::dynamic& obj, void* context);

 private:
  void addDetails(std::stringstream& stream) const override;

  const std::vector<FieldAccessTypedExprPtr> sortingKeys_;
  const std::vector<SortOrder> sortingOrders_;
};

class LocalMergeNode : public PlanNode {
 public:
  LocalMergeNode(
      const PlanNodeId& id,
      std::vector<FieldAccessTypedExprPtr> sortingKeys,
      std::vector<SortOrder> sortingOrders,
      std::vector<PlanNodePtr> sources)
      : PlanNode(id),
        sources_{std::move(sources)},
        sortingKeys_{std::move(sortingKeys)},
        sortingOrders_{std::move(sortingOrders)} {}

  const RowTypePtr& outputType() const override {
    return sources_[0]->outputType();
  }

  const std::vector<PlanNodePtr>& sources() const override {
    return sources_;
  }

  const std::vector<FieldAccessTypedExprPtr>& sortingKeys() const {
    return sortingKeys_;
  }

  const std::vector<SortOrder>& sortingOrders() const {
    return sortingOrders_;
  }

  std::string_view name() const override {
    return "LocalMerge";
  }

  folly::dynamic serialize() const override;

  static PlanNodePtr create(const folly::dynamic& obj, void* context);

 private:
  void addDetails(std::stringstream& stream) const override;

  const std::vector<PlanNodePtr> sources_;
  const std::vector<FieldAccessTypedExprPtr> sortingKeys_;
  const std::vector<SortOrder> sortingOrders_;
};

/// Calculates partition number for each row of the specified vector.
class PartitionFunction {
 public:
  virtual ~PartitionFunction() = default;

  /// @param input RowVector to split into partitions.
  /// @param [out] partitions Computed partition numbers for each row in
  /// 'input'.
  virtual void partition(
      const RowVector& input,
      std::vector<uint32_t>& partitions) = 0;
};

/// Factory class for creating PartitionFunction instances.
class PartitionFunctionSpec : public ISerializable {
 public:
  virtual std::unique_ptr<PartitionFunction> create(
      int numPartitions) const = 0;

  virtual ~PartitionFunctionSpec() = default;

  virtual std::string toString() const = 0;
};

using PartitionFunctionSpecPtr = std::shared_ptr<const PartitionFunctionSpec>;

class GatherPartitionFunctionSpec : public PartitionFunctionSpec {
 public:
  std::unique_ptr<PartitionFunction> create(
      int /* numPartitions */) const override {
    VELOX_UNREACHABLE();
  }

  std::string toString() const override {
    return "gather";
  }

  folly::dynamic serialize() const override {
    folly::dynamic obj = folly::dynamic::object;
    obj["name"] = "GatherPartitionFunctionSpec";
    return obj;
  }

  static PartitionFunctionSpecPtr deserialize(
      const folly::dynamic& /* obj */,
      void* /* context */) {
    return std::make_shared<GatherPartitionFunctionSpec>();
  }
};

/// TODO Remove once Prestissimo is updated.
#ifdef VELOX_ENABLE_BACKWARD_COMPATIBILITY
using PartitionFunctionFactory =
    std::function<std::unique_ptr<PartitionFunction>(int numPartitions)>;

class LegacyPartitionFunctionSpec : public PartitionFunctionSpec {
 public:
  LegacyPartitionFunctionSpec(PartitionFunctionFactory factory)
      : factory_{std::move(factory)} {}

  std::unique_ptr<PartitionFunction> create(int numPartitions) const override {
    return factory_(numPartitions);
  };

  std::string toString() const override {
    return "legacy";
  }

  folly::dynamic serialize() const override {
    VELOX_UNSUPPORTED();
  }

 private:
  const PartitionFunctionFactory factory_;
};
#endif

/// Partitions data using specified partition function. The number of partitions
/// is determined by the parallelism of the upstream pipeline. Can be used to
/// gather data from multiple sources.
class LocalPartitionNode : public PlanNode {
 public:
  enum class Type {
    // N-to-1 exchange.
    kGather,
    // N-to-M shuffle.
    kRepartition,
  };

  static const char* typeName(Type type);

  static Type typeFromName(const std::string& name);

  LocalPartitionNode(
      const PlanNodeId& id,
      Type type,
      PartitionFunctionSpecPtr partitionFunctionSpec,
      std::vector<PlanNodePtr> sources)
      : PlanNode(id),
        type_{type},
        sources_{std::move(sources)},
        partitionFunctionSpec_{std::move(partitionFunctionSpec)} {
    VELOX_USER_CHECK_GT(
        sources_.size(),
        0,
        "Local repartitioning node requires at least one source");

    VELOX_USER_CHECK_NOT_NULL(partitionFunctionSpec_);

    for (auto i = 1; i < sources_.size(); ++i) {
      VELOX_USER_CHECK(
          *sources_[i]->outputType() == *sources_[0]->outputType(),
          "All sources of the LocalPartitionedNode must have the same output type: {} vs. {}.",
          sources_[i]->outputType()->toString(),
          sources_[0]->outputType()->toString());
    }
  }

  /// TODO Remove once Prestissimo is updated.
#ifdef VELOX_ENABLE_BACKWARD_COMPATIBILITY
  LocalPartitionNode(
      const PlanNodeId& id,
      Type type,
      PartitionFunctionFactory partitionFunctionFactory,
      std::vector<PlanNodePtr> sources)
      : LocalPartitionNode(
            id,
            type,
            std::make_shared<LegacyPartitionFunctionSpec>(
                std::move(partitionFunctionFactory)),
            std::move(sources)) {}
#endif

  static std::shared_ptr<LocalPartitionNode> gather(
      const PlanNodeId& id,
      std::vector<PlanNodePtr> sources) {
    return std::make_shared<LocalPartitionNode>(
        id,
        Type::kGather,
        std::make_shared<GatherPartitionFunctionSpec>(),
        std::move(sources));
  }

  Type type() const {
    return type_;
  }

  const RowTypePtr& outputType() const override {
    return sources_[0]->outputType();
  }

  const std::vector<PlanNodePtr>& sources() const override {
    return sources_;
  }

  const PartitionFunctionSpec& partitionFunctionSpec() const {
    return *partitionFunctionSpec_;
  }

  std::string_view name() const override {
    return "LocalPartition";
  }

  folly::dynamic serialize() const override;

  static PlanNodePtr create(const folly::dynamic& obj, void* context);

 private:
  void addDetails(std::stringstream& stream) const override;

  const Type type_;
  const std::vector<PlanNodePtr> sources_;
  const PartitionFunctionSpecPtr partitionFunctionSpec_;
};

class PartitionedOutputNode : public PlanNode {
 public:
  PartitionedOutputNode(
      const PlanNodeId& id,
      const std::vector<TypedExprPtr>& keys,
      int numPartitions,
      bool broadcast,
      bool replicateNullsAndAny,
      PartitionFunctionSpecPtr partitionFunctionSpec,
      RowTypePtr outputType,
      PlanNodePtr source)
      : PlanNode(id),
        sources_{{std::move(source)}},
        keys_(keys),
        numPartitions_(numPartitions),
        broadcast_(broadcast),
        replicateNullsAndAny_(replicateNullsAndAny),
        partitionFunctionSpec_(std::move(partitionFunctionSpec)),
        outputType_(std::move(outputType)) {
    VELOX_CHECK(numPartitions > 0, "numPartitions must be greater than zero");
    if (numPartitions == 1) {
      VELOX_CHECK(
          keys_.empty(),
          "Non-empty partitioning keys require more than one partition");
    }
    if (broadcast) {
      VELOX_CHECK(
          keys_.empty(),
          "Broadcast partitioning doesn't allow for partitioning keys");
    }
  }

#ifdef VELOX_ENABLE_BACKWARD_COMPATIBILITY
  PartitionedOutputNode(
      const PlanNodeId& id,
      const std::vector<TypedExprPtr>& keys,
      int numPartitions,
      bool broadcast,
      bool replicateNullsAndAny,
      PartitionFunctionFactory partitionFunctionFactory,
      RowTypePtr outputType,
      PlanNodePtr source)
      : PartitionedOutputNode(
            id,
            keys,
            numPartitions,
            broadcast,
            replicateNullsAndAny,
            std::make_shared<LegacyPartitionFunctionSpec>(
                std::move(partitionFunctionFactory)),
            std::move(outputType),
            std::move(source)) {}

  PartitionFunctionFactory partitionFunctionFactory() const {
    return [this](int numPartitions) {
      return partitionFunctionSpec_->create(numPartitions);
    };
  }
#endif

  static std::shared_ptr<PartitionedOutputNode> broadcast(
      const PlanNodeId& id,
      int numPartitions,
      RowTypePtr outputType,
      PlanNodePtr source) {
    std::vector<TypedExprPtr> noKeys;
    return std::make_shared<PartitionedOutputNode>(
        id,
        noKeys,
        numPartitions,
        true,
        false,
        std::make_shared<GatherPartitionFunctionSpec>(),
        std::move(outputType),
        std::move(source));
  }

  static std::shared_ptr<PartitionedOutputNode>
  single(const PlanNodeId& id, RowTypePtr outputType, PlanNodePtr source) {
    std::vector<TypedExprPtr> noKeys;
    return std::make_shared<PartitionedOutputNode>(
        id,
        noKeys,
        1,
        false,
        false,
        std::make_shared<GatherPartitionFunctionSpec>(),
        std::move(outputType),
        std::move(source));
  }

  const RowTypePtr& outputType() const override {
    return outputType_;
  }

  const std::vector<PlanNodePtr>& sources() const override {
    return sources_;
  }

  const RowTypePtr& inputType() const {
    return sources_[0]->outputType();
  }

  const std::vector<TypedExprPtr>& keys() const {
    return keys_;
  }

  int numPartitions() const {
    return numPartitions_;
  }

  bool isBroadcast() const {
    return broadcast_;
  }

  /// Returns true if an arbitrary row and all rows with null keys must be
  /// replicated to all destinations. This is used to ensure correct results for
  /// anti-join which requires all nodes to know whether combined build side is
  /// empty and whether it has any entry with null join key.
  bool isReplicateNullsAndAny() const {
    return replicateNullsAndAny_;
  }

  const PartitionFunctionSpecPtr& partitionFunctionSpecPtr() const {
    return partitionFunctionSpec_;
  }

  const PartitionFunctionSpec& partitionFunctionSpec() const {
    return *partitionFunctionSpec_;
  }

  std::string_view name() const override {
    return "PartitionedOutput";
  }

  folly::dynamic serialize() const override;

  static PlanNodePtr create(const folly::dynamic& obj, void* context);

 private:
  void addDetails(std::stringstream& stream) const override;

  const std::vector<PlanNodePtr> sources_;
  const std::vector<TypedExprPtr> keys_;
  const int numPartitions_;
  const bool broadcast_;
  const bool replicateNullsAndAny_;
  const PartitionFunctionSpecPtr partitionFunctionSpec_;
  const RowTypePtr outputType_;
};

enum class JoinType {
  // For each row on the left, find all matching rows on the right and return
  // all combinations.
  kInner,
  // For each row on the left, find all matching rows on the right and return
  // all combinations. In addition, return all rows from the left that have no
  // match on the right with right-side columns filled with nulls.
  kLeft,
  // Opposite of kLeft. For each row on the right, find all matching rows on the
  // left and return all combinations. In addition, return all rows from the
  // right that have no match on the left with left-side columns filled with
  // nulls.
  kRight,
  // A "union" of kLeft and kRight. For each row on the left, find all matching
  // rows on the right and return all combinations. In addition, return all rows
  // from the left that have no
  // match on the right with right-side columns filled with nulls. Also, return
  // all rows from the
  // right that have no match on the left with left-side columns filled with
  // nulls.
  kFull,
  // Return a subset of rows from the left side which have a match on the right
  // side. For this join type, cardinality of the output is less than or equal
  // to the cardinality of the left side.
  kLeftSemiFilter,
  // Return each row from the left side with a boolean flag indicating whether
  // there exists a match on the right side. For this join type, cardinality of
  // the output equals the cardinality of the left side.
  //
  // The handling of the rows with nulls in the join key depends on the
  // 'nullAware' boolean specified separately.
  //
  // Null-aware join follows IN semantic. Regular join follows EXISTS semantic.
  kLeftSemiProject,
  // Opposite of kLeftSemiFilter. Return a subset of rows from the right side
  // which have a match on the left side. For this join type, cardinality of the
  // output is less than or equal to the cardinality of the right side.
  kRightSemiFilter,
  // Opposite of kLeftSemiProject. Return each row from the right side with a
  // boolean flag indicating whether there exists a match on the left side. For
  // this join type, cardinality of the output equals the cardinality of the
  // right side.
  //
  // The handling of the rows with nulls in the join key depends on the
  // 'nullAware' boolean specified separately.
  //
  // Null-aware join follows IN semantic. Regular join follows EXISTS semantic.
  kRightSemiProject,
  // Return each row from the left side which has no match on the right side.
  // The handling of the rows with nulls in the join key depends on the
  // 'nullAware' boolean specified separately.
  //
  // Null-aware join follows NOT IN semantic:
  // (1) return empty result if the right side contains a record with a null in
  // the join key;
  // (2) return left-side row with null in the join key only when
  // the right side is empty.
  //
  // Regular anti join follows NOT EXISTS semantic:
  // (1) ignore right-side rows with nulls in the join keys;
  // (2) unconditionally return left side rows with nulls in the join keys.
  kAnti,
};

const char* joinTypeName(JoinType joinType);

JoinType joinTypeFromName(const std::string& name);

inline bool isInnerJoin(JoinType joinType) {
  return joinType == JoinType::kInner;
}

inline bool isLeftJoin(JoinType joinType) {
  return joinType == JoinType::kLeft;
}

inline bool isRightJoin(JoinType joinType) {
  return joinType == JoinType::kRight;
}

inline bool isFullJoin(JoinType joinType) {
  return joinType == JoinType::kFull;
}

inline bool isLeftSemiFilterJoin(JoinType joinType) {
  return joinType == JoinType::kLeftSemiFilter;
}

inline bool isLeftSemiProjectJoin(JoinType joinType) {
  return joinType == JoinType::kLeftSemiProject;
}

inline bool isRightSemiFilterJoin(JoinType joinType) {
  return joinType == JoinType::kRightSemiFilter;
}

inline bool isRightSemiProjectJoin(JoinType joinType) {
  return joinType == JoinType::kRightSemiProject;
}

inline bool isAntiJoin(JoinType joinType) {
  return joinType == JoinType::kAnti;
}

inline bool isNullAwareSupported(core::JoinType joinType) {
  return joinType == JoinType::kAnti ||
      joinType == JoinType::kLeftSemiProject ||
      joinType == JoinType::kRightSemiProject;
}

/// Abstract class representing inner/outer/semi/anti joins. Used as a base
/// class for specific join implementations, e.g. hash and merge joins.
class AbstractJoinNode : public PlanNode {
 public:
  AbstractJoinNode(
      const PlanNodeId& id,
      JoinType joinType,
      const std::vector<FieldAccessTypedExprPtr>& leftKeys,
      const std::vector<FieldAccessTypedExprPtr>& rightKeys,
      TypedExprPtr filter,
      PlanNodePtr left,
      PlanNodePtr right,
      const RowTypePtr outputType);

  const std::vector<PlanNodePtr>& sources() const override {
    return sources_;
  }

  const RowTypePtr& outputType() const override {
    return outputType_;
  }

  JoinType joinType() const {
    return joinType_;
  }

  bool isInnerJoin() const {
    return joinType_ == JoinType::kInner;
  }

  bool isLeftJoin() const {
    return joinType_ == JoinType::kLeft;
  }

  bool isRightJoin() const {
    return joinType_ == JoinType::kRight;
  }

  bool isFullJoin() const {
    return joinType_ == JoinType::kFull;
  }

  bool isLeftSemiFilterJoin() const {
    return joinType_ == JoinType::kLeftSemiFilter;
  }

  bool isLeftSemiProjectJoin() const {
    return joinType_ == JoinType::kLeftSemiProject;
  }

  bool isRightSemiFilterJoin() const {
    return joinType_ == JoinType::kRightSemiFilter;
  }

  bool isRightSemiProjectJoin() const {
    return joinType_ == JoinType::kRightSemiProject;
  }

  bool isAntiJoin() const {
    return joinType_ == JoinType::kAnti;
  }

  const std::vector<FieldAccessTypedExprPtr>& leftKeys() const {
    return leftKeys_;
  }

  const std::vector<FieldAccessTypedExprPtr>& rightKeys() const {
    return rightKeys_;
  }

  const TypedExprPtr& filter() const {
    return filter_;
  }

 protected:
  void addDetails(std::stringstream& stream) const override;

  folly::dynamic serializeBase() const;

  const JoinType joinType_;
  const std::vector<FieldAccessTypedExprPtr> leftKeys_;
  const std::vector<FieldAccessTypedExprPtr> rightKeys_;
  // Optional join filter, nullptr if absent. This is applied to
  // join hits and if this is false, the hit turns into a miss, which
  // has a special meaning for outer joins. For inner joins, this is
  // equivalent to a Filter above the join.
  const TypedExprPtr filter_;
  const std::vector<PlanNodePtr> sources_;
  const RowTypePtr outputType_;
};

/// Represents inner/outer/semi/anti hash joins. Translates to an
/// exec::HashBuild and exec::HashProbe. A separate pipeline is produced for the
/// build side when generating exec::Operators.
///
/// 'nullAware' boolean applies to semi and anti joins. When true, the join
/// semantic is IN / NOT IN. When false, the join semantic is EXISTS / NOT
/// EXISTS.
class HashJoinNode : public AbstractJoinNode {
 public:
  HashJoinNode(
      const PlanNodeId& id,
      JoinType joinType,
      bool nullAware,
      const std::vector<FieldAccessTypedExprPtr>& leftKeys,
      const std::vector<FieldAccessTypedExprPtr>& rightKeys,
      TypedExprPtr filter,
      PlanNodePtr left,
      PlanNodePtr right,
      const RowTypePtr outputType)
      : AbstractJoinNode(
            id,
            joinType,
            leftKeys,
            rightKeys,
            filter,
            left,
            right,
            outputType),
        nullAware_{nullAware} {
    if (nullAware) {
      VELOX_USER_CHECK(
          isNullAwareSupported(joinType),
          "Null-aware flag is supported only for semi and anti joins");
      VELOX_USER_CHECK_EQ(
          1, leftKeys_.size(), "Null-aware joins allow only one join key");

      if (filter_) {
        VELOX_USER_CHECK(
            !isRightSemiProjectJoin(),
            "Null-aware right semi project join doesn't support extra filter");
      }
    }
  }

  std::string_view name() const override {
    return "HashJoin";
  }

  bool canSpill(const QueryConfig& queryConfig) const override {
    // NOTE: as for now, we don't allow spilling for null-aware anti-join with
    // filter set. It requires to cross join the null-key probe rows with all
    // the build-side rows for filter evaluation which is not supported under
    // spilling.
    return !(isAntiJoin() && nullAware_ && filter() != nullptr) &&
        queryConfig.joinSpillEnabled();
  }

  bool isNullAware() const {
    return nullAware_;
  }

  folly::dynamic serialize() const override;

  static PlanNodePtr create(const folly::dynamic& obj, void* context);

 private:
  void addDetails(std::stringstream& stream) const override;

  const bool nullAware_;
};

/// Represents inner/outer/semi/anti merge joins. Translates to an
/// exec::MergeJoin operator. Assumes that both left and right input data is
/// sorted on the join keys. A separate pipeline that puts its output into
/// exec::MergeJoinSource is produced for the right side when generating
/// exec::Operators.
class MergeJoinNode : public AbstractJoinNode {
 public:
  MergeJoinNode(
      const PlanNodeId& id,
      JoinType joinType,
      const std::vector<FieldAccessTypedExprPtr>& leftKeys,
      const std::vector<FieldAccessTypedExprPtr>& rightKeys,
      TypedExprPtr filter,
      PlanNodePtr left,
      PlanNodePtr right,
      const RowTypePtr outputType)
      : AbstractJoinNode(
            id,
            joinType,
            leftKeys,
            rightKeys,
            filter,
            left,
            right,
            outputType) {}

  std::string_view name() const override {
    return "MergeJoin";
  }

  folly::dynamic serialize() const override;

  static PlanNodePtr create(const folly::dynamic& obj, void* context);
};

// Cross join.
class CrossJoinNode : public PlanNode {
 public:
  CrossJoinNode(
      const PlanNodeId& id,
      PlanNodePtr left,
      PlanNodePtr right,
      RowTypePtr outputType);

  const std::vector<PlanNodePtr>& sources() const override {
    return sources_;
  }

  const RowTypePtr& outputType() const override {
    return outputType_;
  }

  std::string_view name() const override {
    return "CrossJoin";
  }

  folly::dynamic serialize() const override;

  static PlanNodePtr create(const folly::dynamic& obj, void* context);

 private:
  void addDetails(std::stringstream& stream) const override;

  const std::vector<PlanNodePtr> sources_;
  const RowTypePtr outputType_;
};

// Represents the 'SortBy' node in the plan.
class OrderByNode : public PlanNode {
 public:
  OrderByNode(
      const PlanNodeId& id,
      const std::vector<FieldAccessTypedExprPtr>& sortingKeys,
      const std::vector<SortOrder>& sortingOrders,
      bool isPartial,
      const PlanNodePtr& source)
      : PlanNode(id),
        sortingKeys_(sortingKeys),
        sortingOrders_(sortingOrders),
        isPartial_(isPartial),
        sources_{source} {
    VELOX_CHECK(!sortingKeys.empty(), "OrderBy must specify sorting keys");
    VELOX_CHECK_EQ(
        sortingKeys.size(),
        sortingOrders.size(),
        "Number of sorting keys and sorting orders in OrderBy must be the same");
  }

  const std::vector<FieldAccessTypedExprPtr>& sortingKeys() const {
    return sortingKeys_;
  }

  const std::vector<SortOrder>& sortingOrders() const {
    return sortingOrders_;
  }

  bool canSpill(const QueryConfig& queryConfig) const override {
    return queryConfig.orderBySpillEnabled();
  }

  const RowTypePtr& outputType() const override {
    return sources_[0]->outputType();
  }

  const std::vector<PlanNodePtr>& sources() const override {
    return sources_;
  }

  // True if this node only sorts a portion of the final result. If it is
  // true, a local merge or merge exchange is required to merge the sorted
  // runs.
  bool isPartial() const {
    return isPartial_;
  }

  std::string_view name() const override {
    return "OrderBy";
  }

  folly::dynamic serialize() const override;

  static PlanNodePtr create(const folly::dynamic& obj, void* context);

 private:
  void addDetails(std::stringstream& stream) const override;

  const std::vector<FieldAccessTypedExprPtr> sortingKeys_;
  const std::vector<SortOrder> sortingOrders_;
  const bool isPartial_;
  const std::vector<PlanNodePtr> sources_;
};

class TopNNode : public PlanNode {
 public:
  TopNNode(
      const PlanNodeId& id,
      const std::vector<FieldAccessTypedExprPtr>& sortingKeys,
      const std::vector<SortOrder>& sortingOrders,
      int32_t count,
      bool isPartial,
      const PlanNodePtr& source)
      : PlanNode(id),
        sortingKeys_(sortingKeys),
        sortingOrders_(sortingOrders),
        count_(count),
        isPartial_(isPartial),
        sources_{source} {
    VELOX_CHECK(!sortingKeys.empty(), "TopN must specify sorting keys");
    VELOX_CHECK(
        sortingKeys.size() == sortingOrders.size(),
        "Number of sorting keys and sorting orders in TopN must be the same");
    VELOX_CHECK(
        count > 0,
        "TopN must specify greater than zero number of rows to keep");
  }

  const std::vector<FieldAccessTypedExprPtr>& sortingKeys() const {
    return sortingKeys_;
  }

  const std::vector<SortOrder>& sortingOrders() const {
    return sortingOrders_;
  }

  const RowTypePtr& outputType() const override {
    return sources_[0]->outputType();
  }

  const std::vector<PlanNodePtr>& sources() const override {
    return sources_;
  }

  int32_t count() const {
    return count_;
  }

  bool isPartial() const {
    return isPartial_;
  }

  std::string_view name() const override {
    return "TopN";
  }

  folly::dynamic serialize() const override;

  static PlanNodePtr create(const folly::dynamic& obj, void* context);

 private:
  void addDetails(std::stringstream& stream) const override;

  const std::vector<FieldAccessTypedExprPtr> sortingKeys_;
  const std::vector<SortOrder> sortingOrders_;
  const int32_t count_;
  const bool isPartial_;
  const std::vector<PlanNodePtr> sources_;
};

class LimitNode : public PlanNode {
 public:
  // @param isPartial Boolean indicating whether Limit node generates partial
  // results on local workers or finalizes the partial results from `PARTIAL`
  // nodes.
  LimitNode(
      const PlanNodeId& id,
      int32_t offset,
      int32_t count,
      bool isPartial,
      const PlanNodePtr& source)
      : PlanNode(id),
        offset_(offset),
        count_(count),
        isPartial_(isPartial),
        sources_{source} {
    VELOX_CHECK(
        count > 0,
        "Limit must specify greater than zero number of rows to keep");
  }

  const RowTypePtr& outputType() const override {
    return sources_[0]->outputType();
  }

  const std::vector<PlanNodePtr>& sources() const override {
    return sources_;
  }

  int32_t offset() const {
    return offset_;
  }

  int32_t count() const {
    return count_;
  }

  bool isPartial() const {
    return isPartial_;
  }

  std::string_view name() const override {
    return "Limit";
  }

  folly::dynamic serialize() const override;

  static PlanNodePtr create(const folly::dynamic& obj, void* context);

 private:
  void addDetails(std::stringstream& stream) const override;

  const int32_t offset_;
  const int32_t count_;
  const bool isPartial_;
  const std::vector<PlanNodePtr> sources_;
};

/// Expands arrays and maps into separate columns. Arrays are expanded into a
/// single column, and maps are expanded into two columns (key, value). Can be
/// used to expand multiple columns. In this case will produce as many rows as
/// the highest cardinality array or map (the other columns are padded with
/// nulls). Optionally can produce an ordinality column that specifies the row
/// number starting with 1.
class UnnestNode : public PlanNode {
 public:
  /// @param replicateVariables Inputs that are projected as is
  /// @param unnestVariables Inputs that are unnested. Must be of type ARRAY or
  /// MAP.
  /// @param unnestNames Names to use for unnested outputs: one name for each
  /// array (element); two names for each map (key and value). The output names
  /// must appear in the same order as unnestVariables.
  /// @param ordinalityName Optional name for the ordinality columns. If not
  /// present, ordinality column is not produced.
  UnnestNode(
      const PlanNodeId& id,
      std::vector<FieldAccessTypedExprPtr> replicateVariables,
      std::vector<FieldAccessTypedExprPtr> unnestVariables,
      const std::vector<std::string>& unnestNames,
      const std::optional<std::string>& ordinalityName,
      const PlanNodePtr& source);

  /// The order of columns in the output is: replicated columns (in the order
  /// specified), unnested columns (in the order specified, for maps: key comes
  /// before value), optional ordinality column.
  const RowTypePtr& outputType() const override {
    return outputType_;
  }

  const std::vector<PlanNodePtr>& sources() const override {
    return sources_;
  }

  const std::vector<FieldAccessTypedExprPtr>& replicateVariables() const {
    return replicateVariables_;
  }

  const std::vector<FieldAccessTypedExprPtr>& unnestVariables() const {
    return unnestVariables_;
  }

  bool withOrdinality() const {
    return withOrdinality_;
  }

  std::string_view name() const override {
    return "Unnest";
  }

  folly::dynamic serialize() const override;

  static PlanNodePtr create(const folly::dynamic& obj, void* context);

 private:
  void addDetails(std::stringstream& stream) const override;

  const std::vector<FieldAccessTypedExprPtr> replicateVariables_;
  const std::vector<FieldAccessTypedExprPtr> unnestVariables_;
  const bool withOrdinality_;
  const std::vector<PlanNodePtr> sources_;
  RowTypePtr outputType_;
};

/// Checks that input contains at most one row. Return that row as is. If input
/// is empty, returns a single row with all values set to null. If input
/// contains more than one row raises an exception.
///
/// This plan node is used in query plans that use non-correlated sub-queries.
class EnforceSingleRowNode : public PlanNode {
 public:
  EnforceSingleRowNode(const PlanNodeId& id, PlanNodePtr source)
      : PlanNode(id), sources_{std::move(source)} {}

  const RowTypePtr& outputType() const override {
    return sources_[0]->outputType();
  }

  const std::vector<PlanNodePtr>& sources() const override {
    return sources_;
  }

  std::string_view name() const override {
    return "EnforceSingleRow";
  }

  folly::dynamic serialize() const override;

  static PlanNodePtr create(const folly::dynamic& obj, void* context);

 private:
  void addDetails(std::stringstream& stream) const override;

  const std::vector<PlanNodePtr> sources_;
};

/// Adds a new column named `idName` at the end of the input columns
/// with unique int64_t value per input row.
///
/// 64-bit unique id is built in following way:
///  - first 24 bits - task unique id
///  - next 40 bits - operator counter value
///
/// The task unique id is added to ensure the generated id is unique
/// across all the nodes executing the same query stage in a distributed
/// query execution.
class AssignUniqueIdNode : public PlanNode {
 public:
  AssignUniqueIdNode(
      const PlanNodeId& id,
      const std::string& idName,
      const int32_t taskUniqueId,
      PlanNodePtr source);

  const RowTypePtr& outputType() const override {
    return outputType_;
  }

  const std::vector<PlanNodePtr>& sources() const override {
    return sources_;
  }

  std::string_view name() const override {
    return "AssignUniqueId";
  }

  int32_t taskUniqueId() const {
    return taskUniqueId_;
  }

  const std::shared_ptr<std::atomic_int64_t>& uniqueIdCounter() const {
    return uniqueIdCounter_;
  };

  folly::dynamic serialize() const override;

  static PlanNodePtr create(const folly::dynamic& obj, void* context);

 private:
  void addDetails(std::stringstream& stream) const override;

  const int32_t taskUniqueId_;
  const std::vector<PlanNodePtr> sources_;
  RowTypePtr outputType_;
  std::shared_ptr<std::atomic_int64_t> uniqueIdCounter_;
};

/// PlanNode used for evaluating Sql window functions.
/// All window functions evaluated in the operator have the same
/// window spec (partition keys + order columns).
/// If no partition keys are specified, then all input rows
/// are considered to be in a single partition.
/// If no order by columns are specified, then the input rows order
/// is non-deterministic.
/// Each window function also has a frame which specifies the sliding
/// window over which it is computed. The frame
/// could be RANGE (based on peers which are all rows with the same
/// ORDER BY value) or ROWS (position based).
/// The frame bound types are CURRENT_ROW, (expression or UNBOUNDED)
/// ROWS_PRECEDING and (expression or UNBOUNDED) ROWS_FOLLOWING.
/// The WindowNode has one passthrough output column for each input
/// column followed by the results of the window functions.
class WindowNode : public PlanNode {
 public:
  enum class WindowType { kRange, kRows };

  static const char* windowTypeName(WindowType type);

  static WindowType windowTypeFromName(const std::string& name);

  enum class BoundType {
    kUnboundedPreceding,
    kPreceding,
    kCurrentRow,
    kFollowing,
    kUnboundedFollowing
  };

  static const char* boundTypeName(BoundType type);

  static BoundType boundTypeFromName(const std::string& name);

  struct Frame {
    WindowType type;
    BoundType startType;
    TypedExprPtr startValue;
    BoundType endType;
    TypedExprPtr endValue;

    folly::dynamic serialize() const;

    static Frame deserialize(const folly::dynamic& obj);
  };

  struct Function {
    CallTypedExprPtr functionCall;
    Frame frame;
    bool ignoreNulls;

    folly::dynamic serialize() const;

    static Function deserialize(const folly::dynamic& obj);
  };

  /// @param windowColumnNames specifies the output column
  /// names for each window function column. So
  /// windowColumnNames.length() = windowFunctions.length().
  WindowNode(
      PlanNodeId id,
      std::vector<FieldAccessTypedExprPtr> partitionKeys,
      std::vector<FieldAccessTypedExprPtr> sortingKeys,
      std::vector<SortOrder> sortingOrders,
      std::vector<std::string> windowColumnNames,
      std::vector<Function> windowFunctions,
      PlanNodePtr source);

  const std::vector<PlanNodePtr>& sources() const override {
    return sources_;
  }

  /// The outputType is the concatenation of the input columns
  /// with the output columns of each window function.
  const RowTypePtr& outputType() const override {
    return outputType_;
  }

  const std::vector<FieldAccessTypedExprPtr>& partitionKeys() const {
    return partitionKeys_;
  }

  const std::vector<FieldAccessTypedExprPtr>& sortingKeys() const {
    return sortingKeys_;
  }

  const std::vector<SortOrder>& sortingOrders() const {
    return sortingOrders_;
  }

  const std::vector<Function>& windowFunctions() const {
    return windowFunctions_;
  }

  std::string_view name() const override {
    return "Window";
  }

  folly::dynamic serialize() const override;

  static PlanNodePtr create(const folly::dynamic& obj, void* context);

 private:
  void addDetails(std::stringstream& stream) const override;

  const std::vector<FieldAccessTypedExprPtr> partitionKeys_;

  const std::vector<FieldAccessTypedExprPtr> sortingKeys_;
  const std::vector<SortOrder> sortingOrders_;

  const std::vector<Function> windowFunctions_;

  const std::vector<PlanNodePtr> sources_;

  const RowTypePtr outputType_;
};

} // namespace facebook::velox::core
