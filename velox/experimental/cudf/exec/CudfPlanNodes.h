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

#include "velox/experimental/cudf/exec/CudfPlanNode.h"

namespace facebook::velox::cudf_velox {

/// Base class of cuDF plan nodes with one source that take and produce cuDF
/// vectors.
class CudfUnaryPlanNode : public CudfPlanNode {
 public:
  CudfPlanNodeTraits cudfTraits() const override {
    return {{InputRepresentation::kCudf}, VectorRepresentation::kCudf, {}};
  }

  const RowTypePtr& outputType() const override {
    return outputType_;
  }

  const std::vector<core::PlanNodePtr>& sources() const override {
    return sources_;
  }

 protected:
  CudfUnaryPlanNode(
      const core::PlanNodeId& id,
      core::PlanNodePtr source,
      RowTypePtr outputType)
      : CudfPlanNode(id),
        sources_{std::move(source)},
        outputType_(std::move(outputType)) {}

  template <typename Node>
  core::PlanNodePtr copyWithSources(
      std::vector<core::PlanNodePtr> sources) const {
    VELOX_CHECK_EQ(sources.size(), 1);
    auto copy = std::make_shared<Node>(static_cast<const Node&>(*this));
    copy->sources_ = std::move(sources);
    return copy;
  }

  std::vector<core::PlanNodePtr> sources_;
  const RowTypePtr outputType_;
};

/// One physical cuDF node for the Filter, Project, or fused Filter+Project
/// execution unit.
class CudfFilterProjectNode : public CudfUnaryPlanNode {
 public:
  /// 'filter' and 'filterNodeId' are set together, for a node that filters.
  /// 'projectNodeId' is set for a node that projects: 'projections' then has
  /// one expression of the matching type for each column of 'outputType'.
  /// Without it, 'projections' is empty and 'outputType' is the type of
  /// 'source'. 'lazyDereference' is allowed only for a projection without a
  /// filter. Throws if the arguments break these rules, or if
  /// unsupportedReason() has a reason for the node.
  CudfFilterProjectNode(
      const core::PlanNodeId& id,
      core::PlanNodePtr source,
      RowTypePtr outputType,
      std::optional<core::PlanNodeId> filterNodeId,
      std::optional<core::PlanNodeId> projectNodeId,
      core::TypedExprPtr filter,
      std::vector<core::TypedExprPtr> projections,
      bool lazyDereference);

  /// Returns why cuDF cannot run 'filter' followed by 'projections' over an
  /// input of 'inputType', whatever the session, or std::nullopt. 'filter' is
  /// null when there is no filter.
  static std::optional<std::string> unsupportedReason(
      const RowTypePtr& inputType,
      const core::TypedExprPtr& filter,
      const std::vector<core::TypedExprPtr>& projections);

  /// Whether cuDF can evaluate the filter and the projections.
  std::optional<std::string> unsupportedReasonForSession(
      const core::QueryCtx& queryCtx) const override;

  core::PlanNodePtr withSources(
      std::vector<core::PlanNodePtr> sources) const override {
    return copyWithSources<CudfFilterProjectNode>(std::move(sources));
  }

  std::string_view name() const override {
    return "CudfFilterProject";
  }

  bool hasFilter() const {
    return filter_ != nullptr;
  }

  bool hasProject() const {
    return projectNodeId_.has_value();
  }

  const std::optional<core::PlanNodeId>& filterNodeId() const {
    return filterNodeId_;
  }

  const std::optional<core::PlanNodeId>& projectNodeId() const {
    return projectNodeId_;
  }

  const core::TypedExprPtr& filter() const {
    return filter_;
  }

  const std::vector<core::TypedExprPtr>& projections() const {
    return projections_;
  }

  bool isLazyDereference() const {
    return lazyDereference_;
  }

 private:
  void addDetails(std::stringstream& stream) const override;

  const std::optional<core::PlanNodeId> filterNodeId_;
  const std::optional<core::PlanNodeId> projectNodeId_;
  const core::TypedExprPtr filter_;
  const std::vector<core::TypedExprPtr> projections_;
  const bool lazyDereference_;
};

/// Aggregates with cuDF. 'intermediateTypes' has the type of the state of
/// each aggregate, which a partial step produces and a final or intermediate
/// step reads.
class CudfAggregationNode : public CudfUnaryPlanNode {
 public:
  /// Throws if 'intermediateTypes' does not have one type for each aggregate,
  /// or if a type is not the one the aggregate function declares for the raw
  /// input types of the aggregate. The cuDF aggregation operators produce and
  /// read only that state.
  CudfAggregationNode(
      const core::PlanNodeId& id,
      core::PlanNodePtr source,
      RowTypePtr outputType,
      core::AggregationNode::Step step,
      std::vector<core::FieldAccessTypedExprPtr> groupingKeys,
      std::vector<core::AggregationNode::Aggregate> aggregates,
      std::vector<TypePtr> intermediateTypes,
      bool ignoreNullKeys);

  /// Whether cuDF implements the aggregates for this step and can evaluate
  /// their inputs and the grouping keys.
  std::optional<std::string> unsupportedReasonForSession(
      const core::QueryCtx& queryCtx) const override;

  core::PlanNodePtr withSources(
      std::vector<core::PlanNodePtr> sources) const override {
    return copyWithSources<CudfAggregationNode>(std::move(sources));
  }

  std::string_view name() const override {
    return "CudfAggregation";
  }

  core::AggregationNode::Step step() const {
    return step_;
  }

  const std::vector<core::FieldAccessTypedExprPtr>& groupingKeys() const {
    return groupingKeys_;
  }

  const std::vector<core::AggregationNode::Aggregate>& aggregates() const {
    return aggregates_;
  }

  const std::vector<TypePtr>& intermediateTypes() const {
    return intermediateTypes_;
  }

  bool ignoreNullKeys() const {
    return ignoreNullKeys_;
  }

 private:
  void addDetails(std::stringstream& stream) const override;

  const core::AggregationNode::Step step_;
  const std::vector<core::FieldAccessTypedExprPtr> groupingKeys_;
  const std::vector<core::AggregationNode::Aggregate> aggregates_;
  const std::vector<TypePtr> intermediateTypes_;
  const bool ignoreNullKeys_;
};

/// Hash join that runs on cuDF. The first source is the probe side and the
/// second is the build side.
class CudfHashJoinNode : public CudfPlanNode {
 public:
  /// Returns why cuDF does not implement a hash join of 'joinType' with this
  /// null awareness and filter presence, or std::nullopt if it does. cuDF
  /// implements inner, left, right, full, left semi filter, right semi filter,
  /// left semi project and anti joins, except a null-aware anti join with a
  /// filter.
  static std::optional<std::string>
  unsupportedReason(core::JoinType joinType, bool nullAware, bool hasFilter);

  /// Throws if unsupportedReason() has a reason for the join.
  CudfHashJoinNode(
      const core::PlanNodeId& id,
      core::PlanNodePtr probeSource,
      core::PlanNodePtr buildSource,
      RowTypePtr outputType,
      core::JoinType joinType,
      bool nullAware,
      std::vector<core::FieldAccessTypedExprPtr> leftKeys,
      std::vector<core::FieldAccessTypedExprPtr> rightKeys,
      core::TypedExprPtr filter);

  const RowTypePtr& outputType() const override {
    return outputType_;
  }

  const std::vector<core::PlanNodePtr>& sources() const override {
    return sources_;
  }

  bool requiresSingleThread() const override {
    return joinType_ == core::JoinType::kRightSemiProject && nullAware_;
  }

  CudfPlanNodeTraits cudfTraits() const override {
    return {
        {InputRepresentation::kCudf, InputRepresentation::kCudf},
        VectorRepresentation::kCudf,
        {}};
  }

  /// Whether cuDF can evaluate the join filter, and as a cuDF AST if
  /// filterRequiresAst().
  std::optional<std::string> unsupportedReasonForSession(
      const core::QueryCtx& queryCtx) const override;

  /// True for the join types whose filter CudfHashJoinProbe applies only as a
  /// cuDF AST over both sides: full, semi and anti joins. The other join types
  /// can also evaluate the filter over the joined rows.
  bool filterRequiresAst() const;

  core::PlanNodePtr withSources(
      std::vector<core::PlanNodePtr> sources) const override {
    VELOX_CHECK_EQ(sources.size(), 2);
    auto copy = std::make_shared<CudfHashJoinNode>(*this);
    copy->sources_ = std::move(sources);
    return copy;
  }

  std::string_view name() const override {
    return "CudfHashJoin";
  }

  core::JoinType joinType() const {
    return joinType_;
  }

  bool isNullAware() const {
    return nullAware_;
  }

  bool isInnerJoin() const {
    return core::isInnerJoin(joinType_);
  }

  bool isLeftJoin() const {
    return core::isLeftJoin(joinType_);
  }

  bool isRightJoin() const {
    return core::isRightJoin(joinType_);
  }

  bool isFullJoin() const {
    return core::isFullJoin(joinType_);
  }

  bool isLeftSemiFilterJoin() const {
    return core::isLeftSemiFilterJoin(joinType_);
  }

  bool isLeftSemiProjectJoin() const {
    return core::isLeftSemiProjectJoin(joinType_);
  }

  bool isRightSemiFilterJoin() const {
    return core::isRightSemiFilterJoin(joinType_);
  }

  bool isAntiJoin() const {
    return core::isAntiJoin(joinType_);
  }

  const std::vector<core::FieldAccessTypedExprPtr>& leftKeys() const {
    return leftKeys_;
  }

  const std::vector<core::FieldAccessTypedExprPtr>& rightKeys() const {
    return rightKeys_;
  }

  const core::TypedExprPtr& filter() const {
    return filter_;
  }

 private:
  void addDetails(std::stringstream& stream) const override;

  const core::JoinType joinType_;
  const bool nullAware_;
  const std::vector<core::FieldAccessTypedExprPtr> leftKeys_;
  const std::vector<core::FieldAccessTypedExprPtr> rightKeys_;
  const core::TypedExprPtr filter_;
  std::vector<core::PlanNodePtr> sources_;
  const RowTypePtr outputType_;
};

/// Nested loop join that runs on cuDF. The first source is the probe side
/// and the second is the build side.
class CudfNestedLoopJoinNode : public CudfPlanNode {
 public:
  /// Returns why cuDF does not implement a nested loop join of 'joinType'
  /// with 'joinCondition', or std::nullopt if it does. cuDF implements inner,
  /// left, right, full and left semi project joins, with a condition that
  /// compares no type with custom comparison. 'joinCondition' may be null.
  static std::optional<std::string> unsupportedReason(
      core::JoinType joinType,
      const core::TypedExprPtr& joinCondition);

  /// Throws if unsupportedReason() has a reason for the join.
  CudfNestedLoopJoinNode(
      const core::PlanNodeId& id,
      core::PlanNodePtr probeSource,
      core::PlanNodePtr buildSource,
      RowTypePtr outputType,
      core::JoinType joinType,
      core::TypedExprPtr joinCondition);

  const RowTypePtr& outputType() const override {
    return outputType_;
  }

  const std::vector<core::PlanNodePtr>& sources() const override {
    return sources_;
  }

  CudfPlanNodeTraits cudfTraits() const override {
    return {
        {InputRepresentation::kCudf, InputRepresentation::kCudf},
        VectorRepresentation::kCudf,
        {}};
  }

  /// Whether cuDF can evaluate the join condition.
  std::optional<std::string> unsupportedReasonForSession(
      const core::QueryCtx& queryCtx) const override;

  core::PlanNodePtr withSources(
      std::vector<core::PlanNodePtr> sources) const override {
    VELOX_CHECK_EQ(sources.size(), 2);
    auto copy = std::make_shared<CudfNestedLoopJoinNode>(*this);
    copy->sources_ = std::move(sources);
    return copy;
  }

  std::string_view name() const override {
    return "CudfNestedLoopJoin";
  }

  core::JoinType joinType() const {
    return joinType_;
  }

  const core::TypedExprPtr& joinCondition() const {
    return joinCondition_;
  }

 private:
  void addDetails(std::stringstream& stream) const override;

  const core::JoinType joinType_;
  const core::TypedExprPtr joinCondition_;
  std::vector<core::PlanNodePtr> sources_;
  const RowTypePtr outputType_;
};

/// Sorts with cuDF. A final sort runs in one driver and produces the whole
/// input in order. A partial sort runs in every driver of its pipeline, and
/// each driver produces only its own input in order, so the plan needs a
/// CudfLocalMergeNode downstream to produce one ordered stream.
class CudfOrderByNode : public CudfUnaryPlanNode {
 public:
  /// Throws if there is no sorting key, a sorting key is null or repeated, or
  /// the number of sorting orders differs from the number of sorting keys.
  CudfOrderByNode(
      const core::PlanNodeId& id,
      core::PlanNodePtr source,
      RowTypePtr outputType,
      std::vector<core::FieldAccessTypedExprPtr> sortingKeys,
      std::vector<core::SortOrder> sortingOrders,
      bool isPartial);

  core::PlanNodePtr withSources(
      std::vector<core::PlanNodePtr> sources) const override {
    return copyWithSources<CudfOrderByNode>(std::move(sources));
  }

  std::string_view name() const override {
    return "CudfOrderBy";
  }

  const std::vector<core::FieldAccessTypedExprPtr>& sortingKeys() const {
    return sortingKeys_;
  }

  const std::vector<core::SortOrder>& sortingOrders() const {
    return sortingOrders_;
  }

  bool requiresSingleThread() const override {
    return !isPartial_;
  }

  bool isPartial() const {
    return isPartial_;
  }

 private:
  void addDetails(std::stringstream& stream) const override;

  const std::vector<core::FieldAccessTypedExprPtr> sortingKeys_;
  const std::vector<core::SortOrder> sortingOrders_;
  const bool isPartial_;
};

/// Keeps the first 'count' rows in sorting order. A final node runs in one
/// driver. A partial node keeps 'count' rows per driver, so the plan needs a
/// final CudfTopNNode downstream.
class CudfTopNNode : public CudfUnaryPlanNode {
 public:
  /// Throws for the sorting keys and orders CudfOrderByNode rejects, and if
  /// 'count' is not positive.
  CudfTopNNode(
      const core::PlanNodeId& id,
      core::PlanNodePtr source,
      RowTypePtr outputType,
      std::vector<core::FieldAccessTypedExprPtr> sortingKeys,
      std::vector<core::SortOrder> sortingOrders,
      int32_t count,
      bool isPartial);

  core::PlanNodePtr withSources(
      std::vector<core::PlanNodePtr> sources) const override {
    return copyWithSources<CudfTopNNode>(std::move(sources));
  }

  std::string_view name() const override {
    return "CudfTopN";
  }

  const std::vector<core::FieldAccessTypedExprPtr>& sortingKeys() const {
    return sortingKeys_;
  }

  const std::vector<core::SortOrder>& sortingOrders() const {
    return sortingOrders_;
  }

  int32_t count() const {
    return count_;
  }

  bool requiresSingleThread() const override {
    return !isPartial_;
  }

  bool isPartial() const {
    return isPartial_;
  }

 private:
  void addDetails(std::stringstream& stream) const override;

  const std::vector<core::FieldAccessTypedExprPtr> sortingKeys_;
  const std::vector<core::SortOrder> sortingOrders_;
  const int32_t count_;
  const bool isPartial_;
};

/// Keeps the first 'limit' rows of each partition by row_number, rank or
/// dense_rank over the sorting keys.
class CudfTopNRowNumberNode : public CudfUnaryPlanNode {
 public:
  /// Returns why cuDF cannot rank with 'rankFunction' over these keys, or
  /// std::nullopt if it can. row_number takes any keys. rank and dense_rank
  /// compare the keys for equality, so they take no key with custom
  /// comparison and no ARRAY or MAP sorting key.
  static std::optional<std::string> unsupportedReason(
      core::TopNRowNumberNode::RankFunction rankFunction,
      const std::vector<core::FieldAccessTypedExprPtr>& partitionKeys,
      const std::vector<core::FieldAccessTypedExprPtr>& sortingKeys);

  /// Throws if unsupportedReason() has a reason for the ranking, 'limit' is
  /// not positive, there is no sorting key, or the number of sorting orders
  /// differs from the number of sorting keys.
  CudfTopNRowNumberNode(
      const core::PlanNodeId& id,
      core::PlanNodePtr source,
      RowTypePtr outputType,
      core::TopNRowNumberNode::RankFunction rankFunction,
      std::vector<core::FieldAccessTypedExprPtr> partitionKeys,
      std::vector<core::FieldAccessTypedExprPtr> sortingKeys,
      std::vector<core::SortOrder> sortingOrders,
      int32_t limit,
      bool generateRowNumber);

  core::PlanNodePtr withSources(
      std::vector<core::PlanNodePtr> sources) const override {
    return copyWithSources<CudfTopNRowNumberNode>(std::move(sources));
  }

  std::string_view name() const override {
    return "CudfTopNRowNumber";
  }

  core::TopNRowNumberNode::RankFunction rankFunction() const {
    return rankFunction_;
  }

  const std::vector<core::FieldAccessTypedExprPtr>& partitionKeys() const {
    return partitionKeys_;
  }

  const std::vector<core::FieldAccessTypedExprPtr>& sortingKeys() const {
    return sortingKeys_;
  }

  const std::vector<core::SortOrder>& sortingOrders() const {
    return sortingOrders_;
  }

  int32_t limit() const {
    return limit_;
  }

  bool generateRowNumber() const {
    return generateRowNumber_;
  }

 private:
  void addDetails(std::stringstream& stream) const override;

  const core::TopNRowNumberNode::RankFunction rankFunction_;
  const std::vector<core::FieldAccessTypedExprPtr> partitionKeys_;
  const std::vector<core::FieldAccessTypedExprPtr> sortingKeys_;
  const std::vector<core::SortOrder> sortingOrders_;
  const int32_t limit_;
  const bool generateRowNumber_;
};

/// The core base preserves LocalPlanner's source pipelines and Task queues.
/// A boundary-only adapter supplies CudfLocalMerge until core has a factory
/// hook.
class CudfLocalMergeNode : public core::LocalMergeNode {
 public:
  CudfLocalMergeNode(
      const core::PlanNodeId& id,
      std::vector<core::PlanNodePtr> sources,
      RowTypePtr outputType,
      std::vector<core::FieldAccessTypedExprPtr> sortingKeys,
      std::vector<core::SortOrder> sortingOrders)
      : LocalMergeNode(
            id,
            std::move(sortingKeys),
            std::move(sortingOrders),
            std::move(sources)) {
    VELOX_USER_CHECK(!this->sources().empty());
    VELOX_USER_CHECK(*outputType == *this->outputType());
    VELOX_USER_CHECK_EQ(
        this->sortingKeys().size(), this->sortingOrders().size());
  }

  std::string_view name() const override {
    return "CudfLocalMerge";
  }

  /// Throws, like CudfPlanNode::serialize(). The core base class would write
  /// its fields under a name that has no deserializer.
  folly::dynamic serialize() const override {
    VELOX_UNSUPPORTED("CudfLocalMerge plan node is not serializable");
  }

  core::PlanNodePtr withSources(std::vector<core::PlanNodePtr> sources) const {
    VELOX_USER_CHECK(!sources.empty());
    auto type = sources[0]->outputType();
    return std::make_shared<CudfLocalMergeNode>(
        id(), std::move(sources), type, sortingKeys(), sortingOrders());
  }
};

/// Explicit device-vector local exchange. Retaining the core base lets Task
/// provision the existing queues; only producer construction is adapted.
class CudfLocalPartitionNode : public core::LocalPartitionNode {
 public:
  /// Returns why cuDF cannot partition like 'node', or std::nullopt if it can.
  /// cuDF partitions by hash, round robin, round robin by row and gather, and
  /// does not scale writers.
  static std::optional<std::string> unsupportedReason(
      const core::LocalPartitionNode& node);

  /// Throws if unsupportedReason() has a reason for the partition function.
  CudfLocalPartitionNode(
      const core::PlanNodeId& id,
      Type type,
      core::PartitionFunctionSpecPtr partitionFunctionSpec,
      std::vector<core::PlanNodePtr> sources)
      : LocalPartitionNode(
            id,
            type,
            false,
            std::move(partitionFunctionSpec),
            std::move(sources)) {
    checkSupported();
  }

  std::string_view name() const override {
    return "CudfLocalPartition";
  }

  /// Throws, like CudfPlanNode::serialize(). The core base class would write
  /// its fields under a name that has no deserializer.
  folly::dynamic serialize() const override {
    VELOX_UNSUPPORTED("CudfLocalPartition plan node is not serializable");
  }

  /// Returns a cuDF local partition that partitions like 'node' and reads
  /// from 'sources'. Throws if unsupportedReason() has a reason for 'node'.
  static std::shared_ptr<const CudfLocalPartitionNode> create(
      const core::LocalPartitionNode& node,
      std::vector<core::PlanNodePtr> sources);

  core::PlanNodePtr withSources(std::vector<core::PlanNodePtr> sources) const {
    return create(*this, std::move(sources));
  }

 private:
  // Preserve opaque partition specs without retaining a CPU plan node.
  explicit CudfLocalPartitionNode(const core::LocalPartitionNode& node)
      : LocalPartitionNode(node) {
    checkSupported();
  }

  // Throws if unsupportedReason() has a reason for this node.
  void checkSupported() const;
};

/// Skips 'offset' rows and keeps the next 'count'. A final limit runs in one
/// driver. A partial limit applies to each driver by itself.
class CudfLimitNode : public CudfUnaryPlanNode {
 public:
  CudfLimitNode(
      const core::PlanNodeId& id,
      core::PlanNodePtr source,
      RowTypePtr outputType,
      int64_t offset,
      int64_t count,
      bool isPartial)
      : CudfUnaryPlanNode(id, std::move(source), std::move(outputType)),
        offset_(offset),
        count_(count),
        isPartial_(isPartial) {}

  core::PlanNodePtr withSources(
      std::vector<core::PlanNodePtr> sources) const override {
    return copyWithSources<CudfLimitNode>(std::move(sources));
  }

  std::string_view name() const override {
    return "CudfLimit";
  }

  int64_t offset() const {
    return offset_;
  }

  int64_t count() const {
    return count_;
  }

  bool requiresSingleThread() const override {
    return !isPartial_;
  }

  bool isPartial() const {
    return isPartial_;
  }

 private:
  void addDetails(std::stringstream& stream) const override {
    stream << "offset=" << offset_ << ", count=" << count_;
  }

  const int64_t offset_;
  const int64_t count_;
  const bool isPartial_;
};

/// Concatenates small input batches, so that the operator above it gets
/// fewer, larger batches.
class CudfBatchConcatNode : public CudfUnaryPlanNode {
 public:
  CudfBatchConcatNode(const core::PlanNodeId& id, core::PlanNodePtr source)
      : CudfUnaryPlanNode(id, source, source->outputType()) {}

  core::PlanNodePtr withSources(
      std::vector<core::PlanNodePtr> sources) const override {
    return copyWithSources<CudfBatchConcatNode>(std::move(sources));
  }

  std::string_view name() const override {
    return "CudfBatchConcat";
  }

 private:
  void addDetails(std::stringstream& /* stream */) const override {}
};

/// Appends a column of IDs that are unique within the query.
/// 'taskUniqueId' keeps the IDs of different tasks apart.
class CudfAssignUniqueIdNode : public CudfUnaryPlanNode {
 public:
  CudfAssignUniqueIdNode(
      const core::PlanNodeId& id,
      core::PlanNodePtr source,
      RowTypePtr outputType,
      int32_t taskUniqueId)
      : CudfUnaryPlanNode(id, std::move(source), std::move(outputType)),
        taskUniqueId_(taskUniqueId) {}

  core::PlanNodePtr withSources(
      std::vector<core::PlanNodePtr> sources) const override {
    return copyWithSources<CudfAssignUniqueIdNode>(std::move(sources));
  }

  std::string_view name() const override {
    return "CudfAssignUniqueId";
  }

  int32_t taskUniqueId() const {
    return taskUniqueId_;
  }

 private:
  void addDetails(std::stringstream& stream) const override {
    stream << "taskUniqueId=" << taskUniqueId_;
  }

  const int32_t taskUniqueId_;
};

/// Appends one boolean column that is true for the first row of each
/// distinct value of 'distinctKeys'. Has no masks.
class CudfMarkDistinctNode : public CudfUnaryPlanNode {
 public:
  CudfMarkDistinctNode(
      const core::PlanNodeId& id,
      core::PlanNodePtr source,
      RowTypePtr outputType,
      std::vector<core::FieldAccessTypedExprPtr> distinctKeys)
      : CudfUnaryPlanNode(id, std::move(source), std::move(outputType)),
        distinctKeys_(std::move(distinctKeys)) {}

  core::PlanNodePtr withSources(
      std::vector<core::PlanNodePtr> sources) const override {
    return copyWithSources<CudfMarkDistinctNode>(std::move(sources));
  }

  std::string_view name() const override {
    return "CudfMarkDistinct";
  }

  const std::vector<core::FieldAccessTypedExprPtr>& distinctKeys() const {
    return distinctKeys_;
  }

 private:
  void addDetails(std::stringstream& stream) const override;

  const std::vector<core::FieldAccessTypedExprPtr> distinctKeys_;
};

/// Fails if its input has more than one row, and produces one row of nulls
/// if its input is empty.
class CudfEnforceSingleRowNode : public CudfUnaryPlanNode {
 public:
  CudfEnforceSingleRowNode(
      const core::PlanNodeId& id,
      core::PlanNodePtr source,
      RowTypePtr outputType)
      : CudfUnaryPlanNode(id, std::move(source), std::move(outputType)) {}

  core::PlanNodePtr withSources(
      std::vector<core::PlanNodePtr> sources) const override {
    return copyWithSources<CudfEnforceSingleRowNode>(std::move(sources));
  }

  std::string_view name() const override {
    return "CudfEnforceSingleRow";
  }

  bool requiresSingleThread() const override {
    return true;
  }

 private:
  void addDetails(std::stringstream& /* stream */) const override {}
};

/// Produces one copy of the input for each grouping set, with a group ID
/// column that tells the copies apart.
class CudfGroupIdNode : public CudfUnaryPlanNode {
 public:
  CudfGroupIdNode(
      const core::PlanNodeId& id,
      core::PlanNodePtr source,
      RowTypePtr outputType,
      std::vector<std::vector<std::string>> groupingSets,
      std::vector<core::GroupIdNode::GroupingKeyInfo> groupingKeyInfos,
      std::vector<core::FieldAccessTypedExprPtr> aggregationInputs,
      int32_t numGroupingKeys)
      : CudfUnaryPlanNode(id, std::move(source), std::move(outputType)),
        groupingSets_(std::move(groupingSets)),
        groupingKeyInfos_(std::move(groupingKeyInfos)),
        aggregationInputs_(std::move(aggregationInputs)),
        numGroupingKeys_(numGroupingKeys) {}

  core::PlanNodePtr withSources(
      std::vector<core::PlanNodePtr> sources) const override {
    return copyWithSources<CudfGroupIdNode>(std::move(sources));
  }

  std::string_view name() const override {
    return "CudfGroupId";
  }

  const std::vector<std::vector<std::string>>& groupingSets() const {
    return groupingSets_;
  }

  const std::vector<core::GroupIdNode::GroupingKeyInfo>& groupingKeyInfos()
      const {
    return groupingKeyInfos_;
  }

  const std::vector<core::FieldAccessTypedExprPtr>& aggregationInputs() const {
    return aggregationInputs_;
  }

  int32_t numGroupingKeys() const {
    return numGroupingKeys_;
  }

 private:
  void addDetails(std::stringstream& stream) const override;

  const std::vector<std::vector<std::string>> groupingSets_;
  const std::vector<core::GroupIdNode::GroupingKeyInfo> groupingKeyInfos_;
  const std::vector<core::FieldAccessTypedExprPtr> aggregationInputs_;
  const int32_t numGroupingKeys_;
};

/// Evaluates window functions with cuDF. 'inputsSorted' says that the input
/// is already sorted on the partition keys and then the sorting keys.
class CudfWindowNode : public CudfUnaryPlanNode {
 public:
  /// Throws if unsupportedReason() has a reason for the window.
  CudfWindowNode(
      const core::PlanNodeId& id,
      core::PlanNodePtr source,
      RowTypePtr outputType,
      std::vector<core::FieldAccessTypedExprPtr> partitionKeys,
      std::vector<core::FieldAccessTypedExprPtr> sortingKeys,
      std::vector<core::SortOrder> sortingOrders,
      std::vector<core::WindowNode::Function> windowFunctions,
      bool inputsSorted);

  /// Returns why cuDF cannot run a window with these keys and functions over
  /// an input of 'inputType', or std::nullopt if it can. Defined in
  /// CudfWindow.cpp, beside the frame and type helpers that it shares with the
  /// operator.
  static std::optional<std::string> unsupportedReason(
      const RowTypePtr& inputType,
      const std::vector<core::FieldAccessTypedExprPtr>& partitionKeys,
      const std::vector<core::FieldAccessTypedExprPtr>& sortingKeys,
      const std::vector<core::SortOrder>& sortingOrders,
      const std::vector<core::WindowNode::Function>& windowFunctions);

  core::PlanNodePtr withSources(
      std::vector<core::PlanNodePtr> sources) const override {
    return copyWithSources<CudfWindowNode>(std::move(sources));
  }

  std::string_view name() const override {
    return "CudfWindow";
  }

  const RowTypePtr& inputType() const {
    return sources_[0]->outputType();
  }

  const std::vector<core::FieldAccessTypedExprPtr>& partitionKeys() const {
    return partitionKeys_;
  }

  const std::vector<core::FieldAccessTypedExprPtr>& sortingKeys() const {
    return sortingKeys_;
  }

  const std::vector<core::SortOrder>& sortingOrders() const {
    return sortingOrders_;
  }

  const std::vector<core::WindowNode::Function>& windowFunctions() const {
    return windowFunctions_;
  }

  bool inputsSorted() const {
    return inputsSorted_;
  }

 private:
  void addDetails(std::stringstream& stream) const override;

  const std::vector<core::FieldAccessTypedExprPtr> partitionKeys_;
  const std::vector<core::FieldAccessTypedExprPtr> sortingKeys_;
  const std::vector<core::SortOrder> sortingOrders_;
  const std::vector<core::WindowNode::Function> windowFunctions_;
  const bool inputsSorted_;
};

/// Converts Velox vectors to cuDF vectors. Marks where a plan moves from the
/// CPU to the GPU.
class CudfFromVeloxNode : public CudfPlanNode {
 public:
  CudfFromVeloxNode(const core::PlanNodeId& id, core::PlanNodePtr source)
      : CudfPlanNode(id), sources_{std::move(source)} {}

  const std::vector<core::PlanNodePtr>& sources() const override {
    return sources_;
  }

  const RowTypePtr& outputType() const override {
    return sources_[0]->outputType();
  }

  CudfPlanNodeTraits cudfTraits() const override {
    return {{InputRepresentation::kVelox}, VectorRepresentation::kCudf, {}};
  }

  core::PlanNodePtr withSources(
      std::vector<core::PlanNodePtr> sources) const override {
    VELOX_CHECK_EQ(sources.size(), 1);
    return std::make_shared<CudfFromVeloxNode>(id(), std::move(sources[0]));
  }

  std::string_view name() const override {
    return "CudfFromVelox";
  }

 private:
  void addDetails(std::stringstream& /* stream */) const override {}

  std::vector<core::PlanNodePtr> sources_;
};

/// Converts cuDF vectors to Velox vectors. Marks where a plan moves from the
/// GPU to the CPU.
class CudfToVeloxNode : public CudfPlanNode {
 public:
  CudfToVeloxNode(const core::PlanNodeId& id, core::PlanNodePtr source)
      : CudfPlanNode(id), sources_{std::move(source)} {}

  const std::vector<core::PlanNodePtr>& sources() const override {
    return sources_;
  }

  const RowTypePtr& outputType() const override {
    return sources_[0]->outputType();
  }

  CudfPlanNodeTraits cudfTraits() const override {
    return {{InputRepresentation::kCudf}, VectorRepresentation::kVelox, {}};
  }

  core::PlanNodePtr withSources(
      std::vector<core::PlanNodePtr> sources) const override {
    VELOX_CHECK_EQ(sources.size(), 1);
    return std::make_shared<CudfToVeloxNode>(id(), std::move(sources[0]));
  }

  std::string_view name() const override {
    return "CudfToVelox";
  }

 private:
  void addDetails(std::stringstream& /* stream */) const override {}

  std::vector<core::PlanNodePtr> sources_;
};

} // namespace facebook::velox::cudf_velox
