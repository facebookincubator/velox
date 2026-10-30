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
  CudfFilterProjectNode(
      const core::PlanNodeId& id,
      core::PlanNodePtr source,
      RowTypePtr outputType,
      std::optional<core::PlanNodeId> filterNodeId,
      std::optional<core::PlanNodeId> projectNodeId,
      core::TypedExprPtr filter,
      std::vector<core::TypedExprPtr> projections,
      bool lazyDereference)
      : CudfUnaryPlanNode(id, std::move(source), std::move(outputType)),
        filterNodeId_(std::move(filterNodeId)),
        projectNodeId_(std::move(projectNodeId)),
        filter_(std::move(filter)),
        projections_(std::move(projections)),
        lazyDereference_(lazyDereference) {
    VELOX_CHECK(filter_ || projectNodeId_);
  }

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
  void addDetails(std::stringstream& stream) const override {
    stream << "filter=" << (filterNodeId_ ? filterNodeId_.value() : "none")
           << ", project="
           << (projectNodeId_ ? projectNodeId_.value() : "none");
  }

  const std::optional<core::PlanNodeId> filterNodeId_;
  const std::optional<core::PlanNodeId> projectNodeId_;
  const core::TypedExprPtr filter_;
  const std::vector<core::TypedExprPtr> projections_;
  const bool lazyDereference_;
};

class CudfAggregationNode : public CudfUnaryPlanNode {
 public:
  CudfAggregationNode(
      const core::PlanNodeId& id,
      core::PlanNodePtr source,
      RowTypePtr outputType,
      core::AggregationNode::Step step,
      std::vector<core::FieldAccessTypedExprPtr> groupingKeys,
      std::vector<core::AggregationNode::Aggregate> aggregates,
      std::vector<TypePtr> intermediateTypes,
      bool ignoreNullKeys)
      : CudfUnaryPlanNode(id, std::move(source), std::move(outputType)),
        step_(step),
        groupingKeys_(std::move(groupingKeys)),
        aggregates_(std::move(aggregates)),
        intermediateTypes_(std::move(intermediateTypes)),
        ignoreNullKeys_(ignoreNullKeys) {
    VELOX_USER_CHECK_EQ(intermediateTypes_.size(), aggregates_.size());
    for (const auto& type : intermediateTypes_) {
      VELOX_USER_CHECK_NOT_NULL(type);
    }
  }

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
  void addDetails(std::stringstream& stream) const override {
    stream << core::AggregationNode::toName(step_);
  }

  const core::AggregationNode::Step step_;
  const std::vector<core::FieldAccessTypedExprPtr> groupingKeys_;
  const std::vector<core::AggregationNode::Aggregate> aggregates_;
  const std::vector<TypePtr> intermediateTypes_;
  const bool ignoreNullKeys_;
};

class CudfHashJoinNode : public CudfPlanNode {
 public:
  CudfHashJoinNode(
      const core::PlanNodeId& id,
      core::PlanNodePtr probeSource,
      core::PlanNodePtr buildSource,
      RowTypePtr outputType,
      core::JoinType joinType,
      bool nullAware,
      std::vector<core::FieldAccessTypedExprPtr> leftKeys,
      std::vector<core::FieldAccessTypedExprPtr> rightKeys,
      core::TypedExprPtr filter)
      : CudfPlanNode(id),
        joinType_(joinType),
        nullAware_(nullAware),
        leftKeys_(std::move(leftKeys)),
        rightKeys_(std::move(rightKeys)),
        filter_(std::move(filter)),
        sources_{std::move(probeSource), std::move(buildSource)},
        outputType_(std::move(outputType)) {}

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
    return joinType_ == core::JoinType::kInner;
  }

  bool isLeftJoin() const {
    return joinType_ == core::JoinType::kLeft;
  }

  bool isRightJoin() const {
    return joinType_ == core::JoinType::kRight;
  }

  bool isFullJoin() const {
    return joinType_ == core::JoinType::kFull;
  }

  bool isLeftSemiFilterJoin() const {
    return joinType_ == core::JoinType::kLeftSemiFilter;
  }

  bool isLeftSemiProjectJoin() const {
    return joinType_ == core::JoinType::kLeftSemiProject;
  }

  bool isRightSemiFilterJoin() const {
    return joinType_ == core::JoinType::kRightSemiFilter;
  }

  bool isAntiJoin() const {
    return joinType_ == core::JoinType::kAnti;
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
  void addDetails(std::stringstream& stream) const override {
    stream << "joinType=" << core::JoinTypeName::toName(joinType_);
  }

  const core::JoinType joinType_;
  const bool nullAware_;
  const std::vector<core::FieldAccessTypedExprPtr> leftKeys_;
  const std::vector<core::FieldAccessTypedExprPtr> rightKeys_;
  const core::TypedExprPtr filter_;
  std::vector<core::PlanNodePtr> sources_;
  const RowTypePtr outputType_;
};

class CudfNestedLoopJoinNode : public CudfPlanNode {
 public:
  CudfNestedLoopJoinNode(
      const core::PlanNodeId& id,
      core::PlanNodePtr probeSource,
      core::PlanNodePtr buildSource,
      RowTypePtr outputType,
      core::JoinType joinType,
      core::TypedExprPtr joinCondition)
      : CudfPlanNode(id),
        joinType_(joinType),
        joinCondition_(std::move(joinCondition)),
        sources_{std::move(probeSource), std::move(buildSource)},
        outputType_(std::move(outputType)) {}

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
  void addDetails(std::stringstream& stream) const override {
    stream << "joinType=" << core::JoinTypeName::toName(joinType_);
  }

  const core::JoinType joinType_;
  const core::TypedExprPtr joinCondition_;
  std::vector<core::PlanNodePtr> sources_;
  const RowTypePtr outputType_;
};

class CudfOrderByNode : public CudfUnaryPlanNode {
 public:
  CudfOrderByNode(
      const core::PlanNodeId& id,
      core::PlanNodePtr source,
      RowTypePtr outputType,
      std::vector<core::FieldAccessTypedExprPtr> sortingKeys,
      std::vector<core::SortOrder> sortingOrders,
      bool isPartial)
      : CudfUnaryPlanNode(id, std::move(source), std::move(outputType)),
        sortingKeys_(std::move(sortingKeys)),
        sortingOrders_(std::move(sortingOrders)),
        isPartial_(isPartial) {}

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
  void addDetails(std::stringstream& stream) const override {
    stream << (isPartial_ ? "PARTIAL" : "FINAL");
  }

  const std::vector<core::FieldAccessTypedExprPtr> sortingKeys_;
  const std::vector<core::SortOrder> sortingOrders_;
  const bool isPartial_;
};

class CudfTopNNode : public CudfUnaryPlanNode {
 public:
  CudfTopNNode(
      const core::PlanNodeId& id,
      core::PlanNodePtr source,
      RowTypePtr outputType,
      std::vector<core::FieldAccessTypedExprPtr> sortingKeys,
      std::vector<core::SortOrder> sortingOrders,
      int32_t count,
      bool isPartial)
      : CudfUnaryPlanNode(id, std::move(source), std::move(outputType)),
        sortingKeys_(std::move(sortingKeys)),
        sortingOrders_(std::move(sortingOrders)),
        count_(count),
        isPartial_(isPartial) {}

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
  void addDetails(std::stringstream& stream) const override {
    stream << "count=" << count_;
  }

  const std::vector<core::FieldAccessTypedExprPtr> sortingKeys_;
  const std::vector<core::SortOrder> sortingOrders_;
  const int32_t count_;
  const bool isPartial_;
};

/// Physical row_number top-N. Rank and dense_rank are deliberately not part
/// of this node's contract.
class CudfTopNRowNumberNode : public CudfUnaryPlanNode {
 public:
  CudfTopNRowNumberNode(
      const core::PlanNodeId& id,
      core::PlanNodePtr source,
      RowTypePtr outputType,
      std::vector<core::FieldAccessTypedExprPtr> partitionKeys,
      std::vector<core::FieldAccessTypedExprPtr> sortingKeys,
      std::vector<core::SortOrder> sortingOrders,
      int32_t limit,
      bool generateRowNumber)
      : CudfUnaryPlanNode(id, std::move(source), std::move(outputType)),
        partitionKeys_(std::move(partitionKeys)),
        sortingKeys_(std::move(sortingKeys)),
        sortingOrders_(std::move(sortingOrders)),
        limit_(limit),
        generateRowNumber_(generateRowNumber) {
    VELOX_CHECK_GT(limit_, 0);
    VELOX_CHECK(!sortingKeys_.empty());
    VELOX_CHECK_EQ(sortingKeys_.size(), sortingOrders_.size());
  }

  core::PlanNodePtr withSources(
      std::vector<core::PlanNodePtr> sources) const override {
    return copyWithSources<CudfTopNRowNumberNode>(std::move(sources));
  }

  std::string_view name() const override {
    return "CudfTopNRowNumber";
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
  void addDetails(std::stringstream& stream) const override {
    stream << "limit=" << limit_ << ", rowNumber=" << generateRowNumber_;
  }

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

  core::PlanNodePtr withSources(std::vector<core::PlanNodePtr> sources) const {
    VELOX_USER_CHECK(!sources.empty());
    auto type = sources[0]->outputType();
    return std::make_shared<CudfLocalMergeNode>(
        id(), std::move(sources), type, sortingKeys(), sortingOrders());
  }

 private:
  void addDetails(std::stringstream& stream) const override {
    stream << "sources=" << sources().size();
  }
};

/// Explicit device-vector local exchange. Retaining the core base lets Task
/// provision the existing queues; only producer construction is adapted.
class CudfLocalPartitionNode : public core::LocalPartitionNode {
 public:
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
            std::move(sources)) {}

  std::string_view name() const override {
    return "CudfLocalPartition";
  }

  core::PlanNodePtr withSources(std::vector<core::PlanNodePtr> sources) const {
    return detail::rebuildCudfLocalPartition(*this, std::move(sources));
  }

 private:
  friend core::PlanNodePtr detail::rebuildCudfLocalPartition(
      const core::LocalPartitionNode&,
      std::vector<core::PlanNodePtr>);

  // Preserve opaque partition specs without retaining a CPU plan node.
  explicit CudfLocalPartitionNode(const core::LocalPartitionNode& node)
      : LocalPartitionNode(node) {}
};

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
  void addDetails(std::stringstream& stream) const override {
    stream << "distinctKeys=" << distinctKeys_.size();
  }

  const std::vector<core::FieldAccessTypedExprPtr> distinctKeys_;
};

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
  void addDetails(std::stringstream& stream) const override {
    stream << "groupingSets=" << groupingSets_.size();
  }

  const std::vector<std::vector<std::string>> groupingSets_;
  const std::vector<core::GroupIdNode::GroupingKeyInfo> groupingKeyInfos_;
  const std::vector<core::FieldAccessTypedExprPtr> aggregationInputs_;
  const int32_t numGroupingKeys_;
};

class CudfWindowNode : public CudfUnaryPlanNode {
 public:
  CudfWindowNode(
      const core::PlanNodeId& id,
      core::PlanNodePtr source,
      RowTypePtr outputType,
      std::vector<core::FieldAccessTypedExprPtr> partitionKeys,
      std::vector<core::FieldAccessTypedExprPtr> sortingKeys,
      std::vector<core::SortOrder> sortingOrders,
      std::vector<core::WindowNode::Function> windowFunctions,
      bool inputsSorted)
      : CudfUnaryPlanNode(id, std::move(source), std::move(outputType)),
        partitionKeys_(std::move(partitionKeys)),
        sortingKeys_(std::move(sortingKeys)),
        sortingOrders_(std::move(sortingOrders)),
        windowFunctions_(std::move(windowFunctions)),
        inputsSorted_(inputsSorted) {}

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
  void addDetails(std::stringstream& stream) const override {
    stream << "functions=" << windowFunctions_.size();
  }

  const std::vector<core::FieldAccessTypedExprPtr> partitionKeys_;
  const std::vector<core::FieldAccessTypedExprPtr> sortingKeys_;
  const std::vector<core::SortOrder> sortingOrders_;
  const std::vector<core::WindowNode::Function> windowFunctions_;
  const bool inputsSorted_;
};

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
