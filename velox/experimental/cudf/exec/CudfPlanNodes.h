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

#include "velox/core/PlanNode.h"

namespace facebook::velox::cudf_velox {

class CudfUnaryPlanNode : public core::PlanNode {
 public:
  const RowTypePtr& outputType() const override {
    return outputType_;
  }

  const std::vector<core::PlanNodePtr>& sources() const override {
    return sources_;
  }

  int preferredDriverCount() const {
    return preferredDriverCount_;
  }

 protected:
  CudfUnaryPlanNode(
      const core::PlanNodeId& id,
      core::PlanNodePtr source,
      RowTypePtr outputType,
      int preferredDriverCount)
      : PlanNode(id),
        sources_{std::move(source)},
        outputType_(std::move(outputType)),
        preferredDriverCount_(preferredDriverCount) {}

  const std::vector<core::PlanNodePtr> sources_;
  const RowTypePtr outputType_;
  const int preferredDriverCount_;
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
      bool lazyDereference,
      int preferredDriverCount = 4)
      : CudfUnaryPlanNode(
            id,
            std::move(source),
            std::move(outputType),
            preferredDriverCount),
        filterNodeId_(std::move(filterNodeId)),
        projectNodeId_(std::move(projectNodeId)),
        filter_(std::move(filter)),
        projections_(std::move(projections)),
        lazyDereference_(lazyDereference) {
    VELOX_CHECK(filter_ || projectNodeId_);
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
           << ", project=" << (projectNodeId_ ? projectNodeId_.value() : "none")
           << ", preferredDrivers=" << preferredDriverCount_;
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
      bool ignoreNullKeys,
      int preferredDriverCount = 4)
      : CudfUnaryPlanNode(
            id,
            std::move(source),
            std::move(outputType),
            preferredDriverCount),
        step_(step),
        groupingKeys_(std::move(groupingKeys)),
        aggregates_(std::move(aggregates)),
        ignoreNullKeys_(ignoreNullKeys) {}

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

  bool ignoreNullKeys() const {
    return ignoreNullKeys_;
  }

 private:
  void addDetails(std::stringstream& stream) const override {
    stream << "preferredDrivers=" << preferredDriverCount_;
  }

  const core::AggregationNode::Step step_;
  const std::vector<core::FieldAccessTypedExprPtr> groupingKeys_;
  const std::vector<core::AggregationNode::Aggregate> aggregates_;
  const bool ignoreNullKeys_;
};

class CudfHashJoinNode : public core::PlanNode {
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
      core::TypedExprPtr filter,
      int preferredBuildDriverCount = 4,
      int preferredProbeDriverCount = 4)
      : PlanNode(id),
        joinType_(joinType),
        nullAware_(nullAware),
        leftKeys_(std::move(leftKeys)),
        rightKeys_(std::move(rightKeys)),
        filter_(std::move(filter)),
        sources_{std::move(probeSource), std::move(buildSource)},
        outputType_(std::move(outputType)),
        preferredBuildDriverCount_(preferredBuildDriverCount),
        preferredProbeDriverCount_(preferredProbeDriverCount) {}

  const RowTypePtr& outputType() const override {
    return outputType_;
  }

  const std::vector<core::PlanNodePtr>& sources() const override {
    return sources_;
  }

  bool requiresSingleThread() const override {
    return joinType_ == core::JoinType::kRightSemiProject && nullAware_;
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

  int preferredBuildDriverCount() const {
    return preferredBuildDriverCount_;
  }

  int preferredProbeDriverCount() const {
    return preferredProbeDriverCount_;
  }

 private:
  void addDetails(std::stringstream& stream) const override {
    stream << "preferredBuildDrivers=" << preferredBuildDriverCount_
           << ", preferredProbeDrivers=" << preferredProbeDriverCount_;
  }

  const core::JoinType joinType_;
  const bool nullAware_;
  const std::vector<core::FieldAccessTypedExprPtr> leftKeys_;
  const std::vector<core::FieldAccessTypedExprPtr> rightKeys_;
  const core::TypedExprPtr filter_;
  const std::vector<core::PlanNodePtr> sources_;
  const RowTypePtr outputType_;
  const int preferredBuildDriverCount_;
  const int preferredProbeDriverCount_;
};

class CudfNestedLoopJoinNode : public core::PlanNode {
 public:
  CudfNestedLoopJoinNode(
      const core::PlanNodeId& id,
      core::PlanNodePtr probeSource,
      core::PlanNodePtr buildSource,
      RowTypePtr outputType,
      core::JoinType joinType,
      core::TypedExprPtr joinCondition,
      int preferredDriverCount = 4)
      : PlanNode(id),
        joinType_(joinType),
        joinCondition_(std::move(joinCondition)),
        sources_{std::move(probeSource), std::move(buildSource)},
        outputType_(std::move(outputType)),
        preferredDriverCount_(preferredDriverCount) {}

  const RowTypePtr& outputType() const override {
    return outputType_;
  }

  const std::vector<core::PlanNodePtr>& sources() const override {
    return sources_;
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

  int preferredDriverCount() const {
    return preferredDriverCount_;
  }

 private:
  void addDetails(std::stringstream& stream) const override {
    stream << "preferredDrivers=" << preferredDriverCount_;
  }

  const core::JoinType joinType_;
  const core::TypedExprPtr joinCondition_;
  const std::vector<core::PlanNodePtr> sources_;
  const RowTypePtr outputType_;
  const int preferredDriverCount_;
};

class CudfOrderByNode : public CudfUnaryPlanNode {
 public:
  CudfOrderByNode(
      const core::PlanNodeId& id,
      core::PlanNodePtr source,
      RowTypePtr outputType,
      std::vector<core::FieldAccessTypedExprPtr> sortingKeys,
      std::vector<core::SortOrder> sortingOrders,
      bool isPartial,
      int preferredDriverCount = 4)
      : CudfUnaryPlanNode(
            id,
            std::move(source),
            std::move(outputType),
            preferredDriverCount),
        sortingKeys_(std::move(sortingKeys)),
        sortingOrders_(std::move(sortingOrders)),
        isPartial_(isPartial) {}

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
    stream << "preferredDrivers=" << preferredDriverCount_;
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
      bool isPartial,
      int preferredDriverCount = 4)
      : CudfUnaryPlanNode(
            id,
            std::move(source),
            std::move(outputType),
            preferredDriverCount),
        sortingKeys_(std::move(sortingKeys)),
        sortingOrders_(std::move(sortingOrders)),
        count_(count),
        isPartial_(isPartial) {}

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
    stream << "count=" << count_
           << ", preferredDrivers=" << preferredDriverCount_;
  }

  const std::vector<core::FieldAccessTypedExprPtr> sortingKeys_;
  const std::vector<core::SortOrder> sortingOrders_;
  const int32_t count_;
  const bool isPartial_;
};

class CudfLimitNode : public CudfUnaryPlanNode {
 public:
  CudfLimitNode(
      const core::PlanNodeId& id,
      core::PlanNodePtr source,
      RowTypePtr outputType,
      int64_t offset,
      int64_t count,
      bool isPartial,
      int preferredDriverCount = 4)
      : CudfUnaryPlanNode(
            id,
            std::move(source),
            std::move(outputType),
            preferredDriverCount),
        offset_(offset),
        count_(count),
        isPartial_(isPartial) {}

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
    stream << "offset=" << offset_ << ", count=" << count_
           << ", preferredDrivers=" << preferredDriverCount_;
  }

  const int64_t offset_;
  const int64_t count_;
  const bool isPartial_;
};

class CudfBatchConcatNode : public CudfUnaryPlanNode {
 public:
  CudfBatchConcatNode(
      const core::PlanNodeId& id,
      core::PlanNodePtr source,
      int preferredDriverCount = 4)
      : CudfUnaryPlanNode(
            id,
            source,
            source->outputType(),
            preferredDriverCount) {}

  std::string_view name() const override {
    return "CudfBatchConcat";
  }

 private:
  void addDetails(std::stringstream& stream) const override {
    stream << "preferredDrivers=" << preferredDriverCount_;
  }
};

class CudfAssignUniqueIdNode : public CudfUnaryPlanNode {
 public:
  CudfAssignUniqueIdNode(
      const core::PlanNodeId& id,
      core::PlanNodePtr source,
      RowTypePtr outputType,
      int32_t taskUniqueId,
      int preferredDriverCount = 4)
      : CudfUnaryPlanNode(
            id,
            std::move(source),
            std::move(outputType),
            preferredDriverCount),
        taskUniqueId_(taskUniqueId) {}

  std::string_view name() const override {
    return "CudfAssignUniqueId";
  }

  int32_t taskUniqueId() const {
    return taskUniqueId_;
  }

 private:
  void addDetails(std::stringstream& stream) const override {
    stream << "preferredDrivers=" << preferredDriverCount_;
  }

  const int32_t taskUniqueId_;
};

class CudfMarkDistinctNode : public CudfUnaryPlanNode {
 public:
  CudfMarkDistinctNode(
      const core::PlanNodeId& id,
      core::PlanNodePtr source,
      RowTypePtr outputType,
      std::vector<core::FieldAccessTypedExprPtr> distinctKeys,
      int preferredDriverCount = 4)
      : CudfUnaryPlanNode(
            id,
            std::move(source),
            std::move(outputType),
            preferredDriverCount),
        distinctKeys_(std::move(distinctKeys)) {}

  std::string_view name() const override {
    return "CudfMarkDistinct";
  }

  const std::vector<core::FieldAccessTypedExprPtr>& distinctKeys() const {
    return distinctKeys_;
  }

 private:
  void addDetails(std::stringstream& stream) const override {
    stream << "preferredDrivers=" << preferredDriverCount_;
  }

  const std::vector<core::FieldAccessTypedExprPtr> distinctKeys_;
};

class CudfEnforceSingleRowNode : public CudfUnaryPlanNode {
 public:
  CudfEnforceSingleRowNode(
      const core::PlanNodeId& id,
      core::PlanNodePtr source,
      RowTypePtr outputType)
      : CudfUnaryPlanNode(id, std::move(source), std::move(outputType), 1) {}

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
      int32_t numGroupingKeys,
      int preferredDriverCount = 4)
      : CudfUnaryPlanNode(
            id,
            std::move(source),
            std::move(outputType),
            preferredDriverCount),
        groupingSets_(std::move(groupingSets)),
        groupingKeyInfos_(std::move(groupingKeyInfos)),
        aggregationInputs_(std::move(aggregationInputs)),
        numGroupingKeys_(numGroupingKeys) {}

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
    stream << "groupingSets=" << groupingSets_.size()
           << ", preferredDrivers=" << preferredDriverCount_;
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
      bool inputsSorted,
      int preferredDriverCount = 4)
      : CudfUnaryPlanNode(
            id,
            std::move(source),
            std::move(outputType),
            preferredDriverCount),
        partitionKeys_(std::move(partitionKeys)),
        sortingKeys_(std::move(sortingKeys)),
        sortingOrders_(std::move(sortingOrders)),
        windowFunctions_(std::move(windowFunctions)),
        inputsSorted_(inputsSorted) {}

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
    stream << "functions=" << windowFunctions_.size()
           << ", preferredDrivers=" << preferredDriverCount_;
  }

  const std::vector<core::FieldAccessTypedExprPtr> partitionKeys_;
  const std::vector<core::FieldAccessTypedExprPtr> sortingKeys_;
  const std::vector<core::SortOrder> sortingOrders_;
  const std::vector<core::WindowNode::Function> windowFunctions_;
  const bool inputsSorted_;
};

class CudfFromVeloxNode : public core::PlanNode {
 public:
  CudfFromVeloxNode(const core::PlanNodeId& id, core::PlanNodePtr source)
      : PlanNode(id), sources_{std::move(source)} {}

  const std::vector<core::PlanNodePtr>& sources() const override {
    return sources_;
  }

  const RowTypePtr& outputType() const override {
    return sources_[0]->outputType();
  }

  std::string_view name() const override {
    return "CudfFromVelox";
  }

 private:
  void addDetails(std::stringstream& /* stream */) const override {}

  std::vector<core::PlanNodePtr> sources_;
};

class CudfToVeloxNode : public core::PlanNode {
 public:
  CudfToVeloxNode(const core::PlanNodeId& id, core::PlanNodePtr source)
      : PlanNode(id), sources_{std::move(source)} {}

  const std::vector<core::PlanNodePtr>& sources() const override {
    return sources_;
  }

  const RowTypePtr& outputType() const override {
    return sources_[0]->outputType();
  }

  std::string_view name() const override {
    return "CudfToVelox";
  }

 private:
  void addDetails(std::stringstream& /* stream */) const override {}

  std::vector<core::PlanNodePtr> sources_;
};

} // namespace facebook::velox::cudf_velox
