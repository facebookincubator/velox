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

#include <string>

namespace facebook::velox::cudf_velox {

/// Wraps a Velox logical plan node as a cuDF physical plan node. The wrapped
/// node carries the logical arguments consumed by the existing cuDF operator,
/// while sources() exposes only the physical inputs LocalPlanner must visit.
template <typename T>
class CudfPlanNode : public core::PlanNode {
 public:
  CudfPlanNode(
      std::shared_ptr<const T> planNode,
      std::string name,
      int preferredDriverCount = 4)
      : PlanNode(planNode->id()),
        planNode_(std::move(planNode)),
        name_(std::move(name)),
        preferredDriverCount_(preferredDriverCount) {}

  const RowTypePtr& outputType() const override {
    return planNode_->outputType();
  }

  const std::vector<core::PlanNodePtr>& sources() const override {
    return planNode_->sources();
  }

  bool requiresSingleThread() const override {
    return planNode_->requiresSingleThread();
  }

  std::string_view name() const override {
    return name_;
  }

  const std::shared_ptr<const T>& planNode() const {
    return planNode_;
  }

  int preferredDriverCount() const {
    return preferredDriverCount_;
  }

 private:
  void addDetails(std::stringstream& stream) const override {
    stream << "preferredDrivers=" << preferredDriverCount_;
  }

  const std::shared_ptr<const T> planNode_;
  const std::string name_;
  const int preferredDriverCount_;
};

using CudfNestedLoopJoinNode = CudfPlanNode<core::NestedLoopJoinNode>;
using CudfOrderByNode = CudfPlanNode<core::OrderByNode>;
using CudfTopNNode = CudfPlanNode<core::TopNNode>;
using CudfLimitNode = CudfPlanNode<core::LimitNode>;

/// One physical cuDF node for the Filter, Project, or fused Filter+Project
/// execution unit. If both logical nodes are present, sources() deliberately
/// skips them and points at the input below Filter so LocalPlanner creates only
/// one operator.
class CudfFilterProjectNode : public core::PlanNode {
 public:
  CudfFilterProjectNode(
      std::shared_ptr<const core::FilterNode> filterNode,
      std::shared_ptr<const core::ProjectNode> projectNode,
      int preferredDriverCount = 4)
      : PlanNode(projectNode ? projectNode->id() : filterNode->id()),
        filterNode_(std::move(filterNode)),
        projectNode_(std::move(projectNode)),
        sources_(
            filterNode_ ? filterNode_->sources() : projectNode_->sources()),
        preferredDriverCount_(preferredDriverCount) {
    VELOX_CHECK(filterNode_ || projectNode_);
  }

  const RowTypePtr& outputType() const override {
    return projectNode_ ? projectNode_->outputType()
                        : filterNode_->outputType();
  }

  const std::vector<core::PlanNodePtr>& sources() const override {
    return sources_;
  }

  std::string_view name() const override {
    return "CudfFilterProject";
  }

  const std::shared_ptr<const core::FilterNode>& filterNode() const {
    return filterNode_;
  }

  const std::shared_ptr<const core::ProjectNode>& projectNode() const {
    return projectNode_;
  }

  int preferredDriverCount() const {
    return preferredDriverCount_;
  }

 private:
  void addDetails(std::stringstream& stream) const override {
    stream << "filter=" << (filterNode_ ? filterNode_->id() : "none")
           << ", project=" << (projectNode_ ? projectNode_->id() : "none")
           << ", preferredDrivers=" << preferredDriverCount_;
  }

  const std::shared_ptr<const core::FilterNode> filterNode_;
  const std::shared_ptr<const core::ProjectNode> projectNode_;
  const std::vector<core::PlanNodePtr> sources_;
  const int preferredDriverCount_;
};

/// GPU-specific aggregation plan node that signals this aggregation
/// should run on GPU with a specific number of drivers.
/// Contains an AggregationNode and adds GPU-specific metadata.
class CudfAggregationNode : public core::PlanNode {
 public:
  CudfAggregationNode(
      std::shared_ptr<const core::AggregationNode> aggNode,
      int preferredDriverCount = 4)
      : PlanNode(aggNode->id()),
        aggNode_(std::move(aggNode)),
        preferredDriverCount_(preferredDriverCount) {}

  const RowTypePtr& outputType() const override {
    return aggNode_->outputType();
  }

  const std::vector<core::PlanNodePtr>& sources() const override {
    return aggNode_->sources();
  }

  std::string_view name() const override {
    return "CudfAggregation";
  }

  const std::shared_ptr<const core::AggregationNode>& aggregationNode() const {
    return aggNode_;
  }

  int preferredDriverCount() const {
    return preferredDriverCount_;
  }

 private:
  void addDetails(std::stringstream& stream) const override {
    stream << "preferredDrivers=" << preferredDriverCount_;
  }

  const std::shared_ptr<const core::AggregationNode> aggNode_;
  const int preferredDriverCount_;
};

class CudfHashJoinNode : public core::PlanNode {
 public:
  CudfHashJoinNode(
      std::shared_ptr<const core::HashJoinNode> joinNode,
      int preferredBuildDriverCount = 4,
      int preferredProbeDriverCount = 4)
      : PlanNode(joinNode->id()),
        joinNode_(std::move(joinNode)),
        preferredBuildDriverCount_(preferredBuildDriverCount),
        preferredProbeDriverCount_(preferredProbeDriverCount) {}

  const RowTypePtr& outputType() const override {
    return joinNode_->outputType();
  }

  const std::vector<core::PlanNodePtr>& sources() const override {
    return joinNode_->sources();
  }

  std::string_view name() const override {
    return "CudfHashJoin";
  }

  const std::shared_ptr<const core::HashJoinNode>& hashJoinNode() const {
    return joinNode_;
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

  const std::shared_ptr<const core::HashJoinNode> joinNode_;
  const int preferredBuildDriverCount_;
  const int preferredProbeDriverCount_;
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
