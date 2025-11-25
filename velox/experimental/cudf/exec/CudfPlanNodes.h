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
