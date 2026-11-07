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

class CudfPlanNode : public core::PlanNode {
 public:
  using PlanNode::PlanNode;
};

class CudfOrderByNode final : public CudfPlanNode {
 public:
  CudfOrderByNode(
      const core::PlanNodeId& id,
      std::vector<core::FieldAccessTypedExprPtr> sortingKeys,
      std::vector<core::SortOrder> sortingOrders,
      bool isPartial,
      core::PlanNodePtr source)
      : CudfPlanNode(id),
        sortingKeys_(std::move(sortingKeys)),
        sortingOrders_(std::move(sortingOrders)),
        isPartial_(isPartial),
        sources_{std::move(source)} {
    VELOX_USER_CHECK(
        !sortingKeys_.empty(), "CudfOrderBy must specify sorting keys");
    VELOX_USER_CHECK_EQ(
        sortingKeys_.size(),
        sortingOrders_.size(),
        "Number of sorting keys and sorting orders in CudfOrderBy must be the same");
    std::unordered_set<std::string> uniqueKeys;
    for (const auto& sortKey : sortingKeys_) {
      VELOX_USER_CHECK_NOT_NULL(sortKey, "Sorting key cannot be null");
      VELOX_USER_CHECK(
          uniqueKeys.insert(sortKey->name()).second,
          "Duplicate sorting keys are not allowed: {}",
          sortKey->name());
    }
  }

  const RowTypePtr& outputType() const override {
    return sources_[0]->outputType();
  }

  const std::vector<core::PlanNodePtr>& sources() const override {
    return sources_;
  }

  bool requiresSingleThread() const override {
    return !isPartial_;
  }

  std::string_view name() const override {
    return "CudfOrderBy";
  }

  folly::dynamic serialize() const override {
    VELOX_UNSUPPORTED("CudfOrderByNode is not serializable");
  }

  const std::vector<core::FieldAccessTypedExprPtr>& sortingKeys() const {
    return sortingKeys_;
  }

  const std::vector<core::SortOrder>& sortingOrders() const {
    return sortingOrders_;
  }

  bool isPartial() const {
    return isPartial_;
  }

 private:
  void addDetails(std::stringstream& /* stream */) const override {}

  const std::vector<core::FieldAccessTypedExprPtr> sortingKeys_;
  const std::vector<core::SortOrder> sortingOrders_;
  const bool isPartial_;
  const std::vector<core::PlanNodePtr> sources_;
};

class CudfFromVeloxNode final : public CudfPlanNode {
 public:
  CudfFromVeloxNode(const core::PlanNodeId& id, core::PlanNodePtr source)
      : CudfPlanNode(id), sources_{std::move(source)} {}

  const RowTypePtr& outputType() const override {
    return sources_[0]->outputType();
  }

  const std::vector<core::PlanNodePtr>& sources() const override {
    return sources_;
  }

  std::string_view name() const override {
    return "CudfFromVelox";
  }

  folly::dynamic serialize() const override {
    VELOX_UNSUPPORTED("CudfFromVeloxNode is not serializable");
  }

 private:
  void addDetails(std::stringstream& /* stream */) const override {}

  const std::vector<core::PlanNodePtr> sources_;
};

class CudfToVeloxNode final : public CudfPlanNode {
 public:
  CudfToVeloxNode(const core::PlanNodeId& id, core::PlanNodePtr source)
      : CudfPlanNode(id), sources_{std::move(source)} {}

  const RowTypePtr& outputType() const override {
    return sources_[0]->outputType();
  }

  const std::vector<core::PlanNodePtr>& sources() const override {
    return sources_;
  }

  std::string_view name() const override {
    return "CudfToVelox";
  }

  folly::dynamic serialize() const override {
    VELOX_UNSUPPORTED("CudfToVeloxNode is not serializable");
  }

 private:
  void addDetails(std::stringstream& /* stream */) const override {}

  const std::vector<core::PlanNodePtr> sources_;
};

} // namespace facebook::velox::cudf_velox
