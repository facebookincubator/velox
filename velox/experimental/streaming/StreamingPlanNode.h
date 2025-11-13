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

namespace facebook::velox::streaming {

class StreamingPlanNode : public core::PlanNode {
 public:
  StreamingPlanNode(
    const core::PlanNodePtr& node,
    const std::vector<core::PlanNodePtr>& targets)
    : PlanNode(node->id()), node_(std::move(node)), targets_(targets) {}

  const RowTypePtr& outputType() const override {
      return node_->outputType();
  }

  const std::vector<core::PlanNodePtr>& sources() const override;

  std::string_view name() const override {
    return node_->name();
  }

  bool requiresSplits() const override {
    return node_->requiresSplits();
  }

  static void registerSerDe();

  folly::dynamic serialize() const override;

  static core::PlanNodePtr create(const folly::dynamic& obj, void* context);

  const core::PlanNodePtr node() const {
    return node_;
  }

  const std::vector<core::PlanNodePtr> targets() const {
    return targets_;
  }

 private:
  void addDetails(std::stringstream& stream) const override;

  const core::PlanNodePtr node_;
  const std::vector<core::PlanNodePtr> targets_;
};

} // namespace facebook::velox::streaming
