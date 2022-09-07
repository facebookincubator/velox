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
#include "velox/exec/Operator.h"

namespace facebook::velox::exec {

class Shuffle {
 public:
  virtual void collect(int32_t partition, std::string_view data) = 0;
  virtual void noMoreData() = 0;
  virtual bool hasNext(int32_t partition) const = 0;
  virtual BufferPtr next(int32_t partition) = 0;
};

class ShuffleWriteNode : public core::PlanNode {
 public:
  ShuffleWriteNode(
      const core::PlanNodeId& id,
      Shuffle* shuffle,
      core::PlanNodePtr source)
      : core::PlanNode(id), shuffle_{shuffle}, sources_{std::move(source)} {}

  const RowTypePtr& outputType() const override {
    return sources_[0]->outputType();
  }

  const std::vector<core::PlanNodePtr>& sources() const override {
    return sources_;
  }

  Shuffle* shuffle() const {
    return shuffle_;
  }

  std::string_view name() const override {
    return "ShuffleWrite";
  }

 private:
  void addDetails(std::stringstream& stream) const override {}

  Shuffle* shuffle_;

  const std::vector<core::PlanNodePtr> sources_;
};

class ShuffleWriteTranslator : public Operator::PlanNodeTranslator {
 public:
  std::unique_ptr<Operator> toOperator(
      DriverCtx* ctx,
      int32_t id,
      const core::PlanNodePtr& node) override;
};
} // namespace facebook::velox::exec