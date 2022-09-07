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

class PartitionAndSerializeNode : public core::PlanNode {
 public:
  PartitionAndSerializeNode(
      const core::PlanNodeId& id,
      std::vector<core::TypedExprPtr> keys,
      int numPartitions,
      RowTypePtr outputType,
      core::PlanNodePtr source)
      : core::PlanNode(id),
        keys_{std::move(keys)},
        numPartitions_{numPartitions},
        outputType_{std::move(outputType)},
        sources_{std::move(source)} {
    VELOX_CHECK(ROW({"partition", "data"}, {INTEGER(), VARBINARY()})
                    ->equivalent(*outputType_));
  }

  const RowTypePtr& outputType() const override {
    return outputType_;
  }

  const std::vector<core::PlanNodePtr>& sources() const override {
    return sources_;
  }

  const std::vector<core::TypedExprPtr>& keys() const {
    return keys_;
  }

  int numPartitions() const {
    return numPartitions_;
  }

  std::string_view name() const override {
    return "PartitionAndSerialize";
  }

 private:
  void addDetails(std::stringstream& stream) const override;

  const std::vector<core::TypedExprPtr> keys_;
  const int numPartitions_;
  const RowTypePtr outputType_;
  const std::vector<core::PlanNodePtr> sources_;
};

class PartitionAndSerializeTranslator : public Operator::PlanNodeTranslator {
 public:
  std::unique_ptr<Operator> toOperator(
      DriverCtx* ctx,
      int32_t id,
      const core::PlanNodePtr& node) override;
};
} // namespace facebook::velox::exec