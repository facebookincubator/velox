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

namespace facebook::velox::exec::test {

/// Pass-through sink that emulates a slow downstream consumer by applying
/// back-pressure: after accepting a chunk it refuses further input
/// (needsInput() == false) for 'delayCycles' driver iterations while returning
/// null from getOutput(). During that window the driver, unable to push the
/// upstream operator's output down, descends to the TableScan and advances its
/// reader.
///
/// This is used to deterministically reproduce the Driver-level bug where the
/// scan reader advances past a passthrough LazyVector that an intermediate
/// operator still holds unloaded, surfacing as "Loading LazyVector after the
/// enclosing reader has moved". The bug is operator-neutral: any operator that
/// emits one input batch over multiple chunked getOutput() calls while
/// needsInput() == false (e.g. IndexLookupJoin or a cross-product
/// NestedLoopJoin), sitting downstream of a project-forwarded unloaded scan
/// LazyVector, can strand that lazy. Without such back-pressure the test cursor
/// consumes every chunk instantly and the pipeline runs lock-step, so the
/// reader never advances mid-drain and the bug is masked.
class BackpressureNode : public core::PlanNode {
 public:
  BackpressureNode(
      const core::PlanNodeId& id,
      int32_t delayCycles,
      core::PlanNodePtr source)
      : PlanNode(id), delayCycles_(delayCycles), sources_{std::move(source)} {}

  const RowTypePtr& outputType() const override {
    return sources_[0]->outputType();
  }

  const std::vector<core::PlanNodePtr>& sources() const override {
    return sources_;
  }

  std::string_view name() const override {
    return "Backpressure";
  }

  int32_t delayCycles() const {
    return delayCycles_;
  }

 private:
  void addDetails(std::stringstream& /*stream*/) const override {}

  const int32_t delayCycles_;
  const std::vector<core::PlanNodePtr> sources_;
};

class BackpressureOperator : public Operator {
 public:
  BackpressureOperator(
      int32_t operatorId,
      DriverCtx* driverCtx,
      const std::shared_ptr<const BackpressureNode>& node)
      : Operator(
            driverCtx,
            node->outputType(),
            operatorId,
            node->id(),
            "Backpressure"),
        delayCycles_(node->delayCycles()) {}

  bool needsInput() const override {
    return !noMoreInput_ && input_ == nullptr;
  }

  void addInput(RowVectorPtr input) override {
    input_ = std::move(input);
    cyclesLeft_ = delayCycles_;
  }

  RowVectorPtr getOutput() override {
    if (input_ == nullptr) {
      return nullptr;
    }
    if (cyclesLeft_ > 0) {
      --cyclesLeft_;
      return nullptr;
    }
    return std::move(input_);
  }

  BlockingReason isBlocked(ContinueFuture* /*future*/) override {
    return BlockingReason::kNotBlocked;
  }

  bool isFinished() override {
    return noMoreInput_ && input_ == nullptr;
  }

 private:
  const int32_t delayCycles_;
  int32_t cyclesLeft_{0};
};

class BackpressureTranslator : public Operator::PlanNodeTranslator {
 public:
  std::unique_ptr<Operator> toOperator(
      DriverCtx* ctx,
      int32_t id,
      const core::PlanNodePtr& node) override {
    if (auto castedNode =
            std::dynamic_pointer_cast<const BackpressureNode>(node)) {
      return std::make_unique<BackpressureOperator>(id, ctx, castedNode);
    }
    return nullptr;
  }
};

} // namespace facebook::velox::exec::test
