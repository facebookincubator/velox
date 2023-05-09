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

#include "velox/exec/Operator.h"

namespace facebook::velox::exec::test {

class TestingRebatchNode : public core::PlanNode {
 public:
  explicit TestingRebatchNode(core::PlanNodeId id, core::PlanNodePtr input)
      : PlanNode(id), sources_({input}) {
    registerNode();
  }

  const RowTypePtr& outputType() const override {
    return sources_[0]->outputType();
  }

  const std::vector<std::shared_ptr<const PlanNode>>& sources() const override {
    return sources_;
  }

  std::string_view name() const override {
    return "Rebatch";
  }

 private:
  static void registerNode();
  void addDetails(std::stringstream& /* stream */) const override {}

  std::vector<core::PlanNodePtr> sources_;
};

class TestingRebatch : public Operator {
 public:
  enum class Encoding {
    kConstant,
    kSlice,
    kLongFlat,
    kShortFlat,
    kDicts,
    kSameDict,
    kSameDoubleDict,
    kLastEncoding // Must be last in enum.
  };
  static constexpr int32_t kNumEncodings =
      static_cast<int32_t>(Encoding::kLastEncoding);

  TestingRebatch(DriverCtx* ctx, int32_t id, const core::PlanNodePtr& node)
      : Operator(ctx, node->outputType(), id, node->id(), "Rebatch") {}

  bool needsInput() const override {
    return !noMoreInput_ && !input_;
  }

  void addInput(RowVectorPtr input) override {
    input_ = std::move(input);
    currentRow_ = 0;
  }

  void noMoreInput() override {
    Operator::noMoreInput();
  }

  RowVectorPtr getOutput() override;

  BlockingReason isBlocked(ContinueFuture* future) override {
    return BlockingReason::kNotBlocked;
  }

  bool isFinished() override {
    return noMoreInput_ && input_ == nullptr;
  }

 private:
  void nextEncoding();

  // Counter deciding the next action in getOutput().
  int32_t counter_;

  // Flat concatenation of multiple batches of input
  RowVectorPtr output_;

  // Next row of input to be sent to output.
  vector_size_t currentRow_{0};

  Encoding encoding_;
  int32_t nthSlice_{0};

  // Drop rows between batches. Used for introducing a predictable error to test
  // drilldown into minimal breaking fuzziness.
  bool injectError_{false};
};

class TestingRebatchFactory : public Operator::PlanNodeTranslator {
 public:
  TestingRebatchFactory() = default;

  std::unique_ptr<Operator> toOperator(
      DriverCtx* ctx,
      int32_t id,
      const core::PlanNodePtr& node) override {
    if (auto rebatch =
            std::dynamic_pointer_cast<const TestingRebatchNode>(node)) {
      return std::make_unique<TestingRebatch>(ctx, id, rebatch);
    }
    return nullptr;
  }

  std::optional<uint32_t> maxDrivers(const core::PlanNodePtr& node) override {
    return std::nullopt;
  }
};

} // namespace facebook::velox::exec::test
