/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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

namespace facebook::velox::exec {

/// Streaming implementation of EnforceDistinct for pre-grouped input.
/// Compares each row with the previous row to detect duplicates.
/// Memory usage is O(1) - only stores the previous row's key values.
///
/// Use this operator when input is clustered on distinct keys, i.e., rows with
/// the same key values are guaranteed to be adjacent.
class StreamingEnforceDistinct : public Operator {
 public:
  StreamingEnforceDistinct(
      int32_t operatorId,
      DriverCtx* driverCtx,
      const std::shared_ptr<const core::EnforceDistinctNode>& planNode);

  bool preservesOrder() const override {
    return true;
  }

  bool needsInput() const override {
    return !noMoreInput_ && !input_;
  }

  void addInput(RowVectorPtr input) override;

  RowVectorPtr getOutput() override;

  BlockingReason isBlocked(ContinueFuture* /*future*/) override {
    return BlockingReason::kNotBlocked;
  }

  bool isFinished() override;

 private:
  const RowTypePtr inputType_;
  const std::vector<column_index_t> keyChannels_;
  const std::string errorMessage_;

  // Key values from the last row of the previous batch for cross-batch
  // comparison. Lazily initialized on first input batch.
  RowVectorPtr prevKeyValues_;
};

} // namespace facebook::velox::exec
