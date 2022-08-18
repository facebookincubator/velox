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

namespace facebook::velox::exec {

class MarkDistinct : public Operator {
 public:
  MarkDistinct(
      int32_t operatorId,
      DriverCtx* driverCtx,
      const std::shared_ptr<const core::MarkDistinctNode>& planNode,
      std::vector<uint32_t> keyChannels);

  bool isFilter() const override {
    return true;
  }

  bool preservesOrder() const override {
    return true;
  }

  bool needsInput() const override {
    return true;
  }

  void addInput(RowVectorPtr input) override;

  RowVectorPtr getOutput() override;

  BlockingReason isBlocked(ContinueFuture* /*future*/) override {
    return BlockingReason::kNotBlocked;
  }

  bool isFinished() override;

 private:
  std::unordered_set<uint64_t> hashSet_;

  const std::vector<uint32_t> keyChannels_;

  std::vector<DecodedVector> decodedVectors_;

  SelectivityVector allRows_;

  std::vector<bool> columnExists_;
};
} // namespace facebook::velox::exec