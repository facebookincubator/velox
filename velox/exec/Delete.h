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

class Delete : public Operator {
 public:
  Delete(
      int32_t operatorId,
      DriverCtx* driverCtx,
      const std::shared_ptr<const core::DeleteNode>& deleteNode);

  BlockingReason isBlocked(ContinueFuture* /* future */) override {
    return BlockingReason::kNotBlocked;
  }

  virtual bool needsInput() const override {
    return !finished_ && !closed_;
  }

  bool isFinished() override {
    return finished_;
  }

  void addInput(RowVectorPtr input) override;

  RowVectorPtr getOutput() override;

 private:
  const column_index_t rowIdChannel_;
  vector_size_t numDeletedRows_;
  UpdatableDataSourceSupplier updatableDataSourceSupplier_;

  bool finished_;
  bool closed_;
};
} // namespace facebook::velox::exec
