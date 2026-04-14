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

#include "velox/experimental/cudf/exec/NvtxHelper.h"

#include "velox/core/Expressions.h"
#include "velox/exec/Operator.h"

#include <cudf/scalar/scalar.hpp>

namespace facebook::velox::cudf_velox {

class CudfExpand : public exec::Operator, public NvtxHelper {
 public:
  CudfExpand(
      int32_t operatorId,
      exec::DriverCtx* driverCtx,
      const std::shared_ptr<const core::ExpandNode>& expandNode);

  bool needsInput() const override;

  void addInput(RowVectorPtr input) override;

  RowVectorPtr getOutput() override;

  exec::BlockingReason isBlocked(ContinueFuture* /*future*/) override {
    return exec::BlockingReason::kNotBlocked;
  }

  bool isFinished() override {
    return noMoreInput_ && input_ == nullptr;
  }

 private:
  void initialize() override;

  std::vector<std::vector<column_index_t>> fieldProjections_;

  std::vector<std::vector<std::shared_ptr<const core::ConstantTypedExpr>>>
      constantProjections_;

  // Store cuDF scalars for constant values
  std::vector<std::vector<std::unique_ptr<cudf::scalar>>> constantOutputs_;

  // Used to indicate the index of fieldProjections_.
  int32_t rowIndex_{0};
};

} // namespace facebook::velox::cudf_velox
