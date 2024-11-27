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

#include "velox/core/Expressions.h"
#include "velox/core/PlanNode.h"
#include "velox/exec/Driver.h"
#include "velox/exec/Operator.h"
#include "velox/experimental/cudf/vector/CudfVector.h"
#include "velox/vector/ComplexVector.h"

#include <cudf/table/table.hpp>

namespace facebook::velox::cudf_velox {

class CudfOrderBy : public exec::Operator {
 public:
  CudfOrderBy(
      int32_t operatorId,
      exec::DriverCtx* driverCtx,
      const std::shared_ptr<const core::OrderByNode>& orderByNode);

  bool needsInput() const override {
    return !finished_;
  }

  void addInput(RowVectorPtr input) override;

  void noMoreInput() override;

  RowVectorPtr getOutput() override;

  exec::BlockingReason isBlocked(ContinueFuture* /*future*/) override {
    return exec::BlockingReason::kNotBlocked;
  }

  bool isFinished() override {
    return finished_;
  }

  void close() override;

 private:
  CudfVectorPtr outputTable_;
  std::shared_ptr<const core::OrderByNode> orderByNode_;
  std::vector<CudfVectorPtr> inputs_;
  std::vector<cudf::size_type> sort_keys_;
  std::vector<cudf::order> column_order_;
  std::vector<cudf::null_order> null_order_;
  bool finished_{false};
  uint32_t maxOutputRows_;
};

} // namespace facebook::velox::cudf_velox
