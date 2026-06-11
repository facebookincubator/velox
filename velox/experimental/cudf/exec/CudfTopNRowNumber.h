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

#include "velox/experimental/cudf/exec/CudfOperator.h"
#include "velox/experimental/cudf/vector/CudfVector.h"

namespace facebook::velox::cudf_velox {

/// GPU-accelerated TopNRowNumber: partitioned top-N with row_number.
/// Used when the optimizer rewrites ROW_NUMBER() OVER (...) WHERE rn <= limit
/// into a TopNRowNumber plan node.
class CudfTopNRowNumber : public CudfOperatorBase {
 public:
  CudfTopNRowNumber(
      int32_t operatorId,
      exec::DriverCtx* driverCtx,
      const std::shared_ptr<const core::TopNRowNumberNode>& node);

  bool needsInput() const override {
    return !noMoreInput_;
  }

  exec::BlockingReason isBlocked(ContinueFuture* /*future*/) override {
    return exec::BlockingReason::kNotBlocked;
  }

  bool isFinished() override;

 protected:
  void doAddInput(RowVectorPtr input) override;
  RowVectorPtr doGetOutput() override;
  void doNoMoreInput() override;

 private:
  const std::shared_ptr<const core::TopNRowNumberNode> node_;
  const int32_t limit_;
  const bool generateRowNumber_;

  std::vector<cudf::size_type> partitionKeyIndices_;
  std::vector<cudf::size_type> sortKeyIndices_;
  std::vector<cudf::order> sortOrders_;
  std::vector<cudf::null_order> nullOrders_;

  std::vector<CudfVectorPtr> inputBatches_;
  bool finished_{false};
};

} // namespace facebook::velox::cudf_velox
