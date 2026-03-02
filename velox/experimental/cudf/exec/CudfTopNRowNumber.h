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
#include "velox/experimental/cudf/vector/CudfVector.h"

#include "velox/exec/Operator.h"

namespace facebook::velox::cudf_velox {

/// GPU-accelerated TopNRowNumber: partitioned top-N with row_number.
/// Used when the optimizer rewrites ROW_NUMBER() OVER (...) WHERE rn <= limit
/// into a TopNRowNumber plan node.
/// Buffers all input, sorts by partition + order keys, then for each
/// partition group retains only the top `limit` rows.
class CudfTopNRowNumber : public exec::Operator, public NvtxHelper {
 public:
  CudfTopNRowNumber(
      int32_t operatorId,
      exec::DriverCtx* driverCtx,
      const std::shared_ptr<const core::TopNRowNumberNode>& node);

  bool needsInput() const override {
    return !noMoreInput_;
  }

  void addInput(RowVectorPtr input) override;

  RowVectorPtr getOutput() override;

  void noMoreInput() override;

  exec::BlockingReason isBlocked(ContinueFuture* /*future*/) override {
    return exec::BlockingReason::kNotBlocked;
  }

  bool isFinished() override;

 private:
  std::shared_ptr<const core::TopNRowNumberNode> node_;
  int32_t limit_;
  bool generateRowNumber_;

  std::vector<cudf::size_type> partitionKeyIndices_;
  std::vector<cudf::size_type> sortKeyIndices_;
  std::vector<cudf::order> sortOrders_;
  std::vector<cudf::null_order> nullOrders_;

  std::vector<CudfVectorPtr> inputBatches_;
  bool finished_ = false;
};

} // namespace facebook::velox::cudf_velox
