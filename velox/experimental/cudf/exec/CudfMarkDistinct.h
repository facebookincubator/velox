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

#include "velox/core/PlanNode.h"

#include <cudf/join/filtered_join.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>

#include <memory>
#include <vector>

namespace facebook::velox::cudf_velox {

/// GPU operator that marks first occurrences of distinct key combinations.
///
/// For each input row, appends a boolean column indicating whether this is the
/// first time the key combination has been seen.
///
/// Unlike the CPU implementation, this operator does not support spilling.
class CudfMarkDistinct : public CudfOperatorBase {
 public:
  CudfMarkDistinct(
      int32_t operatorId,
      exec::DriverCtx* driverCtx,
      const std::shared_ptr<const core::MarkDistinctNode>& planNode);

  bool isFilter() const override {
    return true;
  }

  bool preservesOrder() const override {
    return false;
  }

  bool needsInput() const override {
    return !noMoreInput_ && input_ == nullptr;
  }

  bool isFinished() override {
    return noMoreInput_ && input_ == nullptr;
  }

  exec::BlockingReason isBlocked(ContinueFuture* /*future*/) override {
    return exec::BlockingReason::kNotBlocked;
  }

 protected:
  void doAddInput(RowVectorPtr input) override;
  RowVectorPtr doGetOutput() override;

 private:
  /// Column indices in the input schema that form the distinct key.
  std::vector<cudf::size_type> distinctKeyIndices_;

  /// Accumulated distinct keys seen across all batches processed so far.
  std::unique_ptr<cudf::table> seenKeys_;

  /// Persistent hash filter built from seenKeys_. Probed each batch to find
  /// new distinct keys. Only rebuilt when seenKeys_ grows (new keys found).
  /// Null until the first batch has been processed.
  std::unique_ptr<cudf::filtered_join> seenFilter_;
};

} // namespace facebook::velox::cudf_velox
