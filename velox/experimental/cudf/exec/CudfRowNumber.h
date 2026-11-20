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

#include <cudf/join/distinct_hash_join.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>

#include <memory>
#include <optional>
#include <vector>

namespace facebook::velox::cudf_velox {

class CudaEvent;

/// GPU operator that streams row numbers per partition, like
/// exec::RowNumber. Output keeps the input row order; rows beyond the
/// optional per-partition limit are dropped.
///
/// With partition keys, it keeps one INT64 count per partition seen so far
/// in a cuDF table, probed through a distinct_hash_join that is rebuilt only
/// when a batch adds new partitions (see CudfMarkDistinct).
///
/// Unlike the CPU implementation, this operator does not support spilling.
class CudfRowNumber : public CudfOperatorBase {
 public:
  CudfRowNumber(
      int32_t operatorId,
      exec::DriverCtx* driverCtx,
      const std::shared_ptr<const core::RowNumberNode>& planNode);

  bool isFilter() const override {
    return true;
  }

  bool preservesOrder() const override {
    return true;
  }

  bool needsInput() const override {
    return !noMoreInput_ && !finishedEarly_ && input_ == nullptr;
  }

  bool isFinished() override {
    return (noMoreInput_ && input_ == nullptr) || finishedEarly_;
  }

  exec::BlockingReason isBlocked(ContinueFuture* /*future*/) override {
    return exec::BlockingReason::kNotBlocked;
  }

 protected:
  void doAddInput(RowVectorPtr input) override;
  RowVectorPtr doGetOutput() override;
  void doClose() override;

 private:
  // Returns the INT64 row numbers of 'keys' rows in arrival order and
  // updates the per-partition state.
  std::unique_ptr<cudf::column> computePartitionedRowNumbers(
      cudf::table_view keys,
      cuda::stream_ref stream);

  const std::optional<int32_t> limit_;
  const bool generateRowNumber_;

  /// Deduplicated partition key channels in first-occurrence order.
  std::vector<cudf::size_type> partitionKeyIndices_;

  /// Number of rows seen so far when there are no partition keys.
  int64_t numTotalInput_{0};

  /// Set once the limit is reached without partition keys.
  bool finishedEarly_{false};

  /// Distinct partition keys seen so far, and their row counts so far.
  /// Row i of stateCounts_ belongs to row i of stateKeys_.
  std::unique_ptr<cudf::table> stateKeys_;
  std::unique_ptr<cudf::column> stateCounts_;

  /// Hash join over stateKeys_. Rebuilt only when stateKeys_ grows.
  std::unique_ptr<cudf::distinct_hash_join> stateJoin_;

  /// Streams on which stateKeys_ (and stateJoin_) and stateCounts_ were
  /// built. Their buffers are freed in stream order on these streams.
  std::optional<cuda::stream_ref> keysStream_;
  std::optional<cuda::stream_ref> countsStream_;

  /// Reusable event for ordering state lifetime across input streams.
  std::unique_ptr<CudaEvent> cudaEvent_;
};

} // namespace facebook::velox::cudf_velox
