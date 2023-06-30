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

#include "velox/exec/HashTable.h"
#include "velox/exec/Operator.h"

namespace facebook::velox::exec {

class RowNumber : public Operator {
 public:
  RowNumber(
      int32_t operatorId,
      DriverCtx* driverCtx,
      const std::shared_ptr<const core::RowNumberNode>& rowNumberNode);

  void addInput(RowVectorPtr input) override;

  RowVectorPtr getOutput() override;

  bool needsInput() const override {
    return !noMoreInput_ && !finishedEarly_;
  }

  BlockingReason isBlocked(ContinueFuture* /* unused */) override {
    return BlockingReason::kNotBlocked;
  }

  bool isFinished() override {
    return (noMoreInput_ && input_ == nullptr) || finishedEarly_;
  }

 private:
  int64_t numRows(char* partition);

  void setNumRows(char* partition, int64_t numRows);

  RowVectorPtr getOutputForSinglePartition();

  FlatVector<int64_t>& getOrCreateRowNumberVector(vector_size_t size);

  const std::optional<int32_t> limit_;

  /// Hash table to store number of rows seen so far per partition. Not used if
  /// there are no partitioning keys.
  std::unique_ptr<BaseHashTable> table_;
  std::unique_ptr<HashLookup> lookup_;
  int32_t numRowsOffset_;

  /// Total number of input rows. Used when there are no partitioning keys and
  /// therefore no hash table.
  int64_t numTotalInput_{0};

  /// Boolean indicating that the operator finished processing before seeing all
  /// the input. This happens when there are no partitioning keys and the
  /// operator already received 'limit_' rows.
  bool finishedEarly_{false};
};
} // namespace facebook::velox::exec
