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

/// Marks each row with a boolean indicating whether it maintains sort order
/// relative to its predecessor. The first row is always marked true.
/// Subsequent rows are marked true if they are sorted relative to the
/// previous row based on the configured sorting keys and orders.
class MarkSorted : public Operator {
 public:
  MarkSorted(
      int32_t operatorId,
      DriverCtx* driverCtx,
      const std::shared_ptr<const core::MarkSortedNode>& planNode);

  bool preservesOrder() const override {
    return true;
  }

  bool isFilter() const override {
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

  void noMoreInput() override;

 private:
  /// Compare row at currentIndex in currentData with row at prevIndex in
  /// prevData. Both vectors must share the same schema (uses
  /// sortingKeyChannels_ to access columns).
  bool isSortedRelativeTo(
      const RowVectorPtr& currentData,
      vector_size_t currentIndex,
      const RowVectorPtr& prevData,
      vector_size_t prevIndex);

  /// Copy only sorting key columns of the last row from input_ into lastRow_.
  /// Creates a single-row RowVector with key columns at sequential indices
  /// (0, 1, 2, ...) to avoid holding a reference to the entire input batch.
  void copyLastRowKeyColumns();

  const std::string markerName_;
  std::vector<column_index_t> sortingKeyChannels_;
  std::vector<CompareFlags> compareFlags_;

  /// Key-only RowType for lastRow_ construction. Columns are at sequential
  /// indices (0, 1, 2, ...) mapping to sortingKeyChannels_ in the input.
  RowTypePtr lastRowType_;

  /// Stores only sorting key column values of the last row from the previous
  /// batch, for cross-batch comparison. Has a different schema than input_
  /// (key columns only), so cross-batch comparison uses inline logic instead
  /// of isSortedRelativeTo().
  RowVectorPtr lastRow_;
};
} // namespace facebook::velox::exec
