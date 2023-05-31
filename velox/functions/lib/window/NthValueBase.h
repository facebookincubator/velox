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

#include "velox/common/base/Exceptions.h"
#include "velox/exec/WindowFunction.h"
#include "velox/vector/FlatVector.h"

namespace facebook::velox::functions::window {

class NthValueBase : public exec::WindowFunction {
 public:
  NthValueBase(
      const std::vector<exec::WindowFunctionArg>& args,
      const TypePtr& resultType,
      velox::memory::MemoryPool* pool)
      : WindowFunction(resultType, pool, nullptr) {
    VELOX_CHECK_EQ(args.size(), 2);
    VELOX_CHECK_NULL(args[0].constantValue);
    valueIndex_ = args[0].index.value();
  }

  void resetPartition(const exec::WindowPartition* partition) override {
    partition_ = partition;
    partitionOffset_ = 0;
  }

  void apply(
      const BufferPtr& /*peerGroupStarts*/,
      const BufferPtr& /*peerGroupEnds*/,
      const BufferPtr& frameStarts,
      const BufferPtr& frameEnds,
      const SelectivityVector& validRows,
      int32_t resultOffset,
      const VectorPtr& result) override {
    auto numRows = frameStarts->size() / sizeof(vector_size_t);
    auto rawFrameStarts = frameStarts->as<vector_size_t>();
    auto rawFrameEnds = frameEnds->as<vector_size_t>();

    rowNumbers_.resize(numRows);
    if (constantOffset_.has_value() || isConstantOffsetNull_) {
      setRowNumbersForConstantOffset(validRows, rawFrameStarts, rawFrameEnds);
    } else {
      setRowNumbers(
        numRows,
        validRows,
        rawFrameStarts,
        rawFrameEnds,
        partitionOffset_,
        partition_,
        offsetIndex_,
        offsets_,
        rowNumbers_);
    }

    auto rowNumbersRange = folly::Range(rowNumbers_.data(), numRows);
    partition_->extractColumn(
        valueIndex_, rowNumbersRange, resultOffset, result);

    partitionOffset_ += numRows;
  }

 protected:
  void initializeConstantOffset(
      const std::optional<vector_size_t>& constantOffset) {
    if (constantOffset.has_value()) {
      VELOX_USER_CHECK_GE(
          constantOffset.value(), 1, "Offset must be at least 1");
      constantOffset_ = constantOffset;
    } else {
      isConstantOffsetNull_ = true;
    }
  }

  void initializeOffsetVector(
      column_index_t offsetIndex,
      const VectorPtr& offsets) {
    offsetIndex_ = offsetIndex;
    offsets_ = offsets;
  }

  // The below 2 functions build the rowNumbers for column extraction.
  // The rowNumbers map for each output row, as per nth_value function
  // semantics, the rowNumber (relative to the start of the partition) from
  // which the input value should be copied.
  // A rowNumber of kNullRow is for nullptr in the result.
  void setRowNumbersForConstantOffset(
      const SelectivityVector& validRows,
      const vector_size_t* frameStarts,
      const vector_size_t* frameEnds) {
    if (isConstantOffsetNull_) {
      std::fill(rowNumbers_.begin(), rowNumbers_.end(), kNullRow);
      return;
    }

    const auto constantOffsetValue = constantOffset_.value();
    validRows.applyToSelected([&](auto i) {
      setRowNumber(i, frameStarts, frameEnds, constantOffsetValue);
    });

    setRowNumbersForEmptyFrames(validRows);
  }

  virtual void setRowNumbers(
      vector_size_t numRows,
      const SelectivityVector& validRows,
      const vector_size_t* frameStarts,
      const vector_size_t* frameEnds,
      const vector_size_t partitionOffset,
      const exec::WindowPartition* partition,
      const column_index_t offsetIndex,
      const VectorPtr& offsets,
      std::vector<vector_size_t>& rowNumbers)= 0;

  void setRowNumbersForEmptyFrames(const SelectivityVector& validRows) {
    if (validRows.isAllSelected()) {
      return;
    }
    // Rows with empty (not-valid) frames have nullptr in the result.
    // So mark rowNumber to copy as kNullRow for it.
    invalidRows_.resizeFill(validRows.size(), true);
    invalidRows_.deselect(validRows);
    invalidRows_.applyToSelected([&](auto i) { rowNumbers_[i] = kNullRow; });
  }

  inline void setRowNumber(
      vector_size_t i,
      const vector_size_t* frameStarts,
      const vector_size_t* frameEnds,
      vector_size_t offset) {
    auto frameStart = frameStarts[i];
    auto frameEnd = frameEnds[i];
    auto rowNumber = frameStart + offset - 1;
    rowNumbers_[i] = rowNumber <= frameEnd ? rowNumber : kNullRow;
  }

 private:
  // These are the argument indices of the nth_value value and offset columns
  // in the input row vector. These are needed to retrieve column values
  // from the partition data.
  column_index_t valueIndex_;
  column_index_t offsetIndex_;

  const exec::WindowPartition* partition_;

  // These fields are set if the offset argument is a constant value.
  std::optional<vector_size_t> constantOffset_;
  bool isConstantOffsetNull_ = false;

  // This vector is used to extract values of the offset argument column
  // (if not a constant offset value).
  VectorPtr offsets_;

  // This offset tracks how far along the partition rows have been output.
  // This can be used to optimize reading offset column values corresponding
  // to the present row set in getOutput.
  vector_size_t partitionOffset_;

  // The NthValue function directly writes from the input column to the
  // resultVector using the extractColumn API specifying the rowNumber mapping
  // to copy between the 2 vectors. This variable is used for the rowNumber
  // vector across getOutput calls.
  std::vector<vector_size_t> rowNumbers_;

  // Member variable re-used for setting null for empty frames.
  SelectivityVector invalidRows_;
};
} // namespace facebook::velox::functions::window
