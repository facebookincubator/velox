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

#include "velox/common/base/Exceptions.h"
#include "velox/exec/WindowFunction.h"
#include "velox/expression/FunctionSignature.h"
#include "velox/vector/FlatVector.h"

namespace facebook::velox::window::prestosql {

namespace {

class NthValueFunction : public exec::WindowFunction {
 public:
  explicit NthValueFunction(
      const std::vector<exec::WindowFunctionArg>& args,
      const TypePtr& resultType,
      velox::memory::MemoryPool* pool)
      : WindowFunction(resultType, pool, nullptr) {
    VELOX_CHECK_EQ(args.size(), 2);
    VELOX_CHECK_NULL(args[0].constantValue);
    valueIndex_ = args[0].index.value();
    if (args[1].constantValue) {
      if (args[1].constantValue->isNullAt(0)) {
        isConstantOffsetNull_ = true;
        return;
      }
      constantOffset_ =
          args[1]
              .constantValue->template as<ConstantVector<int64_t>>()
              ->valueAt(0);
      VELOX_USER_CHECK_GE(
          constantOffset_.value(), 1, "Offset must be at least 1");
      return;
    }
    offsetIndex_ = args[1].index.value();
    offsets_ = BaseVector::create<FlatVector<int64_t>>(BIGINT(), 0, pool);
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
      setRowNumbers(numRows, validRows, rawFrameStarts, rawFrameEnds);
    }

    auto rowNumbersRange = folly::Range(rowNumbers_.data(), numRows);
    partition_->extractColumn(
        valueIndex_, rowNumbersRange, resultOffset, result);

    partitionOffset_ += numRows;
  }

 private:
  // The below 2 functions build the rowNumbers for column extraction.
  // The rowNumbers map for each output row, as per nth_value function
  // semantics, the rowNumber (relative to the start of the partition) from
  // which the input value should be copied.
  // A rowNumber of -1 is for nullptr in the result.
  void setRowNumbersForConstantOffset(
      const SelectivityVector& validRows,
      const vector_size_t* frameStarts,
      const vector_size_t* frameEnds) {
    if (isConstantOffsetNull_) {
      std::fill(rowNumbers_.begin(), rowNumbers_.end(), -1);
      return;
    }

    auto constantOffsetValue = constantOffset_.value();
    validRows.applyToSelected([&](auto i) {
      setRowNumber(i, frameStarts, frameEnds, constantOffsetValue);
    });

    setRowNumbersForEmptyFrames(validRows);
  }

  void setRowNumbers(
      vector_size_t numRows,
      const SelectivityVector& validRows,
      const vector_size_t* frameStarts,
      const vector_size_t* frameEnds) {
    offsets_->resize(numRows);
    partition_->extractColumn(
        offsetIndex_, partitionOffset_, numRows, 0, offsets_);

    validRows.applyToSelected([&](auto i) {
      if (offsets_->isNullAt(i)) {
        rowNumbers_[i] = -1;
      } else {
        vector_size_t offset = offsets_->valueAt(i);
        VELOX_USER_CHECK_GE(offset, 1, "Offset must be at least 1");
        setRowNumber(i, frameStarts, frameEnds, offset);
      }
    });

    setRowNumbersForEmptyFrames(validRows);
  }

  void setRowNumbersForEmptyFrames(const SelectivityVector& validRows) {
    if (validRows.isAllSelected()) {
      return;
    }
    // Rows with empty (not-valid) frames have nullptr in the result.
    // So mark rowNumber to copy as -1 for it.
    invalidRows_.resizeFill(validRows.size(), true);
    invalidRows_.deselect(validRows);
    invalidRows_.applyToSelected([&](auto i) { rowNumbers_[i] = -1; });
  }

  inline void setRowNumber(
      column_index_t i,
      const vector_size_t* frameStarts,
      const vector_size_t* frameEnds,
      vector_size_t offset) {
    auto frameStart = frameStarts[i];
    auto frameEnd = frameEnds[i];
    auto rowNumber = frameStart + offset - 1;
    rowNumbers_[i] = rowNumber <= frameEnd ? rowNumber : -1;
  }

  // These are the argument indices of the nth_value value and offset columns
  // in the input row vector. These are needed to retrieve column values
  // from the partition data.
  column_index_t valueIndex_;
  column_index_t offsetIndex_;

  const exec::WindowPartition* partition_;

  // These fields are set if the offset argument is a constant value.
  std::optional<int64_t> constantOffset_;
  bool isConstantOffsetNull_ = false;

  // This vector is used to extract values of the offset argument column
  // (if not a constant offset value).
  FlatVectorPtr<int64_t> offsets_;

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
} // namespace

void registerNthValue(const std::string& name) {
  std::vector<exec::FunctionSignaturePtr> signatures{
      // (T, bigint) -> T.
      exec::FunctionSignatureBuilder()
          .typeVariable("T")
          .returnType("T")
          .argumentType("T")
          .argumentType("bigint")
          .build(),
  };

  exec::registerWindowFunction(
      name,
      std::move(signatures),
      [name](
          const std::vector<exec::WindowFunctionArg>& args,
          const TypePtr& resultType,
          velox::memory::MemoryPool* pool,
          HashStringAllocator* /*stringAllocator*/)
          -> std::unique_ptr<exec::WindowFunction> {
        return std::make_unique<NthValueFunction>(args, resultType, pool);
      });
}
} // namespace facebook::velox::window::prestosql
