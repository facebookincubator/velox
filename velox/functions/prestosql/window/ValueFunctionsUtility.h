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

enum class ValueType {
  kFirstLast,
  kNth,
  kLead,
  kLag,
};

template <ValueType TValue>
class ValueFunction : public exec::WindowFunction {
 public:
  explicit ValueFunction(
      const std::vector<exec::WindowFunctionArg>& args,
      const TypePtr& resultType,
      velox::memory::MemoryPool* pool)
      : WindowFunction(resultType, pool, nullptr) {
    auto numArgs = args.size();
    if constexpr (TValue == ValueType::kNth) {
      VELOX_CHECK_EQ(numArgs, 2);
    } else if constexpr (TValue == ValueType::kFirstLast) {
      VELOX_CHECK_EQ(numArgs, 1);
    } else {
      VELOX_CHECK_GE(numArgs, 1);
      VELOX_CHECK_LE(numArgs, 3);
    }
    VELOX_CHECK_NULL(args[0].constantValue);
    valueIndex_ = args[0].index.value();

    if constexpr (TValue != ValueType::kFirstLast) {
      if constexpr (TValue == ValueType::kNth) {
        minimumOffset_ = 1;
      } else {
        minimumOffset_ = 0;
      }

      if (numArgs == 2) {
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
              constantOffset_.value(),
              minimumOffset_,
              "Offset must be at least {}",
              minimumOffset_);
          return;
        }
        offsetIndex_ = args[1].index.value();
        offsets_ = BaseVector::create<FlatVector<int64_t>>(BIGINT(), 0, pool);
      } else {
        // If no offset is provided for the Lead|Lag functions, the default
        // offset is set to 1.
        constantOffset_ = 1;
        if (numArgs == 3) {
          VELOX_UNSUPPORTED(
              "Non-NULL default values are currently unsupported");
        }
      }
    }
  }

 protected:
  // The below 2 functions build the rowNumbers for column extraction.
  // The rowNumbers map for each output row, as per nth_value|lead|lag function
  // semantics, the rowNumber (relative to the start of the partition) from
  // which the input value should be copied.
  // A rowNumber of -1 is for nullptr in the result.
  void setRowNumbersForConstantOffset(
      vector_size_t numRows,
      const SelectivityVector& validRows,
      std::optional<const vector_size_t*> frameStarts = std::nullopt,
      std::optional<const vector_size_t*> frameEnds = std::nullopt) {
    if (isConstantOffsetNull_) {
      std::fill(rowNumbers_.begin(), rowNumbers_.end(), -1);
      return;
    }

    auto constantOffsetValue = constantOffset_.value();
    validRows.applyToSelected([&](auto i) {
      if constexpr (TValue == ValueType::kNth) {
        setNthValueRowNumber(
            i, frameStarts.value(), frameEnds.value(), constantOffsetValue);
      } else if constexpr (TValue == ValueType::kLead) {
        setLeadLagRowNumber(i, numPartitionRows_ - 1, constantOffsetValue);
      } else {
        setLeadLagRowNumber(i, 0, constantOffsetValue);
      }
    });

    setRowNumbersForEmptyFrames(validRows);
  }

  void setRowNumbers(
      vector_size_t numRows,
      const SelectivityVector& validRows,
      std::optional<const vector_size_t*> frameStarts = std::nullopt,
      std::optional<const vector_size_t*> frameEnds = std::nullopt) {
    offsets_->resize(numRows);
    partition_->extractColumn(
        offsetIndex_, partitionOffset_, numRows, 0, offsets_);

    validRows.applyToSelected([&](auto i) {
      if (offsets_->isNullAt(i)) {
        rowNumbers_[i] = -1;
      } else {
        vector_size_t offset = offsets_->valueAt(i);
        VELOX_USER_CHECK_GE(
            offset,
            minimumOffset_,
            "Offset must be at least {}",
            minimumOffset_);
        if constexpr (TValue == ValueType::kNth) {
          setNthValueRowNumber(
              i, frameStarts.value(), frameEnds.value(), offset);
        } else if constexpr (TValue == ValueType::kLead) {
          setLeadLagRowNumber(i, numPartitionRows_ - 1, offset);
        } else {
          setLeadLagRowNumber(i, 0, offset);
        }
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

  inline void setNthValueRowNumber(
      column_index_t i,
      const vector_size_t* frameStarts,
      const vector_size_t* frameEnds,
      vector_size_t offset) {
    auto frameStart = frameStarts[i];
    auto frameEnd = frameEnds[i];
    auto rowNumber = frameStart + offset - 1;
    rowNumbers_[i] = rowNumber <= frameEnd ? rowNumber : -1;
  }

  inline void setLeadLagRowNumber(
      const column_index_t i,
      const vector_size_t partitionLimit,
      const vector_size_t offset) {
    auto currentRow = partitionOffset_ + i;
    if constexpr (TValue == ValueType::kLead) {
      auto rowNumber = currentRow + offset;
      rowNumbers_[i] = rowNumber <= partitionLimit ? rowNumber : -1;
    } else {
      auto rowNumber = currentRow - offset;
      rowNumbers_[i] = rowNumber < 0 ? -1 : rowNumber;
    }
  }

  // These are the argument indices of the function's value and offset columns
  // in the input row vector. These are needed to retrieve column values
  // from the partition data.
  column_index_t valueIndex_;
  column_index_t offsetIndex_;

  const exec::WindowPartition* partition_;

  // These fields are set if the offset argument is a constant value.
  std::optional<int64_t> constantOffset_;
  bool isConstantOffsetNull_ = false;

  // Minimum value of the offset, 1 for NthValue function and 0 for Lead|Lag
  // functions.
  int64_t minimumOffset_;

  // This vector is used to extract values of the offset argument column
  // (if not a constant offset value).
  FlatVectorPtr<int64_t> offsets_;

  // This offset tracks how far along the partition rows have been output.
  // This can be used to optimize reading offset column values corresponding
  // to the present row set in getOutput.
  vector_size_t partitionOffset_;

  // Number of rows in the current partition.
  vector_size_t numPartitionRows_;

  // The NthValue|Lead|Lag functions directly write from the input column to the
  // resultVector using the extractColumn API specifying the rowNumber mapping
  // to copy between the 2 vectors. This variable is used for the rowNumber
  // vector across getOutput calls.
  std::vector<vector_size_t> rowNumbers_;

  // Member variable re-used for setting null for empty frames.
  SelectivityVector invalidRows_;
};

} // namespace facebook::velox::window::prestosql
