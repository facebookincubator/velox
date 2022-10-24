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

enum class LeadLagType {
  kLead,
  kLag,
};

template <LeadLagType TValue>
class LeadLagFunction : public exec::WindowFunction {
 public:
  explicit LeadLagFunction(
      const std::vector<exec::WindowFunctionArg>& args,
      const TypePtr& resultType,
      velox::memory::MemoryPool* pool)
      : WindowFunction(resultType, pool, nullptr) {
    auto numArgs = args.size();
    VELOX_CHECK_GE(numArgs, 1);
    VELOX_CHECK_LE(numArgs, 3);
    VELOX_CHECK_NULL(args[0].constantValue);
    columnIndex_ = args[0].index.value();

    switch (numArgs) {
      case 1: {
        constantOffset_ = 1;
        break;
      }
      case 2: {
        if (args[1].constantValue) {
          if (args[1].constantValue->isNullAt(0)) {
            isConstantOffsetNull_ = true;
          }
          constantOffset_ =
              args[1]
                  .constantValue->template as<ConstantVector<int64_t>>()
                  ->valueAt(0);
          VELOX_USER_CHECK_GE(
              constantOffset_.value(), 0, "Offset must be at least 0");
        } else {
          offsetIndex_ = args[1].index.value();
          offsets_ = std::dynamic_pointer_cast<FlatVector<int64_t>>(
              BaseVector::create(BIGINT(), 0, pool));
        }
        break;
      }
      case 3: {
        VELOX_UNSUPPORTED("Non-NULL default values are currently unsupported");
        break;
      }
    }
  }

  void resetPartition(const exec::WindowPartition* partition) override {
    partition_ = partition;
    numPartitionRows_ = partition->numRows();
    partitionOffset_ = 0;
  }

  void apply(
      const BufferPtr& /*peerGroupStarts*/,
      const BufferPtr& /*peerGroupEnds*/,
      const BufferPtr& frameStarts,
      const BufferPtr& /*frameEnds*/,
      const SelectivityVector& validRows,
      int32_t resultOffset,
      const VectorPtr& result) override {
    auto numRows = frameStarts->size() / sizeof(vector_size_t);
    auto partitionLimit = 0;
    if constexpr (TValue == LeadLagType::kLead) {
      partitionLimit = numPartitionRows_ - 1;
    }
    rowNumbers_.resize(numRows);

    if (constantOffset_.has_value() || isConstantOffsetNull_) {
      setRowNumbersForConstantOffset(validRows, numRows, partitionLimit);
    } else {
      setRowNumbers(validRows, numRows, partitionLimit);
    }

    auto rowNumbersRange = folly::Range(rowNumbers_.data(), numRows);
    partition_->extractColumn(
        columnIndex_, rowNumbersRange, resultOffset, result);
    partitionOffset_ += numRows;
  }

 private:
  // The below 2 functions build the rowNumbers for column extraction.
  // For each output row, rowNumbers_ returns the row relative to the start of
  // the partition from which the input value should be copied.
  // A rowNumber of -1 is for nullptr in the result.
  void setRowNumbersForConstantOffset(
      const SelectivityVector& validRows,
      vector_size_t numRows,
      const vector_size_t partitionLimit) {
    if (isConstantOffsetNull_) {
      std::fill(rowNumbers_.begin(), rowNumbers_.end(), -1);
      return;
    }

    auto constantOffsetValue = constantOffset_.value();
    for (int i = 0; i < numRows; i++) {
      setRowNumber(i, partitionLimit, constantOffsetValue);
    }
    setRowNumbersForEmptyFrames(validRows);
  }

  void setRowNumbers(
      const SelectivityVector& validRows,
      vector_size_t numRows,
      const vector_size_t partitionLimit) {
    offsets_->resize(numRows);
    partition_->extractColumn(
        offsetIndex_, partitionOffset_, numRows, 0, offsets_);
    for (int i = 0; i < numRows; i++) {
      if (offsets_->isNullAt(i)) {
        rowNumbers_[i] = -1;
      } else {
        vector_size_t offset = offsets_->valueAt(i);
        VELOX_USER_CHECK_GE(offset, 0, "Offset must be at least 0");
        setRowNumber(i, partitionLimit, offset);
      }
    }

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
      const column_index_t i,
      const vector_size_t partitionLimit,
      const vector_size_t offset) {
    auto currentRow = partitionOffset_ + i;
    if constexpr (TValue == LeadLagType::kLead) {
      auto rowNumber = currentRow + offset;
      rowNumbers_[i] = rowNumber <= partitionLimit ? rowNumber : -1;
    } else {
      auto rowNumber = currentRow - offset;
      rowNumbers_[i] = rowNumber < 0 ? -1 : rowNumber;
    }
  }

  // Index of the column, the first argument, for lead and lag functions.
  column_index_t columnIndex_;

  // Index of the columnar offset, when the offset takes values of a column.
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

  // This variable is used for the rowNumber vector across getOutput calls.
  std::vector<vector_size_t> rowNumbers_;

  // Number of rows in the current partition.
  vector_size_t numPartitionRows_;

  // Member variable re-used for setting null for empty frames.
  SelectivityVector invalidRows_;
};
} // namespace

template <LeadLagType TValue>
void registerLeadLagInternal(const std::string& name) {
  std::vector<exec::FunctionSignaturePtr> signatures{
      exec::FunctionSignatureBuilder()
          .typeVariable("T")
          .returnType("T")
          .argumentType("T")
          .build(),
      exec::FunctionSignatureBuilder()
          .typeVariable("T")
          .returnType("T")
          .argumentType("T")
          .argumentType("bigint")
          .build(),
      exec::FunctionSignatureBuilder()
          .typeVariable("T")
          .returnType("T")
          .argumentType("T")
          .argumentType("bigint")
          .argumentType("T")
          .build(),
  };

  exec::registerWindowFunction(
      name,
      std::move(signatures),
      [](const std::vector<exec::WindowFunctionArg>& args,
         const TypePtr& resultType,
         velox::memory::MemoryPool* pool,
         HashStringAllocator* /*stringAllocator*/)
          -> std::unique_ptr<exec::WindowFunction> {
        return std::make_unique<LeadLagFunction<TValue>>(
            args, resultType, pool);
      });
}

void registerLead(const std::string& name) {
  registerLeadLagInternal<LeadLagType::kLead>(name);
}

void registerLag(const std::string& name) {
  registerLeadLagInternal<LeadLagType::kLag>(name);
}
} // namespace facebook::velox::window::prestosql
