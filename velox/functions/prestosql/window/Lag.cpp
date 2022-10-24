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

namespace facebook::velox::window {

namespace {

template <typename T>
class LagFunction : public exec::WindowFunction {
 public:
  explicit LagFunction(
      const std::vector<exec::WindowFunctionArg>& args,
      const TypePtr& resultType,
      velox::memory::MemoryPool* pool)
      : WindowFunction(resultType, pool) {
    VELOX_CHECK_GE(args.size(), 1);
    VELOX_CHECK_LE(args.size(), 3);
    VELOX_CHECK_NULL(args[0].constantValue);
    valueIndex_ = args[0].index.value();
    hasOffsetValue_ = false;
    hasDefaultValue_ = false;

    if (args.size() > 1) {
      hasOffsetValue_ = true;
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
            constantOffset_.value(), 1, "Offset must be at least 0");
        return;
      }
      offsetIndex_ = args[1].index.value();
      offsets_ = std::dynamic_pointer_cast<FlatVector<int64_t>>(
          BaseVector::create(BIGINT(), 0, pool));
    }

    if (args.size() == 3) {
      hasDefaultValue_ = true;
      if (args[2].constantValue) {
        if (args[2].constantValue->isNullAt(0)) {
          isConstantDefaultValueNull_ = true;
          return;
        }
        constantDefaultValue_ =
            args[2]
                .constantValue->template as<ConstantVector<int64_t>>()
                ->valueAt(0);
        return;
      }
      defaultValueIndex_ = args[2].index.value();
      defaultValueOffsets_ = std::dynamic_pointer_cast<FlatVector<int64_t>>(
          BaseVector::create(BIGINT(), 0, pool));
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
      int32_t resultOffset,
      const VectorPtr& result) override {
    auto numRows = frameStarts->size() / sizeof(vector_size_t);
    if (!hasOffsetValue_) {
      constantOffset_ = 1;
    }
    if (constantOffset_.has_value() || isConstantOffsetNull_) {
      setRowNumbersForConstantOffset(numRows);
    } else {
      setRowNumbers(numRows);
    }

    if (hasDefaultValue_) {
      if (constantDefaultValue_.has_value() || isConstantDefaultValueNull_) {
        setDefaultValuesForConstantOffset(numRows, result);
      } else {
        setDefaultValues(numRows, result);
      }
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
  void setRowNumbersForConstantOffset(vector_size_t numRows) {
    rowNumbers_.resize(numRows);

    if (isConstantOffsetNull_) {
      std::fill(rowNumbers_.begin(), rowNumbers_.end(), -1);
      return;
    }

    auto constantOffsetValue = constantOffset_.value();
    for (int i = 0; i < numRows; i++) {
      setRowNumber(i, constantOffsetValue);
    }
  }

  void setRowNumbers(vector_size_t numRows) {
    rowNumbers_.resize(numRows);
    offsets_->resize(numRows);
    partition_->extractColumn(
        offsetIndex_, partitionOffset_, numRows, 0, offsets_);
    for (int i = 0; i < numRows; i++) {
      if (offsets_->isNullAt(i)) {
        rowNumbers_[i] = -1;
      } else {
        vector_size_t offset = offsets_->valueAt(i);
        VELOX_USER_CHECK_GE(offset, 0, "Offset must be at least 0");
        setRowNumber(i, offset);
      }
    }
  }

  inline void setRowNumber(const column_index_t i, const vector_size_t offset) {
    auto currentRow = partitionOffset_ + i;
    vector_size_t rowNumber = currentRow - offset;
    rowNumbers_[i] = rowNumber < 0 ? -1 : rowNumber;
  }

  void setDefaultValuesForConstantOffset(
      vector_size_t numRows,
      const VectorPtr& result) {
    auto resultFlatVector = result->as<FlatVector<int64_t>>();
    BufferPtr valuesBuffer = resultFlatVector->mutableValues(numRows);
    auto values = valuesBuffer->asMutableRange<int64_t>();

    rowNumbers_.resize(numRows);
    if (isConstantDefaultValueNull_) {
      std::fill(rowNumbers_.begin(), rowNumbers_.end(), -1);
      return;
    }

    auto constantDefaultOffsetValue = constantDefaultValue_.value();
    for (int i = 0; i < numRows; i++) {
      if (rowNumbers_[i] == -1) {
        values[i] = constantDefaultOffsetValue;
      }
    }
  }

  void setDefaultValues(vector_size_t numRows, const VectorPtr& result) {
    auto resultFlatVector = result->as<FlatVector<int64_t>>();
    BufferPtr valuesBuffer = resultFlatVector->mutableValues(numRows);
    auto values = valuesBuffer->asMutableRange<int64_t>();
    rowNumbers_.resize(numRows);
    defaultValueOffsets_->resize(numRows);
    partition_->extractColumn(
        defaultValueIndex_, partitionOffset_, numRows, 0, defaultValueOffsets_);

    for (int i = 0; i < numRows; i++) {
      if (defaultValueOffsets_->isNullAt(i)) {
        rowNumbers_[i] = -1;
      } else {
        vector_size_t offset = defaultValueOffsets_->valueAt(i);
        VELOX_USER_CHECK_GE(offset, 0, "Offset must be at least 0");
        values[i] = offset;
      }
    }
  }

  bool hasOffsetValue_;
  bool hasDefaultValue_;
  column_index_t valueIndex_;
  column_index_t offsetIndex_;
  column_index_t defaultValueIndex_;

  const exec::WindowPartition* partition_;

  // These fields are set if the offset argument is a constant value.
  std::optional<int64_t> constantOffset_;
  bool isConstantOffsetNull_ = false;
  std::optional<int64_t> constantDefaultValue_;
  bool isConstantDefaultValueNull_ = false;

  // This vector is used to extract values of the offset argument column
  // (if not a constant offset value).
  FlatVectorPtr<int64_t> offsets_;
  FlatVectorPtr<int64_t> defaultValueOffsets_;

  // This offset tracks how far along the partition rows have been output.
  // This can be used to optimize reading offset column values corresponding
  // to the present row set in getOutput.
  vector_size_t partitionOffset_;

  std::vector<vector_size_t> rowNumbers_;
  vector_size_t numPartitionRows_;
};

template <TypeKind kind>
std::unique_ptr<exec::WindowFunction> createLagFunction(
    const std::vector<exec::WindowFunctionArg>& args,
    const TypePtr& resultType,
    velox::memory::MemoryPool* pool) {
  using T = typename TypeTraits<kind>::NativeType;
  return std::make_unique<LagFunction<T>>(args, resultType, pool);
}

} // namespace

void registerLag(const std::string& name) {
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
      [name](
          const std::vector<exec::WindowFunctionArg>& args,
          const TypePtr& resultType,
          velox::memory::MemoryPool* pool)
          -> std::unique_ptr<exec::WindowFunction> {
        auto typeKind = args[0].type->kind();
        return VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH(
            createLagFunction, typeKind, args, resultType, pool);
      });
}

} // namespace facebook::velox::window
