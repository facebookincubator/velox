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

#include "velox/functions/prestosql/window/ValueFunctionsUtility.h"

namespace facebook::velox::window::prestosql {

namespace {

template <ValueType TValue>
class LeadLagFunction : public ValueFunction<TValue> {
 public:
  explicit LeadLagFunction(
      const std::vector<exec::WindowFunctionArg>& args,
      const TypePtr& resultType,
      velox::memory::MemoryPool* pool)
      : ValueFunction<TValue>(args, resultType, pool) {}

  void resetPartition(const exec::WindowPartition* partition) override {
    ValueFunction<TValue>::partition_ = partition;
    ValueFunction<TValue>::numPartitionRows_ = partition->numRows();
    ValueFunction<TValue>::partitionOffset_ = 0;
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
    if constexpr (TValue == ValueType::kLead) {
      partitionLimit = ValueFunction<TValue>::numPartitionRows_ - 1;
    }
    ValueFunction<TValue>::rowNumbers_.resize(numRows);

    if (ValueFunction<TValue>::constantOffset_.has_value() ||
        ValueFunction<TValue>::isConstantOffsetNull_) {
      ValueFunction<TValue>::setRowNumbersForConstantOffset(numRows, validRows);
    } else {
      ValueFunction<TValue>::setRowNumbers(numRows, validRows);
    }

    auto rowNumbersRange =
        folly::Range(ValueFunction<TValue>::rowNumbers_.data(), numRows);
    ValueFunction<TValue>::partition_->extractColumn(
        ValueFunction<TValue>::valueIndex_,
        rowNumbersRange,
        resultOffset,
        result);
    ValueFunction<TValue>::partitionOffset_ += numRows;
  }
};
} // namespace

template <ValueType TValue>
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
  registerLeadLagInternal<ValueType::kLead>(name);
}

void registerLag(const std::string& name) {
  registerLeadLagInternal<ValueType::kLag>(name);
}
} // namespace facebook::velox::window::prestosql
