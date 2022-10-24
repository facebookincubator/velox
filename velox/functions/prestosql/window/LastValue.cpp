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
class LastValueFunction : public exec::WindowFunction {
 public:
  explicit LastValueFunction(
      const std::vector<exec::WindowFunctionArg>& args,
      const TypePtr& resultType,
      velox::memory::MemoryPool* pool)
      : WindowFunction(resultType, pool) {
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
      const BufferPtr& /*frameStarts*/,
      const BufferPtr& frameEnds,
      int32_t resultOffset,
      const VectorPtr& result) override {
    auto numRows = frameEnds->size() / sizeof(vector_size_t);
    auto frameEndsPtr = frameEnds->as<vector_size_t>();
    auto rowNumbersRange = folly::Range(frameEndsPtr, numRows);
    partition_->extractColumn(
        valueIndex_, rowNumbersRange, resultOffset, result);
    partitionOffset_ += numRows;
  }

 private:
  // Argument index of the first_value argument column in the input row vector.
  // This is used to retrieve column values from the partition data.
  column_index_t valueIndex_;

  const exec::WindowPartition* partition_;

  // This offset tracks how far along the partition rows have been output.
  // This can be used to optimize reading offset column values corresponding
  // to the present row set in getOutput.
  vector_size_t partitionOffset_;
};

template <TypeKind kind>
std::unique_ptr<exec::WindowFunction> createLastValueFunction(
    const std::vector<exec::WindowFunctionArg>& args,
    const TypePtr& resultType,
    velox::memory::MemoryPool* pool) {
  using T = typename TypeTraits<kind>::NativeType;
  return std::make_unique<LastValueFunction<T>>(args, resultType, pool);
}

} // namespace

void registerLastValue(const std::string& name) {
  std::vector<exec::FunctionSignaturePtr> signatures{
      // (T, bigint) -> T.
      exec::FunctionSignatureBuilder()
          .typeVariable("T")
          .returnType("T")
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
            createLastValueFunction, typeKind, args, resultType, pool);
      });
}
} // namespace facebook::velox::window
