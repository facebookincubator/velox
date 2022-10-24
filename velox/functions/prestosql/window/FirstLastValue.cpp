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

enum class ValueType {
  kFirst,
  kLast,
};

template <ValueType TValue, typename T>
class FirstLastValueFunction : public exec::WindowFunction {
 public:
  explicit FirstLastValueFunction(
      const std::vector<exec::WindowFunctionArg>& args,
      const TypePtr& resultType,
      velox::memory::MemoryPool* pool)
      : WindowFunction(resultType, pool, nullptr) {
    VELOX_CHECK_NULL(args[0].constantValue);
    valueIndex_ = args[0].index.value();
  }

  void resetPartition(const exec::WindowPartition* partition) override {
    partition_ = partition;
  }

  void apply(
      const BufferPtr& /*peerGroupStarts*/,
      const BufferPtr& /*peerGroupEnds*/,
      const BufferPtr& frameStarts,
      const BufferPtr& frameEnds,
      int32_t resultOffset,
      const VectorPtr& result) override {
    auto numRows = frameStarts->size() / sizeof(vector_size_t);
    auto frameStartsPtr = frameStarts->as<vector_size_t>();
    auto frameEndsPtr = frameEnds->as<vector_size_t>();
    if constexpr (TValue == ValueType::kFirst) {
      auto rowNumbersRange = folly::Range(frameStartsPtr, numRows);
      partition_->extractColumn(
          valueIndex_, rowNumbersRange, resultOffset, result);
    } else {
      auto rowNumbersRange = folly::Range(frameEndsPtr, numRows);
      partition_->extractColumn(
          valueIndex_, rowNumbersRange, resultOffset, result);
    }
  }

 private:
  // Argument index of the first_value argument column in the input row vector.
  // This is used to retrieve column values from the partition data.
  column_index_t valueIndex_;

  const exec::WindowPartition* partition_;
};

template <TypeKind kind>
std::unique_ptr<exec::WindowFunction> createFirstLastValueFunction(
    const std::vector<exec::WindowFunctionArg>& args,
    const TypePtr& resultType,
    velox::memory::MemoryPool* pool,
    const ValueType valueType) {
  using T = typename TypeTraits<kind>::NativeType;
  if (valueType == ValueType::kFirst) {
    return std::make_unique<FirstLastValueFunction<ValueType::kFirst, T>>(
        args, resultType, pool);
  }
  return std::make_unique<FirstLastValueFunction<ValueType::kLast, T>>(
      args, resultType, pool);
}

} // namespace

template <ValueType TValue>
void registerFirstLastInternal(const std::string& name) {
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
          velox::memory::MemoryPool* pool,
          HashStringAllocator* /*stringAllocator*/)
          -> std::unique_ptr<exec::WindowFunction> {
        auto typeKind = args[0].type->kind();
        if constexpr (TValue == ValueType::kFirst) {
          return VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH(
              createFirstLastValueFunction,
              typeKind,
              args,
              resultType,
              pool,
              ValueType::kFirst);
        }
        return VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH(
            createFirstLastValueFunction,
            typeKind,
            args,
            resultType,
            pool,
            ValueType::kLast);
      });
}

void registerFirstValue(const std::string& name) {
  registerFirstLastInternal<ValueType::kFirst>(name);
}
void registerLastValue(const std::string& name) {
  registerFirstLastInternal<ValueType::kLast>(name);
}
} // namespace facebook::velox::window
