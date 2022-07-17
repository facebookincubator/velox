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
class NthValueFunction : public exec::WindowFunction {
 public:
  explicit NthValueFunction(
      const std::vector<column_index_t>& argIndices,
      const TypePtr& resultType,
      velox::memory::MemoryPool* pool)
      : WindowFunction(resultType, pool), argIndices_(argIndices) {}

  void resetPartition(const exec::WindowPartition* partition) {
    partition_ = partition;
    partitionOffset_ = 0;
  }

  void apply(
      const BufferPtr& peerGroupStarts,
      const BufferPtr& /*peerGroupEnds*/,
      const BufferPtr& frameStarts,
      const BufferPtr& /*frameEnds*/,
      int32_t resultOffset,
      const VectorPtr& result) {
    auto numRows = peerGroupStarts->size();

    auto rawResultsBuffer = result->as<FlatVector<T>>()->mutableRawValues();
    auto frameStartsVector = frameStarts->as<vector_size_t>();

    auto columnVector = partition_->argColumn(argIndices_[0]);
    auto firstArgVector = columnVector->template as<FlatVector<T>>();
    auto offsetsVector = partition_->argColumn(argIndices_[1])
                             ->template as<FlatVector<int64_t>>();

    for (int i = 0; i < numRows; i++) {
      auto offset = offsetsVector->valueAt(partitionOffset_ + i);
      auto frameStart = frameStartsVector[partitionOffset_ + i];

      if (columnVector->isNullAt(frameStart + offset - 1)) {
        result->setNull(resultOffset + i, true);
      } else {
        rawResultsBuffer[resultOffset + i] =
            firstArgVector->valueAt(frameStart + offset - 1);
      }
    }

    partitionOffset_ += numRows;
  }

 private:
  const exec::WindowPartition* partition_;
  // This is the index of the nth_value arguments in the WindowNode
  // input columns list.
  const std::vector<column_index_t> argIndices_;

  // This offset tracks how far along the partition rows have been output.
  // This is used to index into the argument vectors from the WindowPartition
  // while outputting all the rows for the partition.
  vector_size_t partitionOffset_;
};

template <TypeKind kind>
std::unique_ptr<exec::WindowFunction> createNthValueFunction(
    const std::vector<column_index_t>& argIndices,
    const TypePtr& resultType,
    velox::memory::MemoryPool* pool) {
  using T = typename TypeTraits<kind>::NativeType;
  return std::make_unique<NthValueFunction<T>>(argIndices, resultType, pool);
}

} // namespace

void registerNthValue(const std::string& name) {
  std::vector<exec::FunctionSignaturePtr> signatures{
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
          const std::vector<TypePtr>& argTypes,
          const std::vector<column_index_t>& argIndices,
          const TypePtr& resultType,
          velox::memory::MemoryPool* pool)
          -> std::unique_ptr<exec::WindowFunction> {
        auto typeKind = argTypes[0]->kind();
        return VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH(
            createNthValueFunction, typeKind, argIndices, resultType, pool);
      });
}
} // namespace facebook::velox::window
