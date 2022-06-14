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
      const std::vector<exec::RowColumn>& argColumns,
      const TypePtr& resultType,
      velox::memory::MemoryPool* pool)
      : WindowFunction(resultType, pool), argColumns_(argColumns) {
    columnVector_ = BaseVector::create(resultType, 0, pool);
    offsetsVector_ = BaseVector::create(BIGINT(), 0, pool);
  }

  void resetPartition(const folly::Range<char**>& rows) {
    columnVector_->resize(rows.size());
    exec::RowContainer::extractColumn(
        rows.data(), rows.size(), argColumns_[0], columnVector_);

    offsetsVector_->resize(rows.size());
    exec::RowContainer::extractColumn(
        rows.data(), rows.size(), argColumns_[1], offsetsVector_);

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
    auto offsetsVector = offsetsVector_->as<FlatVector<int64_t>>();
    auto firstArgVector = columnVector_->as<FlatVector<T>>();

    for (int i = 0; i < numRows; i++) {
      auto offset = offsetsVector->valueAt(partitionOffset_ + i);
      auto frameStart = frameStartsVector[partitionOffset_ + i];

      if (columnVector_->isNullAt(frameStart + offset - 1)) {
        result->setNull(resultOffset + i, true);
      } else {
        rawResultsBuffer[resultOffset + i] =
            firstArgVector->valueAt(frameStart + offset - 1);
      }
    }

    partitionOffset_ += numRows;
  }

 private:
  // This needs to be a copy of the argColumns passed to the function
  // as we need to retain them across function calls.
  std::vector<exec::RowColumn> argColumns_;

  // This is a vector for the column that nth_value needs to extract.
  // It is bound for all rows in the partition at a time.
  VectorPtr columnVector_;
  // This is a vector for the offsets column.
  // It is bound for all rows in the partition at a time.
  VectorPtr offsetsVector_;
  // This offset tracks how far along the partition rows have been output.
  // This is used to index into the columnVector_ and offsetsVector_
  // while outputting all the rows for the partition.
  vector_size_t partitionOffset_;
};

template <TypeKind kind>
std::unique_ptr<exec::WindowFunction> createNthValueFunction(
    const std::vector<exec::RowColumn>& argColumns,
    const TypePtr& resultType,
    velox::memory::MemoryPool* pool) {
  using T = typename TypeTraits<kind>::NativeType;
  return std::make_unique<NthValueFunction<T>>(argColumns, resultType, pool);
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
          const std::vector<exec::RowColumn>& argColumns,
          const std::vector<TypePtr>& argTypes,
          const TypePtr& resultType,
          velox::memory::MemoryPool* pool)
          -> std::unique_ptr<exec::WindowFunction> {
        auto typeKind = argTypes[0]->kind();
        return VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH(
            createNthValueFunction, typeKind, argColumns, resultType, pool);
      });
}
} // namespace facebook::velox::window
