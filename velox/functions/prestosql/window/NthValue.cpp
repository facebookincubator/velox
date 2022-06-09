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
#include "velox/functions/prestosql/window/WindowFunctionNames.h"
#include "velox/vector/FlatVector.h"

namespace facebook::velox::window {

namespace {

template <typename T>
class NthValueFunction : public exec::WindowFunction {
 public:
  explicit NthValueFunction(
      const std::vector<exec::RowColumn>& argColumns,
      const TypePtr& resultType)
      : WindowFunction(resultType),
        argColumns_(argColumns),
        partitionRows_(nullptr) {}

  void resetPartition(const std::vector<char*>& rows) {
    partitionRows_ = reinterpret_cast<char* const*>(rows.data());
    /*firstArgVector_->resize(rows.size());
    exec::RowContainer::extractColumn(
      rows.data(), rows.size(), argColumns_[0], firstArgVector_);

    offsetsVector_->resize(rows.size());
    exec::RowContainer::extractColumn(
      rows.data(), rows.size(), argColumns_[1], offsetsVector_); */
  }

  void apply(
      const BufferPtr& peerGroupStarts,
      const BufferPtr& /*peerGroupEnds*/,
      const BufferPtr& frameStarts,
      const BufferPtr& /*frameEnds*/,
      int32_t startPartitionRow,
      int32_t resultIndex,
      const VectorPtr& result) {
    FlatVector<T>* resultVector = result->as<FlatVector<T>>();
    auto firstArgVector =
        BaseVector::create(resultVector->type(), 1, peerGroupStarts->pool());
    auto offsetsVector =
        BaseVector::create(BIGINT(), 1, peerGroupStarts->pool());

    int numRows = peerGroupStarts->size();
    for (int i = 0; i < numRows; i++) {
      exec::RowContainer::extractColumn(
          partitionRows_ + startPartitionRow + i,
          1,
          argColumns_[1],
          offsetsVector);
      const int64_t offset = offsetsVector->values()->as<int64_t>()[0];
      VELOX_CHECK_GE(offset, 1);
      const int64_t frameStart = frameStarts->as<size_t>()[i];
      VELOX_CHECK_GE(frameStart, 0);
      exec::RowContainer::extractColumn(
          partitionRows_ + frameStart + offset - 1,
          1,
          argColumns_[0],
          firstArgVector);
      resultVector->mutableRawValues()[resultIndex + i] =
          firstArgVector->values()->template as<T>()[0];
    }
  }

 private:
  const std::vector<exec::RowColumn>& argColumns_;
  char* const* partitionRows_;
  // VectorPtr offsetsVector_;
  // VectorPtr firstArgVector_;
};

template <TypeKind kind>
std::unique_ptr<exec::WindowFunction> createNthValueFunction(
    const std::vector<exec::RowColumn>& argColumns,
    const TypePtr& resultType) {
  using T = typename TypeTraits<kind>::NativeType;
  return std::make_unique<NthValueFunction<T>>(argColumns, resultType);
}

bool registerNthValue(const std::string& name) {
  std::vector<std::shared_ptr<exec::FunctionSignature>> signatures{
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
          const TypePtr& resultType) -> std::unique_ptr<exec::WindowFunction> {
        VELOX_CHECK_EQ(argColumns.size(), 2, "{} takes two arguments", name);
        VELOX_CHECK_EQ(argTypes.size(), 2, "{} takes two arguments", name);
        auto typeKind = argTypes[0]->kind();
        return VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH(
            createNthValueFunction, typeKind, argColumns, resultType);
      });
  return true;
}

static bool FB_ANONYMOUS_VARIABLE(g_WindowFunction) =
    registerNthValue(kNthValue);

} // namespace
} // namespace facebook::velox::window
