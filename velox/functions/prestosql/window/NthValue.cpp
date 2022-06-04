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
  explicit NthValueFunction(const TypePtr& resultType)
      : WindowFunction(resultType) {}

  void resetPartition(const std::vector<char*>& /*rows*/) {}

  void apply(
      int32_t /* peerGroupStarts */,
      int32_t /* peerGroupEnds */,
      int32_t frameStarts,
      int32_t /* frameEnds */,
      int32_t currentOutputRow,
      const std::vector<VectorPtr>& argVectors,
      const VectorPtr& result) {
    FlatVector<T>* resultVector = result->as<FlatVector<T>>();
    FlatVector<T>* firstArgVector = argVectors[0]->as<FlatVector<T>>();

    const int64_t offset =
        argVectors[1]->as<FlatVector<int64_t>>()->valueAt(currentOutputRow);
    // TODO : Add more validations here.
    VELOX_CHECK_GE(offset, 1);
    VELOX_CHECK_GE(frameStarts, 0);

    resultVector->mutableRawValues()[currentOutputRow] =
        firstArgVector->valueAt(
            partitionStartOffset_ + frameStarts + offset - 1);
    partitionStartOffset_++;
  }

 private:
  // TODO : Kind of ugly hack to know where in the buffer the partition starts
  // as the frameOffset is relative to the partition start.
  int32_t partitionStartOffset_ = 0;
};

template <TypeKind kind>
std::unique_ptr<exec::WindowFunction> createNthValueFunction(
    const TypePtr& resultType) {
  using T = typename TypeTraits<kind>::NativeType;
  return std::make_unique<NthValueFunction<T>>(resultType);
}

bool registerNthValue(const std::string& name) {
  std::vector<std::shared_ptr<exec::FunctionSignature>> signatures{
      exec::FunctionSignatureBuilder()
          .typeVariable("T")
          .returnType("T")
          .argumentType("T")
          .argumentType("integer")
          .build(),
  };

  exec::registerWindowFunction(
      name,
      std::move(signatures),
      [name](const std::vector<TypePtr>& argTypes, const TypePtr& resultType)
          -> std::unique_ptr<exec::WindowFunction> {
        VELOX_CHECK_EQ(argTypes.size(), 2, "{} takes two arguments", name);
        auto typeKind = argTypes[0]->kind();
        return VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH(
            createNthValueFunction, typeKind, resultType);
      });
  return true;
}

static bool FB_ANONYMOUS_VARIABLE(g_WindowFunction) =
    registerNthValue(kNthValue);

} // namespace
} // namespace facebook::velox::window
