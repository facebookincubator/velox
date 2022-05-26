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

class RowNumberFunction : public exec::WindowFunction {
 public:
  explicit RowNumberFunction() : WindowFunction(BIGINT()) {}

  void resetPartition(const std::vector<char*>& /*rows*/) {
    rowNumber_ = 1;
  }

  void apply(
      const BufferPtr& /* peerGroupStarts */,
      const BufferPtr& /* peerGroupEnds */,
      const BufferPtr& /* frameStarts */,
      const BufferPtr& /* frameEnds */,
      const VectorPtr& result) {
    result->asFlatVector<int64_t>()->mutableRawValues()[currentPosition_] =
        rowNumber_++;
    currentPosition_++;
  }

 private:
  int64_t rowNumber_ = 1;
  // TODO : This variable presumes a single output buffer.
  // Enhance this to handle batches of output buffers.
  int64_t currentPosition_ = 0;
};

bool registerRowNumber(const std::string& name) {
  std::vector<std::shared_ptr<exec::FunctionSignature>> signatures{
      exec::FunctionSignatureBuilder().returnType("bigint").build(),
  };

  exec::registerWindowFunction(
      name,
      std::move(signatures),
      [name](
          const std::vector<TypePtr>& argTypes, const TypePtr&
          /*resultType*/) -> std::unique_ptr<exec::WindowFunction> {
        VELOX_CHECK_LE(argTypes.size(), 0, "{} takes no arguments", name);
        return std::make_unique<RowNumberFunction>();
      });
  return true;
}

static bool FB_ANONYMOUS_VARIABLE(g_WindowFunction) =
    registerRowNumber(kRowNumber);

} // namespace
} // namespace facebook::velox::window
