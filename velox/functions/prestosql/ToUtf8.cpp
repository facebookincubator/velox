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
#include "velox/expression/VectorFunction.h"

namespace facebook::velox::functions {
namespace {
class ToUtf8Function : public velox::exec::VectorFunction {
 public:
  void apply(
      const velox::SelectivityVector& rows,
      std::vector<velox::VectorPtr>& args,
      velox::exec::Expr* /* unused */,
      velox::exec::EvalCtx* context,
      velox::VectorPtr* result) const override {
    // Single-argument function, hence, input is flat.
    auto input = args[0]->asFlatVector<velox::StringView>();

    auto stringBuffers = input->stringBuffers();
    auto localResult = std::make_shared<velox::FlatVector<velox::StringView>>(
        context->pool(),
        velox::VARBINARY(),
        input->nulls(),
        rows.end(),
        input->values(),
        std::move(stringBuffers));

    context->moveOrCopyResult(localResult, rows, result);
  }

  static std::vector<std::shared_ptr<velox::exec::FunctionSignature>>
  signatures() {
    // varchar -> varbinary
    return {velox::exec::FunctionSignatureBuilder()
                .returnType("varbinary")
                .argumentType("varchar")
                .build()};
  }
};
} // namespace

VELOX_DECLARE_VECTOR_FUNCTION(
    udf_to_utf8,
    ToUtf8Function::signatures(),
    std::make_unique<ToUtf8Function>());

} // namespace facebook::velox::functions
