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
#include "velox/functions/sparksql/AssertNotNull.h"

namespace facebook::velox::functions::sparksql {
namespace {

// Asserts that the input value is not null. If any value in the specified rows
// is null, throws a user error. Used by Spark's TableOutputResolver to enforce
// NOT NULL column constraints during table inserts.
class AssertNotNullFunction final : public exec::VectorFunction {
 public:
  explicit AssertNotNullFunction(
      std::string errMsg = "Null value appeared in non-nullable field")
      : errMsg_(std::move(errMsg)) {}

  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& /* outputType */,
      exec::EvalCtx& context,
      VectorPtr& result) const final {
    const auto& input = args[0];
    if (input->mayHaveNulls() && input->rawNulls()) {
      context.applyToSelectedNoThrow(rows, [&](vector_size_t row) {
        VELOX_USER_CHECK(!input->isNullAt(row), "{}", errMsg_);
      });
    }
    context.moveOrCopyResult(input, rows, result);
  }

 private:
  const std::string errMsg_;
};
} // namespace

std::vector<std::shared_ptr<exec::FunctionSignature>>
assertNotNullSignatures() {
  return {
      exec::FunctionSignatureBuilder()
          .typeVariable("T")
          .returnType("T")
          .argumentType("T")
          .build(),
      exec::FunctionSignatureBuilder()
          .typeVariable("T")
          .returnType("T")
          .argumentType("T")
          .constantArgumentType("varchar")
          .build()};
}

std::shared_ptr<exec::VectorFunction> makeAssertNotNull(
    const std::string& /* name */,
    const std::vector<exec::VectorFunctionArg>& inputArgs,
    const core::QueryConfig& /* config */) {
  if (inputArgs.size() == 2 && inputArgs[1].constantValue &&
      !inputArgs[1].constantValue->isNullAt(0)) {
    auto constantExpr =
        inputArgs[1].constantValue->as<ConstantVector<StringView>>();
    return std::make_shared<AssertNotNullFunction>(
        constantExpr->valueAt(0).str());
  }
  return std::make_shared<AssertNotNullFunction>();
}

} // namespace facebook::velox::functions::sparksql
