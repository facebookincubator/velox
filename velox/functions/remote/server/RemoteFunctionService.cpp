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

#include "velox/functions/remote/server/RemoteFunctionService.h"
#include "velox/expression/Expr.h"
#include "velox/functions/remote/if/GetSerde.h"
#include "velox/type/fbhive/HiveTypeParser.h"
#include "velox/vector/VectorStream.h"

namespace facebook::velox::functions {
namespace {

std::string getFunctionName(
    const std::string& prefix,
    const std::string& functionName) {
  return fmt::format("{}.{}", prefix, functionName);
}

TypePtr deserializeType(const std::string& input) {
  // Use hive type parser/serializer.
  return type::fbhive::HiveTypeParser().parse(input);
}

RowTypePtr deserializeArgTypes(const std::vector<std::string>& argTypes) {
  const size_t argCount = argTypes.size();

  std::vector<TypePtr> argumentTypes;
  std::vector<std::string> typeNames;
  argumentTypes.reserve(argCount);
  typeNames.reserve(argCount);

  for (size_t i = 0; i < argCount; ++i) {
    argumentTypes.emplace_back(deserializeType(argTypes[i]));
    typeNames.emplace_back(fmt::format("c{}", i));
  }
  return ROW(std::move(typeNames), std::move(argumentTypes));
}

} // namespace

std::vector<core::TypedExprPtr> getExpressions(
    const RowTypePtr& inputType,
    const TypePtr& returnType,
    const std::string& functionName) {
  std::vector<core::TypedExprPtr> inputs;
  for (size_t i = 0; i < inputType->size(); ++i) {
    inputs.push_back(std::make_shared<core::FieldAccessTypedExpr>(
        inputType->childAt(i), inputType->nameOf(i)));
  }

  return {std::make_shared<core::CallTypedExpr>(
      returnType, std::move(inputs), functionName)};
}

void RemoteFunctionServiceHandler::invokeFunction(
    remote::RemoteFunctionResponse& response,
    std::unique_ptr<remote::RemoteFunctionRequest> request) {
  const auto& functionHandle = request->get_remoteFunctionHandle();
  LOG(INFO) << "Got a request for '" << functionHandle.get_name() << "'.";

  if (!request->get_throwOnError()) {
    VELOX_NYI("throwOnError not implemented yet on remote server.");
  }

  // Deserialize types and data.
  auto inputType = deserializeArgTypes(functionHandle.get_argumentTypes());
  auto outputType = deserializeType(functionHandle.get_returnType());

  auto serdeFormat = request->get_inputs().get_pageFormat();
  auto serde = getSerde(serdeFormat);

  auto inputVector = IOBufToRowVector(
      request->get_inputs().get_payload(), inputType, *pool_, serde.get());

  // Execute the expression.
  const vector_size_t numRows = inputVector->size();
  SelectivityVector rows{numRows};

  // Expression boilerplate.
  core::QueryCtx queryCtx;
  core::ExecCtx execCtx{pool_.get(), &queryCtx};
  exec::ExprSet exprSet{
      getExpressions(
          inputType,
          outputType,
          getFunctionName(functionPrefix_, functionHandle.get_name())),
      &execCtx};
  exec::EvalCtx evalCtx(&execCtx, &exprSet, inputVector.get());

  std::vector<VectorPtr> expressionResult;
  exprSet.eval(rows, evalCtx, expressionResult);

  // Create output vector.
  auto outputRowVector = std::make_shared<RowVector>(
      pool_.get(), ROW({outputType}), BufferPtr(), numRows, expressionResult);

  auto result = response.result_ref();
  result->rowCount_ref() = outputRowVector->size();
  result->pageFormat_ref() = serdeFormat;
  result->payload_ref() =
      rowVectorToIOBuf(outputRowVector, rows.end(), *pool_, serde.get());
}

} // namespace facebook::velox::functions
