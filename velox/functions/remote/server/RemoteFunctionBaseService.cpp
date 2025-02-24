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

#include "velox/functions/remote/server/RemoteFunctionBaseService.h"

#include "velox/type/fbhive/HiveTypeParser.h"

namespace facebook::velox::functions {
namespace {

std::string getFunctionName(
    const std::string& prefix,
    const std::string& functionName) {
  return prefix.empty() ? functionName
                        : fmt::format("{}.{}", prefix, functionName);
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

std::vector<core::TypedExprPtr> getExpressions(
    const RowTypePtr& inputType,
    const TypePtr& returnType,
    const std::string& functionName) {
  std::vector<core::TypedExprPtr> inputs;
  for (size_t i = 0; i < inputType->size(); ++i) {
    inputs.push_back(
        std::make_shared<core::FieldAccessTypedExpr>(
            inputType->childAt(i), inputType->nameOf(i)));
  }

  return {std::make_shared<core::CallTypedExpr>(
      returnType, std::move(inputs), functionName)};
}

} // namespace

RowVectorPtr RemoteFunctionBaseService::invokeFunctionInternal(
    const folly::IOBuf& payload,
    const std::vector<std::string>& argTypeNames,
    const std::string& returnTypeName,
    const std::string& functionName,
    bool throwOnError,
    VectorSerde* serde) {
  auto inputType = deserializeArgTypes(argTypeNames);
  auto outputType = deserializeType(returnTypeName);

  auto inputVector = IOBufToRowVector(payload, inputType, *pool_, serde);

  const vector_size_t numRows = inputVector->size();
  SelectivityVector rows{numRows};

  queryCtx_ = core::QueryCtx::create();
  execCtx_ = std::make_unique<core::ExecCtx>(pool_.get(), queryCtx_.get());
  exec::ExprSet exprSet{
      getExpressions(
          inputType,
          outputType,
          getFunctionName(functionPrefix_, functionName)),
      execCtx_.get()};
  evalCtx_ = std::make_unique<exec::EvalCtx>(
      execCtx_.get(), &exprSet, inputVector.get());
  *evalCtx_->mutableThrowOnError() = throwOnError;

  std::vector<VectorPtr> expressionResult;
  exprSet.eval(rows, *evalCtx_, expressionResult);

  return std::make_shared<RowVector>(
      pool_.get(), ROW({outputType}), BufferPtr(), numRows, expressionResult);
}

} // namespace facebook::velox::functions
