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
#include "velox/functions/prestosql/types/PrestoTypes.h"

namespace facebook::velox::functions {
namespace {

class TypeOfFunction : public exec::VectorFunction {
 public:
  TypeOfFunction(const TypePtr& type)
      : typeName_{PrestoTypes::displayName(*type)} {}

  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& /* outputType */,
      exec::EvalCtx& context,
      VectorPtr& result) const override {
    auto localResult = BaseVector::createConstant(
        VARCHAR(), typeName_, rows.size(), context.pool());
    context.moveOrCopyResult(localResult, rows, result);
  }

  static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
    // T -> varchar
    return {exec::FunctionSignatureBuilder()
                .typeVariable("T")
                .returnType("varchar")
                .argumentType("T")
                .build()};
  }

  static std::shared_ptr<exec::VectorFunction> create(
      const std::string& /*name*/,
      const std::vector<exec::VectorFunctionArg>& inputArgs,
      const core::QueryConfig& /*config*/) {
    try {
      return std::make_shared<TypeOfFunction>(inputArgs[0].type);
    } catch (...) {
      return std::make_shared<exec::AlwaysFailingVectorFunction>(
          std::current_exception());
    }
  }

 private:
  const std::string typeName_;
};
} // namespace

VELOX_DECLARE_STATEFUL_VECTOR_FUNCTION_WITH_METADATA(
    udf_typeof,
    TypeOfFunction::signatures(),
    exec::VectorFunctionMetadataBuilder().defaultNullBehavior(false).build(),
    TypeOfFunction::create);

} // namespace facebook::velox::functions
