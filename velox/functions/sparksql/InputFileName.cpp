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

#include "velox/functions/sparksql/InputFileName.h"
#include <iostream>
#include <utility>
#include "velox/exec/Driver.h"
#include "velox/exec/Operator.h"
#include "velox/expression/VectorWriters.h"
#include "velox/functions/prestosql/URLFunctions.h"
#include "velox/type/StringView.h"

namespace facebook::velox::functions::sparksql {
namespace {
class InputFileName final : public exec::VectorFunction {
 public:
  InputFileName() {}

  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& /* outputType */,
      exec::EvalCtx& context,
      VectorPtr& result) const override {
    context.ensureWritable(rows, VARCHAR(), result);
    auto* output = result->as<FlatVector<StringView>>();
    auto driverCtx = context.driverCtx();
    rows.applyToSelected([&](vector_size_t row) {
      auto writer = exec::StringWriter<>(output, row);
      auto inputFileName = driverCtx != nullptr ? driverCtx->inputFileName : "";
      facebook::velox::functions::urlEscape(
          writer, driverCtx->inputFileName, false, "!$&'()*+,;=/:@");
      writer.finalize();
    });
  }
};
} // namespace
std::shared_ptr<exec::VectorFunction> makeInputFileName(
    const std::string& /*name*/,
    const std::vector<exec::VectorFunctionArg>& inputArgs,
    const core::QueryConfig& /*config*/) {
  return std::make_shared<InputFileName>();
}

std::vector<std::shared_ptr<exec::FunctionSignature>>
inputFileNameSignatures() {
  return {exec::FunctionSignatureBuilder().returnType("varchar").build()};
}
} // namespace facebook::velox::functions::sparksql
