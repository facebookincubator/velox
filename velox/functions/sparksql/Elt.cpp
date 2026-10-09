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
#include "velox/functions/sparksql/Elt.h"

#include "velox/expression/DecodedArgs.h"
#include "velox/functions/sparksql/SparkQueryConfig.h"
#include "velox/vector/FlatVector.h"

namespace facebook::velox::functions::sparksql {
namespace {

class EltFunction : public exec::VectorFunction {
 public:
  explicit EltFunction(bool ansiEnabled) : ansiEnabled_{ansiEnabled} {}

  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& outputType,
      exec::EvalCtx& context,
      VectorPtr& result) const override {
    exec::DecodedArgs decodedArgs(rows, args, context);
    const auto* indices = decodedArgs.at(0);
    const auto numInputs = static_cast<int32_t>(args.size()) - 1;

    context.ensureWritable(rows, outputType, result);
    auto* flatResult = result->asFlatVector<StringView>();
    // The result references the input strings without copying them.
    for (auto i = 1; i <= numInputs; ++i) {
      flatResult->acquireSharedStringBuffers(args[i].get());
    }

    context.applyToSelectedNoThrow(rows, [&](vector_size_t row) {
      if (indices->isNullAt(row)) {
        flatResult->setNull(row, true);
        return;
      }
      const auto index = indices->valueAt<int32_t>(row);
      if (index < 1 || index > numInputs) {
        if (ansiEnabled_) {
          VELOX_USER_FAIL(
              "Index is out of bounds: {}. Number of inputs: {}",
              index,
              numInputs);
        }
        flatResult->setNull(row, true);
        return;
      }
      const auto* input = decodedArgs.at(index);
      if (input->isNullAt(row)) {
        flatResult->setNull(row, true);
        return;
      }
      flatResult->setNoCopy(row, input->valueAt<StringView>(row));
    });
  }

 private:
  // Throws on out-of-range index instead of returning NULL (Spark ANSI mode).
  const bool ansiEnabled_;
};

} // namespace

std::vector<std::shared_ptr<exec::FunctionSignature>> eltSignatures() {
  std::vector<std::shared_ptr<exec::FunctionSignature>> signatures;
  for (const auto& type : {"varchar", "varbinary"}) {
    signatures.emplace_back(
        exec::FunctionSignatureBuilder()
            .returnType(type)
            .argumentType("integer")
            .argumentType(type)
            .variableArity()
            .build());
  }
  return signatures;
}

std::shared_ptr<exec::VectorFunction> makeElt(
    const std::string& /*name*/,
    const std::vector<exec::VectorFunctionArg>& /*inputArgs*/,
    const core::QueryConfig& config) {
  return std::make_shared<EltFunction>(SparkQueryConfig{config}.ansiEnabled());
}

} // namespace facebook::velox::functions::sparksql
