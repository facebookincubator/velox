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

#include "velox/functions/sparksql/ToPrettyString.h"

#include "velox/expression/VectorFunction.h"
#include "velox/functions/sparksql/specialforms/SparkCastKernel.h"

namespace facebook::velox::functions::sparksql {
namespace {

static const StringView kNull = "NULL";

class ToPrettyStringFunction final : public exec::VectorFunction {
 public:
  explicit ToPrettyStringFunction(const core::QueryConfig& config)
      : castKernel_(config, /*allowOverflow=*/false) {}

  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& outputType,
      exec::EvalCtx& context,
      VectorPtr& result) const final {
    const auto& inputType = args[0]->type();

    // Create a selectivity vector for non-null rows only.
    SelectivityVector nonNullRows = rows;
    if (args[0]->mayHaveNulls()) {
      nonNullRows.deselectNulls(args[0]->rawNulls(), rows.begin(), rows.end());
    }

    if (!nonNullRows.hasSelections()) {
      // All rows are null, set result directly to "NULL" string.
      context.ensureWritable(rows, outputType, result);
      result->clearNulls(rows);
      auto* flatResult = result->asFlatVector<StringView>();
      rows.applyToSelected(
          [&](auto row) { flatResult->setNoCopy(row, kNull); });
      return;
    }

    VectorPtr castResult;
    if (inputType->isDate()) {
      castResult = castKernel_.castFromDate(
          nonNullRows, *args[0], context, VARCHAR(), false);
    } else {
      castResult = castKernel_.castToVarchar(
          nonNullRows, *args[0], context, VARCHAR(), false);
    }

    // Set "NULL" for null rows directly in castResult.
    auto* flatCastResult = castResult->asFlatVector<StringView>();
    castResult->clearNulls(rows);
    rows.applyToSelected([&](auto row) {
      if (args[0]->isNullAt(row)) {
        flatCastResult->setNoCopy(row, kNull);
      }
    });

    context.moveOrCopyResult(castResult, rows, result);
  }

 private:
  SparkCastKernel castKernel_;
};

} // namespace

std::vector<std::shared_ptr<exec::FunctionSignature>>
toPrettyStringSignatures() {
  return {
      exec::FunctionSignatureBuilder()
          .returnType("varchar")
          .argumentType("tinyint")
          .build(),
      exec::FunctionSignatureBuilder()
          .returnType("varchar")
          .argumentType("smallint")
          .build(),
      exec::FunctionSignatureBuilder()
          .returnType("varchar")
          .argumentType("integer")
          .build(),
      exec::FunctionSignatureBuilder()
          .returnType("varchar")
          .argumentType("bigint")
          .build(),
      exec::FunctionSignatureBuilder()
          .returnType("varchar")
          .argumentType("real")
          .build(),
      exec::FunctionSignatureBuilder()
          .returnType("varchar")
          .argumentType("double")
          .build(),
      exec::FunctionSignatureBuilder()
          .returnType("varchar")
          .argumentType("boolean")
          .build(),
      exec::FunctionSignatureBuilder()
          .returnType("varchar")
          .argumentType("varchar")
          .build(),
      exec::FunctionSignatureBuilder()
          .returnType("varchar")
          .argumentType("date")
          .build(),
      exec::FunctionSignatureBuilder()
          .returnType("varchar")
          .argumentType("unknown")
          .build(),
  };
}

std::unique_ptr<exec::VectorFunction> makeToPrettyString(
    const std::string& /*name*/,
    const std::vector<exec::VectorFunctionArg>& inputArgs,
    const core::QueryConfig& config) {
  return std::make_unique<ToPrettyStringFunction>(config);
}

} // namespace facebook::velox::functions::sparksql
