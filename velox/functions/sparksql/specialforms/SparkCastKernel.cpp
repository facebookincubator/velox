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

#include "velox/functions/sparksql/specialforms/SparkCastKernel.h"

#include "velox/functions/lib/string/StringImpl.h"

namespace facebook::velox::functions::sparksql {
SparkCastKernel::SparkCastKernel(
    const velox::core::QueryConfig& config,
    bool allowOverflow)
    : exec::PrestoCastKernel(config),
      config_(config),
      allowOverflow_(allowOverflow) {}

VectorPtr SparkCastKernel::castToDate(
    const SelectivityVector& rows,
    const BaseVector& input,
    exec::EvalCtx& context,
    bool setNullInResultAtError) const {
  const auto& fromType = input.type();
  switch (fromType->kind()) {
    case TypeKind::VARCHAR: {
      auto* inputVector = input.as<SimpleVector<StringView>>();
      VectorPtr result;
      initializeResultVector(rows, DATE(), context, result);
      auto* resultFlatVector = result->as<FlatVector<int32_t>>();
      applyToSelectedNoThrowLocal(
          rows, context, result, setNullInResultAtError, [&](int row) {
            // Allows all patterns supported by Spark:
            // `[+-]yyyy*`
            // `[+-]yyyy*-[m]m`
            // `[+-]yyyy*-[m]m-[d]d`
            // `[+-]yyyy*-[m]m-[d]d *`
            // `[+-]yyyy*-[m]m-[d]dT*`
            // The asterisk `*` in `yyyy*` stands for any numbers.
            // For the last two patterns, the trailing `*` can represent none
            // or any sequence of characters, e.g:
            //   "1970-01-01 123"
            //   "1970-01-01 (BC)"
            const auto resultValue = util::fromDateString(
                removeWhiteSpaces(inputVector->valueAt(row)),
                util::ParseMode::kSparkCast);

            if (resultValue.hasValue()) {
              resultFlatVector->set(row, resultValue.value());
            } else {
              setError(
                  input,
                  context,
                  *result,
                  row,
                  resultValue.error().message(),
                  setNullInResultAtError);
            }
          });

      return result;
    }
    default:
      // Otherwise default back to Presto's behavior.
      return exec::PrestoCastKernel::castToDate(
          rows, input, context, setNullInResultAtError);
  }
}

StringView SparkCastKernel::removeWhiteSpaces(const StringView& view) const {
  StringView output;
  stringImpl::trimUnicodeWhiteSpace<true, true, StringView, StringView>(
      output, view);
  return output;
}
} // namespace facebook::velox::functions::sparksql
