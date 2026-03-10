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

#include "velox/functions/sparksql/types/TimestampNTZCastUtil.h"
#include "velox/expression/EvalCtx.h"
#include "velox/functions/lib/string/StringImpl.h"
#include "velox/type/TimestampConversion.h"

namespace facebook::velox::functions::sparksql {
namespace {

void setError(
    exec::EvalCtx& context,
    vector_size_t row,
    const std::string& errorDetails) {
  if (context.captureErrorDetails()) {
    context.setStatus(row, Status::UserError("{}", errorDetails));
  } else {
    context.setStatus(row, Status::UserError());
  }
}

StringView removeWhiteSpaces(const StringView& view) {
  StringView output;
  stringImpl::trimUnicodeWhiteSpace<true, true, StringView, StringView>(
      output, view);
  return output;
}

} // namespace

void castFromString(
    const SimpleVector<StringView>& inputVector,
    exec::EvalCtx& context,
    const SelectivityVector& rows,
    int64_t* rawResults) {
  context.applyToSelectedNoThrow(rows, [&](vector_size_t row) {
    auto view = inputVector.valueAt(row);
    view = removeWhiteSpaces(view);
    if (view.size() == 0) {
      // 'CastExpr::applyPeeled' will choose to set NULL or throw based on the
      // output of 'setNullInResultAtError()'.
      const auto errorDetails = fmt::format(
          "Cannot cast VARCHAR '{}' to TIMESTAMP_NTZ. {}",
          view,
          "Empty string");
      setError(context, row, errorDetails);
      return;
    }

    // The 'fromTimestampString' cannot be used here because it does not allow
    // timezone in the input string, while Spark allows timezone in the input
    // string for TIMESTAMP_NTZ type but ignores it.
    auto conversionResult = util::fromTimestampWithTimezoneString(
        view.data(), view.size(), util::TimestampParseMode::kSparkCast);
    if (conversionResult.hasError()) {
      const auto errorDetails = fmt::format(
          "Cannot cast VARCHAR '{}' to TIMESTAMP_NTZ. {}",
          view,
          conversionResult.error().message());
      setError(context, row, errorDetails);
      return;
    }
    rawResults[row] = conversionResult.value().timestamp.toMicros();
  });
}

} // namespace facebook::velox::functions::sparksql
