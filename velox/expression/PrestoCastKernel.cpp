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

#include "velox/expression/PrestoCastKernel.h"

#include "velox/expression/StringWriter.h"

namespace facebook::velox::exec {

PrestoCastKernel::PrestoCastKernel(const core::QueryConfig& config)
    : legacyCast_(config.isLegacyCast()) {}

VectorPtr PrestoCastKernel::castFromDate(
    const SelectivityVector& rows,
    const BaseVector& input,
    exec::EvalCtx& context,
    const TypePtr& toType,
    bool setNullInResultAtError) const {
  const auto* inputFlatVector = input.as<SimpleVector<int32_t>>();

  VectorPtr result;
  initializeResultVector(rows, toType, context, result);

  switch (toType->kind()) {
    case TypeKind::VARCHAR: {
      auto* resultFlatVector = result->as<FlatVector<StringView>>();
      applyToSelectedNoThrowLocal(
          rows, context, result, setNullInResultAtError, [&](int row) {
            // TODO Optimize to avoid creating an intermediate string.
            auto output = DATE()->toString(inputFlatVector->valueAt(row));
            auto writer = exec::StringWriter(resultFlatVector, row);
            writer.resize(output.size());
            ::memcpy(writer.data(), output.data(), output.size());
            writer.finalize();
          });

      return result;
    }
    case TypeKind::TIMESTAMP: {
      static const int64_t kMillisPerDay{86'400'000};
      const auto* timeZone =
          getTimeZoneFromConfig(context.execCtx()->queryCtx()->queryConfig());
      auto* resultFlatVector = result->as<FlatVector<Timestamp>>();
      applyToSelectedNoThrowLocal(
          rows, context, result, setNullInResultAtError, [&](int row) {
            auto timestamp = Timestamp::fromMillis(
                inputFlatVector->valueAt(row) * kMillisPerDay);
            if (timeZone) {
              timestamp.toGMT(*timeZone);
            }
            resultFlatVector->set(row, timestamp);
          });

      return result;
    }
    default:
      VELOX_UNSUPPORTED(
          "Cast from DATE to {} is not supported", toType->toString());
  }
}

VectorPtr PrestoCastKernel::castToDate(
    const SelectivityVector& rows,
    const BaseVector& input,
    exec::EvalCtx& context,
    bool setNullInResultAtError) const {
  const auto& fromType = input.type();

  VectorPtr result;
  initializeResultVector(rows, DATE(), context, result);
  auto* resultFlatVector = result->as<FlatVector<int32_t>>();

  switch (fromType->kind()) {
    case TypeKind::VARCHAR: {
      auto* inputVector = input.as<SimpleVector<StringView>>();
      applyToSelectedNoThrowLocal(
          rows, context, result, setNullInResultAtError, [&](int row) {
            // Cast from string to date allows only complete ISO 8601
            // formatted strings :
            // [+-](YYYY-MM-DD).
            const auto resultValue = util::fromDateString(
                inputVector->valueAt(row), util::ParseMode::kPrestoCast);

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
    case TypeKind::TIMESTAMP: {
      auto* inputVector = input.as<SimpleVector<Timestamp>>();
      const auto* timeZone =
          getTimeZoneFromConfig(context.execCtx()->queryCtx()->queryConfig());
      applyToSelectedNoThrowLocal(
          rows, context, result, setNullInResultAtError, [&](int row) {
            const auto days = util::toDate(inputVector->valueAt(row), timeZone);
            resultFlatVector->set(row, days);
          });

      return result;
    }
    default:
      VELOX_UNSUPPORTED(
          "Cast from {} to DATE is not supported", fromType->toString());
  }
}

} // namespace facebook::velox::exec
