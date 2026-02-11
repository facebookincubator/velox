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

#define VELOX_DYNAMIC_DECIMAL_TYPE_DISPATCH(       \
    TEMPLATE_FUNC, decimalTypePtr, ...)            \
  [&]() {                                          \
    if (decimalTypePtr->isLongDecimal()) {         \
      return TEMPLATE_FUNC<int128_t>(__VA_ARGS__); \
    } else {                                       \
      return TEMPLATE_FUNC<int64_t>(__VA_ARGS__);  \
    }                                              \
  }()

#define VELOX_DYNAMIC_DECIMAL_TEMPLATE_TYPE_DISPATCH( \
    TEMPLATE_FUNC, decimalTypePtr, T, ...)            \
  [&]() {                                             \
    if (decimalTypePtr->isLongDecimal()) {            \
      return TEMPLATE_FUNC<int128_t, T>(__VA_ARGS__); \
    } else {                                          \
      return TEMPLATE_FUNC<int64_t, T>(__VA_ARGS__);  \
    }                                                 \
  }()

namespace facebook::velox::functions::sparksql {
SparkCastKernel::SparkCastKernel(
    const velox::core::QueryConfig& config,
    bool allowOverflow)
    : exec::PrestoCastKernel(config),
      config_(config),
      allowOverflow_(allowOverflow) {
  const auto sessionTzName = config.sessionTimezone();
  if (!sessionTzName.empty()) {
    timestampToStringOptions_.timeZone = tz::locateZone(sessionTzName);
  }
}

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

VectorPtr SparkCastKernel::castFromDecimal(
    const SelectivityVector& rows,
    const BaseVector& input,
    exec::EvalCtx& context,
    const TypePtr& toType,
    bool setNullInResultAtError) const {
  const auto& fromType = input.type();

  switch (toType->kind()) {
    case TypeKind::TINYINT:
      return VELOX_DYNAMIC_DECIMAL_TEMPLATE_TYPE_DISPATCH(
          applyDecimalToIntegralCast,
          fromType,
          TypeKind::TINYINT,
          rows,
          input,
          context,
          fromType,
          toType,
          setNullInResultAtError);
    case TypeKind::SMALLINT:
      return VELOX_DYNAMIC_DECIMAL_TEMPLATE_TYPE_DISPATCH(
          applyDecimalToIntegralCast,
          fromType,
          TypeKind::SMALLINT,
          rows,
          input,
          context,
          fromType,
          toType,
          setNullInResultAtError);
    case TypeKind::INTEGER:
      return VELOX_DYNAMIC_DECIMAL_TEMPLATE_TYPE_DISPATCH(
          applyDecimalToIntegralCast,
          fromType,
          TypeKind::INTEGER,
          rows,
          input,
          context,
          fromType,
          toType,
          setNullInResultAtError);
    case TypeKind::BIGINT:
      return VELOX_DYNAMIC_DECIMAL_TEMPLATE_TYPE_DISPATCH(
          applyDecimalToIntegralCast,
          fromType,
          TypeKind::BIGINT,
          rows,
          input,
          context,
          fromType,
          toType,
          setNullInResultAtError);
    default:
      return exec::PrestoCastKernel::castFromDecimal(
          rows, input, context, toType, setNullInResultAtError);
  }
}

VectorPtr SparkCastKernel::castToDecimal(
    const SelectivityVector& rows,
    const BaseVector& input,
    exec::EvalCtx& context,
    const TypePtr& toType,
    bool setNullInResultAtError) const {
  const auto& fromType = input.type();

  switch (fromType->kind()) {
    case TypeKind::VARCHAR: {
      VectorPtr result;
      initializeResultVector(rows, toType, context, result);

      VELOX_DYNAMIC_DECIMAL_TYPE_DISPATCH(
          applyVarcharToDecimalCast,
          toType,
          rows,
          input,
          context,
          toType,
          setNullInResultAtError,
          result);

      return result;
    }
    default:
      return exec::PrestoCastKernel::castToDecimal(
          rows, input, context, toType, setNullInResultAtError);
  }
}

VectorPtr SparkCastKernel::castToTinyInt(
    const SelectivityVector& rows,
    const BaseVector& input,
    exec::EvalCtx& context,
    const TypePtr& toType,
    bool setNullInResultAtError) const {
  const auto& fromType = input.type();

  if (fromType->kind() == TypeKind::TIMESTAMP) {
    VectorPtr result;
    context.ensureWritable(rows, toType, result);
    result->clearNulls(rows);
    applyTimestampToIntegerCast<TypeKind::TINYINT>(
        rows, input, context, setNullInResultAtError, result);
    return result;
  }

  return VELOX_DYNAMIC_TEMPLATE_TYPE_DISPATCH(
      castToIntegerImpl,
      TypeKind::TINYINT,
      fromType->kind(),
      rows,
      input,
      context,
      toType,
      setNullInResultAtError);
}

VectorPtr SparkCastKernel::castToSmallInt(
    const SelectivityVector& rows,
    const BaseVector& input,
    exec::EvalCtx& context,
    const TypePtr& toType,
    bool setNullInResultAtError) const {
  const auto& fromType = input.type();

  if (fromType->kind() == TypeKind::TIMESTAMP) {
    VectorPtr result;
    context.ensureWritable(rows, toType, result);
    result->clearNulls(rows);
    applyTimestampToIntegerCast<TypeKind::SMALLINT>(
        rows, input, context, setNullInResultAtError, result);
    return result;
  }

  return VELOX_DYNAMIC_TEMPLATE_TYPE_DISPATCH(
      castToIntegerImpl,
      TypeKind::SMALLINT,
      fromType->kind(),
      rows,
      input,
      context,
      toType,
      setNullInResultAtError);
}

VectorPtr SparkCastKernel::castToInteger(
    const SelectivityVector& rows,
    const BaseVector& input,
    exec::EvalCtx& context,
    const TypePtr& toType,
    bool setNullInResultAtError) const {
  const auto& fromType = input.type();

  if (fromType->kind() == TypeKind::TIMESTAMP) {
    VectorPtr result;
    context.ensureWritable(rows, toType, result);
    result->clearNulls(rows);
    applyTimestampToIntegerCast<TypeKind::INTEGER>(
        rows, input, context, setNullInResultAtError, result);
    return result;
  }

  return VELOX_DYNAMIC_TEMPLATE_TYPE_DISPATCH(
      castToIntegerImpl,
      TypeKind::INTEGER,
      fromType->kind(),
      rows,
      input,
      context,
      toType,
      setNullInResultAtError);
}

VectorPtr SparkCastKernel::castToBigInt(
    const SelectivityVector& rows,
    const BaseVector& input,
    exec::EvalCtx& context,
    const TypePtr& toType,
    bool setNullInResultAtError) const {
  const auto& fromType = input.type();

  if (fromType->kind() == TypeKind::TIMESTAMP) {
    VectorPtr result;
    context.ensureWritable(rows, toType, result);
    result->clearNulls(rows);
    applyTimestampToIntegerCast<TypeKind::BIGINT>(
        rows, input, context, setNullInResultAtError, result);
    return result;
  }

  return VELOX_DYNAMIC_TEMPLATE_TYPE_DISPATCH(
      castToIntegerImpl,
      TypeKind::BIGINT,
      fromType->kind(),
      rows,
      input,
      context,
      toType,
      setNullInResultAtError);
}

VectorPtr SparkCastKernel::castToHugeInt(
    const SelectivityVector& rows,
    const BaseVector& input,
    exec::EvalCtx& context,
    const TypePtr& toType,
    bool setNullInResultAtError) const {
  const auto& fromType = input.type();

  return VELOX_DYNAMIC_TEMPLATE_TYPE_DISPATCH(
      castToIntegerImpl,
      TypeKind::HUGEINT,
      fromType->kind(),
      rows,
      input,
      context,
      toType,
      setNullInResultAtError);
}

VectorPtr SparkCastKernel::castToReal(
    const SelectivityVector& rows,
    const BaseVector& input,
    exec::EvalCtx& context,
    const TypePtr& toType,
    bool setNullInResultAtError) const {
  const auto& fromType = input.type();

  switch (fromType->kind()) {
    case TypeKind::VARCHAR:
    case TypeKind::VARBINARY: {
      VectorPtr result;
      initializeResultVector(rows, toType, context, result);

      applyStringToFloatingPointCast<TypeKind::REAL>(
          rows, input, context, setNullInResultAtError, result);

      return result;
    }
    case TypeKind::DOUBLE: {
      if (!allowOverflow_) {
        return exec::PrestoCastKernel::castToReal(
            rows, input, context, toType, setNullInResultAtError);
      }

      VectorPtr result;
      initializeResultVector(rows, toType, context, result);

      const auto simpleInput = input.as<SimpleVector<double>>();
      auto* resultFlatVector = result->as<FlatVector<float>>();

      applyToSelectedNoThrowLocal(
          rows,
          context,
          result,
          setNullInResultAtError,
          [&](vector_size_t row) {
            resultFlatVector->set(row, simpleInput->valueAt(row));
          });

      return result;
    }
    default:
      return exec::PrestoCastKernel::castToReal(
          rows, input, context, toType, setNullInResultAtError);
  }
}

VectorPtr SparkCastKernel::castToDouble(
    const SelectivityVector& rows,
    const BaseVector& input,
    exec::EvalCtx& context,
    const TypePtr& toType,
    bool setNullInResultAtError) const {
  const auto& fromType = input.type();

  switch (fromType->kind()) {
    case TypeKind::VARCHAR:
    case TypeKind::VARBINARY: {
      VectorPtr result;
      initializeResultVector(rows, toType, context, result);

      applyStringToFloatingPointCast<TypeKind::DOUBLE>(
          rows, input, context, setNullInResultAtError, result);

      return result;
    }
    default:
      return exec::PrestoCastKernel::castToDouble(
          rows, input, context, toType, setNullInResultAtError);
  }
}

StringView SparkCastKernel::removeWhiteSpaces(const StringView& view) const {
  StringView output;
  stringImpl::trimUnicodeWhiteSpace<true, true, StringView, StringView>(
      output, view);
  return output;
}
} // namespace facebook::velox::functions::sparksql
