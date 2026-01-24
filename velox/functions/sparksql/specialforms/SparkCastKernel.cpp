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

#define CAST_TO_INTEGER_CASES(TO_TYPE_KIND, TO_TYPE, PRESTO_CAST_FUNC) \
  case TypeKind::VARCHAR:                                              \
  case TypeKind::VARBINARY: {                                          \
    VectorPtr result;                                                  \
    context.ensureWritable(rows, TO_TYPE, result);                     \
    (*result).clearNulls(rows);                                        \
                                                                       \
    applyStringToIntegerCast<TO_TYPE_KIND>(                            \
        rows, input, context, setNullInResultAtError, result);         \
                                                                       \
    return result;                                                     \
  }                                                                    \
  case TypeKind::TINYINT: {                                            \
    if (!allowOverflow_) {                                             \
      return exec::PrestoCastKernel::PRESTO_CAST_FUNC(                 \
          rows, input, context, setNullInResultAtError);               \
    }                                                                  \
                                                                       \
    VectorPtr result;                                                  \
    context.ensureWritable(rows, TO_TYPE, result);                     \
    (*result).clearNulls(rows);                                        \
                                                                       \
    applyIntegerToIntegerCast<TypeKind::TINYINT, TO_TYPE_KIND>(        \
        rows, input, context, setNullInResultAtError, result);         \
                                                                       \
    return result;                                                     \
  }                                                                    \
  case TypeKind::SMALLINT: {                                           \
    if (!allowOverflow_) {                                             \
      return exec::PrestoCastKernel::PRESTO_CAST_FUNC(                 \
          rows, input, context, setNullInResultAtError);               \
    }                                                                  \
                                                                       \
    VectorPtr result;                                                  \
    context.ensureWritable(rows, TO_TYPE, result);                     \
    (*result).clearNulls(rows);                                        \
                                                                       \
    applyIntegerToIntegerCast<TypeKind::SMALLINT, TO_TYPE_KIND>(       \
        rows, input, context, setNullInResultAtError, result);         \
                                                                       \
    return result;                                                     \
  }                                                                    \
  case TypeKind::INTEGER: {                                            \
    if (!allowOverflow_) {                                             \
      return exec::PrestoCastKernel::PRESTO_CAST_FUNC(                 \
          rows, input, context, setNullInResultAtError);               \
    }                                                                  \
                                                                       \
    VectorPtr result;                                                  \
    context.ensureWritable(rows, TO_TYPE, result);                     \
    (*result).clearNulls(rows);                                        \
                                                                       \
    applyIntegerToIntegerCast<TypeKind::INTEGER, TO_TYPE_KIND>(        \
        rows, input, context, setNullInResultAtError, result);         \
                                                                       \
    return result;                                                     \
  }                                                                    \
  case TypeKind::BIGINT: {                                             \
    if (!allowOverflow_) {                                             \
      return exec::PrestoCastKernel::PRESTO_CAST_FUNC(                 \
          rows, input, context, setNullInResultAtError);               \
    }                                                                  \
                                                                       \
    VectorPtr result;                                                  \
    context.ensureWritable(rows, TO_TYPE, result);                     \
    (*result).clearNulls(rows);                                        \
                                                                       \
    applyIntegerToIntegerCast<TypeKind::BIGINT, TO_TYPE_KIND>(         \
        rows, input, context, setNullInResultAtError, result);         \
                                                                       \
    return result;                                                     \
  }                                                                    \
  case TypeKind::REAL: {                                               \
    VectorPtr result;                                                  \
    context.ensureWritable(rows, TO_TYPE, result);                     \
    (*result).clearNulls(rows);                                        \
                                                                       \
    applyFloatingPointToIntegerCast<TypeKind::REAL, TO_TYPE_KIND>(     \
        rows, input, context, setNullInResultAtError, result);         \
                                                                       \
    return result;                                                     \
  }                                                                    \
  case TypeKind::DOUBLE: {                                             \
    VectorPtr result;                                                  \
    context.ensureWritable(rows, TO_TYPE, result);                     \
    (*result).clearNulls(rows);                                        \
                                                                       \
    applyFloatingPointToIntegerCast<TypeKind::DOUBLE, TO_TYPE_KIND>(   \
        rows, input, context, setNullInResultAtError, result);         \
                                                                       \
    return result;                                                     \
  }                                                                    \
  default:                                                             \
    return exec::PrestoCastKernel::PRESTO_CAST_FUNC(                   \
        rows, input, context, setNullInResultAtError);

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
      context.ensureWritable(rows, DATE(), result);
      (*result).clearNulls(rows);
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
              if (setNullInResultAtError) {
                resultFlatVector->setNull(row, true);
              } else if (context.captureErrorDetails()) {
                context.setStatus(
                    row,
                    Status::UserError(
                        "{} {}",
                        makeErrorMessage(input, row, DATE()),
                        resultValue.error().message()));
              } else {
                context.setStatus(row, Status::UserError());
              }
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
      context.ensureWritable(rows, toType, result);
      (*result).clearNulls(rows);

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

VectorPtr SparkCastKernel::castToBoolean(
    const SelectivityVector& rows,
    const BaseVector& input,
    exec::EvalCtx& context,
    bool setNullInResultAtError) const {
  const auto& fromType = input.type();

  switch (fromType->kind()) {
    case TypeKind::VARCHAR:
    case TypeKind::VARBINARY: {
      VectorPtr result;
      context.ensureWritable(rows, BOOLEAN(), result);
      (*result).clearNulls(rows);
      auto* resultFlatVector = result->as<FlatVector<bool>>();

      const auto simpleInput = input.as<SimpleVector<StringView>>();

      applyToSelectedNoThrowLocal(
          rows,
          context,
          result,
          setNullInResultAtError,
          [&](vector_size_t row) {
            const auto& TU = static_cast<int (*)(int)>(std::toupper);

            StringView inputStr = simpleInput->valueAt(row);
            inputStr = removeWhiteSpaces(inputStr);
            const auto len = inputStr.size();
            const auto data = inputStr.data();

            if (len == 1) {
              auto character = TU(data[0]);
              if (character == 'T' || character == '1' || character == 'Y') {
                resultFlatVector->set(row, true);

                return;
              }
              if (character == 'F' || character == '0' || character == 'N') {
                resultFlatVector->set(row, false);

                return;
              }
            }

            // Case-insensitive 'true'.
            if ((len == 4) && (TU(data[0]) == 'T') && (TU(data[1]) == 'R') &&
                (TU(data[2]) == 'U') && (TU(data[3]) == 'E')) {
              resultFlatVector->set(row, true);

              return;
            }

            // Case-insensitive 'false'.
            if ((len == 5) && (TU(data[0]) == 'F') && (TU(data[1]) == 'A') &&
                (TU(data[2]) == 'L') && (TU(data[3]) == 'S') &&
                (TU(data[4]) == 'E')) {
              resultFlatVector->set(row, false);

              return;
            }

            // Case-insensitive 'yes'.
            if ((len == 3) && (TU(data[0]) == 'Y') && (TU(data[1]) == 'E') &&
                (TU(data[2]) == 'S')) {
              resultFlatVector->set(row, true);

              return;
            }

            // Case-insensitive 'no'.
            if ((len == 2) && (TU(data[0]) == 'N') && (TU(data[1]) == 'O')) {
              resultFlatVector->set(row, false);

              return;
            }

            if (setNullInResultAtError) {
              resultFlatVector->setNull(row, true);
            } else if (context.captureErrorDetails()) {
              context.setStatus(
                  row,
                  Status::UserError(
                      "{} Cannot cast {} to BOOLEAN",
                      makeErrorMessage(input, row, BOOLEAN()),
                      std::string_view(data, len)));
            } else {
              context.setStatus(row, Status::UserError());
            }
          });

      return result;
    }
    case TypeKind::TINYINT: {
      if (!allowOverflow_) {
        return exec::PrestoCastKernel::castToBoolean(
            rows, input, context, setNullInResultAtError);
      }

      VectorPtr result;
      context.ensureWritable(rows, BOOLEAN(), result);
      (*result).clearNulls(rows);

      applyIntegerToBooleanCast<TypeKind::TINYINT>(
          rows, input, context, setNullInResultAtError, result);

      return result;
    }
    case TypeKind::SMALLINT: {
      if (!allowOverflow_) {
        return exec::PrestoCastKernel::castToBoolean(
            rows, input, context, setNullInResultAtError);
      }

      VectorPtr result;
      context.ensureWritable(rows, BOOLEAN(), result);
      (*result).clearNulls(rows);

      applyIntegerToBooleanCast<TypeKind::SMALLINT>(
          rows, input, context, setNullInResultAtError, result);

      return result;
    }
    case TypeKind::INTEGER: {
      if (!allowOverflow_) {
        return exec::PrestoCastKernel::castToBoolean(
            rows, input, context, setNullInResultAtError);
      }

      VectorPtr result;
      context.ensureWritable(rows, BOOLEAN(), result);
      (*result).clearNulls(rows);

      applyIntegerToBooleanCast<TypeKind::INTEGER>(
          rows, input, context, setNullInResultAtError, result);

      return result;
    }
    case TypeKind::BIGINT: {
      if (!allowOverflow_) {
        return exec::PrestoCastKernel::castToBoolean(
            rows, input, context, setNullInResultAtError);
      }

      VectorPtr result;
      context.ensureWritable(rows, BOOLEAN(), result);
      (*result).clearNulls(rows);

      applyIntegerToBooleanCast<TypeKind::BIGINT>(
          rows, input, context, setNullInResultAtError, result);

      return result;
    }
    case TypeKind::HUGEINT: {
      if (!allowOverflow_) {
        return exec::PrestoCastKernel::castToBoolean(
            rows, input, context, setNullInResultAtError);
      }

      if (setNullInResultAtError) {
        return BaseVector::createNullConstant(
            BOOLEAN(), rows.end(), context.pool());
      }

      context.setStatuses(
          rows, Status::UserError("Conversion to BOOLEAN is not supported"));

      return VectorPtr();
    }
    case TypeKind::REAL: {
      if (!allowOverflow_) {
        return exec::PrestoCastKernel::castToBoolean(
            rows, input, context, setNullInResultAtError);
      }

      VectorPtr result;
      context.ensureWritable(rows, BOOLEAN(), result);
      (*result).clearNulls(rows);

      applyFloatingPointToBooleanCast<TypeKind::REAL>(
          rows, input, context, setNullInResultAtError, result);

      return result;
    }
    case TypeKind::DOUBLE: {
      if (!allowOverflow_) {
        return exec::PrestoCastKernel::castToBoolean(
            rows, input, context, setNullInResultAtError);
      }

      VectorPtr result;
      context.ensureWritable(rows, BOOLEAN(), result);
      (*result).clearNulls(rows);

      applyFloatingPointToBooleanCast<TypeKind::DOUBLE>(
          rows, input, context, setNullInResultAtError, result);

      return result;
    }
    default:
      return exec::PrestoCastKernel::castToBoolean(
          rows, input, context, setNullInResultAtError);
  }
}

VectorPtr SparkCastKernel::castToTinyInt(
    const SelectivityVector& rows,
    const BaseVector& input,
    exec::EvalCtx& context,
    bool setNullInResultAtError) const {
  const auto& fromType = input.type();

  switch (fromType->kind()) {
    case TypeKind::TIMESTAMP: {
      VectorPtr result;
      context.ensureWritable(rows, TINYINT(), result);
      (*result).clearNulls(rows);

      applyTimestampToIntegerCast<TypeKind::TINYINT>(
          rows, input, context, setNullInResultAtError, result);

      return result;
    }
      CAST_TO_INTEGER_CASES(TypeKind::TINYINT, TINYINT(), castToTinyInt)
  }
}

VectorPtr SparkCastKernel::castToSmallInt(
    const SelectivityVector& rows,
    const BaseVector& input,
    exec::EvalCtx& context,
    bool setNullInResultAtError) const {
  const auto& fromType = input.type();

  switch (fromType->kind()) {
    case TypeKind::TIMESTAMP: {
      VectorPtr result;
      context.ensureWritable(rows, SMALLINT(), result);
      (*result).clearNulls(rows);

      applyTimestampToIntegerCast<TypeKind::SMALLINT>(
          rows, input, context, setNullInResultAtError, result);

      return result;
    }
      CAST_TO_INTEGER_CASES(TypeKind::SMALLINT, SMALLINT(), castToSmallInt)
  }
}

VectorPtr SparkCastKernel::castToInteger(
    const SelectivityVector& rows,
    const BaseVector& input,
    exec::EvalCtx& context,
    bool setNullInResultAtError) const {
  const auto& fromType = input.type();

  switch (fromType->kind()) {
    case TypeKind::TIMESTAMP: {
      VectorPtr result;
      context.ensureWritable(rows, INTEGER(), result);
      (*result).clearNulls(rows);

      applyTimestampToIntegerCast<TypeKind::INTEGER>(
          rows, input, context, setNullInResultAtError, result);

      return result;
    }
      CAST_TO_INTEGER_CASES(TypeKind::INTEGER, INTEGER(), castToInteger)
  }
}

VectorPtr SparkCastKernel::castToBigInt(
    const SelectivityVector& rows,
    const BaseVector& input,
    exec::EvalCtx& context,
    bool setNullInResultAtError) const {
  const auto& fromType = input.type();

  VectorPtr result;

  switch (fromType->kind()) {
    case TypeKind::TIMESTAMP: {
      context.ensureWritable(rows, BIGINT(), result);
      (*result).clearNulls(rows);

      applyTimestampToIntegerCast<TypeKind::BIGINT>(
          rows, input, context, setNullInResultAtError, result);

      return result;
    }
      CAST_TO_INTEGER_CASES(TypeKind::BIGINT, BIGINT(), castToBigInt)
  }
}

VectorPtr SparkCastKernel::castToHugeInt(
    const SelectivityVector& rows,
    const BaseVector& input,
    exec::EvalCtx& context,
    bool setNullInResultAtError) const {
  const auto& fromType = input.type();

  switch (fromType->kind()) {
    CAST_TO_INTEGER_CASES(TypeKind::HUGEINT, HUGEINT(), castToHugeInt)
  }
}

VectorPtr SparkCastKernel::castToReal(
    const SelectivityVector& rows,
    const BaseVector& input,
    exec::EvalCtx& context,
    bool setNullInResultAtError) const {
  const auto& fromType = input.type();

  switch (fromType->kind()) {
    case TypeKind::VARCHAR:
    case TypeKind::VARBINARY: {
      VectorPtr result;
      context.ensureWritable(rows, REAL(), result);
      (*result).clearNulls(rows);

      applyStringToFloatingPointCast<TypeKind::REAL>(
          rows, input, context, setNullInResultAtError, result);

      return result;
    }
    case TypeKind::DOUBLE: {
      if (!allowOverflow_) {
        return exec::PrestoCastKernel::castToReal(
            rows, input, context, setNullInResultAtError);
      }

      VectorPtr result;
      context.ensureWritable(rows, REAL(), result);
      (*result).clearNulls(rows);

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
          rows, input, context, setNullInResultAtError);
  }
}

VectorPtr SparkCastKernel::castToDouble(
    const SelectivityVector& rows,
    const BaseVector& input,
    exec::EvalCtx& context,
    bool setNullInResultAtError) const {
  const auto& fromType = input.type();

  switch (fromType->kind()) {
    case TypeKind::VARCHAR:
    case TypeKind::VARBINARY: {
      VectorPtr result;
      context.ensureWritable(rows, DOUBLE(), result);
      (*result).clearNulls(rows);

      applyStringToFloatingPointCast<TypeKind::DOUBLE>(
          rows, input, context, setNullInResultAtError, result);

      return result;
    }
    default:
      return exec::PrestoCastKernel::castToDouble(
          rows, input, context, setNullInResultAtError);
  }
}

StringView SparkCastKernel::removeWhiteSpaces(const StringView& view) const {
  StringView output;
  stringImpl::trimUnicodeWhiteSpace<true, true, StringView, StringView>(
      output, view);
  return output;
}

VectorPtr SparkCastKernel::castToTimestamp(
    const SelectivityVector& rows,
    const BaseVector& input,
    exec::EvalCtx& context,
    bool setNullInResultAtError) const {
  const auto& fromType = input.type();

  switch (fromType->kind()) {
    case TypeKind::TINYINT: {
      VectorPtr result;
      context.ensureWritable(rows, TIMESTAMP(), result);
      (*result).clearNulls(rows);

      applyNumberToTimestampCast<TypeKind::TINYINT>(
          rows, input, context, setNullInResultAtError, result);

      return result;
    }
    case TypeKind::SMALLINT: {
      VectorPtr result;
      context.ensureWritable(rows, TIMESTAMP(), result);
      (*result).clearNulls(rows);

      applyNumberToTimestampCast<TypeKind::SMALLINT>(
          rows, input, context, setNullInResultAtError, result);

      return result;
    }
    case TypeKind::INTEGER: {
      VectorPtr result;
      context.ensureWritable(rows, TIMESTAMP(), result);
      (*result).clearNulls(rows);

      applyNumberToTimestampCast<TypeKind::INTEGER>(
          rows, input, context, setNullInResultAtError, result);

      return result;
    }
    case TypeKind::BIGINT: {
      VectorPtr result;
      context.ensureWritable(rows, TIMESTAMP(), result);
      (*result).clearNulls(rows);

      applyNumberToTimestampCast<TypeKind::BIGINT>(
          rows, input, context, setNullInResultAtError, result);

      return result;
    }
    case TypeKind::REAL: {
      VectorPtr result;
      context.ensureWritable(rows, TIMESTAMP(), result);
      (*result).clearNulls(rows);

      applyNumberToTimestampCast<TypeKind::REAL>(
          rows, input, context, setNullInResultAtError, result);

      return result;
    }
    case TypeKind::DOUBLE: {
      VectorPtr result;
      context.ensureWritable(rows, TIMESTAMP(), result);
      (*result).clearNulls(rows);

      applyNumberToTimestampCast<TypeKind::DOUBLE>(
          rows, input, context, setNullInResultAtError, result);

      return result;
    }
    case TypeKind::BOOLEAN: {
      VectorPtr result;
      context.ensureWritable(rows, TIMESTAMP(), result);
      (*result).clearNulls(rows);

      const auto simpleInput = input.as<SimpleVector<bool>>();
      auto* resultFlatVector = result->as<FlatVector<Timestamp>>();

      applyToSelectedNoThrowLocal(
          rows,
          context,
          result,
          setNullInResultAtError,
          [&](vector_size_t row) {
            resultFlatVector->set(
                row,
                Timestamp::fromMicrosNoError(
                    simpleInput->valueAt(row) ? 1 : 0));
          });

      return result;
    }
    case TypeKind::VARCHAR:
    case TypeKind::VARBINARY: {
      VectorPtr result;
      context.ensureWritable(rows, TIMESTAMP(), result);
      (*result).clearNulls(rows);

      const auto simpleInput = input.as<SimpleVector<StringView>>();
      auto* resultFlatVector = result->as<FlatVector<Timestamp>>();
      auto sessionTimezone = config_.sessionTimezone().empty()
          ? nullptr
          : tz::locateZone(config_.sessionTimezone());
      applyToSelectedNoThrowLocal(
          rows,
          context,
          result,
          setNullInResultAtError,
          [&](vector_size_t row) {
            StringView inputStr = simpleInput->valueAt(row);
            inputStr = removeWhiteSpaces(inputStr);

            auto conversionResult = util::fromTimestampWithTimezoneString(
                inputStr.data(),
                inputStr.size(),
                util::TimestampParseMode::kSparkCast);
            if (conversionResult.hasError()) {
              if (setNullInResultAtError) {
                resultFlatVector->setNull(row, true);
              } else if (context.captureErrorDetails()) {
                const auto errorDetails = makeErrorMessage(
                    input,
                    row,
                    result->type(),
                    conversionResult.error().message());
                context.setStatus(row, Status::UserError("{}", errorDetails));
              } else {
                context.setStatus(row, Status::UserError());
              }

              return;
            }

            resultFlatVector->set(
                row,
                util::fromParsedTimestampWithTimeZone(
                    conversionResult.value(), sessionTimezone));
          });

      return result;
    }
    default:
      return exec::PrestoCastKernel::castToTimestamp(
          rows, input, context, setNullInResultAtError);
  }
}
} // namespace facebook::velox::functions::sparksql
