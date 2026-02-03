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

namespace facebook::velox::functions::sparksql {

namespace {
/// Initializes a result vector with the specified type and clears nulls
/// for the selected rows.
inline void initializeResultVector(
    const SelectivityVector& rows,
    const TypePtr& toType,
    exec::EvalCtx& context,
    VectorPtr& result) {
  context.ensureWritable(rows, toType, result);
  result->clearNulls(rows);
}
} // namespace

template <TypeKind ToTypeKind>
std::optional<VectorPtr> SparkCastKernel::castToIntegerImpl(
    const SelectivityVector& rows,
    const BaseVector& input,
    exec::EvalCtx& context,
    bool setNullInResultAtError) const {
  using ToType = typename TypeTraits<ToTypeKind>::ImplType;
  const auto toType = ToType::create();
  const auto& fromType = input.type();

  switch (fromType->kind()) {
    case TypeKind::VARCHAR:
    case TypeKind::VARBINARY: {
      VectorPtr result;
      initializeResultVector(rows, toType, context, result);
      applyStringToIntegerCast<ToTypeKind>(
          rows, input, context, setNullInResultAtError, result);
      return result;
    }
    case TypeKind::TINYINT: {
      if (!allowOverflow_) {
        return std::nullopt;
      }
      VectorPtr result;
      initializeResultVector(rows, toType, context, result);
      applyIntegerToIntegerCast<TypeKind::TINYINT, ToTypeKind>(
          rows, input, context, setNullInResultAtError, result);
      return result;
    }
    case TypeKind::SMALLINT: {
      if (!allowOverflow_) {
        return std::nullopt;
      }
      VectorPtr result;
      initializeResultVector(rows, toType, context, result);
      applyIntegerToIntegerCast<TypeKind::SMALLINT, ToTypeKind>(
          rows, input, context, setNullInResultAtError, result);
      return result;
    }
    case TypeKind::INTEGER: {
      if (!allowOverflow_) {
        return std::nullopt;
      }
      VectorPtr result;
      initializeResultVector(rows, toType, context, result);
      applyIntegerToIntegerCast<TypeKind::INTEGER, ToTypeKind>(
          rows, input, context, setNullInResultAtError, result);
      return result;
    }
    case TypeKind::BIGINT: {
      if (!allowOverflow_) {
        return std::nullopt;
      }
      VectorPtr result;
      initializeResultVector(rows, toType, context, result);
      applyIntegerToIntegerCast<TypeKind::BIGINT, ToTypeKind>(
          rows, input, context, setNullInResultAtError, result);
      return result;
    }
    case TypeKind::REAL: {
      VectorPtr result;
      initializeResultVector(rows, toType, context, result);
      applyFloatingPointToIntegerCast<TypeKind::REAL, ToTypeKind>(
          rows, input, context, setNullInResultAtError, result);
      return result;
    }
    case TypeKind::DOUBLE: {
      VectorPtr result;
      initializeResultVector(rows, toType, context, result);
      applyFloatingPointToIntegerCast<TypeKind::DOUBLE, ToTypeKind>(
          rows, input, context, setNullInResultAtError, result);
      return result;
    }
    default:
      return std::nullopt;
  }
}

// Explicit template instantiations for castToIntegerImpl
template std::optional<VectorPtr>
SparkCastKernel::castToIntegerImpl<TypeKind::TINYINT>(
    const SelectivityVector&,
    const BaseVector&,
    exec::EvalCtx&,
    bool) const;
template std::optional<VectorPtr>
SparkCastKernel::castToIntegerImpl<TypeKind::SMALLINT>(
    const SelectivityVector&,
    const BaseVector&,
    exec::EvalCtx&,
    bool) const;
template std::optional<VectorPtr>
SparkCastKernel::castToIntegerImpl<TypeKind::INTEGER>(
    const SelectivityVector&,
    const BaseVector&,
    exec::EvalCtx&,
    bool) const;
template std::optional<VectorPtr>
SparkCastKernel::castToIntegerImpl<TypeKind::BIGINT>(
    const SelectivityVector&,
    const BaseVector&,
    exec::EvalCtx&,
    bool) const;
template std::optional<VectorPtr>
SparkCastKernel::castToIntegerImpl<TypeKind::HUGEINT>(
    const SelectivityVector&,
    const BaseVector&,
    exec::EvalCtx&,
    bool) const;

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
      initializeResultVector(rows, BOOLEAN(), context, result);
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
      initializeResultVector(rows, BOOLEAN(), context, result);

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
      initializeResultVector(rows, BOOLEAN(), context, result);

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
      initializeResultVector(rows, BOOLEAN(), context, result);

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
      initializeResultVector(rows, BOOLEAN(), context, result);

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
      initializeResultVector(rows, BOOLEAN(), context, result);

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
      initializeResultVector(rows, BOOLEAN(), context, result);

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

  // Handle TIMESTAMP specially
  if (fromType->kind() == TypeKind::TIMESTAMP) {
    VectorPtr result;
    context.ensureWritable(rows, TINYINT(), result);
    result->clearNulls(rows);
    applyTimestampToIntegerCast<TypeKind::TINYINT>(
        rows, input, context, setNullInResultAtError, result);
    return result;
  }

  // Use template function for other source types
  auto result = castToIntegerImpl<TypeKind::TINYINT>(
      rows, input, context, setNullInResultAtError);
  if (result.has_value()) {
    return result.value();
  }

  // Fallback to parent implementation
  return exec::PrestoCastKernel::castToTinyInt(
      rows, input, context, setNullInResultAtError);
}

VectorPtr SparkCastKernel::castToSmallInt(
    const SelectivityVector& rows,
    const BaseVector& input,
    exec::EvalCtx& context,
    bool setNullInResultAtError) const {
  const auto& fromType = input.type();

  // Handle TIMESTAMP specially
  if (fromType->kind() == TypeKind::TIMESTAMP) {
    VectorPtr result;
    context.ensureWritable(rows, SMALLINT(), result);
    result->clearNulls(rows);
    applyTimestampToIntegerCast<TypeKind::SMALLINT>(
        rows, input, context, setNullInResultAtError, result);
    return result;
  }

  // Use template function for other source types
  auto result = castToIntegerImpl<TypeKind::SMALLINT>(
      rows, input, context, setNullInResultAtError);
  if (result.has_value()) {
    return result.value();
  }

  // Fallback to parent implementation
  return exec::PrestoCastKernel::castToSmallInt(
      rows, input, context, setNullInResultAtError);
}

VectorPtr SparkCastKernel::castToInteger(
    const SelectivityVector& rows,
    const BaseVector& input,
    exec::EvalCtx& context,
    bool setNullInResultAtError) const {
  const auto& fromType = input.type();

  // Handle TIMESTAMP specially
  if (fromType->kind() == TypeKind::TIMESTAMP) {
    VectorPtr result;
    context.ensureWritable(rows, INTEGER(), result);
    result->clearNulls(rows);
    applyTimestampToIntegerCast<TypeKind::INTEGER>(
        rows, input, context, setNullInResultAtError, result);
    return result;
  }

  // Use template function for other source types
  auto result = castToIntegerImpl<TypeKind::INTEGER>(
      rows, input, context, setNullInResultAtError);
  if (result.has_value()) {
    return result.value();
  }

  // Fallback to parent implementation
  return exec::PrestoCastKernel::castToInteger(
      rows, input, context, setNullInResultAtError);
}

VectorPtr SparkCastKernel::castToBigInt(
    const SelectivityVector& rows,
    const BaseVector& input,
    exec::EvalCtx& context,
    bool setNullInResultAtError) const {
  const auto& fromType = input.type();

  // Handle TIMESTAMP specially
  if (fromType->kind() == TypeKind::TIMESTAMP) {
    VectorPtr result;
    context.ensureWritable(rows, BIGINT(), result);
    result->clearNulls(rows);
    applyTimestampToIntegerCast<TypeKind::BIGINT>(
        rows, input, context, setNullInResultAtError, result);
    return result;
  }

  // Use template function for other source types
  auto result = castToIntegerImpl<TypeKind::BIGINT>(
      rows, input, context, setNullInResultAtError);
  if (result.has_value()) {
    return result.value();
  }

  // Fallback to parent implementation
  return exec::PrestoCastKernel::castToBigInt(
      rows, input, context, setNullInResultAtError);
}

VectorPtr SparkCastKernel::castToHugeInt(
    const SelectivityVector& rows,
    const BaseVector& input,
    exec::EvalCtx& context,
    bool setNullInResultAtError) const {
  // Use template function for source types
  auto result = castToIntegerImpl<TypeKind::HUGEINT>(
      rows, input, context, setNullInResultAtError);
  if (result.has_value()) {
    return result.value();
  }

  // Fallback to parent implementation
  return exec::PrestoCastKernel::castToHugeInt(
      rows, input, context, setNullInResultAtError);
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
      initializeResultVector(rows, REAL(), context, result);

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
      initializeResultVector(rows, REAL(), context, result);

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
      initializeResultVector(rows, DOUBLE(), context, result);

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
      initializeResultVector(rows, TIMESTAMP(), context, result);

      applyNumberToTimestampCast<TypeKind::TINYINT>(
          rows, input, context, setNullInResultAtError, result);

      return result;
    }
    case TypeKind::SMALLINT: {
      VectorPtr result;
      initializeResultVector(rows, TIMESTAMP(), context, result);

      applyNumberToTimestampCast<TypeKind::SMALLINT>(
          rows, input, context, setNullInResultAtError, result);

      return result;
    }
    case TypeKind::INTEGER: {
      VectorPtr result;
      initializeResultVector(rows, TIMESTAMP(), context, result);

      applyNumberToTimestampCast<TypeKind::INTEGER>(
          rows, input, context, setNullInResultAtError, result);

      return result;
    }
    case TypeKind::BIGINT: {
      VectorPtr result;
      initializeResultVector(rows, TIMESTAMP(), context, result);

      applyNumberToTimestampCast<TypeKind::BIGINT>(
          rows, input, context, setNullInResultAtError, result);

      return result;
    }
    case TypeKind::REAL: {
      VectorPtr result;
      initializeResultVector(rows, TIMESTAMP(), context, result);

      applyNumberToTimestampCast<TypeKind::REAL>(
          rows, input, context, setNullInResultAtError, result);

      return result;
    }
    case TypeKind::DOUBLE: {
      VectorPtr result;
      initializeResultVector(rows, TIMESTAMP(), context, result);

      applyNumberToTimestampCast<TypeKind::DOUBLE>(
          rows, input, context, setNullInResultAtError, result);

      return result;
    }
    case TypeKind::BOOLEAN: {
      VectorPtr result;
      initializeResultVector(rows, TIMESTAMP(), context, result);

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
      initializeResultVector(rows, TIMESTAMP(), context, result);

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
