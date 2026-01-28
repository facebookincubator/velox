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

namespace facebook::velox::exec {

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

PrestoCastKernel::PrestoCastKernel(const core::QueryConfig& config)
    : legacyCast_(config.isLegacyCast()) {
  timestampToStringOptions_ = TimestampToStringOptions{
      .precision = TimestampToStringOptions::Precision::kMilliseconds};
  if (!legacyCast_) {
    timestampToStringOptions_.zeroPaddingYear = true;
    timestampToStringOptions_.dateTimeSeparator = ' ';
    const auto sessionTzName = config.sessionTimezone();
    if (config.adjustTimestampToTimezone() && !sessionTzName.empty()) {
      timestampToStringOptions_.timeZone = tz::locateZone(sessionTzName);
    }
  }
}

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

VectorPtr PrestoCastKernel::castFromIntervalDayTime(
    const SelectivityVector& rows,
    const BaseVector& input,
    exec::EvalCtx& context,
    const TypePtr& toType,
    bool setNullInResultAtError) const {
  auto* inputFlatVector = input.as<SimpleVector<int64_t>>();

  VectorPtr result;
  initializeResultVector(rows, toType, context, result);

  switch (toType->kind()) {
    case TypeKind::VARCHAR: {
      auto* resultFlatVector = result->as<FlatVector<StringView>>();
      applyToSelectedNoThrowLocal(
          rows, context, result, setNullInResultAtError, [&](int row) {
            // TODO Optimize to avoid creating an intermediate string.
            auto output = INTERVAL_DAY_TIME()->valueToString(
                inputFlatVector->valueAt(row));
            auto writer = exec::StringWriter(resultFlatVector, row);
            writer.resize(output.size());
            ::memcpy(writer.data(), output.data(), output.size());
            writer.finalize();
          });

      return result;
    }
    default:
      VELOX_UNSUPPORTED(
          "Cast from {} to {} is not supported",
          INTERVAL_DAY_TIME()->toString(),
          toType->toString());
  }
}

VectorPtr PrestoCastKernel::castToIntervalDayTime(
    const SelectivityVector& /*rows*/,
    const BaseVector& input,
    exec::EvalCtx& /*context*/,
    bool /*setNullInResultAtError*/) const {
  VELOX_UNSUPPORTED(
      "Cast from {} to {} is not supported",
      input.type()->toString(),
      INTERVAL_DAY_TIME()->toString());
}

VectorPtr PrestoCastKernel::castFromTime(
    const SelectivityVector& rows,
    const BaseVector& input,
    exec::EvalCtx& context,
    const TypePtr& toType,
    bool setNullInResultAtError) const {
  auto* inputFlatVector = input.as<SimpleVector<int64_t>>();

  switch (toType->kind()) {
    case TypeKind::VARCHAR: {
      VectorPtr result;
      initializeResultVector(rows, toType, context, result);

      // Get session timezone
      const auto* timeZone =
          getTimeZoneFromConfig(context.execCtx()->queryCtx()->queryConfig());
      // Get session start time
      const auto startTimeMs =
          context.execCtx()->queryCtx()->queryConfig().sessionStartTimeMs();
      auto systemDay = std::chrono::milliseconds{startTimeMs} / kMillisInDay;

      auto* resultFlatVector = result->as<FlatVector<StringView>>();

      Buffer* buffer = resultFlatVector->getBufferWithSpace(
          rows.countSelected() * TimeType::kTimeToVarcharRowSize,
          true /*exactSize*/);
      char* rawBuffer = buffer->asMutable<char>() + buffer->size();

      applyToSelectedNoThrowLocal(
          rows, context, result, setNullInResultAtError, [&](int row) {
            // Use timezone-aware conversion
            auto systemTime = systemDay.count() * kMillisInDay +
                inputFlatVector->valueAt(row);

            int64_t adjustedTime{0};
            if (timeZone) {
              adjustedTime =
                  (timeZone->to_local(std::chrono::milliseconds{systemTime}) %
                   kMillisInDay)
                      .count();
            } else {
              adjustedTime = systemTime % kMillisInDay;
            }

            if (adjustedTime < 0) {
              adjustedTime += kMillisInDay;
            }

            auto output = TIME()->valueToString(adjustedTime, rawBuffer);
            resultFlatVector->setNoCopy(row, output);
            rawBuffer += output.size();
          });

      buffer->setSize(rawBuffer - buffer->asMutable<char>());

      return result;
    }
    case TypeKind::BIGINT: {
      // if input is constant, create a constant output vector
      if (input.isConstantEncoding()) {
        auto constantInput = input.as<ConstantVector<int64_t>>();
        if (constantInput->isNullAt(0)) {
          return BaseVector::createNullConstant(
              toType, rows.end(), context.pool());
        }
        auto constantValue = constantInput->valueAt(0);
        return std::make_shared<ConstantVector<int64_t>>(
            context.pool(),
            rows.end(),
            false, // isNull
            toType,
            std::move(constantValue));
      }

      // fallback to element-wise copy for non-constant inputs
      VectorPtr result;
      initializeResultVector(rows, toType, context, result);
      auto* resultFlatVector = result->as<FlatVector<int64_t>>();
      applyToSelectedNoThrowLocal(
          rows, context, result, setNullInResultAtError, [&](int row) {
            resultFlatVector->set(row, inputFlatVector->valueAt(row));
          });

      return result;
    }
    case TypeKind::TIMESTAMP: {
      // if input is constant, create a constant output vector
      if (input.isConstantEncoding()) {
        auto constantInput = input.as<ConstantVector<int64_t>>();
        if (constantInput->isNullAt(0)) {
          return BaseVector::createNullConstant(
              toType, rows.end(), context.pool());
        }
        auto timeMillis = constantInput->valueAt(0);
        return std::make_shared<ConstantVector<Timestamp>>(
            context.pool(),
            rows.end(),
            false, // isNull
            toType,
            Timestamp::fromMillis(timeMillis));
      }

      // fallback to element-wise copy for non-constant inputs
      VectorPtr result;
      initializeResultVector(rows, toType, context, result);
      auto* resultFlatVector = result->as<FlatVector<Timestamp>>();
      applyToSelectedNoThrowLocal(
          rows, context, result, setNullInResultAtError, [&](int row) {
            auto timeMillis = inputFlatVector->valueAt(row);
            resultFlatVector->set(row, Timestamp::fromMillis(timeMillis));
          });

      return result;
    }
    default:
      VELOX_UNSUPPORTED(
          "Cast from TIME to {} is not supported", toType->toString());
  }
}

VectorPtr PrestoCastKernel::castToTime(
    const SelectivityVector& rows,
    const BaseVector& input,
    exec::EvalCtx& context,
    bool setNullInResultAtError) const {
  const auto& fromType = input.type();

  switch (fromType->kind()) {
    case TypeKind::VARCHAR: {
      VectorPtr result;
      initializeResultVector(rows, TIME(), context, result);

      // Get session timezone and start time for timezone conversions
      const auto* timeZone =
          getTimeZoneFromConfig(context.execCtx()->queryCtx()->queryConfig());
      const auto sessionStartTimeMs =
          context.execCtx()->queryCtx()->queryConfig().sessionStartTimeMs();

      auto* inputVector = input.as<SimpleVector<StringView>>();
      auto* resultFlatVector = result->as<FlatVector<int64_t>>();

      applyToSelectedNoThrowLocal(
          rows, context, result, setNullInResultAtError, [&](int row) {
            const auto inputString = inputVector->valueAt(row);
            int64_t time =
                TIME()->valueToTime(inputString, timeZone, sessionStartTimeMs);
            resultFlatVector->set(row, time);
          });

      return result;
    }
    case TypeKind::TIMESTAMP: {
      VectorPtr result;
      initializeResultVector(rows, TIME(), context, result);
      auto* inputVector = input.as<SimpleVector<Timestamp>>();
      auto* resultFlatVector = result->as<FlatVector<int64_t>>();

      // Cast from TIMESTAMP to TIME extracts the time-of-day component
      // (milliseconds since midnight) from the timestamp
      applyToSelectedNoThrowLocal(
          rows, context, result, setNullInResultAtError, [&](int row) {
            const auto timestamp = inputVector->valueAt(row);
            // Extract time-of-day using std::chrono.
            // floor() also rounds towards negative infinity, so this correctly
            // handles negative timestamps.
            auto millis = std::chrono::milliseconds{timestamp.toMillis()};
            auto timeOfDay =
                millis - std::chrono::floor<std::chrono::days>(millis);
            resultFlatVector->set(row, timeOfDay.count());
          });

      return result;
    }
    default:
      VELOX_UNSUPPORTED(
          "Cast from {} to TIME is not supported", fromType->toString());
  }
}

VectorPtr PrestoCastKernel::castFromDecimal(
    const SelectivityVector& rows,
    const BaseVector& input,
    exec::EvalCtx& context,
    const TypePtr& toType,
    bool setNullInResultAtError) const {
  const auto& fromType = input.type();

  switch (toType->kind()) {
    case TypeKind::VARCHAR:
      return VELOX_DYNAMIC_DECIMAL_TYPE_DISPATCH(
          applyDecimalToVarcharCast,
          fromType,
          rows,
          input,
          context,
          fromType,
          setNullInResultAtError);
    case TypeKind::BOOLEAN:
      return VELOX_DYNAMIC_DECIMAL_TYPE_DISPATCH(
          applyDecimalToBooleanCast,
          fromType,
          rows,
          input,
          context,
          setNullInResultAtError);
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
      if (toType->isShortDecimal()) {
        return VELOX_DYNAMIC_DECIMAL_TEMPLATE_TYPE_DISPATCH(
            applyDecimalToDecimalCast,
            fromType,
            int64_t,
            rows,
            input,
            context,
            toType,
            setNullInResultAtError);
      }
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
    case TypeKind::HUGEINT:
      if (toType->isLongDecimal()) {
        return VELOX_DYNAMIC_DECIMAL_TEMPLATE_TYPE_DISPATCH(
            applyDecimalToDecimalCast,
            fromType,
            int128_t,
            rows,
            input,
            context,
            toType,
            setNullInResultAtError);
      }
      VELOX_UNSUPPORTED(
          "Cast from {} to {} is not supported",
          fromType->toString(),
          toType->toString());
    case TypeKind::REAL:
      return VELOX_DYNAMIC_DECIMAL_TEMPLATE_TYPE_DISPATCH(
          applyDecimalToFloatCast,
          fromType,
          TypeKind::REAL,
          rows,
          input,
          context,
          fromType,
          toType,
          setNullInResultAtError);
    case TypeKind::DOUBLE:
      return VELOX_DYNAMIC_DECIMAL_TEMPLATE_TYPE_DISPATCH(
          applyDecimalToFloatCast,
          fromType,
          TypeKind::DOUBLE,
          rows,
          input,
          context,
          fromType,
          toType,
          setNullInResultAtError);
    default:
      VELOX_UNSUPPORTED(
          "Cast from {} to {} is not supported",
          fromType->toString(),
          toType->toString());
  }
}

VectorPtr PrestoCastKernel::castToDecimal(
    const SelectivityVector& rows,
    const BaseVector& input,
    exec::EvalCtx& context,
    const TypePtr& toType,
    bool setNullInResultAtError) const {
  const auto& fromType = input.type();

  switch (fromType->kind()) {
    case TypeKind::BOOLEAN:
      return VELOX_DYNAMIC_DECIMAL_TEMPLATE_TYPE_DISPATCH(
          applyIntToDecimalCast,
          toType,
          bool,
          rows,
          input,
          context,
          toType,
          setNullInResultAtError);
    case TypeKind::TINYINT:
      return VELOX_DYNAMIC_DECIMAL_TEMPLATE_TYPE_DISPATCH(
          applyIntToDecimalCast,
          toType,
          int8_t,
          rows,
          input,
          context,
          toType,
          setNullInResultAtError);
    case TypeKind::SMALLINT:
      return VELOX_DYNAMIC_DECIMAL_TEMPLATE_TYPE_DISPATCH(
          applyIntToDecimalCast,
          toType,
          int16_t,
          rows,
          input,
          context,
          toType,
          setNullInResultAtError);
    case TypeKind::INTEGER:
      return VELOX_DYNAMIC_DECIMAL_TEMPLATE_TYPE_DISPATCH(
          applyIntToDecimalCast,
          toType,
          int32_t,
          rows,
          input,
          context,
          toType,
          setNullInResultAtError);
    case TypeKind::REAL:
      return VELOX_DYNAMIC_DECIMAL_TEMPLATE_TYPE_DISPATCH(
          applyFloatingPointToDecimalCast,
          toType,
          float,
          rows,
          input,
          context,
          toType,
          setNullInResultAtError);
    case TypeKind::DOUBLE:
      return VELOX_DYNAMIC_DECIMAL_TEMPLATE_TYPE_DISPATCH(
          applyFloatingPointToDecimalCast,
          toType,
          double,
          rows,
          input,
          context,
          toType,
          setNullInResultAtError);
    case TypeKind::BIGINT: {
      return VELOX_DYNAMIC_DECIMAL_TEMPLATE_TYPE_DISPATCH(
          applyIntToDecimalCast,
          toType,
          int64_t,
          rows,
          input,
          context,
          toType,
          setNullInResultAtError);
    }
    case TypeKind::HUGEINT: {
      return VELOX_DYNAMIC_DECIMAL_TEMPLATE_TYPE_DISPATCH(
          applyIntToDecimalCast,
          toType,
          int128_t,
          rows,
          input,
          context,
          toType,
          setNullInResultAtError);
    }
    case TypeKind::VARCHAR:
      return VELOX_DYNAMIC_DECIMAL_TYPE_DISPATCH(
          applyVarcharToDecimalCast,
          toType,
          rows,
          input,
          context,
          toType,
          setNullInResultAtError);
    default:
      VELOX_UNSUPPORTED(
          "Cast from {} to {} is not supported",
          fromType->toString(),
          toType->toString());
  }
}

VectorPtr PrestoCastKernel::applyTimestampToVarcharCast(
    const TypePtr& toType,
    const SelectivityVector& rows,
    exec::EvalCtx& context,
    const BaseVector& input,
    bool setNullInResultAtError) const {
  VectorPtr result;
  initializeResultVector(rows, toType, context, result);
  auto flatResult = result->asFlatVector<StringView>();
  const auto simpleInput = input.as<SimpleVector<Timestamp>>();

  const uint32_t rowSize = getMaxStringLength(timestampToStringOptions());

  Buffer* buffer = flatResult->getBufferWithSpace(
      rows.countSelected() * rowSize, true /*exactSize*/);
  char* rawBuffer = buffer->asMutable<char>() + buffer->size();

  const TimestampToStringOptions& options = timestampToStringOptions();

  applyToSelectedNoThrowLocal(
      rows, context, result, setNullInResultAtError, [&](vector_size_t row) {
        // Adjust input timestamp according the session timezone.
        Timestamp inputValue(simpleInput->valueAt(row));
        if (options.timeZone) {
          inputValue.toTimezone(*(options.timeZone));
        }
        const auto stringView =
            Timestamp::tsToStringView(inputValue, options, rawBuffer);
        flatResult->setNoCopy(row, stringView);
        // The result of both Presto and Spark contains more than 12
        // digits even when 'zeroPaddingYear' is disabled.
        VELOX_DCHECK(!stringView.isInline());
        rawBuffer += stringView.size();
      });

  // Update the exact buffer size.
  buffer->setSize(rawBuffer - buffer->asMutable<char>());
  return result;
}

VectorPtr PrestoCastKernel::castToVarchar(
    const SelectivityVector& rows,
    const BaseVector& input,
    exec::EvalCtx& context,
    bool setNullInResultAtError) const {
  if (input.typeKind() == TypeKind::TIMESTAMP) {
    return applyTimestampToVarcharCast(
        VARCHAR(), rows, context, input, setNullInResultAtError);
  }

  return applyCastPrimitivesDispatch<TypeKind::VARCHAR>(
      input.type(), VARCHAR(), rows, context, input, setNullInResultAtError);
}

VectorPtr PrestoCastKernel::castToVarbinary(
    const SelectivityVector& rows,
    const BaseVector& input,
    exec::EvalCtx& context,
    const TypePtr& toType,
    bool setNullInResultAtError) const {
  switch (input.typeKind()) {
    case TypeKind::TIMESTAMP:
      return applyTimestampToVarcharCast(
          toType, rows, context, input, setNullInResultAtError);
    case TypeKind::TINYINT:
      return applyIntToBinaryCast<int8_t>(
          rows, context, input, toType, setNullInResultAtError);
    case TypeKind::SMALLINT:
      return applyIntToBinaryCast<int16_t>(
          rows, context, input, toType, setNullInResultAtError);
    case TypeKind::INTEGER:
      return applyIntToBinaryCast<int32_t>(
          rows, context, input, toType, setNullInResultAtError);
    case TypeKind::BIGINT:
      return applyIntToBinaryCast<int64_t>(
          rows, context, input, toType, setNullInResultAtError);
    default:
      // Handle primitive type conversions.
      return applyCastPrimitivesDispatch<TypeKind::VARBINARY>(
          input.type(), toType, rows, context, input, setNullInResultAtError);
  }
}
} // namespace facebook::velox::exec
