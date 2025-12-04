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

#include <double-conversion/double-conversion.h>
#include <folly/Expected.h>

#include "velox/expression/StringWriter.h"
#include "velox/functions/lib/RowsTranslationUtil.h"

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

const tz::TimeZone* getTimeZoneFromConfig(const core::QueryConfig& config) {
  if (config.adjustTimestampToTimezone()) {
    const auto sessionTzName = config.sessionTimezone();
    if (!sessionTzName.empty()) {
      return tz::locateZone(sessionTzName);
    }
  }
  return nullptr;
}

void propagateErrorsOrSetNulls(
    bool setNullInResultAtError,
    EvalCtx& context,
    const SelectivityVector& nestedRows,
    const BufferPtr& elementToTopLevelRows,
    VectorPtr& result,
    exec::EvalErrorsPtr& oldErrors) {
  if (context.errors()) {
    if (setNullInResultAtError) {
      // Errors in context.errors() should be translated to nulls in
      // the top level rows.
      context.convertElementErrorsToTopLevelNulls(
          nestedRows, elementToTopLevelRows, result);
    } else {
      context.addElementErrorsToTopLevel(
          nestedRows, elementToTopLevelRows, oldErrors);
    }
  }
}
} // namespace

PrestoCastKernel::PrestoCastKernel(const core::QueryConfig& config)
    : legacyCast_(config.isLegacyCast()),
      matchStructFieldsByName_(config.isMatchStructByName()) {
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
  context.ensureWritable(rows, toType, result);
  (*result).clearNulls(rows);

  switch (toType->kind()) {
    case TypeKind::VARCHAR: {
      auto* resultFlatVector = result->as<FlatVector<StringView>>();
      applyToSelectedNoThrowLocal(
          rows, input, context, result, setNullInResultAtError, [&](int row) {
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
          rows, input, context, result, setNullInResultAtError, [&](int row) {
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
    const TypePtr& toType,
    bool setNullInResultAtError) const {
  const auto& fromType = input.type();

  VectorPtr result;
  context.ensureWritable(rows, toType, result);
  (*result).clearNulls(rows);
  auto* resultFlatVector = result->as<FlatVector<int32_t>>();

  switch (fromType->kind()) {
    case TypeKind::VARCHAR: {
      auto* inputVector = input.as<SimpleVector<StringView>>();
      applyToSelectedNoThrowLocal(
          rows, input, context, result, setNullInResultAtError, [&](int row) {
            // Cast from string to date allows only complete ISO 8601 formatted
            // strings :
            // [+-](YYYY-MM-DD).
            const auto resultValue = util::fromDateString(
                inputVector->valueAt(row), util::ParseMode::kPrestoCast);

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
    case TypeKind::TIMESTAMP: {
      auto* inputVector = input.as<SimpleVector<Timestamp>>();
      const auto* timeZone =
          getTimeZoneFromConfig(context.execCtx()->queryCtx()->queryConfig());
      applyToSelectedNoThrowLocal(
          rows, input, context, result, setNullInResultAtError, [&](int row) {
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

VectorPtr PrestoCastKernel::castFromIntervalDayTime(
    const SelectivityVector& rows,
    const BaseVector& input,
    exec::EvalCtx& context,
    const TypePtr& toType,
    bool setNullInResultAtError) const {
  auto* inputFlatVector = input.as<SimpleVector<int64_t>>();

  VectorPtr result;
  context.ensureWritable(rows, toType, result);
  (*result).clearNulls(rows);

  switch (toType->kind()) {
    case TypeKind::VARCHAR: {
      auto* resultFlatVector = result->as<FlatVector<StringView>>();
      applyToSelectedNoThrowLocal(
          rows, input, context, result, setNullInResultAtError, [&](int row) {
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
    const SelectivityVector& rows,
    const BaseVector& input,
    exec::EvalCtx& context,
    const TypePtr& toType,
    bool setNullInResultAtError) const {
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

  VectorPtr result;
  context.ensureWritable(rows, toType, result);
  (*result).clearNulls(rows);

  switch (toType->kind()) {
    case TypeKind::VARCHAR: {
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
          rows, input, context, result, setNullInResultAtError, [&](int row) {
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
        } else {
          auto constantValue = constantInput->valueAt(0);
          return std::make_shared<ConstantVector<int64_t>>(
              context.pool(),
              rows.end(),
              false, // isNull
              toType,
              std::move(constantValue));
        }
      }

      // fallback to element-wise copy for non-constant inputs
      auto* resultFlatVector = result->as<FlatVector<int64_t>>();
      applyToSelectedNoThrowLocal(
          rows, input, context, result, setNullInResultAtError, [&](int row) {
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
        } else {
          auto timeMillis = constantInput->valueAt(0);
          return std::make_shared<ConstantVector<Timestamp>>(
              context.pool(),
              rows.end(),
              false, // isNull
              toType,
              Timestamp::fromMillis(timeMillis));
        }
      }

      // fallback to element-wise copy for non-constant inputs
      auto* resultFlatVector = result->as<FlatVector<Timestamp>>();
      applyToSelectedNoThrowLocal(
          rows, input, context, result, setNullInResultAtError, [&](int row) {
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
    const TypePtr& toType,
    bool setNullInResultAtError) const {
  const auto& fromType = input.type();

  switch (fromType->kind()) {
    case TypeKind::VARCHAR: {
      VectorPtr result;
      context.ensureWritable(rows, TIME(), result);
      (*result).clearNulls(rows);

      // Get session timezone and start time for timezone conversions
      const auto* timeZone =
          getTimeZoneFromConfig(context.execCtx()->queryCtx()->queryConfig());
      const auto sessionStartTimeMs =
          context.execCtx()->queryCtx()->queryConfig().sessionStartTimeMs();

      auto* inputVector = input.as<SimpleVector<StringView>>();
      auto* resultFlatVector = result->as<FlatVector<int64_t>>();

      applyToSelectedNoThrowLocal(
          rows, input, context, result, setNullInResultAtError, [&](int row) {
            const auto inputString = inputVector->valueAt(row);
            int64_t time =
                TIME()->valueToTime(inputString, timeZone, sessionStartTimeMs);
            resultFlatVector->set(row, time);
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
      break;
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

  VectorPtr result;
  context.ensureWritable(rows, toType, result);
  (*result).clearNulls(rows);

  switch (fromType->kind()) {
    case TypeKind::BOOLEAN:
      VELOX_DYNAMIC_DECIMAL_TEMPLATE_TYPE_DISPATCH(
          applyIntToDecimalCast,
          toType,
          bool,
          rows,
          input,
          context,
          toType,
          setNullInResultAtError,
          result);
      break;
    case TypeKind::TINYINT:
      VELOX_DYNAMIC_DECIMAL_TEMPLATE_TYPE_DISPATCH(
          applyIntToDecimalCast,
          toType,
          int8_t,
          rows,
          input,
          context,
          toType,
          setNullInResultAtError,
          result);
      break;
    case TypeKind::SMALLINT:
      VELOX_DYNAMIC_DECIMAL_TEMPLATE_TYPE_DISPATCH(
          applyIntToDecimalCast,
          toType,
          int16_t,
          rows,
          input,
          context,
          toType,
          setNullInResultAtError,
          result);
      break;
    case TypeKind::INTEGER:
      VELOX_DYNAMIC_DECIMAL_TEMPLATE_TYPE_DISPATCH(
          applyIntToDecimalCast,
          toType,
          int32_t,
          rows,
          input,
          context,
          toType,
          setNullInResultAtError,
          result);
      break;
    case TypeKind::REAL:
      VELOX_DYNAMIC_DECIMAL_TEMPLATE_TYPE_DISPATCH(
          applyFloatingPointToDecimalCast,
          toType,
          float,
          rows,
          input,
          context,
          toType,
          setNullInResultAtError,
          result);
      break;
    case TypeKind::DOUBLE:
      VELOX_DYNAMIC_DECIMAL_TEMPLATE_TYPE_DISPATCH(
          applyFloatingPointToDecimalCast,
          toType,
          double,
          rows,
          input,
          context,
          toType,
          setNullInResultAtError,
          result);
      break;
    case TypeKind::BIGINT: {
      VELOX_DYNAMIC_DECIMAL_TEMPLATE_TYPE_DISPATCH(
          applyIntToDecimalCast,
          toType,
          int64_t,
          rows,
          input,
          context,
          toType,
          setNullInResultAtError,
          result);
      break;
    }
    case TypeKind::HUGEINT: {
      VELOX_DYNAMIC_DECIMAL_TEMPLATE_TYPE_DISPATCH(
          applyIntToDecimalCast,
          toType,
          int128_t,
          rows,
          input,
          context,
          toType,
          setNullInResultAtError,
          result);
      break;
    }
    case TypeKind::VARCHAR:
      VELOX_DYNAMIC_DECIMAL_TYPE_DISPATCH(
          applyVarcharToDecimalCast,
          toType,
          rows,
          input,
          context,
          toType,
          setNullInResultAtError,
          result);
      break;
    default:
      VELOX_UNSUPPORTED(
          "Cast from {} to {} is not supported",
          fromType->toString(),
          toType->toString());
  }

  return result;
}

VectorPtr PrestoCastKernel::applyTimestampToVarcharCast(
    const TypePtr& toType,
    const SelectivityVector& rows,
    exec::EvalCtx& context,
    const BaseVector& input,
    bool setNullInResultAtError) const {
  VectorPtr result;
  context.ensureWritable(rows, toType, result);
  (*result).clearNulls(rows);
  auto flatResult = result->asFlatVector<StringView>();
  const auto simpleInput = input.as<SimpleVector<Timestamp>>();

  const uint32_t rowSize = getMaxStringLength(timestampToStringOptions());

  Buffer* buffer = flatResult->getBufferWithSpace(
      rows.countSelected() * rowSize, true /*exactSize*/);
  char* rawBuffer = buffer->asMutable<char>() + buffer->size();

  const TimestampToStringOptions& options = timestampToStringOptions();

  applyToSelectedNoThrowLocal(
      rows,
      input,
      context,
      result,
      setNullInResultAtError,
      [&](vector_size_t row) {
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
    const TypePtr& toType,
    bool setNullInResultAtError) const {
  if (input.typeKind() == TypeKind::TIMESTAMP) {
    return applyTimestampToVarcharCast(
        toType, rows, context, input, setNullInResultAtError);
  }

  return applyCastPrimitivesDispatch<TypeKind::VARCHAR>(
      input.type(), toType, rows, context, input, setNullInResultAtError);
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
      break;
    case TypeKind::SMALLINT:
      return applyIntToBinaryCast<int16_t>(
          rows, context, input, toType, setNullInResultAtError);
      break;
    case TypeKind::INTEGER:
      return applyIntToBinaryCast<int32_t>(
          rows, context, input, toType, setNullInResultAtError);
      break;
    case TypeKind::BIGINT:
      return applyIntToBinaryCast<int64_t>(
          rows, context, input, toType, setNullInResultAtError);
      break;
    default:
      // Handle primitive type conversions.
      return applyCastPrimitivesDispatch<TypeKind::VARBINARY>(
          input.type(), toType, rows, context, input, setNullInResultAtError);
  }
}

VectorPtr PrestoCastKernel::castArray(
    const SelectivityVector& rows,
    const BaseVector& input,
    exec::EvalCtx& context,
    const TypePtr& toType,
    bool setNullInResultAtError,
    ApplyCallback apply) const {
  // Cast input array elements to output array elements based on their
  // types using their linear selectivity vector
  const auto* arrayVector = input.as<ArrayVector>();
  auto arrayElements = arrayVector->elements();
  const auto fromType = arrayVector->type()->asArray();
  const auto& toArrayType = toType->asArray();

  auto nestedRows =
      functions::toElementRows(arrayElements->size(), rows, arrayVector);
  auto elementToTopLevelRows = functions::getElementToTopLevelRows(
      arrayElements->size(), rows, arrayVector, context.pool());

  EvalErrorsPtr oldErrors;
  context.swapErrors(oldErrors);

  VectorPtr newElements;
  {
    apply(
        nestedRows,
        arrayElements,
        context,
        fromType.elementType(),
        toArrayType.elementType(),
        newElements);
  }

  // Returned array vector should be addressable for every element,
  // even those that are not selected.
  BufferPtr sizes = arrayVector->sizes();
  if (newElements->isConstantEncoding()) {
    // If the newElements we extends its size since that is cheap.
    newElements->resize(arrayVector->elements()->size());
  } else if (newElements->size() < arrayVector->elements()->size()) {
    sizes =
        AlignedBuffer::allocate<vector_size_t>(rows.end(), context.pool(), 0);
    auto* inputSizes = arrayVector->rawSizes();
    auto* rawSizes = sizes->asMutable<vector_size_t>();
    rows.applyToSelected(
        [&](vector_size_t row) { rawSizes[row] = inputSizes[row]; });
  }

  VectorPtr result = std::make_shared<ArrayVector>(
      context.pool(),
      toType,
      arrayVector->nulls(),
      rows.end(),
      arrayVector->offsets(),
      sizes,
      newElements);

  propagateErrorsOrSetNulls(
      setNullInResultAtError,
      context,
      nestedRows,
      elementToTopLevelRows,
      result,
      oldErrors);
  // Restore original state.
  context.swapErrors(oldErrors);
  return result;
}

VectorPtr PrestoCastKernel::castMap(
    const SelectivityVector& rows,
    const BaseVector& input,
    exec::EvalCtx& context,
    const TypePtr& toType,
    bool setNullInResultAtError,
    ApplyCallback apply) const {
  // Cast input keys/values vector to output keys/values vector using
  // their element selectivity vector
  const auto* mapVector = input.as<MapVector>();
  const auto fromType = mapVector->type()->asMap();
  const auto& toMapType = toType->asMap();

  // Initialize nested rows
  auto mapKeys = mapVector->mapKeys();
  auto mapValues = mapVector->mapValues();

  SelectivityVector nestedRows;
  BufferPtr elementToTopLevelRows;
  if (fromType.keyType() != toMapType.keyType() ||
      fromType.valueType() != toMapType.valueType()) {
    nestedRows = functions::toElementRows(mapKeys->size(), rows, mapVector);
    elementToTopLevelRows = functions::getElementToTopLevelRows(
        mapKeys->size(), rows, mapVector, context.pool());
  }

  EvalErrorsPtr oldErrors;
  context.swapErrors(oldErrors);

  // Cast keys
  VectorPtr newMapKeys;
  if (*fromType.keyType() == *toMapType.keyType()) {
    newMapKeys = mapVector->mapKeys();
  } else {
    {
      apply(
          nestedRows,
          mapKeys,
          context,
          fromType.keyType(),
          toMapType.keyType(),
          newMapKeys);
    }
  }

  // Cast values
  VectorPtr newMapValues;
  if (*fromType.valueType() == *toMapType.valueType()) {
    newMapValues = mapValues;
  } else {
    {
      apply(
          nestedRows,
          mapValues,
          context,
          fromType.valueType(),
          toMapType.valueType(),
          newMapValues);
    }
  }

  // Returned map vector should be addressable for every element, even
  // those that are not selected.
  BufferPtr sizes = mapVector->sizes();
  if (newMapKeys->isConstantEncoding() && newMapValues->isConstantEncoding()) {
    // We extends size since that is cheap.
    newMapKeys->resize(mapVector->mapKeys()->size());
    newMapValues->resize(mapVector->mapValues()->size());
  } else if (
      newMapKeys->size() < mapVector->mapKeys()->size() ||
      newMapValues->size() < mapVector->mapValues()->size()) {
    sizes =
        AlignedBuffer::allocate<vector_size_t>(rows.end(), context.pool(), 0);
    auto* inputSizes = mapVector->rawSizes();
    auto* rawSizes = sizes->asMutable<vector_size_t>();

    rows.applyToSelected(
        [&](vector_size_t row) { rawSizes[row] = inputSizes[row]; });
  }

  // Assemble the output map
  VectorPtr result = std::make_shared<MapVector>(
      context.pool(),
      MAP(toMapType.keyType(), toMapType.valueType()),
      mapVector->nulls(),
      rows.end(),
      mapVector->offsets(),
      sizes,
      newMapKeys,
      newMapValues);

  propagateErrorsOrSetNulls(
      setNullInResultAtError,
      context,
      nestedRows,
      elementToTopLevelRows,
      result,
      oldErrors);

  // Restore original state.
  context.swapErrors(oldErrors);
  return result;
}

VectorPtr PrestoCastKernel::castRow(
    const SelectivityVector& rows,
    const BaseVector& input,
    exec::EvalCtx& context,
    const TypePtr& toType,
    bool setNullInResultAtError,
    ApplyCallback apply) const {
  const auto* rowVector = input.as<RowVector>();
  const auto& fromType = rowVector->type()->asRow();
  int numInputChildren = rowVector->children().size();
  const RowType& toRowType = toType->asRow();
  int numOutputChildren = toRowType.size();

  // Cast each row child to its corresponding output child
  std::vector<VectorPtr> newChildren;
  newChildren.reserve(numOutputChildren);

  EvalErrorsPtr oldErrors;
  if (setNullInResultAtError) {
    // We need to isolate errors that happen during the cast from
    // previous errors since those translate to nulls, unlike
    // exisiting errors.
    context.swapErrors(oldErrors);
  }

  for (auto toChildrenIndex = 0; toChildrenIndex < numOutputChildren;
       toChildrenIndex++) {
    // For each child, find the corresponding column index in the
    // output
    const auto& toFieldName = toRowType.nameOf(toChildrenIndex);
    bool matchNotFound = false;

    // If match is by field name and the input field name is not found
    // in the output row type, do not consider it in the output
    int fromChildrenIndex = -1;
    if (matchStructFieldsByName_) {
      if (!fromType.containsChild(toFieldName)) {
        matchNotFound = true;
      } else {
        fromChildrenIndex = fromType.getChildIdx(toFieldName);
        toChildrenIndex = toRowType.getChildIdx(toFieldName);
      }
    } else {
      fromChildrenIndex = toChildrenIndex;
      if (fromChildrenIndex >= numInputChildren) {
        matchNotFound = true;
      }
    }

    // Updating output types and names
    VectorPtr outputChild;
    const auto& toChildType = toRowType.childAt(toChildrenIndex);

    if (matchNotFound) {
      // Create a vector for null for this child
      context.ensureWritable(rows, toChildType, outputChild);
      outputChild->addNulls(rows);
    } else {
      const auto& inputChild = rowVector->children()[fromChildrenIndex];
      if (*toChildType == *inputChild->type()) {
        outputChild = inputChild;
      } else {
        // Apply cast for the child.
        apply(
            rows,
            inputChild,
            context,
            inputChild->type(),
            toChildType,
            outputChild);
      }
    }
    newChildren.emplace_back(std::move(outputChild));
  }

  // Assemble the output row
  VectorPtr result = std::make_shared<RowVector>(
      context.pool(),
      toType,
      rowVector->nulls(),
      rows.end(),
      std::move(newChildren));

  if (setNullInResultAtError) {
    // Set errors as nulls.
    if (auto errors = context.errors()) {
      rows.applyToSelected([&](auto row) {
        if (errors->hasErrorAt(row)) {
          result->setNull(row, true);
        }
      });
    }
    // Restore original state.
    context.swapErrors(oldErrors);
  }

  return result;
}
} // namespace facebook::velox::exec
