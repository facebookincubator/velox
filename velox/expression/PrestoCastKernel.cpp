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
          input.type()->toString(),
          toType->toString());
  }
}

VectorPtr PrestoCastKernel::castToIntervalDayTime(
    const SelectivityVector& /*rows*/,
    const BaseVector& input,
    exec::EvalCtx& /*context*/,
    const TypePtr& toType,
    bool /*setNullInResultAtError*/) const {
  VELOX_UNSUPPORTED(
      "Cast from {} to {} is not supported",
      input.type()->toString(),
      toType->toString());
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
    const TypePtr& toType,
    bool setNullInResultAtError) const {
  const auto& fromType = input.type();

  switch (fromType->kind()) {
    case TypeKind::VARCHAR: {
      VectorPtr result;
      initializeResultVector(rows, toType, context, result);

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
      initializeResultVector(rows, toType, context, result);
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
} // namespace facebook::velox::exec
