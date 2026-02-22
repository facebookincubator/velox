/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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
#include "velox/functions/prestosql/types/TimeWithTimezoneRegistration.h"

#include "velox/expression/CastExpr.h"
#include "velox/external/tzdb/time_zone.h"
#include "velox/functions/prestosql/types/TimeWithTimezoneType.h"
#include "velox/functions/prestosql/types/fuzzer_utils/TimeWithTimezoneInputGenerator.h"
#include "velox/type/Time.h"
#include "velox/type/Type.h"
#include "velox/type/tz/TimeZoneMap.h"
#include "velox/vector/DecodedVector.h"

namespace facebook::velox {

namespace {
void castToTime(
    const BaseVector& input,
    exec::EvalCtx& context,
    const SelectivityVector& rows,
    BaseVector& result) {
  auto* flatResult = result.asFlatVector<int64_t>();

  auto convertToLocalTime = [](int64_t timeWithTimezone) {
    int64_t millisUtc = util::unpackMillisUtc(timeWithTimezone);
    auto timezoneMinutes = util::unpackZoneOffset(timeWithTimezone);
    int16_t offsetMinutes = util::decodeTimezoneOffset(timezoneMinutes);
    return util::utcToLocalTime(millisUtc, offsetMinutes);
  };

  if (input.isConstantEncoding()) {
    const auto timeWithTimezone =
        input.as<ConstantVector<int64_t>>()->valueAt(0);
    context.applyToSelectedNoThrow(rows, [&](vector_size_t row) {
      flatResult->set(row, convertToLocalTime(timeWithTimezone));
    });
    return;
  }
  const auto timeWithTimezones = input.as<FlatVector<int64_t>>();
  context.applyToSelectedNoThrow(rows, [&](vector_size_t row) {
    const auto timeWithTimezone = timeWithTimezones->valueAt(row);
    flatResult->set(row, convertToLocalTime(timeWithTimezone));
  });
}

void castToString(
    const BaseVector& input,
    exec::EvalCtx& context,
    const SelectivityVector& rows,
    BaseVector& result) {
  auto* flatResult = result.asFlatVector<StringView>();
  DecodedVector decoded(input, rows);
  Buffer* buffer = flatResult->getBufferWithSpace(
      rows.countSelected() *
          TimeWithTimezoneType::kTimeWithTimezoneToVarcharRowSize,
      true /*exactSize*/);
  char* rawBuffer = buffer->asMutable<char>() + buffer->size();
  context.applyToSelectedNoThrow(rows, [&](vector_size_t row) {
    const auto timeWithTimezone = decoded.valueAt<int64_t>(row);
    auto output =
        TIME_WITH_TIME_ZONE()->valueToString(timeWithTimezone, rawBuffer);
    flatResult->setNoCopy(row, output);
    rawBuffer += output.size();
  });
  buffer->setSize(rawBuffer - buffer->asMutable<char>());
}

void castFromString(
    const BaseVector& input,
    exec::EvalCtx& context,
    const SelectivityVector& rows,
    BaseVector& result) {
  auto* flatResult = result.asFlatVector<int64_t>();
  DecodedVector decoded(input, rows);

  context.applyToSelectedNoThrow(rows, [&](vector_size_t row) {
    const auto stringValue = decoded.valueAt<StringView>(row);

    auto parseResult = util::fromTimeWithTimezoneString(stringValue);
    if (parseResult.hasError()) {
      context.setStatus(row, parseResult.error());
      return;
    }

    flatResult->set(row, parseResult.value());
  });
}

const tz::TimeZone* getTimeZoneFromConfig(const core::QueryConfig& config) {
  const auto sessionTzName = config.sessionTimezone();

  if (!sessionTzName.empty()) {
    return tz::locateZone(sessionTzName);
  }

  return tz::locateZone(0); // GMT
}

// Calculate timezone offset in minutes at the given timestamp.
// Since TIME has no date component, we use session start time to determine
// which DST offset to apply.
inline int16_t getTimezoneOffsetMinutes(
    const tz::TimeZone* sessionTimeZone,
    int64_t sessionStartTimeMs) {
  if (auto offset = sessionTimeZone->offset()) {
    return static_cast<int16_t>(offset->count());
  }
  auto sysTime = std::chrono::time_point<std::chrono::system_clock>(
      std::chrono::milliseconds(sessionStartTimeMs));
  auto info = sessionTimeZone->tz()->get_info(sysTime);
  auto offsetMinutes =
      std::chrono::duration_cast<std::chrono::minutes>(info.offset);
  return static_cast<int16_t>(offsetMinutes.count());
}

// Pack time (in milliseconds since midnight) and timezone offset minutes
inline int64_t packTimeWithTimeZone(
    int64_t timeMillis,
    int16_t timezoneOffsetMinutes) {
  auto encodedOffset = util::biasEncode(timezoneOffsetMinutes);
  return util::pack(timeMillis, encodedOffset);
}

void castFromTime(
    const SimpleVector<int64_t>& inputVector,
    exec::EvalCtx& context,
    const SelectivityVector& rows,
    int64_t* rawResults,
    int16_t offsetMinutes) {
  context.applyToSelectedNoThrow(rows, [&](vector_size_t row) {
    const auto timeMillis = inputVector.valueAt(row);
    // Convert local time to UTC before packing
    const auto utcMillis = util::localToUtcTime(timeMillis, offsetMinutes);
    rawResults[row] = packTimeWithTimeZone(utcMillis, offsetMinutes);
  });
}

class TimeWithTimeZoneCastOperator final : public exec::CastOperator {
  TimeWithTimeZoneCastOperator() = default;

 public:
  static std::shared_ptr<const CastOperator> get() {
    VELOX_CONSTEXPR_SINGLETON TimeWithTimeZoneCastOperator kInstance;
    return {std::shared_ptr<const CastOperator>{}, &kInstance};
  }

  bool isSupportedFromType(const TypePtr& other) const override {
    switch (other->kind()) {
      case TypeKind::BIGINT:
        return other->isTime();
      case TypeKind::VARCHAR:
        return true;
      default:
        return false;
    }
  }

  bool isSupportedToType(const TypePtr& other) const override {
    switch (other->kind()) {
      case TypeKind::BIGINT:
        return other->isTime();
      case TypeKind::VARCHAR:
        return true;
      default:
        return false;
    }
  }

  void castTo(
      const BaseVector& input,
      exec::EvalCtx& context,
      const SelectivityVector& rows,
      const TypePtr& resultType,
      VectorPtr& result) const override {
    context.ensureWritable(rows, resultType, result);

    // Handle VARCHAR to TIME WITH TIME ZONE
    if (input.type() == VARCHAR()) {
      castFromString(input, context, rows, *result);
      return;
    }

    if (input.type()->isTime()) {
      const auto& config = context.execCtx()->queryCtx()->queryConfig();
      const auto* sessionTimeZone = getTimeZoneFromConfig(config);
      const auto sessionStartTimeMs = config.sessionStartTimeMs();
      int16_t offsetMinutes =
          getTimezoneOffsetMinutes(sessionTimeZone, sessionStartTimeMs);

      if (input.isConstantEncoding()) {
        auto constantInput = input.as<ConstantVector<int64_t>>();
        if (constantInput->isNullAt(0)) {
          result = BaseVector::createNullConstant(
              resultType, rows.end(), context.pool());
          return;
        }

        const auto timeMillis = constantInput->valueAt(0);
        // Convert local time to UTC before packing
        const auto utcMillis = util::localToUtcTime(timeMillis, offsetMinutes);
        auto packedValue = packTimeWithTimeZone(utcMillis, offsetMinutes);

        result = std::make_shared<ConstantVector<int64_t>>(
            context.pool(),
            rows.end(),
            false, // isNull
            resultType,
            std::move(packedValue));
        return;
      }

      auto* timeWithTzResult = result->asFlatVector<int64_t>();
      timeWithTzResult->clearNulls(rows);
      auto* rawResults = timeWithTzResult->mutableRawValues();

      const auto inputVector = input.as<SimpleVector<int64_t>>();
      castFromTime(*inputVector, context, rows, rawResults, offsetMinutes);
      return;
    } else {
      VELOX_UNSUPPORTED(
          "Cast from {} to TIME WITH TIME ZONE not yet supported",
          input.type()->toString());
    }
  }

  void castFrom(
      const BaseVector& input,
      exec::EvalCtx& context,
      const SelectivityVector& rows,
      const TypePtr& resultType,
      VectorPtr& result) const override {
    context.ensureWritable(rows, resultType, result);

    switch (resultType->kind()) {
      case TypeKind::BIGINT:
        if (resultType->isTime()) {
          castToTime(input, context, rows, *result);
          return;
        }
        break;
      case TypeKind::VARCHAR:
        castToString(input, context, rows, *result);
        return;
      default:
        break;
    }

    VELOX_UNSUPPORTED(
        "Cast from TIME WITH TIME ZONE to {} not yet supported",
        resultType->toString());
  }
};

class TimeWithTimezoneTypeFactory : public CustomTypeFactory {
 public:
  TimeWithTimezoneTypeFactory() = default;

  TypePtr getType(const std::vector<TypeParameter>& parameters) const override {
    VELOX_CHECK(parameters.empty());
    return TIME_WITH_TIME_ZONE();
  }

  exec::CastOperatorPtr getCastOperator() const override {
    return TimeWithTimeZoneCastOperator::get();
  }

  AbstractInputGeneratorPtr getInputGenerator(
      const InputGeneratorConfig& config) const override {
    return std::make_shared<fuzzer::TimeWithTimezoneInputGenerator>(
        config.seed_, config.nullRatio_);
  }
};

} // namespace

void registerTimeWithTimezoneType() {
  registerCustomType(
      "time with time zone",
      std::make_unique<const TimeWithTimezoneTypeFactory>());
}

} // namespace facebook::velox
