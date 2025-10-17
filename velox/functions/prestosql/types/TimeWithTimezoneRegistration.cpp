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

#include "velox/functions/prestosql/types/TimeWithTimezoneType.h"
#include "velox/functions/prestosql/types/TimestampWithTimeZoneType.h"

#include "velox/expression/CastExpr.h"
#include "velox/type/Type.h"
#include "velox/type/tz/TimeZoneMap.h"

namespace facebook::velox {

folly::dynamic TimeWithTimezoneType::serialize() const {
  folly::dynamic obj = folly::dynamic::object;
  obj["name"] = "Type";
  obj["type"] = name();
  return obj;
}

std::string TimeWithTimezoneType::valueToString(int64_t value) const {
  // TIME WITH TIME ZONE is encoded similarly to TIMESTAMP WITH TIME ZONE
  // with the most significnat 52 bits representing the time component and the
  // least 12 bits representing the timezone minutes. This is different from
  // TIMESTAMP WITH TIMEZONE where the last 12 bits represent the timezone
  // offset. The timezone offset minutes are stored by value, encoded in the
  // type itself. This allows the type to be used in a timezone-agnostic manner.
  //
  // The time component is a 52 bit value representing the number of
  // milliseconds since midnight in UTC.

  int64_t timeComponent = unpackMillisUtc(value);

  // Ensure time component is within valid range
  VELOX_CHECK_GE(timeComponent, 0, "Time component is negative");
  VELOX_CHECK_LE(timeComponent, kMillisInDay, "Time component is too large");

  int64_t hours = timeComponent / kMillisInHour;
  int64_t remainingMs = timeComponent % kMillisInHour;
  int64_t minutes = remainingMs / kMillisInMinute;
  remainingMs = remainingMs % kMillisInMinute;
  int64_t seconds = remainingMs / kMillisInSecond;
  int64_t millis = remainingMs % kMillisInSecond;

  // TimeZone's are encoded as a 12 bit value.
  // This represents a range of -14:00 to +14:00, with 0 representing UTC.
  // The range is from -840 to 840 minutes, we thus encode by doing bias
  // encoding and taking 840 as the bias.
  auto timezoneMinutes = unpackZoneKeyId(value);

  VELOX_CHECK_GE(timezoneMinutes, 0, "Timezone offset is less than -14:00");
  VELOX_CHECK_LE(
      timezoneMinutes, 1680, "Timezone offset is greater than +14:00");

  auto decodedMinutes = timezoneMinutes >= kTimeZoneBias
      ? timezoneMinutes - kTimeZoneBias
      : kTimeZoneBias - timezoneMinutes;

  const auto isBehindUTCString = timezoneMinutes >= kTimeZoneBias ? "+" : "-";

  int16_t offsetHours = decodedMinutes / kMinutesInHour;
  int16_t remainingOffsetMinutes = decodedMinutes % kMinutesInHour;

  return fmt::format(
      "{:02d}:{:02d}:{:02d}.{:03d}{}{:02d}:{:02d}",
      hours,
      minutes,
      seconds,
      millis,
      isBehindUTCString,
      offsetHours,
      remainingOffsetMinutes);
}

namespace {

const tz::TimeZone* getTimeZoneFromConfig(const core::QueryConfig& config) {
  const auto sessionTzName = config.sessionTimezone();

  if (!sessionTzName.empty()) {
    return tz::locateZone(sessionTzName);
  }

  return tz::locateZone(0); // GMT
}

// Pack time (in milliseconds since midnight) and timezone offset minutes
inline int64_t packTimeWithTimeZone(
    int64_t timeMillis,
    int16_t timezoneOffsetMinutes) {
  auto encodedOffset = TimeWithTimezoneType::biasEncode(timezoneOffsetMinutes);
  return pack(timeMillis, encodedOffset);
}

void castFromTime(
    const SimpleVector<int64_t>& inputVector,
    exec::EvalCtx& context,
    const SelectivityVector& rows,
    int64_t* rawResults) {
  const auto& config = context.execCtx()->queryCtx()->queryConfig();
  const auto* sessionTimeZone = getTimeZoneFromConfig(config);

  const auto adjustTimestampToTimezone = config.adjustTimestampToTimezone();

  context.applyToSelectedNoThrow(rows, [&](auto row) {
    const auto timeMillis = inputVector.valueAt(row);

    int16_t offsetMinutes = 0;

    if (adjustTimestampToTimezone) {
      // Get the offset for the session timezone at epoch (1970-01-01)
      // We use epoch to get a consistent offset (ignoring DST variations)
      // toGMT converts FROM local time TO GMT, so the offset is negative of the
      // result
      auto ts = Timestamp::fromMillis(0);
      ts.toGMT(*sessionTimeZone);
      int64_t offsetMillis = ts.toMillis();
      // The offset is negative because toGMT added the offset to convert to GMT
      offsetMinutes = static_cast<int16_t>(-offsetMillis / (60 * 1000));
    }

    rawResults[row] = packTimeWithTimeZone(timeMillis, offsetMinutes);
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
      default:
        return false;
    }
  }

  bool isSupportedToType(const TypePtr& other) const override {
    // Currently no casting from TIME WITH TIME ZONE to other types
    return false;
  }

  void castTo(
      const BaseVector& input,
      exec::EvalCtx& context,
      const SelectivityVector& rows,
      const TypePtr& resultType,
      VectorPtr& result) const override {
    context.ensureWritable(rows, resultType, result);

    // Optimization for constant TIME input vectors
    if (input.typeKind() == TypeKind::BIGINT && input.type()->isTime() &&
        input.isConstantEncoding()) {
      auto constantInput = input.as<ConstantVector<int64_t>>();
      if (constantInput->isNullAt(0)) {
        result = BaseVector::createNullConstant(
            resultType, rows.end(), context.pool());
        return;
      }

      const auto& config = context.execCtx()->queryCtx()->queryConfig();
      const auto* sessionTimeZone = getTimeZoneFromConfig(config);
      const auto adjustTimestampToTimezone = config.adjustTimestampToTimezone();

      const auto timeMillis = constantInput->valueAt(0);

      int16_t offsetMinutes = 0;

      if (adjustTimestampToTimezone) {
        auto ts = Timestamp::fromMillis(0);
        ts.toGMT(*sessionTimeZone);
        int64_t offsetMillis = ts.toMillis();
        offsetMinutes = static_cast<int16_t>(-offsetMillis / (60 * 1000));
      }

      auto packedValue = packTimeWithTimeZone(timeMillis, offsetMinutes);

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

    if (input.typeKind() == TypeKind::BIGINT) {
      VELOX_CHECK(input.type()->isTime());
      const auto inputVector = input.as<SimpleVector<int64_t>>();
      castFromTime(*inputVector, context, rows, rawResults);
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
    return nullptr;
  }
};

} // namespace

void registerTimeWithTimezoneType() {
  registerCustomType(
      "time with time zone",
      std::make_unique<const TimeWithTimezoneTypeFactory>());
}

} // namespace facebook::velox
