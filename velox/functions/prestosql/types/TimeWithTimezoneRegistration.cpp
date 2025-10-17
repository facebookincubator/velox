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

#include "velox/expression/CastExpr.h"
#include "velox/functions/prestosql/types/TimeWithTimezoneType.h"
#include "velox/functions/prestosql/types/TimestampWithTimeZoneType.h"
#include "velox/type/Type.h"

namespace facebook::velox {

folly::dynamic TimeWithTimezoneType::serialize() const {
  folly::dynamic obj = folly::dynamic::object;
  obj["name"] = "Type";
  obj["type"] = name();
  return obj;
}

StringView TimeWithTimezoneType::valueToString(
    int64_t value,
    char* const startPos) const {
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

  fmt::format_to_n(
      startPos,
      kTimeWithTimezoneToVarcharRowSize,
      "{:02d}:{:02d}:{:02d}.{:03d}{}{:02d}:{:02d}",
      hours,
      minutes,
      seconds,
      millis,
      isBehindUTCString,
      offsetHours,
      remainingOffsetMinutes);
  return StringView{startPos, kTimeWithTimezoneToVarcharRowSize};
}

namespace {

void castToString(
    const BaseVector& input,
    exec::EvalCtx& context,
    const SelectivityVector& rows,
    BaseVector& result) {
  auto* flatResult = result.asFlatVector<StringView>();
  auto* flatInput = input.asFlatVector<int64_t>();
  Buffer* buffer = flatResult->getBufferWithSpace(
      rows.countSelected() *
          TimeWithTimezoneType::kTimeWithTimezoneToVarcharRowSize,
      true /*exactSize*/);
  char* rawBuffer = buffer->asMutable<char>() + buffer->size();
  context.applyToSelectedNoThrow(rows, [&](vector_size_t row) {
    const auto timeWithTimezone = flatInput->valueAt(row);
    auto output =
        TIME_WITH_TIME_ZONE()->valueToString(timeWithTimezone, rawBuffer);
    flatResult->setNoCopy(row, output);
    rawBuffer += output.size();
  });
  buffer->setSize(rawBuffer - buffer->asMutable<char>());
}

class TimeWithTimeZoneCastOperator final : public exec::CastOperator {
  TimeWithTimeZoneCastOperator() = default;

 public:
  static std::shared_ptr<const exec::CastOperator> get() {
    VELOX_CONSTEXPR_SINGLETON TimeWithTimeZoneCastOperator kInstance;
    return {std::shared_ptr<const exec::CastOperator>{}, &kInstance};
  }

  bool isSupportedToType(const TypePtr& other) const override {
    switch (other->kind()) {
      case TypeKind::VARCHAR:
        return true;
      default:
        return false;
    }
  }

  bool isSupportedFromType(const TypePtr& other) const override {
    switch (other->kind()) {
      default:
        return false;
    }
  }

  void castTo(
      const BaseVector& input,
      exec::EvalCtx& context,
      const SelectivityVector& rows,
      const TypePtr& resultType,
      VectorPtr& result) const override {}

  void castFrom(
      const BaseVector& input,
      exec::EvalCtx& context,
      const SelectivityVector& rows,
      const TypePtr& resultType,
      VectorPtr& result) const override {
    context.ensureWritable(rows, resultType, result);
    switch (resultType->kind()) {
      case TypeKind::VARCHAR:
        castToString(input, context, rows, *result);
        break;
      default:
        VELOX_UNREACHABLE(
            "Cast from TIME WITH TIME ZONE to {} not yet supported",
            resultType->toString());
    }
  }
};

class TimeWithTimezoneTypeFactory : public CustomTypeFactory {
 public:
  TimeWithTimezoneTypeFactory() = default;

  TypePtr getType(const std::vector<TypeParameter>& parameters) const override {
    VELOX_CHECK(parameters.empty());
    return TIME_WITH_TIME_ZONE();
  }

  // Type casting from and to TimestampWithTimezone is not supported yet.
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
