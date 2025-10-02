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
#include "velox/type/Type.h"

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

class TimeWithTimezoneTypeFactory : public CustomTypeFactory {
 public:
  TimeWithTimezoneTypeFactory() = default;

  TypePtr getType(const std::vector<TypeParameter>& parameters) const override {
    VELOX_CHECK(parameters.empty());
    return TIME_WITH_TIME_ZONE();
  }

  // Type casting from and to TimestampWithTimezone is not supported yet.
  exec::CastOperatorPtr getCastOperator() const override {
    return nullptr;
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
