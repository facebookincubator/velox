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

#include "velox/connectors/hive/ConstantFromString.h"

#include "velox/connectors/hive/PartitionValue.h"
#include "velox/functions/prestosql/types/TimestampWithTimeZoneType.h"
#include "velox/type/TimestampConversion.h"
#include "velox/type/tz/TimeZoneMap.h"

namespace facebook::velox::connector::hive {
namespace {

// Zone key for UTC, resolved once instead of hardcoding its numeric value.
TimeZoneKey utcTimeZoneKey() {
  static const TimeZoneKey kUtc = tz::getTimeZoneID("UTC");
  return kUtc;
}

// Parses a TIMESTAMP WITH TIME ZONE value using Presto cast semantics and
// returns the UTC millis packed with the zone key, or nullopt if 'value' is not
// a timestamp string.
std::optional<int64_t> tryPackTimestampWithTimeZone(const std::string& value) {
  auto parsed = util::fromTimestampWithTimezoneString(
      StringView(value), util::TimestampParseMode::kPrestoCast);
  if (parsed.hasError()) {
    return std::nullopt;
  }

  auto [timestamp, timeZone, offsetMillis] = std::move(parsed).value();
  if (timeZone == nullptr) {
    // An offset outside the range covered by named zones parses but has no zone
    // key to pack it with, so reject it instead of packing the wrong millis.
    VELOX_USER_CHECK(
        !offsetMillis.has_value(),
        "Unknown timezone in TIMESTAMP WITH TIME ZONE value: {}",
        value);
    // No zone in the string, so the value is already UTC.
    return pack(timestamp.toMillis(), utcTimeZoneKey());
  }
  timestamp.toGMT(*timeZone);
  return pack(timestamp.toMillis(), timeZone->id());
}

} // namespace

VectorPtr newConstantFromString(
    const TypePtr& type,
    const std::optional<std::string>& value,
    velox::memory::MemoryPool* pool,
    bool isLocalTimestamp,
    bool isDaysSinceEpoch,
    const tz::TimeZone* timezone) {
  if (!value.has_value()) {
    return BaseVector::createNullConstant(type, 1, pool);
  }

  if (isTimestampWithTimeZoneType(type)) {
    if (const auto packed = tryPackTimestampWithTimeZone(*value)) {
      return BaseVector::createConstant(type, Variant(*packed), 1, pool);
    }
    // Not a timestamp string. Fall through to plain BIGINT parsing, which
    // accepts an already packed value and reports the error otherwise.
  }

  return BaseVector::createConstant(
      type,
      PartitionValue::fromString(
          *value,
          *type,
          isLocalTimestamp ? PartitionValue::TimestampMode::kLocalTime
                           : PartitionValue::TimestampMode::kUtc,
          isDaysSinceEpoch ? PartitionValue::DateMode::kDaysSinceEpoch
                           : PartitionValue::DateMode::kIsoString,
          timezone),
      1,
      pool);
}

} // namespace facebook::velox::connector::hive
