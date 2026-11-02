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

#include "velox/connectors/hive/PartitionValue.h"

#include <string>
#include <type_traits>

#include <folly/Conv.h>

#include "velox/common/base/Exceptions.h"
#include "velox/functions/prestosql/types/TimestampWithTimeZoneType.h"
#include "velox/type/Conversions.h"
#include "velox/type/DecimalUtil.h"
#include "velox/type/TimestampConversion.h"
#include "velox/type/tz/TimeZoneMap.h"

namespace facebook::velox::connector::hive {
namespace {

// Parses 'value' as a TIMESTAMP WITH TIME ZONE using CAST(varchar AS TIMESTAMP
// WITH TIME ZONE) syntax and returns its UTC milliseconds packed with its time
// zone key. A value naming a zone or carrying an offset is shifted to UTC and
// recorded with that zone. A value with no zone is taken as already UTC, which
// diverges from the cast: CAST would interpret it in the session time zone,
// but a partition value has no session and would otherwise be ambiguous.
// Sub-millisecond precision is truncated, since the packed layout has no room
// for it.
//
// Returns nullopt if 'value' is not a timestamp string at all, leaving the
// caller to decide how to interpret it.
std::optional<int64_t> tryPackTimestampWithTimeZone(std::string_view value) {
  auto parsed = util::fromTimestampWithTimezoneString(
      StringView(value), util::TimestampParseMode::kPrestoCast);
  if (parsed.hasError()) {
    return std::nullopt;
  }

  auto [timestamp, timeZone, offsetMillis] = std::move(parsed).value();
  if (timeZone == nullptr) {
    // An offset outside the range covered by named zones parses but has no key
    // to pack with, so reject it rather than record the instant under a
    // different zone.
    VELOX_USER_CHECK(
        !offsetMillis.has_value(),
        "Unknown timezone in TIMESTAMP WITH TIME ZONE value: {}",
        value);
    // The string carries no zone, so the value is already UTC.
    static const TimeZoneKey kUtcKey = tz::getTimeZoneID("UTC");
    return pack(timestamp.toMillis(), kUtcKey);
  }

  timestamp.toGMT(*timeZone);
  return pack(timestamp.toMillis(), timeZone->id());
}

Variant fromTimestampWithTimeZoneString(std::string_view value) {
  if (const auto packed = tryPackTimestampWithTimeZone(value)) {
    return Variant(*packed);
  }
  // Not a timestamp string. An already packed integer is accepted too, so a
  // table can store pre-packed values, matching how an unannotated INT64
  // Parquet column read as this type passes its values through.
  const auto packed = folly::tryTo<int64_t>(value);
  VELOX_USER_CHECK(
      packed.hasValue(),
      "Cannot convert value to TIMESTAMP WITH TIME ZONE: {}",
      value);
  return Variant(packed.value());
}


template <TypeKind kind>
Variant fromStringImpl(
    std::string_view value,
    const Type& type,
    PartitionValue::TimestampMode timestampMode,
    PartitionValue::DateMode dateMode,
    const tz::TimeZone* timezone) {
  using NativeType = typename TypeTraits<kind>::NativeType;

  if (type.isDate()) {
    const auto days = dateMode == PartitionValue::DateMode::kDaysSinceEpoch
        ? folly::to<int32_t>(value)
        : DATE()->toDays(value);
    return Variant(days);
  }

  if constexpr (
      std::is_same_v<NativeType, int64_t> ||
      std::is_same_v<NativeType, int128_t>) {
    if (type.isDecimal()) {
      NativeType decimalValue{0};
      const auto [precision, scale] = getDecimalPrecisionScale(type);
      const auto status = DecimalUtil::castFromString(
          StringView(value), precision, scale, decimalValue);
      VELOX_USER_CHECK(status.ok(), "{}", status.message());
      return Variant::create<kind>(decimalValue);
    }
  }

  if constexpr (std::is_same_v<NativeType, StringView>) {
    return Variant::create<kind>(std::string(value));
  } else {
    auto converted = util::Converter<kind>::tryCast(value).thenOrThrow(
        folly::identity,
        [&](const Status& status) { VELOX_USER_FAIL("{}", status.message()); });
    if constexpr (kind == TypeKind::TIMESTAMP) {
      // A TIMESTAMP_UTC value is always UTC and is never shifted.
      if (type.equivalent(*TIMESTAMP())) {
        if (timezone != nullptr) {
          converted.toGMT(*timezone);
        } else if (timestampMode == PartitionValue::TimestampMode::kLocalTime) {
          converted.toGMT(Timestamp::defaultTimezone());
        }
      }
    }
    return Variant::create<kind>(converted);
  }
}

} // namespace

// static
Variant PartitionValue::fromString(
    std::string_view value,
    const Type& type,
    TimestampMode timestampMode,
    DateMode dateMode,
    const tz::TimeZone* timezone) {
  // TimestampWithTimeZoneType has kind BIGINT, so the dispatch below would
  // route it to the integer parse. Handle it up front instead.
  if (isTimestampWithTimeZoneType(type)) {
    return fromTimestampWithTimeZoneString(value);
  }
  return VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH(
      fromStringImpl,
      type.kind(),
      value,
      type,
      timestampMode,
      dateMode,
      timezone);
}

} // namespace facebook::velox::connector::hive
