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

#pragma once

#include <string_view>

#include "velox/type/Type.h"
#include "velox/type/Variant.h"

namespace facebook::velox::tz {
class TimeZone;
}

namespace facebook::velox::connector::hive {

/// Converts partition key strings to typed values. TIMESTAMP and DATE have
/// more than one encoding, so the caller states which one applies.
///
///   auto value = PartitionValue::fromString(
///       "2020-01-01 12:34:56",
///       *TIMESTAMP(),
///       PartitionValue::TimestampMode::kLocalTime,
///       PartitionValue::DateMode::kIsoString);
class PartitionValue {
 public:
  enum class TimestampMode {
    /// Interprets the value as local time and shifts it to UTC. A
    /// TIMESTAMP_UTC value is not shifted.
    kLocalTime,

    /// Interprets the value as UTC. No shift.
    kUtc,
  };

  enum class DateMode {
    /// Parses an ISO date, for example "2020-01-02".
    kIsoString,

    /// Parses an integer count of days since the epoch, for example "18263".
    kDaysSinceEpoch,
  };

  /// 'value' must be non-null. Accepted input per type:
  /// - BOOLEAN: t, f, 1, 0, true or false, case-insensitively.
  /// - TINYINT, SMALLINT, INTEGER, BIGINT: an integer, range-checked against
  ///   the native type rather than widened.
  /// - REAL, DOUBLE: a floating point literal.
  /// - DECIMAL: a decimal literal, scaled by the type's scale.
  /// - VARCHAR, VARBINARY: taken verbatim.
  /// - TIMESTAMP: parsed as TimestampParseMode::kPrestoCast, then shifted per
  ///   'timezone' and 'timestampMode'.
  /// - DATE: parsed per 'dateMode'.
  ///
  /// 'timezone', when non-null, is the zone a TIMESTAMP value is read in
  /// before being shifted to UTC. It takes precedence over 'timestampMode',
  /// which falls back to the process default zone for kLocalTime. Delta Lake
  /// and Iceberg use it to honor the session timezone.
  ///
  /// Fails for a non-scalar type, and for a value that does not parse as
  /// 'type'.
  static Variant fromString(
      std::string_view value,
      const Type& type,
      TimestampMode timestampMode,
      DateMode dateMode,
      const tz::TimeZone* timezone = nullptr);
};

} // namespace facebook::velox::connector::hive
