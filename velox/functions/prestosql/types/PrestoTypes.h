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

#include <fmt/format.h>
#include "velox/functions/prestosql/types/BingTileType.h"
#include "velox/functions/prestosql/types/IPAddressType.h"
#include "velox/functions/prestosql/types/IPPrefixType.h"
#include "velox/functions/prestosql/types/TimeWithTimezoneType.h"
#include "velox/functions/prestosql/types/TimestampWithTimeZoneType.h"
#include "velox/functions/prestosql/types/UuidType.h"
#include "velox/type/Type.h"
#include "velox/vector/DecodedVector.h"

namespace facebook::velox {

/// Groups Presto-specific type operations: SQL type rendering and
/// human-readable value formatting for Presto custom types (BINGTILE,
/// IPADDRESS, UUID, IPPREFIX, TIMESTAMP WITH TIME ZONE, TIME WITH TIME ZONE)
/// on top of the core Type::valueToString template.
class PrestoTypes {
 public:
  /// Returns the Presto SQL string representation of the given type.
  static std::string toSql(const TypePtr& type);

  /// Formats a scalar value as a human-readable string using Presto type
  /// conventions. Handles Presto custom types (IPADDRESS, UUID,
  /// TIMESTAMP WITH TIME ZONE, TIME WITH TIME ZONE) in addition to core types
  /// handled by Type::valueToString.
  template <typename T>
  static std::string valueToString(T value, const TypePtr& type);

  /// Formats a value from a decoded vector as a human-readable string
  /// matching Presto's output format. Returns "null" for null values.
  /// Handles scalar types, composite types (IPPREFIX), and complex types
  /// (ARRAY, MAP, ROW) with recursive rendering.
  static std::string valueToString(
      const DecodedVector& decoded,
      vector_size_t index,
      const TypePtr& type);

 private:
  // Formats a Timestamp in Presto format: space separator, millisecond
  // precision, zero-padded year. Consistent with PrestoCastHooks (non-legacy)
  // options for CAST(timestamp AS varchar).
  static std::string timestampToPrestoString(Timestamp value);

  // Formats binary data as space-separated hex bytes (e.g., "02 0c 01 00").
  static std::string toHex(StringView value);
};

template <typename T>
std::string PrestoTypes::valueToString(T value, const TypePtr& type) {
  if constexpr (std::is_same_v<T, int128_t>) {
    if (isIPAddressType(type)) {
      return IPADDRESS()->valueToString(value);
    }
    if (isUuidType(type)) {
      char buffer[UuidType::kStringSize];
      return std::string(UUID()->valueToString(value, buffer));
    }
  } else if constexpr (std::is_same_v<T, int64_t>) {
    if (isTimestampWithTimeZoneType(type)) {
      return TIMESTAMP_WITH_TIME_ZONE()->valueToString(value);
    }
    if (isTimeWithTimeZone(type)) {
      char buffer[TimeWithTimezoneType::kTimeWithTimezoneToVarcharRowSize];
      return std::string(TIME_WITH_TIME_ZONE()->valueToString(value, buffer));
    }
    if (isBingTileType(type)) {
      return fmt::format(
          "{{x={}, y={}, zoom={}}}",
          BingTileType::bingTileX(value),
          BingTileType::bingTileY(value),
          BingTileType::bingTileZoom(value));
    }
  } else if constexpr (std::is_same_v<T, Timestamp>) {
    return timestampToPrestoString(value);
  } else if constexpr (std::is_same_v<T, StringView>) {
    if (type->isVarbinary()) {
      return toHex(value);
    }
  }
  return type->valueToString(value);
}

} // namespace facebook::velox
