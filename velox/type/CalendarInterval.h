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

#include <cstdint>
#include <string>

#include "velox/type/Type.h"

namespace facebook::velox {

/// C++ equivalent of Spark's CalendarInterval.
/// Three independent fields: months, days, microseconds.
/// Months and days are calendar units and must NOT be normalized.
///
/// Packed into int128_t matching Spark UnsafeRow layout:
///   low  64 bits: months (low 32) | days (high 32)
///   high 64 bits: microseconds
struct CalendarInterval {
  int32_t months{0};
  int32_t days{0};
  int64_t microseconds{0};

  CalendarInterval() = default;

  CalendarInterval(int32_t months, int32_t days, int64_t microseconds)
      : months(months), days(days), microseconds(microseconds) {}

  /// Pack into int128_t for storage in FlatVector<int128_t>.
  int128_t pack() const {
    static_assert(
        sizeof(int128_t) == 16,
        "int128_t must be 16 bytes for CalendarInterval packing");
    // Use unsigned arithmetic to avoid undefined behavior from
    // left-shifting negative signed values.
    const uint64_t low = static_cast<uint64_t>(static_cast<uint32_t>(months)) |
        (static_cast<uint64_t>(static_cast<uint32_t>(days)) << 32);
    using u128 = unsigned __int128;
    const u128 packed = static_cast<u128>(low) |
        (static_cast<u128>(static_cast<uint64_t>(microseconds)) << 64);
    return static_cast<int128_t>(packed);
  }

  /// Unpack from int128_t stored in FlatVector<int128_t>.
  static CalendarInterval unpack(int128_t packed) {
    int64_t low = static_cast<int64_t>(packed);
    int32_t months = static_cast<int32_t>(low);
    int32_t days = static_cast<int32_t>(static_cast<uint64_t>(low) >> 32);
    int64_t microseconds = static_cast<int64_t>(packed >> 64);
    return CalendarInterval(months, days, microseconds);
  }

  bool operator==(const CalendarInterval& other) const {
    return months == other.months && days == other.days &&
        microseconds == other.microseconds;
  }

  bool operator!=(const CalendarInterval& other) const {
    return !(*this == other);
  }

  /// Lexicographic comparison by months, then days, then microseconds.
  /// Used for grouping-like operations (equality partitioning).
  /// Note: CalendarInterval is comparable but NOT orderable in Spark
  /// because months and days are not fixed-duration units.
  int compare(const CalendarInterval& other) const {
    if (months != other.months) {
      return months < other.months ? -1 : 1;
    }
    if (days != other.days) {
      return days < other.days ? -1 : 1;
    }
    if (microseconds != other.microseconds) {
      return microseconds < other.microseconds ? -1 : 1;
    }
    return 0;
  }

  /// Format matching Spark's CalendarInterval.toString().
  /// Examples:
  ///   {0, 0, 0}        -> "0 seconds"
  ///   {14, 0, 0}       -> "1 years 2 months"
  ///   {0, 5, 0}        -> "5 days"
  ///   {0, 0, 3723000000} -> "1 hours 2 minutes 3 seconds"
  ///   {14, 5, 3723000000} -> "1 years 2 months 5 days 1 hours 2 minutes 3
  ///   seconds"
  std::string toString() const;
};

} // namespace facebook::velox
