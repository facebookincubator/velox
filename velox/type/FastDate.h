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

// The Gregorian-date algorithms below are adapted from supplementary code
// to the paper:
//
//   Cassio Neri and Lorenz Schneider,
//   "Euclidean Affine Functions and their Application to Calendar
//   Algorithms" (2022).
//
// Original source:
//   https://github.com/benjoffe/fast-date-benchmarks/blob/main/algorithms/neri_schneider.hpp
//   (which mirrors the reference implementation from the paper)
//
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2022 Cassio Neri <cassio.neri@gmail.com>
// SPDX-FileCopyrightText: 2022 Lorenz Schneider <schneider@em-lyon.com>

#include <cstdint>

namespace facebook::velox {

/// Result of converting an epoch-day to a Gregorian (year, month, day).
/// Year is signed (proleptic Gregorian), month is 1-12, day is 1-31.
struct YearMonthDay {
  int32_t year;
  uint32_t month;
  uint32_t day;
};

namespace fast_date {

// Era shift used by the Neri-Schneider algorithm. The paper calls this
// `s`; 82 was chosen so the supported range covers everything Velox can
// natively store.
inline constexpr uint32_t kEraShift = 82u;

// Epoch-day offset (paper's `K`): added to dayNumber so the algorithm
// operates on a non-negative integer for the entire supported range.
inline constexpr uint32_t kEpochOffset = 719468u + 146097u * kEraShift;

// Year offset (paper's `L`): subtracted from the algorithm's internal
// "year-within-era" representation to recover a real proleptic-Gregorian
// year.
inline constexpr uint32_t kYearOffset = 400u * kEraShift;

// Supported input ranges (computed from the algorithm's affine arithmetic).
//
// Note on the boundary years: the algorithm is parameterized from March,
// not January. Its exact range is [Mar 1 kYearMin, Feb 28 kYearMax], not
// [Jan 1 kYearMin, Dec 31 kYearMax]. For the inverse direction, callers
// should restrict to year ∈ (kYearMin, kYearMax) — strictly excluding the
// boundary years — to avoid the month-dependent wrinkle. The forward
// direction has no such caveat because rata-die is monotonic.
inline constexpr int32_t kRataDieMin = -12'699'422; // 1 Mar -32800
inline constexpr int32_t kRataDieMax = 1'061'042'401; // 28 Feb 2906945
inline constexpr int32_t kYearMin = -32800;
inline constexpr int32_t kYearMax = 2'906'945;

} // namespace fast_date

/// Converts an epoch-day count (days since 1970-01-01) to the corresponding
/// proleptic Gregorian (year, month, day). Exact for any dayNumber in
/// [fast_date::kRataDieMin, fast_date::kRataDieMax]; results outside that
/// range are undefined. The supported range covers ~3 million years
/// centered on the epoch — vastly wider than any practical Velox DATE.
///
/// Variable mapping (Neri-Schneider 2022, §5):
///   Paper | Local                | Meaning
///   ------|----------------------|---------------------------------------
///   N     | shiftedDay           | day-since-epoch rebased to be non-neg
///   N_1   | centuryNumerator     | numerator for the century division
///   C     | century              | century within the era
///   N_C   | dayWithinCentury     | day within the century
///   N_2   | yearNumerator        | numerator for year-within-century
///   P_2   | yearProduct          | 64-bit multiply-shift reciprocal
///   Z     | yearWithinCentury    | year within the century
///   N_Y   | dayWithinYear        | day within the (March-based) year
///   Y     | yearWithinEra        | year within the era
///   N_3   | monthNumerator       | numerator for the month/day division
///   M     | monthFromMarch       | month index counted from March
///   D     | dayOfMonthZeroBased  | zero-based day-of-month
///   J     | janFebAdjust         | 1 if Jan/Feb of next calendar year
inline YearMonthDay daysToYmd(int32_t dayNumber) {
  using namespace fast_date;
  const uint32_t shiftedDay = static_cast<uint32_t>(dayNumber) + kEpochOffset;
  // Century.
  const uint32_t centuryNumerator = 4u * shiftedDay + 3u;
  const uint32_t century = centuryNumerator / 146097u;
  const uint32_t dayWithinCentury = centuryNumerator % 146097u / 4u;
  // Year within century.
  const uint32_t yearNumerator = 4u * dayWithinCentury + 3u;
  const uint64_t yearProduct = uint64_t{2939745u} * yearNumerator;
  const uint32_t yearWithinCentury = static_cast<uint32_t>(yearProduct >> 32);
  const uint32_t dayWithinYear =
      static_cast<uint32_t>(yearProduct & 0xFFFF'FFFFull) / 2939745u / 4u;
  const uint32_t yearWithinEra = 100u * century + yearWithinCentury;
  // Month and day within year (year here starts at March).
  const uint32_t monthNumerator = 2141u * dayWithinYear + 197913u;
  const uint32_t monthFromMarch = monthNumerator / 65536u;
  const uint32_t dayOfMonthZeroBased = monthNumerator % 65536u / 2141u;
  // Years are counted from March, so January/February belong to the next
  // calendar year; correct that here.
  const uint32_t janFebAdjust = dayWithinYear >= 306u ? 1u : 0u;
  YearMonthDay out;
  out.year = static_cast<int32_t>(yearWithinEra - kYearOffset) +
      static_cast<int32_t>(janFebAdjust);
  out.month = janFebAdjust ? monthFromMarch - 12u : monthFromMarch;
  out.day = dayOfMonthZeroBased + 1u;
  return out;
}

/// Inverse of daysToYmd. Safe for year ∈ (fast_date::kYearMin,
/// fast_date::kYearMax) — strictly between the two boundaries — for any
/// month and day. At the boundary years themselves, only the algorithm's
/// March-based partial year is exact (see the constants' comment); use
/// WideRangeDateConversion::daysSinceEpochFromDate for those instead.
///
/// Variable mapping (Neri-Schneider 2022, §6):
///   Paper  | Local                | Meaning
///   -------|----------------------|--------------------------------------
///   J      | janFebAdjust         | 1 if input month is Jan/Feb
///   Y      | shiftedYear          | year within the era after Jan/Feb fix
///   M      | monthFromMarch       | month index counted from March
///   D      | dayOfMonthZeroBased  | zero-based day-of-month
///   C      | century              | century within the era
///   y_star | yearDays             | days from era start to year start
///   m_star | monthDays            | days from year start to month start
///   N      | shiftedRataDie       | shifted day-since-epoch (pre-offset)
inline int32_t ymdToDays(int32_t year, uint32_t month, uint32_t day) {
  using namespace fast_date;
  const uint32_t janFebAdjust = month <= 2u ? 1u : 0u;
  const uint32_t shiftedYear =
      (static_cast<uint32_t>(year) + kYearOffset) - janFebAdjust;
  const uint32_t monthFromMarch = janFebAdjust ? month + 12u : month;
  const uint32_t dayOfMonthZeroBased = day - 1u;
  const uint32_t century = shiftedYear / 100u;
  const uint32_t yearDays = 1461u * shiftedYear / 4u - century + century / 4u;
  const uint32_t monthDays = (979u * monthFromMarch - 2919u) / 32u;
  const uint32_t shiftedRataDie = yearDays + monthDays + dayOfMonthZeroBased;
  return static_cast<int32_t>(shiftedRataDie - kEpochOffset);
}

} // namespace facebook::velox
