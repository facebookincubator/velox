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

// Era shift used by the Neri-Schneider algorithm. s = 82 chosen so the
// supported range covers everything Velox can natively store.
inline constexpr uint32_t kS = 82u;
inline constexpr uint32_t kK = 719468u + 146097u * kS;
inline constexpr uint32_t kL = 400u * kS;

// Supported input ranges (computed from the algorithm's affine arithmetic).
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
inline YearMonthDay daysToYmd(int32_t dayNumber) {
  using namespace fast_date;
  const uint32_t N = static_cast<uint32_t>(dayNumber) + kK;
  // Century.
  const uint32_t N_1 = 4u * N + 3u;
  const uint32_t C = N_1 / 146097u;
  const uint32_t N_C = N_1 % 146097u / 4u;
  // Year within century.
  const uint32_t N_2 = 4u * N_C + 3u;
  const uint64_t P_2 = uint64_t{2939745u} * N_2;
  const uint32_t Z = static_cast<uint32_t>(P_2 >> 32);
  const uint32_t N_Y = static_cast<uint32_t>(P_2 & 0xFFFF'FFFFull) / 2939745u / 4u;
  const uint32_t Y = 100u * C + Z;
  // Month and day within year.
  const uint32_t N_3 = 2141u * N_Y + 197913u;
  const uint32_t M = N_3 / 65536u;
  const uint32_t D = N_3 % 65536u / 2141u;
  // Map: years are counted from March, so January/February belong to next
  // calendar year; correct that here.
  const uint32_t J = N_Y >= 306u ? 1u : 0u;
  YearMonthDay out;
  out.year = static_cast<int32_t>(Y - kL) + static_cast<int32_t>(J);
  out.month = J ? M - 12u : M;
  out.day = D + 1u;
  return out;
}

/// Inverse of daysToYmd. Year domain is
/// [fast_date::kYearMin, fast_date::kYearMax]; results outside that range
/// are undefined.
inline int32_t ymdToDays(int32_t year, uint32_t month, uint32_t day) {
  using namespace fast_date;
  const uint32_t J = month <= 2u ? 1u : 0u;
  const uint32_t Y = (static_cast<uint32_t>(year) + kL) - J;
  const uint32_t M = J ? month + 12u : month;
  const uint32_t D = day - 1u;
  const uint32_t C = Y / 100u;
  const uint32_t y_star = 1461u * Y / 4u - C + C / 4u;
  const uint32_t m_star = (979u * M - 2919u) / 32u;
  const uint32_t N = y_star + m_star + D;
  return static_cast<int32_t>(N - kK);
}

} // namespace facebook::velox
