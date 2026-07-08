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
#include <ctime>
#include "velox/common/base/Status.h"

namespace facebook::velox {

/// Loop-based date conversion that handles the full validation range
/// [kMinYear, kMaxYear]. Slower than the Neri-Schneider fast path in
/// FastDate.h, but covers inputs outside that fast path's exact domain
/// (~3M years centered on the epoch). Used as the fallback inside
/// Timestamp::epochToCalendarUtc and util::daysSinceEpochFromDate.
///
/// Exposed as a real production class (not a test-only helper) so the
/// fast/fallback equivalence can be asserted directly from tests, and so
/// the benchmark file can compare without duplicating the loop body.
class WideRangeDateConversion {
 public:
  /// Converts seconds-since-epoch to a calendar-UTC std::tm using the
  /// loop-based path. Returns false if the result year does not fit in
  /// std::tm::tm_year or the input is otherwise unrepresentable.
  static bool epochToCalendarUtc(int64_t epoch, std::tm& tm);

  /// Converts a (year, month, day) triple to days-since-epoch using the
  /// loop-based path. Validates the input against [kMinYear, kMaxYear];
  /// returns Status::UserError ("Date out of range: ...") on invalid
  /// input.
  static Expected<int64_t>
  daysSinceEpochFromDate(int32_t year, int32_t month, int32_t day);
};

} // namespace facebook::velox
