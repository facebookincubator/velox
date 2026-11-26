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

#include <gtest/gtest.h>

#include "velox/type/TimestampConversion.h"
#include "velox/type/tz/TimeZoneMap.h"
#include "velox/vector/DecodedVector.h"

namespace facebook::velox::functions::sparksql::test {

struct TimeOnlyTimestampTestCase {
  std::string input;
  std::string effectiveZone;
  int64_t secondsOfDay;
};

inline const std::vector<TimeOnlyTimestampTestCase>
    kTimeOnlyTimestampTestCases = {
        {"10:20:30", "America/Los_Angeles", 37'230},
        {"T01", "America/Los_Angeles", 3'600},
        {"10:20", "America/Los_Angeles", 37'200},
        {"10:20:30 Asia/Tokyo", "Asia/Tokyo", 37'230},
        {"10:20:30+14:00", "+14:00", 37'230},
        {"10:20:30-12:00", "-12:00", 37'230},
        {"T10:20:30 ", "America/Los_Angeles", 37'230},
        {"T01\t", "America/Los_Angeles", 3'600},
        {" 10:20:30", "America/Los_Angeles", 37'230},
        {"\t10:20:30\t", "America/Los_Angeles", 37'230},
};

inline const std::vector<std::string> kInvalidTimeOnlyTimestampInputs = {
    " T10:20:30",
    "\tT01",
    "\nT10:20:30",
    "\rT01",
    "\fT01",
    "\vT01",
    "\tT10:20:30Z",
    " T10:20:30 GMT+01:00",
};

template <typename Evaluate>
void assertTimeOnlyTimestamp(
    const TimeOnlyTimestampTestCase& testCase,
    Evaluate&& evaluate) {
  SCOPED_TRACE(testCase.input);
  const auto* zone = tz::locateZone(testCase.effectiveZone);
  const auto before = util::toDate(Timestamp::now(), zone);
  const auto result = evaluate();
  const auto after = util::toDate(Timestamp::now(), zone);
  DecodedVector decoded(*result);
  ASSERT_FALSE(decoded.isNullAt(0));
  const auto actual = decoded.valueAt<Timestamp>(0);
  const auto expected = [&](int32_t day) {
    Timestamp timestamp(int64_t{day} * 86'400 + testCase.secondsOfDay, 0);
    timestamp.toGMT(*zone);
    return timestamp;
  };
  // Permit evaluation to cross midnight in the effective timezone.
  EXPECT_TRUE(actual == expected(before) || actual == expected(after))
      << actual.toString();
}

} // namespace facebook::velox::functions::sparksql::test
