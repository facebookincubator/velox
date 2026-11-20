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

#include <gtest/gtest.h>

#include <array>

#include "velox/external/tzdb/tzdb_list.h"

namespace facebook::velox::tzdb {
namespace {

using namespace std::chrono;

void compareInfo(const sys_info& expected, const sys_info& actual) {
  EXPECT_EQ(expected.begin, actual.begin);
  EXPECT_EQ(expected.end, actual.end);
  EXPECT_EQ(expected.offset, actual.offset);
  EXPECT_EQ(expected.save, actual.save);
  EXPECT_TRUE(actual.abbrev.empty());
}

template <typename Duration>
void checkConversion(
    const time_zone& zone,
    const date::local_time<Duration>& local) {
  const auto info = zone.get_info(local);
  const auto conversionInfo = zone.get_info_for_conversion(local);
  ASSERT_EQ(info.result, conversionInfo.result);
  compareInfo(info.first, conversionInfo.first);
  compareInfo(info.second, conversionInfo.second);

  const auto expected = [&](const sys_info& selected) {
    return date::sys_time<Duration>{local.time_since_epoch() - selected.offset};
  };

  // Exercise both choose policies, including gaps and overlaps. The original
  // full-info lookup is the oracle for the numeric conversion.
  EXPECT_EQ(expected(info.first), zone.to_sys(local, choose::earliest));
  EXPECT_EQ(
      expected(info.result == local_info::ambiguous ? info.second : info.first),
      zone.to_sys(local, choose::latest));

  if (info.result == local_info::unique) {
    EXPECT_EQ(expected(info.first), zone.to_sys(local));
  } else {
    const std::string message = info.result == local_info::nonexistent
        ? nonexistent_local_time(local, info).what()
        : ambiguous_local_time(local, info).what();
    try {
      (void)zone.to_sys(local);
      FAIL() << "Expected a local-time conversion error";
    } catch (const nonexistent_local_time& e) {
      EXPECT_EQ(local_info::nonexistent, info.result);
      EXPECT_EQ(message, e.what());
    } catch (const ambiguous_local_time& e) {
      EXPECT_EQ(local_info::ambiguous, info.result);
      EXPECT_EQ(message, e.what());
    }
  }

  // UTC-to-local conversion must also preserve subsecond precision.
  const date::sys_time<Duration> sys{local.time_since_epoch()};
  const auto sysInfo = zone.get_info(sys);
  EXPECT_EQ(
      (date::local_time<Duration>{sys.time_since_epoch() + sysInfo.offset}),
      zone.to_local(sys));

  // Conversions must not mutate the abbreviations used by formatting callers.
  const auto after = zone.get_info(local);
  EXPECT_EQ(info.first.abbrev, after.first.abbrev);
  EXPECT_EQ(info.second.abbrev, after.second.abbrev);
  EXPECT_FALSE(after.first.abbrev.empty());
}

date::local_seconds localTime(
    int year,
    unsigned month,
    unsigned day,
    int hour = 0,
    int minute = 0,
    int second = 0) {
  return date::local_seconds{
      date::local_days{date::year{year} / month / day}.time_since_epoch() +
      hours{hour} + minutes{minute} + seconds{second}};
}

TEST(TzdbConversionTest, historicalAndFutureConversions) {
  for (const auto* name :
       {"UTC",
        "Asia/Shanghai",
        "America/New_York",
        "Europe/London",
        "Pacific/Apia",
        "Australia/Lord_Howe"}) {
    SCOPED_TRACE(name);
    const auto* zone = locate_zone(name);
    for (const auto local :
         {localTime(1800, 1, 1),
          localTime(1900, 1, 1),
          localTime(1969, 12, 31, 23, 59, 59),
          localTime(2026, 7, 1),
          localTime(2100, 1, 1),
          localTime(2100, 7, 1),
          localTime(2400, 7, 1)}) {
      SCOPED_TRACE(local.time_since_epoch().count());
      checkConversion(*zone, local);
      checkConversion(
          *zone,
          date::local_time<milliseconds>{
              duration_cast<milliseconds>(local.time_since_epoch()) +
              milliseconds{123}});
      checkConversion(
          *zone,
          date::local_time<microseconds>{
              duration_cast<microseconds>(local.time_since_epoch()) +
              microseconds{123456}});
    }
  }
}

TEST(TzdbConversionTest, transitionBoundaries) {
  struct Transition {
    const char* name;
    date::local_seconds begin;
    date::local_seconds end;
    date::sys_seconds instant;
    seconds before;
    seconds after;
    int result;
  };
  const auto sysTime =
      [](int year, unsigned month, unsigned day, int hour, int minute = 0) {
        return date::sys_seconds{
            localTime(year, month, day, hour, minute).time_since_epoch()};
      };
  // State the civil-time interval, UTC instant and offsets explicitly. Cover
  // precomputed transitions and future rules, including non-hour changes.
  const std::vector<Transition> transitions = {
      {"America/New_York",
       localTime(2024, 3, 10, 2),
       localTime(2024, 3, 10, 3),
       sysTime(2024, 3, 10, 7),
       hours{-5},
       hours{-4},
       local_info::nonexistent},
      {"America/New_York",
       localTime(2024, 11, 3, 1),
       localTime(2024, 11, 3, 2),
       sysTime(2024, 11, 3, 6),
       hours{-4},
       hours{-5},
       local_info::ambiguous},
      {"America/New_York",
       localTime(2100, 3, 14, 2),
       localTime(2100, 3, 14, 3),
       sysTime(2100, 3, 14, 7),
       hours{-5},
       hours{-4},
       local_info::nonexistent},
      {"America/New_York",
       localTime(2100, 11, 7, 1),
       localTime(2100, 11, 7, 2),
       sysTime(2100, 11, 7, 6),
       hours{-4},
       hours{-5},
       local_info::ambiguous},
      {"Australia/Lord_Howe",
       localTime(2024, 4, 7, 1, 30),
       localTime(2024, 4, 7, 2),
       sysTime(2024, 4, 6, 15),
       hours{11},
       hours{10} + minutes{30},
       local_info::ambiguous},
      {"Australia/Lord_Howe",
       localTime(2024, 10, 6, 2),
       localTime(2024, 10, 6, 2, 30),
       sysTime(2024, 10, 5, 15, 30),
       hours{10} + minutes{30},
       hours{11},
       local_info::nonexistent},
      {"Pacific/Apia",
       localTime(2011, 12, 30),
       localTime(2011, 12, 31),
       sysTime(2011, 12, 30, 10),
       hours{-10},
       hours{14},
       local_info::nonexistent},
  };
  for (const auto& transition : transitions) {
    SCOPED_TRACE(transition.name);
    SCOPED_TRACE(transition.begin.time_since_epoch().count());
    const auto* zone = locate_zone(transition.name);
    const date::local_time<milliseconds> begin{transition.begin};
    const date::local_time<milliseconds> end{transition.end};
    struct LocalCase {
      date::local_time<milliseconds> local;
      int result;
      seconds firstOffset;
      seconds secondOffset;
    };
    // Both gaps and overlaps include their beginning and exclude their end.
    const std::array<LocalCase, 6> cases = {{
        {begin - milliseconds{1},
         local_info::unique,
         transition.before,
         transition.before},
        {begin, transition.result, transition.before, transition.after},
        {begin + milliseconds{1},
         transition.result,
         transition.before,
         transition.after},
        {end - milliseconds{1},
         transition.result,
         transition.before,
         transition.after},
        {end, local_info::unique, transition.after, transition.after},
        {end + milliseconds{1},
         local_info::unique,
         transition.after,
         transition.after},
    }};
    for (const auto& test : cases) {
      SCOPED_TRACE(test.local.time_since_epoch().count());
      const auto info = zone->get_info_for_conversion(test.local);
      ASSERT_EQ(test.result, info.result);
      EXPECT_EQ(test.firstOffset, info.first.offset);
      if (test.result != local_info::unique) {
        EXPECT_EQ(test.secondOffset, info.second.offset);
      }
      EXPECT_EQ(
          (date::sys_time<milliseconds>{
              test.local.time_since_epoch() - test.firstOffset}),
          zone->to_sys(test.local, choose::earliest));
      const auto latestOffset = test.result == local_info::ambiguous
          ? test.secondOffset
          : test.firstOffset;
      EXPECT_EQ(
          (date::sys_time<milliseconds>{
              test.local.time_since_epoch() - latestOffset}),
          zone->to_sys(test.local, choose::latest));
      checkConversion(*zone, test.local);
    }

    // Also exercise the seconds instantiation at both boundaries and inside
    // the interval, including the middle of Apia's skipped calendar date.
    for (const auto local :
         {transition.begin,
          transition.begin + (transition.end - transition.begin) / 2,
          transition.end}) {
      SCOPED_TRACE(local.time_since_epoch().count());
      checkConversion(*zone, local);
    }

    // UTC instants are unambiguous. At the transition itself the new offset
    // applies, and the millisecond component must survive the conversion.
    const date::sys_time<milliseconds> instant{transition.instant};
    for (const auto delta :
         {milliseconds{-1}, milliseconds{0}, milliseconds{1}}) {
      SCOPED_TRACE(delta.count());
      const auto sys = instant + delta;
      const auto offset =
          delta < milliseconds{0} ? transition.before : transition.after;
      EXPECT_EQ(offset, zone->get_info(sys).offset);
      EXPECT_EQ(
          (date::local_time<milliseconds>{sys.time_since_epoch() + offset}),
          zone->to_local(sys));
    }
  }
}

} // namespace
} // namespace facebook::velox::tzdb
