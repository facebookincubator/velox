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

TEST(TzdbConversionTest, transitionsAndExceptionMessages) {
  // US one-hour DST changes, Lord Howe's half-hour changes, and Apia's
  // skipped date. Include future DST rules beyond the precomputed table.
  const std::vector<std::pair<const char*, date::local_seconds>> cases = {
      {"America/New_York", localTime(2024, 3, 10, 2, 30)},
      {"America/New_York", localTime(2024, 11, 3, 1, 30)},
      {"America/New_York", localTime(2100, 3, 14, 2, 30)},
      {"America/New_York", localTime(2100, 11, 7, 1, 30)},
      {"Australia/Lord_Howe", localTime(2024, 4, 7, 1, 45)},
      {"Australia/Lord_Howe", localTime(2024, 10, 6, 2, 15)},
      {"Pacific/Apia", localTime(2011, 12, 30, 12)},
  };
  for (const auto& [name, local] : cases) {
    SCOPED_TRACE(name);
    const auto* zone = locate_zone(name);
    ASSERT_NE(local_info::unique, zone->get_info(local).result);
    for (const auto delta : {-hours{2}, hours{0}, hours{2}}) {
      checkConversion(*zone, local + delta);
      checkConversion(
          *zone,
          date::local_time<milliseconds>{
              duration_cast<milliseconds>((local + delta).time_since_epoch()) +
              milliseconds{999}});
    }
  }
}

} // namespace
} // namespace facebook::velox::tzdb
