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

#include "velox/type/WideRangeDateConversion.h"

#include <algorithm>
#include <limits>
#include "velox/common/base/VeloxException.h"
#include "velox/type/TimestampConversion.h"

namespace facebook::velox {
namespace {

constexpr int kTmYearBase = 1900;
constexpr int64_t kLeapYearOffset = 4'000'000'000LL;
constexpr int64_t kSecondsPerHour = 3600;
constexpr int64_t kSecondsPerDay = 24 * kSecondsPerHour;

inline bool isLeap(int64_t y) {
  return y % 4 == 0 && (y % 100 != 0 || y % 400 == 0);
}

inline int64_t leapThroughEndOf(int64_t y) {
  // Add a large offset so the calculation works for negative years.
  y += kLeapYearOffset;
  return y / 4 - y / 100 + y / 400;
}

inline int64_t daysBetweenYears(int64_t y1, int64_t y2) {
  return 365 * (y2 - y1) + leapThroughEndOf(y2 - 1) - leapThroughEndOf(y1 - 1);
}

constexpr int16_t kDaysBeforeFirstDayOfMonth[][12] = {
    {0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334},
    {0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335},
};

constexpr int32_t kCumulativeDays[] =
    {0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365};
constexpr int32_t kCumulativeLeapDays[] =
    {0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335, 366};

// Days from 1970-01-01 to the start of each year in [1970, 2370].
// The inverse path uses this table for years inside the era and
// 400-year-step loops for years outside it.
constexpr int32_t kCumulativeYearDays[] = {
    0,      365,    730,    1096,   1461,   1826,   2191,   2557,   2922,
    3287,   3652,   4018,   4383,   4748,   5113,   5479,   5844,   6209,
    6574,   6940,   7305,   7670,   8035,   8401,   8766,   9131,   9496,
    9862,   10227,  10592,  10957,  11323,  11688,  12053,  12418,  12784,
    13149,  13514,  13879,  14245,  14610,  14975,  15340,  15706,  16071,
    16436,  16801,  17167,  17532,  17897,  18262,  18628,  18993,  19358,
    19723,  20089,  20454,  20819,  21184,  21550,  21915,  22280,  22645,
    23011,  23376,  23741,  24106,  24472,  24837,  25202,  25567,  25933,
    26298,  26663,  27028,  27394,  27759,  28124,  28489,  28855,  29220,
    29585,  29950,  30316,  30681,  31046,  31411,  31777,  32142,  32507,
    32872,  33238,  33603,  33968,  34333,  34699,  35064,  35429,  35794,
    36160,  36525,  36890,  37255,  37621,  37986,  38351,  38716,  39082,
    39447,  39812,  40177,  40543,  40908,  41273,  41638,  42004,  42369,
    42734,  43099,  43465,  43830,  44195,  44560,  44926,  45291,  45656,
    46021,  46387,  46752,  47117,  47482,  47847,  48212,  48577,  48942,
    49308,  49673,  50038,  50403,  50769,  51134,  51499,  51864,  52230,
    52595,  52960,  53325,  53691,  54056,  54421,  54786,  55152,  55517,
    55882,  56247,  56613,  56978,  57343,  57708,  58074,  58439,  58804,
    59169,  59535,  59900,  60265,  60630,  60996,  61361,  61726,  62091,
    62457,  62822,  63187,  63552,  63918,  64283,  64648,  65013,  65379,
    65744,  66109,  66474,  66840,  67205,  67570,  67935,  68301,  68666,
    69031,  69396,  69762,  70127,  70492,  70857,  71223,  71588,  71953,
    72318,  72684,  73049,  73414,  73779,  74145,  74510,  74875,  75240,
    75606,  75971,  76336,  76701,  77067,  77432,  77797,  78162,  78528,
    78893,  79258,  79623,  79989,  80354,  80719,  81084,  81450,  81815,
    82180,  82545,  82911,  83276,  83641,  84006,  84371,  84736,  85101,
    85466,  85832,  86197,  86562,  86927,  87293,  87658,  88023,  88388,
    88754,  89119,  89484,  89849,  90215,  90580,  90945,  91310,  91676,
    92041,  92406,  92771,  93137,  93502,  93867,  94232,  94598,  94963,
    95328,  95693,  96059,  96424,  96789,  97154,  97520,  97885,  98250,
    98615,  98981,  99346,  99711,  100076, 100442, 100807, 101172, 101537,
    101903, 102268, 102633, 102998, 103364, 103729, 104094, 104459, 104825,
    105190, 105555, 105920, 106286, 106651, 107016, 107381, 107747, 108112,
    108477, 108842, 109208, 109573, 109938, 110303, 110669, 111034, 111399,
    111764, 112130, 112495, 112860, 113225, 113591, 113956, 114321, 114686,
    115052, 115417, 115782, 116147, 116513, 116878, 117243, 117608, 117974,
    118339, 118704, 119069, 119435, 119800, 120165, 120530, 120895, 121260,
    121625, 121990, 122356, 122721, 123086, 123451, 123817, 124182, 124547,
    124912, 125278, 125643, 126008, 126373, 126739, 127104, 127469, 127834,
    128200, 128565, 128930, 129295, 129661, 130026, 130391, 130756, 131122,
    131487, 131852, 132217, 132583, 132948, 133313, 133678, 134044, 134409,
    134774, 135139, 135505, 135870, 136235, 136600, 136966, 137331, 137696,
    138061, 138427, 138792, 139157, 139522, 139888, 140253, 140618, 140983,
    141349, 141714, 142079, 142444, 142810, 143175, 143540, 143905, 144271,
    144636, 145001, 145366, 145732, 146097,
};

} // namespace

bool WideRangeDateConversion::epochToCalendarUtc(int64_t epoch, std::tm& tm) {
  constexpr int kDaysPerYear = 365;
  int64_t days = epoch / kSecondsPerDay;
  int64_t rem = epoch % kSecondsPerDay;
  if (rem < 0) {
    rem += kSecondsPerDay;
    --days;
  }
  tm.tm_hour = rem / kSecondsPerHour;
  rem = rem % kSecondsPerHour;
  tm.tm_min = rem / 60;
  tm.tm_sec = rem % 60;
  tm.tm_wday = (4 + days) % 7;
  if (tm.tm_wday < 0) {
    tm.tm_wday += 7;
  }
  int64_t y = 1970;
  if (y + days / kDaysPerYear <= -kLeapYearOffset + 10) {
    return false;
  }
  bool leapYear;
  while (days < 0 || days >= kDaysPerYear + (leapYear = isLeap(y))) {
    auto newy = y + days / kDaysPerYear - (days < 0);
    days -= daysBetweenYears(y, newy);
    y = newy;
  }
  y -= kTmYearBase;
  if (y > std::numeric_limits<decltype(tm.tm_year)>::max() ||
      y < std::numeric_limits<decltype(tm.tm_year)>::min()) {
    return false;
  }
  tm.tm_year = y;
  tm.tm_yday = days;
  const auto* months = kDaysBeforeFirstDayOfMonth[leapYear];
  tm.tm_mon = std::upper_bound(months, months + 12, days) - months - 1;
  tm.tm_mday = days - months[tm.tm_mon] + 1;
  tm.tm_isdst = 0;
  return true;
}

Expected<int64_t> WideRangeDateConversion::daysSinceEpochFromDate(
    int32_t year,
    int32_t month,
    int32_t day) {
  if (!util::isValidDate(year, month, day)) {
    if (threadSkipErrorDetails()) {
      return folly::makeUnexpected(Status::UserError());
    }
    return folly::makeUnexpected(
        Status::UserError("Date out of range: {}-{}-{}", year, month, day));
  }
  int64_t daysSinceEpoch = 0;
  while (year < 1970) {
    year += util::kYearInterval;
    daysSinceEpoch -= util::kDaysPerYearInterval;
  }
  while (year >= 2370) {
    year -= util::kYearInterval;
    daysSinceEpoch += util::kDaysPerYearInterval;
  }
  daysSinceEpoch += kCumulativeYearDays[year - 1970];
  daysSinceEpoch += util::isLeapYear(year) ? kCumulativeLeapDays[month - 1]
                                           : kCumulativeDays[month - 1];
  daysSinceEpoch += day - 1;
  return daysSinceEpoch;
}

} // namespace facebook::velox
