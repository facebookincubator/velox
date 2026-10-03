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

// Runs GPU-registered functions on the device and checks their results.
// Compiled with gpu_shadows/ ahead of the Velox source root, so it instantiates
// the same functions GpuPrestoFunctions.cu registers.

#include "velox/experimental/cudf/functions/GpuDateTimeFunctions.cuh"
#include "velox/experimental/cudf/functions/GpuLogicalFunctions.cuh"
#include "velox/experimental/cudf/tests/MapOnDevice.h"
#include "velox/experimental/cudf/tests/TimeZoneReference.h"

// For FOLLY_ALWAYS_INLINE, which the checked-arithmetic structs carry.
#include "folly/CPortability.h"
#include "velox/functions/lib/CheckedArithmetic.h"
#include "velox/functions/prestosql/Arithmetic.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <ctime>
#include <limits>
#include <string>
#include <vector>

namespace facebook::velox::cudf_velox::gpu_sfi {
namespace {

using facebook::velox::gpu::GpuExec;

// ---------------------------------------------------------------------------
// DATE field extraction
// ---------------------------------------------------------------------------

struct DateFields {
  int64_t year;
  int64_t month;
  int64_t day;
  int64_t quarter;
  int64_t dayOfYear;
  int64_t dayOfWeek;
};

__global__ void
extractDateFields(const int32_t* days, DateFields* out, int count) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) {
    return;
  }
  GpuYearFunction<GpuExec>{}.call(out[i].year, days[i]);
  GpuMonthFunction<GpuExec>{}.call(out[i].month, days[i]);
  GpuDayFunction<GpuExec>{}.call(out[i].day, days[i]);
  GpuQuarterFunction<GpuExec>{}.call(out[i].quarter, days[i]);
  GpuDayOfYearFunction<GpuExec>{}.call(out[i].dayOfYear, days[i]);
  GpuDayOfWeekFunction<GpuExec>{}.call(out[i].dayOfWeek, days[i]);
}

// Compares against gmtime_r rather than the CPU extractors, which share the
// calendar code in TimeUtilsCore.h with the GPU ones.
TEST(GpuFunctionSemanticsTest, dateFieldsMatchTheCLibrary) {
  std::vector<int32_t> days;
  // The epoch and its neighbours, both sides of a leap day, a century non-leap
  // year, the TPC-H range, and dates far outside any real query's range.
  for (int32_t day :
       {0,
        -1,
        1,
        365,
        366,
        -365,
        8035,
        10592,
        19000,
        7305,
        7304,
        -25567,
        50000,
        -700000,
        700000,
        100000,
        -50000,
        250000,
        -250000}) {
    days.push_back(day);
  }
  for (int32_t day = -3000; day <= 3000; day += 7) {
    days.push_back(day);
  }

  const auto got = mapOnDevice<int32_t, DateFields>(
      days, [](const int32_t* in, DateFields* out, int count) {
        extractDateFields<<<(count + 255) / 256, 256>>>(in, out, count);
      });

  for (size_t i = 0; i < days.size(); ++i) {
    SCOPED_TRACE(days[i]);
    const time_t seconds = static_cast<time_t>(days[i]) * 86400;
    std::tm expected{};
    ASSERT_NE(gmtime_r(&seconds, &expected), nullptr);

    EXPECT_EQ(got[i].year, 1900 + expected.tm_year);
    EXPECT_EQ(got[i].month, 1 + expected.tm_mon);
    EXPECT_EQ(got[i].day, expected.tm_mday);
    EXPECT_EQ(got[i].quarter, expected.tm_mon / 3 + 1);
    EXPECT_EQ(got[i].dayOfYear, expected.tm_yday + 1);
    // tm_wday counts from Sunday; Presto counts Monday as 1 through Sunday 7.
    EXPECT_EQ(got[i].dayOfWeek, expected.tm_wday == 0 ? 7 : expected.tm_wday);
  }
}

// ---------------------------------------------------------------------------
// GpuTimeZone: local time back to UTC
// ---------------------------------------------------------------------------

// Fixed offsets, zones with and without daylight saving time, a 30-minute
// daylight shift (Lord_Howe), a 45-minute offset (Kathmandu), a zone that
// abolished daylight saving time (Sao_Paulo), and one that skipped a whole day
// (Apia).
const std::vector<std::string> kTimeZones{
    "UTC",
    "+05:30",
    "-08:00",
    "America/Los_Angeles",
    "Europe/London",
    "Asia/Kolkata",
    "Asia/Kathmandu",
    "Australia/Lord_Howe",
    "America/Sao_Paulo",
    "Pacific/Apia",
};

// Zones whose daylight saving rules keep running past the end of the device
// table.
const std::vector<std::string> kDaylightSavingTimeZones{
    "America/Los_Angeles",
    "Australia/Lord_Howe",
    "Europe/London",
};

struct LocalToUtc {
  int64_t utcSeconds;
  uint8_t exists;
  // toLocal() of utcSeconds, so the round trip runs on the device too.
  int64_t backToLocal;
};

__global__ void localToUtc(
    const int64_t* locals,
    LocalToUtc* out,
    int count,
    GpuTimeZone zone) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) {
    return;
  }
  const auto instant = zone.toUtc(locals[i]);
  out[i].utcSeconds = instant.utcSeconds;
  out[i].exists = instant.exists ? 1 : 0;
  out[i].backToLocal = zone.toLocal(instant.utcSeconds);
}

struct UtcRoundTrip {
  int64_t localSeconds;
  int64_t backToUtc;
  uint8_t exists;
};

__global__ void utcToLocalAndBack(
    const int64_t* instants,
    UtcRoundTrip* out,
    int count,
    GpuTimeZone zone) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) {
    return;
  }
  out[i].localSeconds = zone.toLocal(instants[i]);
  const auto back = zone.toUtc(out[i].localSeconds);
  out[i].backToUtc = back.utcSeconds;
  out[i].exists = back.exists ? 1 : 0;
}

// Appends `count` deterministic values spread over [from, to).
void appendSpread(
    std::vector<int64_t>& values,
    int64_t from,
    int64_t to,
    int32_t count) {
  uint64_t state = 0x9e37'79b9'7f4a'7c15;
  const auto span = static_cast<uint64_t>(to - from);
  for (int32_t i = 0; i < count; ++i) {
    state = state * 6'364'136'223'846'793'005ULL + 1'442'695'040'888'963'407ULL;
    values.push_back(from + static_cast<int64_t>((state >> 11) % span));
  }
}

// Seconds either side of a point that reach into, and just past, an hour-long
// gap or overlap.
const std::vector<int64_t> kEdgeDeltas{-3'600, -1, 0, 1, 3'599};

// Local times either side of the wall-clock reading each offset change ends on
// and the one it starts on, plus a spread over the whole span.
std::vector<int64_t> localTimesAroundChanges(
    const std::vector<OffsetChange>& changes,
    int32_t fromYear,
    int32_t toYear,
    int32_t spreadCount) {
  std::vector<int64_t> values;
  for (const auto& change : changes) {
    for (const int32_t offset : {change.offsetBefore, change.offsetAfter}) {
      for (const int64_t delta : kEdgeDeltas) {
        values.push_back(change.utcSeconds + offset + delta);
      }
    }
  }
  appendSpread(
      values,
      yearStartSeconds(fromYear),
      yearStartSeconds(toYear),
      spreadCount);
  return values;
}

// Instants either side of each offset change, plus a spread over the span.
std::vector<int64_t> instantsAroundChanges(
    const std::vector<OffsetChange>& changes,
    int32_t fromYear,
    int32_t toYear,
    int32_t spreadCount) {
  std::vector<int64_t> values;
  for (const auto& change : changes) {
    for (const int64_t delta : kEdgeDeltas) {
      values.push_back(change.utcSeconds + delta);
    }
  }
  appendSpread(
      values,
      yearStartSeconds(fromYear),
      yearStartSeconds(toYear),
      spreadCount);
  return values;
}

// Runs toUtc() on the device over `locals` and holds each result to
// Timestamp::toGMT(): the same instant, and a gap exactly where the CPU raises.
// In a gap the instant must be the one date_add's correction produces. Outside
// a gap, toLocal() must take the instant back to the local time. Returns the
// number of gaps, so a caller can check the zone exercised one.
int32_t assertToUtcMatchesCpu(
    const std::string& timeZone,
    const std::vector<int64_t>& locals) {
  const auto zone = deviceTimeZone(timeZone);
  const auto got = mapOnDevice<int64_t, LocalToUtc>(
      locals, [&](const int64_t* in, LocalToUtc* out, int count) {
        localToUtc<<<(count + 255) / 256, 256>>>(in, out, count, zone);
      });

  int32_t numGaps{0};
  for (size_t i = 0; i < locals.size(); ++i) {
    SCOPED_TRACE(fmt::format("{} local {}", timeZone, locals[i]));
    const auto expected = cpuToUtc(timeZone, locals[i]);
    EXPECT_EQ(got[i].exists != 0, expected.has_value());
    if (expected.has_value()) {
      EXPECT_EQ(got[i].utcSeconds, *expected);
      EXPECT_EQ(got[i].backToLocal, locals[i]);
    } else {
      ++numGaps;
      EXPECT_EQ(got[i].utcSeconds, cpuCorrectedToUtc(timeZone, locals[i]));
    }
  }
  return numGaps;
}

// Runs toLocal() then toUtc() on the device over `instants`. The local time is
// held to Timestamp::toTimezone(), and the way back must land on the instant
// itself, except where an offset decrease repeats the local time: there the
// earlier of the two instants is the answer, so the later one comes back
// shifted by the decrease. Returns how many instants were shifted, so a caller
// can check the zone exercised an overlap.
int32_t assertUtcRoundTrip(
    const std::string& timeZone,
    const std::vector<OffsetChange>& changes,
    const std::vector<int64_t>& instants) {
  const auto zone = deviceTimeZone(timeZone);
  const auto got = mapOnDevice<int64_t, UtcRoundTrip>(
      instants, [&](const int64_t* in, UtcRoundTrip* out, int count) {
        utcToLocalAndBack<<<(count + 255) / 256, 256>>>(in, out, count, zone);
      });

  int32_t numShifted{0};
  for (size_t i = 0; i < instants.size(); ++i) {
    SCOPED_TRACE(fmt::format("{} instant {}", timeZone, instants[i]));
    EXPECT_EQ(got[i].localSeconds, cpuToLocal(timeZone, instants[i]));
    // An instant's own local time always exists.
    EXPECT_TRUE(got[i].exists != 0);

    int64_t expected = instants[i];
    for (const auto& change : changes) {
      const int64_t decrease = change.offsetBefore - change.offsetAfter;
      if (decrease > 0 && instants[i] >= change.utcSeconds &&
          instants[i] < change.utcSeconds + decrease) {
        expected -= decrease;
        ++numShifted;
      }
    }
    EXPECT_EQ(got[i].backToUtc, expected);
  }
  return numShifted;
}

bool hasIncrease(const std::vector<OffsetChange>& changes) {
  for (const auto& change : changes) {
    if (change.offsetAfter > change.offsetBefore) {
      return true;
    }
  }
  return false;
}

bool hasDecrease(const std::vector<OffsetChange>& changes) {
  for (const auto& change : changes) {
    if (change.offsetAfter < change.offsetBefore) {
      return true;
    }
  }
  return false;
}

// Every offset change over the years a nanosecond column can hold, which
// reach back past the zones' local mean time, with a spread over the same
// span.
TEST(GpuFunctionSemanticsTest, localTimeToUtcMatchesCpu) {
  for (const auto& timeZone : kTimeZones) {
    SCOPED_TRACE(timeZone);
    const auto changes = offsetChanges(timeZone, 1678, 2262);
    const auto numGaps = assertToUtcMatchesCpu(
        timeZone, localTimesAroundChanges(changes, 1678, 2262, 20'000));
    EXPECT_EQ(numGaps > 0, hasIncrease(changes));
    const auto numShifted = assertUtcRoundTrip(
        timeZone, changes, instantsAroundChanges(changes, 1678, 2262, 20'000));
    EXPECT_EQ(numShifted > 0, hasDecrease(changes));
  }
}

// Past 2800 both directions fold by whole 400-year cycles. A local time and
// the instant it names can fall on different sides of the table's end, so the
// edges of the end and of later cycle boundaries are covered from both sides.
TEST(
    GpuFunctionSemanticsTest,
    localTimeToUtcMatchesCpuBeyondTheTabulatedRange) {
  std::vector<int64_t> edges;
  for (const int64_t cycles : {0, 1, 3}) {
    const int64_t boundary =
        GpuTimeZone::kTableEndSeconds + cycles * GpuTimeZone::kCycleSeconds;
    for (const int64_t delta :
         {-86'400, -43'200, -3'600, -1, 0, 1, 3'600, 43'200, 86'400}) {
      edges.push_back(boundary + delta);
    }
  }

  for (const auto& timeZone : kDaylightSavingTimeZones) {
    SCOPED_TRACE(timeZone);
    const auto changes = offsetChanges(timeZone, 2700, 4100);
    ASSERT_TRUE(hasIncrease(changes));
    ASSERT_TRUE(hasDecrease(changes));

    auto locals = localTimesAroundChanges(changes, 2700, 4100, 5'000);
    locals.insert(locals.end(), edges.begin(), edges.end());
    EXPECT_GT(assertToUtcMatchesCpu(timeZone, locals), 0);

    auto instants = instantsAroundChanges(changes, 2700, 4100, 5'000);
    instants.insert(instants.end(), edges.begin(), edges.end());
    EXPECT_GT(assertUtcRoundTrip(timeZone, changes, instants), 0);
  }
}

// ---------------------------------------------------------------------------
// Kleene logic
// ---------------------------------------------------------------------------

/// -1 null, 0 false, 1 true, for both inputs and results.
struct Tristate {
  int8_t terms[3];
};

struct Conjunctions {
  int8_t conjunction;
  int8_t disjunction;
};

__global__ void
evaluateLogical(const Tristate* cases, Conjunctions* out, int count) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) {
    return;
  }
  // Three one-row columns, held per thread because each thread evaluates its
  // own combination.
  bool values[3];
  cudf::bitmask_type masks[3];
  GpuArgView arguments[3];
  for (int term = 0; term < 3; ++term) {
    values[term] = cases[i].terms[term] == 1;
    // Validity is carried by the mask, as in a cudf column.
    masks[term] = cases[i].terms[term] >= 0 ? 1u : 0u;
    arguments[term] = GpuArgView{
        &values[term],
        &masks[term],
        0,
        /*isConstant=*/true,
        /*ticksPerSecond=*/0};
  }

  GpuVariadicView<bool> terms{arguments, 3, 0};

  bool result{};
  out[i].conjunction =
      GpuAndFunction<GpuExec>{}.callNullable(result, terms) ? result : -1;
  out[i].disjunction =
      GpuOrFunction<GpuExec>{}.callNullable(result, terms) ? result : -1;
}

// Covers every true/false/null combination, including those where an input is
// null and the result is not: a single false decides a conjunction.
TEST(GpuFunctionSemanticsTest, kleeneLogicOverAllTristateCombinations) {
  std::vector<Tristate> cases;
  for (int8_t a = -1; a <= 1; ++a) {
    for (int8_t b = -1; b <= 1; ++b) {
      for (int8_t c = -1; c <= 1; ++c) {
        cases.push_back(Tristate{{a, b, c}});
      }
    }
  }
  ASSERT_EQ(cases.size(), 27u);

  const auto got = mapOnDevice<Tristate, Conjunctions>(
      cases, [](const Tristate* in, Conjunctions* out, int count) {
        evaluateLogical<<<1, 32>>>(in, out, count);
      });

  for (size_t i = 0; i < cases.size(); ++i) {
    const auto& terms = cases[i].terms;
    SCOPED_TRACE(fmt::format("({}, {}, {})", terms[0], terms[1], terms[2]));

    bool sawNull = false;
    bool sawFalse = false;
    bool sawTrue = false;
    for (int8_t term : terms) {
      sawNull |= term < 0;
      sawFalse |= term == 0;
      sawTrue |= term == 1;
    }

    const int8_t expectedAnd = sawFalse ? 0 : (sawNull ? -1 : 1);
    const int8_t expectedOr = sawTrue ? 1 : (sawNull ? -1 : 0);
    EXPECT_EQ(got[i].conjunction, expectedAnd);
    EXPECT_EQ(got[i].disjunction, expectedOr);
  }
}

// ---------------------------------------------------------------------------
// round and truncate
// ---------------------------------------------------------------------------

struct RoundCase {
  double value;
  int32_t decimals;
};

struct RoundResults {
  double rounded;
  double truncated;
};

__global__ void
roundAndTruncate(const RoundCase* cases, RoundResults* out, int count) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) {
    return;
  }
  functions::RoundFunction<void>{}.call(
      out[i].rounded, cases[i].value, cases[i].decimals);
  functions::TruncateFunction<void>{}.call(
      out[i].truncated, cases[i].value, cases[i].decimals);
}

// Both sides run the same Velox source, so this tests the floating-point
// environment: a device libm one ulp off, or a contracted multiply-add, would
// make the GPU disagree with the CPU. Results must be bit-identical.
TEST(GpuFunctionSemanticsTest, roundAndTruncateAgreeWithHostBitForBit) {
  std::vector<RoundCase> cases;
  for (double value :
       {0.0,
        -0.0,
        0.5,
        -0.5,
        1.5,
        2.5,
        -2.5,
        1.005,
        2.675,
        123.456789,
        -123.456789,
        0.000001234,
        1e15,
        -1e15,
        // Either side of the threshold where round() switches
        // from the factor path to splitting the number.
        17592186044415.5,
        17592186044416.5,
        1e300,
        3.14159265358979,
        -9.99999999}) {
    for (int32_t decimals : {-3, -1, 0, 1, 2, 3, 7, 15}) {
      cases.push_back(RoundCase{value, decimals});
    }
  }

  const auto got = mapOnDevice<RoundCase, RoundResults>(
      cases, [](const RoundCase* in, RoundResults* out, int count) {
        roundAndTruncate<<<(count + 127) / 128, 128>>>(in, out, count);
      });

  // Compared as bits, so -0.0 and NaN must match exactly too.
  auto bits = [](double value) {
    uint64_t pattern{};
    std::memcpy(&pattern, &value, sizeof(pattern));
    return pattern;
  };

  for (size_t i = 0; i < cases.size(); ++i) {
    SCOPED_TRACE(
        fmt::format("round({}, {})", cases[i].value, cases[i].decimals));

    double expectedRound{};
    functions::RoundFunction<void>{}.call(
        expectedRound, cases[i].value, cases[i].decimals);
    double expectedTruncate{};
    functions::TruncateFunction<void>{}.call(
        expectedTruncate, cases[i].value, cases[i].decimals);

    EXPECT_EQ(bits(got[i].rounded), bits(expectedRound));
    EXPECT_EQ(bits(got[i].truncated), bits(expectedTruncate));
  }
}

// ---------------------------------------------------------------------------
// A check that fails
// ---------------------------------------------------------------------------

struct CheckedAddCase {
  int64_t a;
  int64_t b;
};

struct CheckedAddResult {
  int64_t value;
  uint8_t raised;
};

// Does what the adapter's kernel does around a function body: clear this
// thread's error byte, run the body, read the byte back.
__global__ void checkedAddRecordingErrors(
    const CheckedAddCase* cases,
    CheckedAddResult* out,
    int count) {
  gpuErrorBytes[threadIdx.x] = static_cast<uint8_t>(GpuErrorKind::kNone);
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) {
    return;
  }
  int64_t value{-1};
  facebook::velox::functions::CheckedPlusFunction<GpuExec>{}.call(
      value, cases[i].a, cases[i].b);
  out[i].value = value;
  out[i].raised = gpuErrorBytes[threadIdx.x];
}

// An overflow inside checkedPlus reaches VELOX_ARITHMETIC_ERROR, which records
// the row on the device.
TEST(GpuFunctionSemanticsTest, aFailedCheckIsRecordedPerRow) {
  constexpr int64_t kMax = std::numeric_limits<int64_t>::max();
  constexpr int64_t kMin = std::numeric_limits<int64_t>::min();
  const std::vector<CheckedAddCase> cases{
      {1, 2},
      {kMax, 1},
      {3, 4},
      {kMin, -1},
      {kMax, 0},
      {-1, -1},
  };
  const std::vector<bool> shouldRaise{false, true, false, true, false, false};

  auto got = mapOnDevice<CheckedAddCase, CheckedAddResult>(
      cases, [&](const CheckedAddCase* in, CheckedAddResult* out, int count) {
        checkedAddRecordingErrors<<<1, count, count * sizeof(uint8_t)>>>(
            in, out, count);
      });

  for (size_t i = 0; i < cases.size(); ++i) {
    SCOPED_TRACE(fmt::format("case {}: {} + {}", i, cases[i].a, cases[i].b));
    if (shouldRaise[i]) {
      // A user error, as VELOX_ARITHMETIC_ERROR is on the CPU, so a TRY may
      // swallow it.
      EXPECT_EQ(got[i].raised, static_cast<uint8_t>(GpuErrorKind::kUserError));
    } else {
      // A clean row is not declined by its neighbour's failure.
      EXPECT_EQ(got[i].raised, static_cast<uint8_t>(GpuErrorKind::kNone));
      EXPECT_EQ(got[i].value, cases[i].a + cases[i].b);
    }
  }
}

} // namespace
} // namespace facebook::velox::cudf_velox::gpu_sfi
