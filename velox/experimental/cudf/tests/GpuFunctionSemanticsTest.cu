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

// What GPU-registered functions actually compute, run on the device.
//
// The companion to GpuShadowCompileTest.cu, which only asks whether a function
// compiles for the device. Compiling is the weaker claim: a function can be
// perfectly device-callable and still return a different answer than the CPU
// does, because the floating-point environment differs, because a std::tm field
// was never populated, or because the three-valued logic was written against
// the wrong truth table. Each test here pins a result rather than a signature.
//
// Compiled with the gpu_shadows/ include path ahead of the Velox source root,
// so the functions instantiated are the same ones GpuPrestoFunctions.cu
// registers.

#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <ctime>
#include <vector>

#include "velox/experimental/cudf/functions/GpuDateTimeFunctions.cuh"
#include "velox/experimental/cudf/functions/GpuLogicalFunctions.cuh"
#include "velox/functions/prestosql/Arithmetic.h"

namespace facebook::velox::cudf_velox::gpu_sfi {
namespace {

using facebook::velox::gpu::GpuExec;

/// Runs one device call per element of `input` and brings the results back.
///
/// Written once because every test here has the same shape -- copy up, one
/// thread per case, copy down -- and because a hand-rolled cudaMalloc per test
/// is where leaks and missing synchronisation come from.
template <typename TIn, typename TOut, typename Kernel>
std::vector<TOut> mapOnDevice(const std::vector<TIn>& input, Kernel kernel) {
  const int count = static_cast<int>(input.size());
  TIn* deviceInput{nullptr};
  TOut* deviceOutput{nullptr};
  EXPECT_EQ(cudaMalloc(&deviceInput, count * sizeof(TIn)), cudaSuccess);
  EXPECT_EQ(cudaMalloc(&deviceOutput, count * sizeof(TOut)), cudaSuccess);
  EXPECT_EQ(
      cudaMemcpy(
          deviceInput,
          input.data(),
          count * sizeof(TIn),
          cudaMemcpyHostToDevice),
      cudaSuccess);

  kernel(deviceInput, deviceOutput, count);
  EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess)
      << cudaGetErrorString(cudaGetLastError());

  std::vector<TOut> output(count);
  EXPECT_EQ(
      cudaMemcpy(
          output.data(),
          deviceOutput,
          count * sizeof(TOut),
          cudaMemcpyDeviceToHost),
      cudaSuccess);
  EXPECT_EQ(cudaFree(deviceInput), cudaSuccess);
  EXPECT_EQ(cudaFree(deviceOutput), cudaSuccess);
  return output;
}

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

// gmtime_r is the point of this test: the GPU extractors and the CPU ones share
// the calendar code in TimeUtilsCore.h, so comparing those two would largely be
// comparing shared code against itself. The C library is an independent
// implementation, which is what makes a disagreement informative.
TEST(GpuFunctionSemanticsTest, dateFieldsMatchTheCLibrary) {
  std::vector<int32_t> days;
  // Epoch and its neighbours, both sides of a leap day, a century non-leap
  // year, the TPC-H range, and far enough out either way to leave the range any
  // real query touches.
  for (int32_t day : {0,      -1,     1,      365,    366,    -365,  8035,
                      10592,  19000,  7305,   7304,   -25567, 50000, -700000,
                      700000, 100000, -50000, 250000, -250000}) {
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
  // Three one-row columns, so the view reads element 0 of each. Values and
  // masks are per-thread locals: each thread evaluates its own combination, and
  // sharing either across the block would let threads overwrite each other.
  bool values[3];
  cudf::bitmask_type masks[3];
  GpuArgView arguments[3];
  for (int term = 0; term < 3; ++term) {
    values[term] = cases[i].terms[term] == 1;
    // Validity is carried by the mask, exactly as a cudf column carries it, so
    // that a null term reaches the function as an absent element rather than as
    // some sentinel value it would have to know about.
    masks[term] = cases[i].terms[term] >= 0 ? 1u : 0u;
    arguments[term] =
        GpuArgView{&values[term], &masks[term], 0, /*isConstant=*/true};
  }

  GpuVariadicView<bool> terms{arguments, 3, 0};

  bool result{};
  out[i].conjunction =
      GpuAndFunction<GpuExec>{}.callNullable(result, terms) ? result : -1;
  out[i].disjunction =
      GpuOrFunction<GpuExec>{}.callNullable(result, terms) ? result : -1;
}

// The cases that matter are the ones where an input is null and the result is
// not: a single false decides a conjunction no matter how many unknowns sit
// beside it. Default null behaviour returns null for those, so a function that
// used call() instead of callNullable() would pass any test built only from
// non-null input.
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
    SCOPED_TRACE(
        fmt::format("({}, {}, {})", terms[0], terms[1], terms[2]));

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

// Not a test of the algorithm -- both sides run the identical Velox source --
// but of the floating-point environment under it. round() multiplies by a power
// of ten, rounds, and divides back, and its own comment notes that precision
// loss "plagues it on both paths". A device libm differing by one ulp, or a
// contracted multiply-add, would make the GPU disagree with the CPU on the same
// query, so the bar is bit equality rather than approximate equality.
TEST(GpuFunctionSemanticsTest, roundAndTruncateAgreeWithHostBitForBit) {
  std::vector<RoundCase> cases;
  for (double value : {0.0,
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

  // Compared as bits so that -0.0 and NaN are held to the same standard as any
  // other value rather than comparing equal to something they are not.
  auto bits = [](double value) {
    uint64_t pattern{};
    std::memcpy(&pattern, &value, sizeof(pattern));
    return pattern;
  };

  for (size_t i = 0; i < cases.size(); ++i) {
    SCOPED_TRACE(fmt::format("round({}, {})", cases[i].value, cases[i].decimals));

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

} // namespace
} // namespace facebook::velox::cudf_velox::gpu_sfi
