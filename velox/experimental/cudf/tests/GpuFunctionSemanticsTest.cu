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

#include "velox/experimental/cudf/functions/GpuLogicalFunctions.cuh"
#include "velox/experimental/cudf/tests/MapOnDevice.h"

// For FOLLY_ALWAYS_INLINE, which the checked-arithmetic structs carry.
#include "folly/CPortability.h"
#include "velox/common/base/BitUtil.h"
#include "velox/functions/lib/CheckedArithmetic.h"
#include "velox/functions/prestosql/Arithmetic.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

namespace facebook::velox::cudf_velox::gpu_sfi {
namespace {

using facebook::velox::gpu::GpuExec;

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
// the row on the device as a user error, as it is on the CPU, so a TRY may
// swallow it; a clean row is not declined by its neighbour's failure.
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

  auto got = mapOnDevice<CheckedAddCase, CheckedAddResult>(
      cases, [&](const CheckedAddCase* in, CheckedAddResult* out, int count) {
        checkedAddRecordingErrors<<<1, count, count * sizeof(uint8_t)>>>(
            in, out, count);
      });

  for (size_t i = 0; i < cases.size(); ++i) {
    SCOPED_TRACE(fmt::format("case {}: {} + {}", i, cases[i].a, cases[i].b));
    // The host's overflow test decides which rows must be declined.
    int64_t sum{};
    if (__builtin_add_overflow(cases[i].a, cases[i].b, &sum)) {
      EXPECT_EQ(got[i].raised, static_cast<uint8_t>(GpuErrorKind::kUserError));
    } else {
      EXPECT_EQ(got[i].raised, static_cast<uint8_t>(GpuErrorKind::kNone));
      EXPECT_EQ(got[i].value, sum);
    }
  }
}

struct BitRange {
  int32_t begin;
  int32_t end;
};

// Counts the bits of each range over two words, 9 then all ones, with this
// thread's error byte holding `errorByte`: clean, or set as a failed check
// leaves it. A count that enters the second word shows as 64 or more.
__global__ void countBitsWithErrorByte(
    const BitRange* ranges,
    int32_t* out,
    int count,
    uint8_t errorByte) {
  gpuErrorBytes[threadIdx.x] = errorByte;
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) {
    return;
  }
  const uint64_t words[2] = {9, ~uint64_t{0}};
  out[i] =
      facebook::velox::bits::countBits(words, ranges[i].begin, ranges[i].end);
}

// The BitUtil.h shadow keeps the real countBits across words while the row is
// clean. Once a check has failed, the body runs on and BitCountFunction passes
// a width its check rejected, so the count must return before it reads past
// the one word that caller holds.
TEST(GpuFunctionSemanticsTest, countBitsReadsNothingAfterAFailedCheck) {
  const auto countWith = [](const std::vector<BitRange>& ranges,
                            GpuErrorKind errorKind) {
    return mapOnDevice<BitRange, int32_t>(
        ranges, [&](const BitRange* in, int32_t* out, int count) {
          countBitsWithErrorByte<<<1, count, count * sizeof(uint8_t)>>>(
              in, out, count, static_cast<uint8_t>(errorKind));
        });
  };

  const std::vector<BitRange> withinTheWords{
      {0, 8}, {0, 64}, {0, 65}, {0, 128}, {70, 100}};
  EXPECT_THAT(
      countWith(withinTheWords, GpuErrorKind::kNone),
      testing::ElementsAre(2, 2, 3, 66, 30));

  // Past the second word too: nothing may be read for these to pass.
  const std::vector<BitRange> rejectedWidths{
      {0, 8},
      {0, 65},
      {0, 1'048'576},
      {0, std::numeric_limits<int32_t>::max()},
  };
  EXPECT_THAT(
      countWith(rejectedWidths, GpuErrorKind::kUserError),
      testing::ElementsAre(0, 0, 0, 0));
}

} // namespace
} // namespace facebook::velox::cudf_velox::gpu_sfi
