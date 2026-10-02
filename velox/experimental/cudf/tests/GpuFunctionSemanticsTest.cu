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

#include "velox/functions/prestosql/Arithmetic.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
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

} // namespace
} // namespace facebook::velox::cudf_velox::gpu_sfi
