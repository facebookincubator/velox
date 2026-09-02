/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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

// ALP transform benchmark: two comparisons on the same fixed (exponent,
// factor) over the same input buffer.
//
//   Table 1 (per-row transform): scalar (`scalarTransformOne`) vs vector
//   (`batchTransform`) writing full output buffers. Isolates the per-row
//   cost of the transform primitive; used to sanity-check SIMD lane
//   throughput separate from any accumulator work.
//
//   Table 2 (scoreCombination inner loop): scalar-only sweep vs the exact
//   prod SIMD loop shape from `ALPEncoding<T>::scoreCombination`. Accumulates
//   (zigZagMin, zigZagMax, exceptionCount) only — no per-row buffer stores —
//   so μs/Mv here is directly comparable to `scoreCombination`'s hot path
//   and is the number that the xsimd-vectorize commit's encode-wall speedup
//   is expected to move. Includes a drift-guard: the SIMD path here is byte-
//   compared against `ALPEncoding<T>::scoreCombination` (which internally uses
//   the same batchTransform).
//
// Datasets vary exception rate from 0% (clean 2-decimal values, e=f=2) to
// ~100% (chaotic random doubles, no rational form) so the reader can see
// how the SIMD win scales with representability: when almost every lane
// is representable, batchTransform stays on the FMA/round/compare path
// end-to-end; when exceptions dominate, both variants degrade to similar
// per-row cost because scalarTransformOne is already just a few FP ops.
//
// The build target is a plain executable (not folly-benchmark).

#include <folly/init/Init.h>
#include <glog/logging.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <random>
#include <span>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include "velox/common/memory/Memory.h"
#include "velox/dwio/nimble/common/Types.h"
#include "velox/dwio/nimble/common/Vector.h"
#include "velox/dwio/nimble/encodings/ALPEncoding.h"

namespace {

using facebook::nimble::ALPEncoding;
using facebook::nimble::TypeTraits;
using facebook::nimble::Vector;
namespace alp = facebook::nimble::detail::alp;

std::shared_ptr<facebook::velox::memory::MemoryPool>& benchPool() {
  static auto pool = facebook::velox::memory::memoryManager()->addLeafPool(
      "alp_batch_transform_benchmark");
  return pool;
}

// -----------------------------------------------------------------------------
// Datasets — each generator returns rows suited to one (e, f) pair.
// -----------------------------------------------------------------------------

constexpr uint64_t kSeed = 0x51ED0FF1CEULL;
constexpr uint32_t kDefaultRows = 1 << 20; // 1M rows keeps L2 pressure honest.

template <typename T>
Vector<T> makeEmpty(uint32_t n) {
  Vector<T> v{benchPool().get()};
  v.reserve(n);
  return v;
}

// (1) 2-decimal, no exceptions. Every value round-trips at (e=2, f=0).
template <typename T>
Vector<T> makeCleanTwoDecimal(uint32_t n) {
  auto v = makeEmpty<T>(n);
  std::mt19937_64 rng(kSeed);
  std::uniform_int_distribution<int32_t> dist(0, 999'999);
  for (uint32_t i = 0; i < n; ++i) {
    v.push_back(static_cast<T>(dist(rng)) / static_cast<T>(100));
  }
  return v;
}

// (2) Sensor-like at outlierRate. 3-decimal base with a controlled fraction of
// 6-decimal high-precision outliers -> deterministic mixed exception rate.
template <typename T>
Vector<T> makeSensorLike(uint32_t n, double outlierRate) {
  auto v = makeEmpty<T>(n);
  std::mt19937_64 rng(kSeed);
  std::uniform_int_distribution<int32_t> clean(0, 99'999);
  std::uniform_real_distribution<double> pOut(0.0, 1.0);
  std::uniform_int_distribution<int64_t> outlier(
      5'000'000'000LL, 9'999'999'999LL);
  for (uint32_t i = 0; i < n; ++i) {
    if (pOut(rng) < outlierRate) {
      v.push_back(static_cast<T>(outlier(rng)) / static_cast<T>(1'000'000));
    } else {
      v.push_back(static_cast<T>(clean(rng)) / static_cast<T>(1'000));
    }
  }
  return v;
}

// (3) Chaotic doubles with no rational decimal form. Almost every value is
// an exception at any (e, f), so the exception-branch cost dominates.
template <typename T>
Vector<T> makeChaotic(uint32_t n) {
  auto v = makeEmpty<T>(n);
  std::mt19937_64 rng(kSeed);
  std::uniform_real_distribution<double> dist(-1e6, 1e6);
  for (uint32_t i = 0; i < n; ++i) {
    v.push_back(static_cast<T>(dist(rng)));
  }
  return v;
}

// -----------------------------------------------------------------------------
// Micro-runner
// -----------------------------------------------------------------------------
//
// Each run does two passes over the same input: one all-scalar, one batched
// with a scalar tail. Output buffers are pre-allocated once and reused so we
// only measure the transform. Warm-up pass is done outside the timed
// section.

struct RunResult {
  std::string dataset;
  std::string dtype;
  uint32_t rows;
  uint8_t exponent;
  uint8_t factor;
  double scalarNs;
  double batchNs;
  uint32_t scalarExceptions;
  uint32_t batchExceptions;
};

template <typename T>
RunResult runOne(
    const std::string& datasetName,
    const Vector<T>& values,
    uint8_t exponent,
    uint8_t factor) {
  using physicalType = typename TypeTraits<T>::physicalType;
  using Alp = ALPEncoding<T>;

  const uint32_t rows = static_cast<uint32_t>(values.size());
  const double exponentMultiplier = Alp::kPow10Double[exponent];
  const double factorMultiplier = Alp::kPow10Double[factor];

  // Pre-compute physicals once. Both variants consume the same slice so any
  // divergence in the exception count is a scalar-vs-batch bug, not input
  // drift.
  std::vector<physicalType> physicals(rows);
  for (uint32_t i = 0; i < rows; ++i) {
    physicals[i] = alp::toPhysical<T>(values[i]);
  }

  std::vector<uint64_t> outZigZag(rows, 0);
  std::vector<uint8_t> outMask(rows, 0); // 1 -> representable, 0 -> exception.

  auto scalarPass = [&](bool timed) -> std::pair<double, uint32_t> {
    uint32_t exceptions = 0;
    const auto t0 = std::chrono::steady_clock::now();
    for (uint32_t i = 0; i < rows; ++i) {
      uint64_t zz = 0;
      const bool ok = Alp::scalarTransformOne(
          values[i], physicals[i], exponentMultiplier, factorMultiplier, zz);
      outZigZag[i] = zz;
      outMask[i] = ok ? 1 : 0;
      if (!ok) {
        ++exceptions;
      }
    }
    const auto t1 = std::chrono::steady_clock::now();
    // touch outZigZag[0] to defeat DCE.
    if (!timed && outZigZag.empty()) {
      LOG(FATAL) << "unreachable";
    }
    const double ns =
        std::chrono::duration<double, std::nano>(t1 - t0).count();
    return {ns, exceptions};
  };

  constexpr std::size_t kBatch = Alp::kBatchSize;
  auto batchPass = [&](bool timed) -> std::pair<double, uint32_t> {
    uint32_t exceptions = 0;
    alignas(64) uint64_t zigZagLanes[kBatch];
    alignas(64) bool okLanes[kBatch];
    const auto t0 = std::chrono::steady_clock::now();
    uint32_t i = 0;
    for (; i + kBatch <= rows; i += kBatch) {
      Alp::batchTransform(
          values.data() + i,
          physicals.data() + i,
          exponentMultiplier,
          factorMultiplier,
          zigZagLanes,
          okLanes);
      for (std::size_t k = 0; k < kBatch; ++k) {
        outZigZag[i + k] = zigZagLanes[k];
        outMask[i + k] = okLanes[k] ? 1 : 0;
        if (!okLanes[k]) {
          ++exceptions;
        }
      }
    }
    // Scalar tail. Kept identical to the production hot path in
    // scoreCombination so this micro-benchmark measures exactly what runs
    // in production.
    for (; i < rows; ++i) {
      uint64_t zz = 0;
      const bool ok = Alp::scalarTransformOne(
          values[i], physicals[i], exponentMultiplier, factorMultiplier, zz);
      outZigZag[i] = zz;
      outMask[i] = ok ? 1 : 0;
      if (!ok) {
        ++exceptions;
      }
    }
    const auto t1 = std::chrono::steady_clock::now();
    if (!timed && outZigZag.empty()) {
      LOG(FATAL) << "unreachable";
    }
    const double ns =
        std::chrono::duration<double, std::nano>(t1 - t0).count();
    return {ns, exceptions};
  };

  // Warm-up, discarded.
  scalarPass(/*timed=*/false);
  batchPass(/*timed=*/false);

  // Interleave a few passes and take the min per side — reduces noise from a
  // stray context switch without hiding a real regression (min is the same
  // regardless of noise floor if the underlying cost is stable).
  constexpr int kIters = 5;
  double scalarBest = std::numeric_limits<double>::infinity();
  double batchBest = std::numeric_limits<double>::infinity();
  uint32_t scalarExceptions = 0;
  uint32_t batchExceptions = 0;
  for (int iter = 0; iter < kIters; ++iter) {
    const auto [ns, exc] = scalarPass(/*timed=*/true);
    scalarBest = std::min(scalarBest, ns);
    scalarExceptions = exc;
  }
  for (int iter = 0; iter < kIters; ++iter) {
    const auto [ns, exc] = batchPass(/*timed=*/true);
    batchBest = std::min(batchBest, ns);
    batchExceptions = exc;
  }

  return RunResult{
      .dataset = datasetName,
      .dtype = std::is_same_v<T, float> ? "float" : "double",
      .rows = rows,
      .exponent = exponent,
      .factor = factor,
      .scalarNs = scalarBest / static_cast<double>(rows),
      .batchNs = batchBest / static_cast<double>(rows),
      .scalarExceptions = scalarExceptions,
      .batchExceptions = batchExceptions,
  };
}

void printHeader() {
  {
    std::ostringstream oss;
    oss << "| " << std::left << std::setw(26) << "dataset" << " | "
        << std::setw(6) << "dtype" << " | " << std::right << std::setw(7)
        << "rows" << " | (e, f) | scalar ns/row | batch ns/row |  speedup"
        << " | exc rate | agree |";
    LOG(INFO) << oss.str();
  }
  {
    std::ostringstream oss;
    oss << "|" << std::string(28, '-') << "|" << std::string(8, '-') << "|"
        << std::string(9, '-') << "|" << std::string(9, '-') << "|"
        << std::string(15, '-') << "|" << std::string(14, '-') << "|"
        << std::string(10, '-') << "|" << std::string(10, '-') << "|"
        << std::string(7, '-') << "|";
    LOG(INFO) << oss.str();
  }
}

void printRow(const RunResult& r) {
  const double speedup = r.batchNs > 0 ? r.scalarNs / r.batchNs : 0.0;
  const double excRate = r.rows > 0
      ? 100.0 * static_cast<double>(r.batchExceptions) /
          static_cast<double>(r.rows)
      : 0.0;
  const char* agree =
      r.scalarExceptions == r.batchExceptions ? "yes" : "MISMATCH";
  std::ostringstream oss;
  oss << "| " << std::left << std::setw(26) << r.dataset << " | "
      << std::setw(6) << r.dtype << " | " << std::right << std::setw(7)
      << r.rows << " | (" << std::setw(2) << int(r.exponent) << ","
      << std::setw(2) << int(r.factor) << ") | " << std::setw(13)
      << std::fixed << std::setprecision(2) << r.scalarNs << " | "
      << std::setw(12) << r.batchNs << " | " << std::setw(6)
      << std::setprecision(2) << speedup << "x | " << std::setw(7)
      << std::setprecision(2) << excRate << "% | " << std::setw(5) << agree
      << " |";
  LOG(INFO) << oss.str();
}

// -----------------------------------------------------------------------------
// scoreCombination-shape inner-loop A/B
// -----------------------------------------------------------------------------
//
// Replicates the outer loop of `ALPEncoding<T>::scoreCombination` (see
// ALPEncoding.h) byte-exactly on the SIMD side, and runs a pure-scalar sweep
// on the other side. Both compute (zigZagMin, zigZagMax, exceptionCount) over
// the same N rows — no output buffers, no allocations in the hot loop — so
// μs/Mv reflects exactly the encode-wall work xsimd-vectorize moves.

struct ScoreResult {
  std::string dataset;
  std::string dtype;
  uint32_t rows;
  uint8_t exponent;
  uint8_t factor;
  double scalarUs; // total us for one full pass
  double batchUs;
  uint32_t exceptionCount;
  bool driftGuardOk; // batch == prod scoreCombination on same input
};

template <typename T>
ScoreResult runScoreLoopAB(
    const std::string& datasetName,
    const Vector<T>& values,
    uint8_t exponent,
    uint8_t factor) {
  using physicalType = typename TypeTraits<T>::physicalType;
  using Alp = ALPEncoding<T>;

  const uint32_t rows = static_cast<uint32_t>(values.size());
  const double exponentMultiplier = Alp::kPow10Double[exponent];
  const double factorMultiplier = Alp::kPow10Double[factor];
  constexpr std::size_t kBatch = Alp::kBatchSize;

  // Pure-scalar sweep: matches what scoreCombination looked like before the
  // xsimd-vectorize change (batchTransform path replaced with per-row
  // scalarTransformOne).
  auto scalarSweep = [&]() -> std::tuple<double, uint64_t, uint64_t, uint32_t> {
    uint64_t zigZagMin = std::numeric_limits<uint64_t>::max();
    uint64_t zigZagMax = 0;
    uint32_t exceptions = 0;
    const auto t0 = std::chrono::steady_clock::now();
    for (uint32_t i = 0; i < rows; ++i) {
      const auto logical = values[i];
      uint64_t zz = 0;
      if (!Alp::scalarTransformOne(
              logical,
              alp::toPhysical<T>(logical),
              exponentMultiplier,
              factorMultiplier,
              zz)) {
        ++exceptions;
        continue;
      }
      zigZagMin = std::min(zigZagMin, zz);
      zigZagMax = std::max(zigZagMax, zz);
    }
    const auto t1 = std::chrono::steady_clock::now();
    return {
        std::chrono::duration<double, std::micro>(t1 - t0).count(),
        zigZagMin,
        zigZagMax,
        exceptions};
  };

  // SIMD sweep: byte-exact copy of the main loop in
  // ALPEncoding<T>::scoreCombination — physicals precomputed per batch,
  // batchTransform on kBatch lanes, then the inner accumulator loop, plus a
  // scalar tail. Keeping this in lock-step with prod is what makes the
  // reported us/Mv attributable to xsimd-vectorize.
  auto batchSweep = [&]() -> std::tuple<double, uint64_t, uint64_t, uint32_t> {
    uint64_t zigZagMin = std::numeric_limits<uint64_t>::max();
    uint64_t zigZagMax = 0;
    uint32_t exceptions = 0;
    alignas(64) uint64_t zigZagLanes[kBatch];
    alignas(64) bool okLanes[kBatch];
    alignas(64) physicalType physLanes[kBatch];
    const auto t0 = std::chrono::steady_clock::now();
    uint64_t i = 0;
    for (; i + kBatch <= rows; i += kBatch) {
      for (std::size_t k = 0; k < kBatch; ++k) {
        physLanes[k] = alp::toPhysical<T>(values[i + k]);
      }
      Alp::batchTransform(
          values.data() + i,
          physLanes,
          exponentMultiplier,
          factorMultiplier,
          zigZagLanes,
          okLanes);
      for (std::size_t k = 0; k < kBatch; ++k) {
        if (!okLanes[k]) {
          ++exceptions;
          continue;
        }
        const uint64_t zz = zigZagLanes[k];
        zigZagMin = std::min(zigZagMin, zz);
        zigZagMax = std::max(zigZagMax, zz);
      }
    }
    for (; i < rows; ++i) {
      const auto logical = values[i];
      uint64_t zz = 0;
      if (!Alp::scalarTransformOne(
              logical,
              alp::toPhysical<T>(logical),
              exponentMultiplier,
              factorMultiplier,
              zz)) {
        ++exceptions;
        continue;
      }
      zigZagMin = std::min(zigZagMin, zz);
      zigZagMax = std::max(zigZagMax, zz);
    }
    const auto t1 = std::chrono::steady_clock::now();
    return {
        std::chrono::duration<double, std::micro>(t1 - t0).count(),
        zigZagMin,
        zigZagMax,
        exceptions};
  };

  // Warm-up, discarded.
  (void)scalarSweep();
  (void)batchSweep();

  constexpr int kIters = 5;
  double scalarBest = std::numeric_limits<double>::infinity();
  double batchBest = std::numeric_limits<double>::infinity();
  uint64_t scalarMin = 0, scalarMax = 0;
  uint64_t batchMin = 0, batchMax = 0;
  uint32_t scalarExc = 0, batchExc = 0;
  for (int iter = 0; iter < kIters; ++iter) {
    const auto [us, mn, mx, exc] = scalarSweep();
    if (us < scalarBest) {
      scalarBest = us;
    }
    scalarMin = mn;
    scalarMax = mx;
    scalarExc = exc;
  }
  for (int iter = 0; iter < kIters; ++iter) {
    const auto [us, mn, mx, exc] = batchSweep();
    if (us < batchBest) {
      batchBest = us;
    }
    batchMin = mn;
    batchMax = mx;
    batchExc = exc;
  }

  // Drift guard 1: scalar and SIMD must agree bit-for-bit — they see the
  // same input with the same fixed (e, f).
  const bool scalarBatchAgree =
      (scalarMin == batchMin) && (scalarMax == batchMax) &&
      (scalarExc == batchExc);

  // Drift guard 2: SIMD accumulators here must match the prod
  // scoreCombination call byte-for-byte. If someone later changes prod's
  // main loop shape without touching this benchmark, we'll catch it.
  const facebook::nimble::Encoding::Options options{};
  const auto prodScore = Alp::scoreCombination(
      std::span<const T>{values.data(), values.size()},
      exponent,
      factor,
      options);
  bool prodAgree;
  if (prodScore.estimatedBytes == Alp::kUnusableScore) {
    // Prod reports "unusable"; our sweep should have counted every row as an
    // exception too.
    prodAgree = (batchExc == rows);
  } else {
    prodAgree = (prodScore.zigZagMin == batchMin) &&
        (prodScore.zigZagMax == batchMax) &&
        (prodScore.exceptionCount == batchExc);
  }

  return ScoreResult{
      .dataset = datasetName,
      .dtype = std::is_same_v<T, float> ? "float" : "double",
      .rows = rows,
      .exponent = exponent,
      .factor = factor,
      .scalarUs = scalarBest,
      .batchUs = batchBest,
      .exceptionCount = batchExc,
      .driftGuardOk = scalarBatchAgree && prodAgree,
  };
}

void printScoreHeader() {
  {
    std::ostringstream oss;
    oss << "| " << std::left << std::setw(26) << "dataset" << " | "
        << std::setw(6) << "dtype" << " | " << std::right << std::setw(7)
        << "rows" << " | (e, f) | scalar us/Mv | batch us/Mv |  speedup"
        << " | exc rate | drift |";
    LOG(INFO) << oss.str();
  }
  {
    std::ostringstream oss;
    oss << "|" << std::string(28, '-') << "|" << std::string(8, '-') << "|"
        << std::string(9, '-') << "|" << std::string(9, '-') << "|"
        << std::string(14, '-') << "|" << std::string(13, '-') << "|"
        << std::string(10, '-') << "|" << std::string(10, '-') << "|"
        << std::string(7, '-') << "|";
    LOG(INFO) << oss.str();
  }
}

void printScoreRow(const ScoreResult& r) {
  const double speedup = r.batchUs > 0 ? r.scalarUs / r.batchUs : 0.0;
  const double megaVals = static_cast<double>(r.rows) / 1'000'000.0;
  const double scalarUsPerMv = megaVals > 0 ? r.scalarUs / megaVals : 0.0;
  const double batchUsPerMv = megaVals > 0 ? r.batchUs / megaVals : 0.0;
  const double excRate = r.rows > 0
      ? 100.0 * static_cast<double>(r.exceptionCount) /
          static_cast<double>(r.rows)
      : 0.0;
  const char* drift = r.driftGuardOk ? "ok" : "DRIFT";
  std::ostringstream oss;
  oss << "| " << std::left << std::setw(26) << r.dataset << " | "
      << std::setw(6) << r.dtype << " | " << std::right << std::setw(7)
      << r.rows << " | (" << std::setw(2) << int(r.exponent) << ","
      << std::setw(2) << int(r.factor) << ") | " << std::setw(12)
      << std::fixed << std::setprecision(2) << scalarUsPerMv << " | "
      << std::setw(11) << batchUsPerMv << " | " << std::setw(6)
      << std::setprecision(2) << speedup << "x | " << std::setw(7)
      << std::setprecision(2) << excRate << "% | " << std::setw(5) << drift
      << " |";
  LOG(INFO) << oss.str();
}

template <typename T>
void runAll() {
  const uint32_t n = kDefaultRows;
  std::vector<RunResult> rows;
  // 0% exceptions: clean 2-decimal at the exact matching (e, f).
  rows.push_back(runOne<T>(
      "clean 2dp (0% exc)",
      makeCleanTwoDecimal<T>(n),
      /*exponent=*/2,
      /*factor=*/0));
  // ~2% exceptions: sensor-like with 2% outliers, encoded at (e=3, f=0).
  rows.push_back(runOne<T>(
      "sensor 2% (~2% exc)",
      makeSensorLike<T>(n, 0.02),
      /*exponent=*/3,
      /*factor=*/0));
  // ~30% exceptions: heavy mixed precision.
  rows.push_back(runOne<T>(
      "sensor 30% (~30% exc)",
      makeSensorLike<T>(n, 0.30),
      /*exponent=*/3,
      /*factor=*/0));
  // ~100% exceptions: chaotic. Every lane hits the exception branch.
  rows.push_back(runOne<T>(
      "chaotic (~100% exc)",
      makeChaotic<T>(n),
      /*exponent=*/2,
      /*factor=*/0));
  for (const auto& r : rows) {
    printRow(r);
  }
}

template <typename T>
void runAllScoreLoop() {
  const uint32_t n = kDefaultRows;
  // Rebuild the same datasets so both tables score the same shape of input.
  std::vector<ScoreResult> rows;
  rows.push_back(runScoreLoopAB<T>(
      "clean 2dp (0% exc)",
      makeCleanTwoDecimal<T>(n),
      /*exponent=*/2,
      /*factor=*/0));
  rows.push_back(runScoreLoopAB<T>(
      "sensor 2% (~2% exc)",
      makeSensorLike<T>(n, 0.02),
      /*exponent=*/3,
      /*factor=*/0));
  rows.push_back(runScoreLoopAB<T>(
      "sensor 30% (~30% exc)",
      makeSensorLike<T>(n, 0.30),
      /*exponent=*/3,
      /*factor=*/0));
  rows.push_back(runScoreLoopAB<T>(
      "chaotic (~100% exc)",
      makeChaotic<T>(n),
      /*exponent=*/2,
      /*factor=*/0));
  for (const auto& r : rows) {
    printScoreRow(r);
  }
}

} // namespace

int main(int argc, char** argv) {
  folly::Init init{&argc, &argv};
  facebook::velox::memory::MemoryManager::initialize({});

  LOG(INFO) << "=== ALP per-row transform micro-benchmark: "
            << "scalarTransformOne vs batchTransform ===";
  LOG(INFO) << "rows per dataset: " << kDefaultRows
            << "; iterations per side: 5 (reporting min).";

  printHeader();
  runAll<double>();
  runAll<float>();

  LOG(INFO) << "Legend: ns/row = wall-clock ns per input row for a single "
            << "pass (min of 5).  speedup = scalar ns/row / batch ns/row.  "
            << "exc rate = observed exception fraction.  agree = whether "
            << "scalar and batch produced the same exception count (a "
            << "'MISMATCH' means the SIMD path diverged from scalar — bug).";

  // Table 2: scoreCombination-shape inner loop. Same rows/(e, f) as Table 1,
  // but now accumulating (zigZagMin, zigZagMax, exceptionCount) only — no
  // per-row buffer writes — so us/Mv matches the encode-wall work moved by
  // xsimd-vectorize. Drift-guarded against ALPEncoding<T>::scoreCombination.
  LOG(INFO) << "=== ALP scoreCombination inner loop: pure-scalar vs "
            << "prod SIMD shape ===";
  LOG(INFO) << "rows per dataset: " << kDefaultRows
            << "; iterations per side: 5 (reporting min).";
  printScoreHeader();
  runAllScoreLoop<double>();
  runAllScoreLoop<float>();

  LOG(INFO) << "Legend: us/Mv = wall-clock us per megavalue for one full "
            << "pass over the rows (min of 5).  speedup = scalar us/Mv / "
            << "batch us/Mv — this is the encode-wall speedup xsimd-"
            << "vectorize is expected to move on scoreCombination's hot "
            << "path.  drift = 'ok' means the SIMD-path accumulators match "
            << "the pure-scalar sweep AND the prod ALPEncoding::"
            << "scoreCombination result byte-for-byte on the same input; "
            << "'DRIFT' flags a divergence — this benchmark's SIMD shape has "
            << "fallen out of sync with prod, or one of the paths has a bug.";
  return 0;
}
