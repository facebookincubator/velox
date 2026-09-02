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

// ALP transform benchmark: two comparisons of the scalar and vectorized
// transform over the same inputs.
//
//   Table 1 (per-row transform): scalar (`scalarTransformOne`) vs vector
//   (`batchTransform`) at one fixed (exponent, factor), writing masks and valid
//   ZigZag outputs. Isolates the per-row cost of the transform primitive, so
//   SIMD lane throughput can be read separately from any surrounding work.
//
//   Table 2 (selection grid): the (exponent, factor) search that runs once per
//   encode, swept scalar-only against the vectorized shape. This mirrors what
//   `findBestExponentFactorByCount` walks -- the full grid over a 1024-value
//   sample, ~276 reachable combinations -- and is the cost the vectorized
//   transform actually removes from an encode. Reported per whole sweep
//   because that is the unit an encode pays.
//
// Datasets are named after how they are generated, not after an exception rate,
// because the rate a generator produces depends on the value type: float's
// coarser mantissa makes values representable that double rejects, so the same
// generator yields very different rates for the two. The measured rate is
// reported per row in the `exc rate` column.
//
// They span from fully representable (clean 2-decimal values at e=2, f=0) to
// chaotic random values with no short rational form, so the reader can see how
// the SIMD win scales with representability: when almost every lane is
// representable, batchTransform stays on the round/compare path end-to-end;
// when exceptions dominate, both variants degrade toward similar per-row cost
// because scalarTransformOne is already just a few floating-point operations.
//
// The build target is a plain executable (not folly-benchmark).

#include <folly/init/Init.h>
#include <glog/logging.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <limits>
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
// Datasets -- each generator returns rows suited to one (exponent, factor)
// pair.
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

// (3) Chaotic doubles with no rational decimal form. Almost every value is an
// exception at any (exponent, factor), so the exception branch dominates.
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
// Table 1: per-row transform
// -----------------------------------------------------------------------------
//
// Each run does two passes over the same input: one all-scalar, one batched
// with a scalar tail. Output buffers are pre-allocated once and reused so only
// the transform is measured. The warm-up pass sits outside the timed section.

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
  bool outputsAgree;
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
  // output divergence is a scalar-vs-batch bug, not input drift.
  std::vector<physicalType> physicals(rows);
  for (uint32_t i = 0; i < rows; ++i) {
    physicals[i] = alp::toPhysical<T>(values[i]);
  }

  std::vector<uint64_t> outZigZag(rows, 0);
  std::vector<uint8_t> outMask(rows, 0); // 1 -> representable, 0 -> exception.

  auto scalarPass = [&]() -> std::pair<double, uint32_t> {
    uint32_t exceptions = 0;
    const auto start = std::chrono::steady_clock::now();
    for (uint32_t i = 0; i < rows; ++i) {
      uint64_t zigZag = 0;
      const bool ok = Alp::scalarTransformOne(
          values[i],
          physicals[i],
          exponentMultiplier,
          factorMultiplier,
          zigZag);
      outZigZag[i] = zigZag;
      outMask[i] = ok ? 1 : 0;
      if (!ok) {
        ++exceptions;
      }
    }
    const auto end = std::chrono::steady_clock::now();
    return {
        std::chrono::duration<double, std::nano>(end - start).count(),
        exceptions};
  };

  constexpr std::size_t kBatch = Alp::kBatchSize;
  auto batchPass = [&]() -> std::pair<double, uint32_t> {
    uint32_t exceptions = 0;
    alignas(64) uint64_t zigZagLanes[kBatch];
    alignas(64) bool okLanes[kBatch];
    const auto start = std::chrono::steady_clock::now();
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
        const bool ok = okLanes[k];
        outMask[i + k] = ok ? 1 : 0;
        if (ok) {
          // batchTransform deliberately leaves ZigZag undefined for exception
          // lanes, so only read the output of representable lanes.
          outZigZag[i + k] = zigZagLanes[k];
        } else {
          ++exceptions;
        }
      }
    }
    // Scalar tail, matching the shape of the production encode loop.
    for (; i < rows; ++i) {
      uint64_t zigZag = 0;
      const bool ok = Alp::scalarTransformOne(
          values[i],
          physicals[i],
          exponentMultiplier,
          factorMultiplier,
          zigZag);
      outZigZag[i] = zigZag;
      outMask[i] = ok ? 1 : 0;
      if (!ok) {
        ++exceptions;
      }
    }
    const auto end = std::chrono::steady_clock::now();
    return {
        std::chrono::duration<double, std::nano>(end - start).count(),
        exceptions};
  };

  // Warm-up, discarded.
  scalarPass();
  batchPass();

  // Take the min per side over several passes -- that suppresses a stray
  // context switch without hiding a real regression, because the min is
  // unchanged by noise when the underlying cost is stable.
  constexpr int kIters = 5;
  double scalarBest = std::numeric_limits<double>::infinity();
  double batchBest = std::numeric_limits<double>::infinity();
  uint32_t scalarExceptions = 0;
  uint32_t batchExceptions = 0;
  for (int iter = 0; iter < kIters; ++iter) {
    const auto [nanos, exceptions] = scalarPass();
    scalarBest = std::min(scalarBest, nanos);
    scalarExceptions = exceptions;
  }
  // Preserve the final scalar output outside the timed region. The batch
  // passes reuse the same output buffers to keep the measured memory-access
  // shape unchanged.
  const auto scalarZigZag = outZigZag;
  const auto scalarMask = outMask;
  for (int iter = 0; iter < kIters; ++iter) {
    const auto [nanos, exceptions] = batchPass();
    batchBest = std::min(batchBest, nanos);
    batchExceptions = exceptions;
  }

  bool outputsAgree = scalarExceptions == batchExceptions;
  for (uint32_t i = 0; outputsAgree && i < rows; ++i) {
    if (scalarMask[i] != outMask[i] ||
        (scalarMask[i] != 0 && scalarZigZag[i] != outZigZag[i])) {
      outputsAgree = false;
      break;
    }
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
      .outputsAgree = outputsAgree,
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

void printRow(const RunResult& result) {
  const double speedup =
      result.batchNs > 0 ? result.scalarNs / result.batchNs : 0.0;
  const double exceptionRate = result.rows > 0
      ? 100.0 * static_cast<double>(result.batchExceptions) /
          static_cast<double>(result.rows)
      : 0.0;
  const char* agree = result.outputsAgree ? "yes" : "MISMATCH";
  std::ostringstream oss;
  oss << "| " << std::left << std::setw(26) << result.dataset << " | "
      << std::setw(6) << result.dtype << " | " << std::right << std::setw(7)
      << result.rows << " | (" << std::setw(2) << int(result.exponent) << ","
      << std::setw(2) << int(result.factor) << ") | " << std::setw(13)
      << std::fixed << std::setprecision(2) << result.scalarNs << " | "
      << std::setw(12) << result.batchNs << " | " << std::setw(6)
      << std::setprecision(2) << speedup << "x | " << std::setw(7)
      << std::setprecision(2) << exceptionRate << "% | " << std::setw(5)
      << agree << " |";
  LOG(INFO) << oss.str();
}

// -----------------------------------------------------------------------------
// Table 2: (exponent, factor) selection grid
// -----------------------------------------------------------------------------
//
// Reproduces the search that runs once per encode: for every candidate in the
// grid, count how many sampled values the transform represents exactly, and
// keep the best. The scalar arm is the search as it stood before the
// vectorized transform; the batch arm is the current shape. Both arms must
// pick the same (exponent, factor) and the same count -- that equality is the
// correctness check reported in the `agree` column.
//
// Mirrors findBestExponentFactorByCount rather than calling it: that function
// is private, so the loop bounds and the early exit are restated here. Keep
// them in step if the production search changes.

// Largest exponent and factor the search walks, bounded by kPow10Double.
constexpr int kMaxExponent = 23;
constexpr int kMaxFactor = 23;
// Values the production search samples from the head of the input.
constexpr uint32_t kSampleSize = 1024;

struct GridResult {
  std::string dataset;
  std::string dtype;
  uint32_t sampleSize;
  uint8_t exponent;
  uint8_t factor;
  double scalarUs; // One full grid sweep.
  double batchUs;
  uint32_t representableCount;
  bool armsAgree;
};

// One (exponent, factor) candidate, counted all-scalar.
template <typename T>
uint32_t countScalar(
    std::span<const T> values,
    const typename TypeTraits<T>::physicalType* physicals,
    int exponent,
    int factor) {
  using Alp = ALPEncoding<T>;
  const double exponentMultiplier = Alp::kPow10Double[exponent];
  const double factorMultiplier = Alp::kPow10Double[factor];
  uint32_t count = 0;
  for (std::size_t i = 0; i < values.size(); ++i) {
    uint64_t zigZag = 0;
    if (Alp::scalarTransformOne(
            values[i],
            physicals[i],
            exponentMultiplier,
            factorMultiplier,
            zigZag)) {
      ++count;
    }
  }
  return count;
}

// The same candidate, routed through the vectorized transform with a scalar
// tail -- the shape countRepresentable() uses in production.
template <typename T>
uint32_t countBatch(
    std::span<const T> values,
    const typename TypeTraits<T>::physicalType* physicals,
    int exponent,
    int factor) {
  using Alp = ALPEncoding<T>;
  constexpr std::size_t kBatch = Alp::kBatchSize;
  const double exponentMultiplier = Alp::kPow10Double[exponent];
  const double factorMultiplier = Alp::kPow10Double[factor];
  const std::size_t size = values.size();

  uint32_t count = 0;
  alignas(64) uint64_t zigZagLanes[kBatch];
  alignas(64) bool okLanes[kBatch];

  std::size_t i = 0;
  for (; i + kBatch <= size; i += kBatch) {
    Alp::batchTransform(
        values.data() + i,
        physicals + i,
        exponentMultiplier,
        factorMultiplier,
        zigZagLanes,
        okLanes);
    for (std::size_t k = 0; k < kBatch; ++k) {
      count += okLanes[k] ? 1 : 0;
    }
  }
  for (; i < size; ++i) {
    uint64_t zigZag = 0;
    if (Alp::scalarTransformOne(
            values[i],
            physicals[i],
            exponentMultiplier,
            factorMultiplier,
            zigZag)) {
      ++count;
    }
  }
  return count;
}

// Walks the candidate grid with `counter`, returning the winning pair, its
// count, and the elapsed microseconds for the whole sweep.
template <typename T, typename Counter>
std::tuple<double, uint8_t, uint8_t, uint32_t> sweepGrid(
    std::span<const T> sample,
    const typename TypeTraits<T>::physicalType* physicals,
    Counter counter) {
  const uint32_t sampleSize = static_cast<uint32_t>(sample.size());
  uint8_t bestExponent = 0;
  uint8_t bestFactor = 0;
  uint32_t bestCount = 0;

  const auto start = std::chrono::steady_clock::now();
  for (int e = 0; e <= kMaxExponent; ++e) {
    const uint32_t countNoFactor = counter(sample, physicals, e, /*factor=*/0);
    if (countNoFactor > bestCount) {
      bestCount = countNoFactor;
      bestExponent = static_cast<uint8_t>(e);
      bestFactor = 0;
    }
    if (bestCount == sampleSize) {
      break;
    }
    for (int f = 1; f <= std::min(e, kMaxFactor); ++f) {
      const uint32_t countWithFactor = counter(sample, physicals, e, f);
      if (countWithFactor > bestCount) {
        bestCount = countWithFactor;
        bestExponent = static_cast<uint8_t>(e);
        bestFactor = static_cast<uint8_t>(f);
      }
    }
    if (bestCount == sampleSize) {
      break;
    }
  }
  const auto end = std::chrono::steady_clock::now();
  return {
      std::chrono::duration<double, std::micro>(end - start).count(),
      bestExponent,
      bestFactor,
      bestCount};
}

template <typename T>
GridResult runGridAB(const std::string& datasetName, const Vector<T>& values) {
  using physicalType = typename TypeTraits<T>::physicalType;

  const uint32_t sampleSize =
      std::min(static_cast<uint32_t>(values.size()), kSampleSize);
  const std::span<const T> sample{values.data(), sampleSize};

  std::vector<physicalType> physicals(sampleSize);
  for (uint32_t i = 0; i < sampleSize; ++i) {
    physicals[i] = alp::toPhysical<T>(sample[i]);
  }

  auto scalarSweep = [&]() {
    return sweepGrid<T>(sample, physicals.data(), &countScalar<T>);
  };
  auto batchSweep = [&]() {
    return sweepGrid<T>(sample, physicals.data(), &countBatch<T>);
  };

  // Warm-up, discarded.
  (void)scalarSweep();
  (void)batchSweep();

  constexpr int kIters = 5;
  double scalarBest = std::numeric_limits<double>::infinity();
  double batchBest = std::numeric_limits<double>::infinity();
  uint8_t scalarExponent = 0, scalarFactor = 0;
  uint8_t batchExponent = 0, batchFactor = 0;
  uint32_t scalarCount = 0, batchCount = 0;
  for (int iter = 0; iter < kIters; ++iter) {
    const auto [micros, exponent, factor, count] = scalarSweep();
    scalarBest = std::min(scalarBest, micros);
    scalarExponent = exponent;
    scalarFactor = factor;
    scalarCount = count;
  }
  for (int iter = 0; iter < kIters; ++iter) {
    const auto [micros, exponent, factor, count] = batchSweep();
    batchBest = std::min(batchBest, micros);
    batchExponent = exponent;
    batchFactor = factor;
    batchCount = count;
  }

  return GridResult{
      .dataset = datasetName,
      .dtype = std::is_same_v<T, float> ? "float" : "double",
      .sampleSize = sampleSize,
      .exponent = batchExponent,
      .factor = batchFactor,
      .scalarUs = scalarBest,
      .batchUs = batchBest,
      .representableCount = batchCount,
      .armsAgree = scalarExponent == batchExponent &&
          scalarFactor == batchFactor && scalarCount == batchCount,
  };
}

void printGridHeader() {
  {
    std::ostringstream oss;
    oss << "| " << std::left << std::setw(26) << "dataset" << " | "
        << std::setw(6) << "dtype" << " | " << std::right << std::setw(7)
        << "sample" << " | (e, f) | scalar us | batch us |  speedup"
        << " | repr | agree |";
    LOG(INFO) << oss.str();
  }
  {
    std::ostringstream oss;
    oss << "|" << std::string(28, '-') << "|" << std::string(8, '-') << "|"
        << std::string(9, '-') << "|" << std::string(9, '-') << "|"
        << std::string(11, '-') << "|" << std::string(10, '-') << "|"
        << std::string(10, '-') << "|" << std::string(8, '-') << "|"
        << std::string(7, '-') << "|";
    LOG(INFO) << oss.str();
  }
}

void printGridRow(const GridResult& result) {
  const double speedup =
      result.batchUs > 0 ? result.scalarUs / result.batchUs : 0.0;
  const double representableRate = result.sampleSize > 0
      ? 100.0 * static_cast<double>(result.representableCount) /
          static_cast<double>(result.sampleSize)
      : 0.0;
  const char* agree = result.armsAgree ? "yes" : "MISMATCH";
  std::ostringstream oss;
  oss << "| " << std::left << std::setw(26) << result.dataset << " | "
      << std::setw(6) << result.dtype << " | " << std::right << std::setw(7)
      << result.sampleSize << " | (" << std::setw(2) << int(result.exponent)
      << "," << std::setw(2) << int(result.factor) << ") | " << std::setw(9)
      << std::fixed << std::setprecision(2) << result.scalarUs << " | "
      << std::setw(8) << result.batchUs << " | " << std::setw(6)
      << std::setprecision(2) << speedup << "x | " << std::setw(3)
      << std::setprecision(0) << representableRate << "% | " << std::setw(5)
      << agree << " |";
  LOG(INFO) << oss.str();
}

// -----------------------------------------------------------------------------
// Drivers
// -----------------------------------------------------------------------------

template <typename T>
void runAll() {
  const uint32_t n = kDefaultRows;
  std::vector<RunResult> rows;
  // Fully representable for both types: clean 2-decimal at the exact matching
  // (exponent, factor).
  rows.push_back(
      runOne<T>(
          "clean 2dp",
          makeCleanTwoDecimal<T>(n),
          /*exponent=*/2,
          /*factor=*/0));
  // Sensor-like with 2% wide-range outliers, encoded at (e=3, f=0).
  rows.push_back(
      runOne<T>(
          "sensor 2% outliers",
          makeSensorLike<T>(n, 0.02),
          /*exponent=*/3,
          /*factor=*/0));
  // Same shape with 30% outliers: heavy mixed precision.
  rows.push_back(
      runOne<T>(
          "sensor 30% outliers",
          makeSensorLike<T>(n, 0.30),
          /*exponent=*/3,
          /*factor=*/0));
  // Chaotic random values with no short rational form.
  rows.push_back(
      runOne<T>(
          "chaotic",
          makeChaotic<T>(n),
          /*exponent=*/2,
          /*factor=*/0));
  for (const auto& row : rows) {
    printRow(row);
  }
}

template <typename T>
void runAllGrid() {
  const uint32_t n = kDefaultRows;
  // Same datasets as Table 1 so both tables score the same shape of input.
  std::vector<GridResult> rows;
  rows.push_back(runGridAB<T>("clean 2dp", makeCleanTwoDecimal<T>(n)));
  rows.push_back(
      runGridAB<T>("sensor 2% outliers", makeSensorLike<T>(n, 0.02)));
  rows.push_back(
      runGridAB<T>("sensor 30% outliers", makeSensorLike<T>(n, 0.30)));
  rows.push_back(runGridAB<T>("chaotic", makeChaotic<T>(n)));
  for (const auto& row : rows) {
    printGridRow(row);
  }
}

} // namespace

int main(int argc, char** argv) {
  folly::Init init{&argc, &argv};
  facebook::velox::memory::MemoryManager::initialize({});

  LOG(INFO) << "=== ALP per-row transform: scalarTransformOne vs "
            << "batchTransform ===";
  LOG(INFO) << "rows per dataset: " << kDefaultRows
            << "; iterations per side: 5 (reporting min).";
  printHeader();
  runAll<double>();
  runAll<float>();
  LOG(INFO) << "Legend: ns/row = wall-clock ns per input row for a single "
            << "pass (min of 5).  speedup = scalar ns/row / batch ns/row.  "
            << "exc rate = observed exception fraction.  agree = whether "
            << "scalar and batch produced identical masks and ZigZag values "
            << "for every representable lane; "
            << "'MISMATCH' means the vectorized path diverged from scalar, "
            << "which is a bug.";

  LOG(INFO) << "=== ALP (exponent, factor) selection grid: scalar vs "
            << "vectorized ===";
  LOG(INFO) << "sample per dataset: " << kSampleSize
            << " values; iterations per side: 5 (reporting min).";
  printGridHeader();
  runAllGrid<double>();
  runAllGrid<float>();
  LOG(INFO) << "Legend: us = wall-clock us for one full sweep of the "
            << "candidate grid over the sample, which is what an encode pays "
            << "once (min of 5).  speedup = scalar us / batch us.  repr = "
            << "share of the sample the winning pair represents exactly.  "
            << "agree = whether both arms selected the same (exponent, "
            << "factor) with the same count; 'MISMATCH' is a bug.";
  return 0;
}
