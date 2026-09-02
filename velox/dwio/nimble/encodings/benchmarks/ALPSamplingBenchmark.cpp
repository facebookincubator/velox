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

// ALP sampling A/B benchmark: legacy strided-singleton vs chunked-stride.
//
// The estimator draws kSampleSize=1024 rows from the input to pick an
// (exponent, factor). The legacy sampler pulled 1024 far-apart singletons;
// the chunked-stride sampler now draws kSamplingChunks short contiguous
// runs whose total length is still 1024. Contiguous reads let the prefetcher
// stream cache lines, so on large inputs the gather stage is much cheaper —
// but the (e, f) chosen from the shorter contiguous runs must still be
// identical for well-behaved inputs, because both samplers cover the input
// evenly.
//
// This benchmark reports two comparisons on the same datasets:
//
//   Table 1 (gather stage in isolation): times just the sample gather ns —
//   the piece the chunked-stride change touched. This isolates the memory-
//   traversal win from the size selector's cost.
//
//   Table 2 (gather + selector together): times gather + full
//   `pickBySizeFromSample` sweep — the substantive work of prod's
//   `ALPEncoding<T>::findBestExponentFactorBySize`. Reports us/Mv so the
//   number is directly comparable to `findBestExponentFactorBySize` and
//   captures the -16% headline the chunked-stride commit's message claimed.
//
// Both tables include a drift guard: on each dataset the local chunked-stride
// sampler + local size selector must pick the same (e, f) as prod
// `ALPEncoding<T>::findBestExponentFactorBySize`, so the local reimplementation
// cannot silently drift from prod.
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
#include <utility>
#include <vector>

#include "velox/common/memory/Memory.h"
#include "velox/dwio/nimble/common/Vector.h"
#include "velox/dwio/nimble/encodings/ALPEncoding.h"

namespace {

using facebook::nimble::ALPEncoding;
using facebook::nimble::Vector;

std::shared_ptr<facebook::velox::memory::MemoryPool>& benchPool() {
  static auto pool = facebook::velox::memory::memoryManager()->addLeafPool(
      "alp_sampling_benchmark");
  return pool;
}

// -----------------------------------------------------------------------------
// Sample gatherers
// -----------------------------------------------------------------------------
//
// Both return exactly `sampleSize` values. The legacy sampler mirrors the
// original implementation: sampleIndex * rowCount / sampleSize. The chunked-
// stride sampler mirrors the current in-tree layout (kept in sync with
// ALPEncoding<T>::estimateSize).

constexpr uint32_t kSampleSize = 1024;
constexpr uint32_t kSamplingChunks = 32;

template <typename T>
void gatherLegacyStride(
    std::span<const T> input,
    std::vector<T>& sampleOut) {
  const uint64_t rowCount = input.size();
  const uint32_t sampleSize = std::min<uint32_t>(rowCount, kSampleSize);
  sampleOut.clear();
  sampleOut.reserve(sampleSize);
  for (uint32_t i = 0; i < sampleSize; ++i) {
    const uint64_t idx = static_cast<uint64_t>(i) * rowCount / sampleSize;
    sampleOut.push_back(input[idx]);
  }
}

template <typename T>
void gatherChunkedStride(
    std::span<const T> input,
    std::vector<T>& sampleOut) {
  const uint64_t rowCount = input.size();
  const uint32_t sampleSize = std::min<uint32_t>(rowCount, kSampleSize);
  sampleOut.clear();
  sampleOut.reserve(sampleSize);
  if (rowCount <= kSamplingChunks || sampleSize <= kSamplingChunks) {
    // Fallback path shared with prod: dense strided singletons for tiny
    // inputs. Not exercised in this benchmark (rows always in the millions).
    for (uint32_t i = 0; i < sampleSize; ++i) {
      const uint64_t idx = static_cast<uint64_t>(i) * rowCount / sampleSize;
      sampleOut.push_back(input[idx]);
    }
    return;
  }
  const uint32_t chunkSize = sampleSize / kSamplingChunks;
  for (uint32_t chunk = 0; chunk < kSamplingChunks; ++chunk) {
    const uint64_t chunkStart =
        static_cast<uint64_t>(chunk) * rowCount / kSamplingChunks;
    const uint64_t chunkLen =
        std::min<uint64_t>(chunkSize, rowCount - chunkStart);
    const T* p = input.data() + chunkStart;
    for (uint64_t j = 0; j < chunkLen; ++j) {
      sampleOut.push_back(p[j]);
    }
  }
  // Top up if integer division dropped a few slots.
  while (sampleOut.size() < sampleSize) {
    const uint64_t idx = static_cast<uint64_t>(sampleOut.size()) * rowCount /
        sampleSize;
    sampleOut.push_back(input[idx]);
  }
}

// -----------------------------------------------------------------------------
// Datasets — large enough that the prefetcher-win is visible.
// -----------------------------------------------------------------------------

constexpr uint64_t kSeed = 0x51ED0FF1CEULL;

template <typename T>
Vector<T> makeEmpty(uint32_t n) {
  Vector<T> v{benchPool().get()};
  v.reserve(n);
  return v;
}

template <typename T>
Vector<T> makeTwoDecimalUniform(uint32_t n) {
  auto v = makeEmpty<T>(n);
  std::mt19937_64 rng(kSeed);
  std::uniform_int_distribution<int32_t> dist(0, 999'999);
  for (uint32_t i = 0; i < n; ++i) {
    v.push_back(static_cast<T>(dist(rng)) / static_cast<T>(100));
  }
  return v;
}

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

// Cluster the outliers into short runs. Under a strided sampler this hides
// the outlier basin between stride steps; under chunked-stride it shows up
// in whichever chunk lands on the cluster. If the two samplers disagree on
// (e, f) we expect it here — treat any divergence as a diagnostic signal,
// not a failure of one strategy.
template <typename T>
Vector<T> makeClusteredOutliers(uint32_t n, uint32_t clusterEvery, uint32_t clusterLen) {
  auto v = makeEmpty<T>(n);
  std::mt19937_64 rng(kSeed);
  std::uniform_int_distribution<int32_t> clean(0, 99'999);
  std::uniform_int_distribution<int64_t> outlier(
      5'000'000'000LL, 9'999'999'999LL);
  for (uint32_t i = 0; i < n; ++i) {
    const bool inCluster = (i % clusterEvery) < clusterLen;
    if (inCluster) {
      v.push_back(static_cast<T>(outlier(rng)) / static_cast<T>(1'000'000));
    } else {
      v.push_back(static_cast<T>(clean(rng)) / static_cast<T>(1'000));
    }
  }
  return v;
}

// -----------------------------------------------------------------------------
// A/B runner
// -----------------------------------------------------------------------------

struct SamplingResult {
  std::string dataset;
  std::string dtype;
  uint32_t rows;
  double legacyGatherNs;
  double chunkedGatherNs;
  std::pair<uint8_t, uint8_t> legacyPick;
  std::pair<uint8_t, uint8_t> chunkedPick;
};

// Local size-based selector that operates on a pre-gathered sample. Mirrors
// findBestExponentFactorBySize but consumes an already-materialized sample
// so we can isolate the sampling stage.
template <typename T>
std::pair<uint8_t, uint8_t> pickBySizeFromSample(std::span<const T> sample) {
  using Alp = ALPEncoding<T>;
  const facebook::nimble::Encoding::Options options{};

  uint8_t bestExponent = 0;
  uint8_t bestFactor = 0;
  uint64_t bestBytes = Alp::kUnusableScore;

  // Iterate the same (e, f) grid the production selector walks; ascending
  // order with `<=` recovers the DuckDB tie-break rule (larger e wins on
  // ties, then larger f).
  for (int e = 0; e <= 18; ++e) {
    // Cap factor at min(e, kMaxFactor). The runtime kMaxExponent differs
    // between float (10) and double (18), so gate on both here.
    const int fCap = std::min<int>(e, std::is_same_v<T, float> ? 10 : 18);
    if (e > (std::is_same_v<T, float> ? 10 : 18)) {
      break;
    }
    for (int f = 0; f <= fCap; ++f) {
      const auto score = Alp::scoreCombination(sample, e, f, options);
      if (score.estimatedBytes == Alp::kUnusableScore) {
        continue;
      }
      if (score.estimatedBytes <= bestBytes) {
        bestBytes = score.estimatedBytes;
        bestExponent = static_cast<uint8_t>(e);
        bestFactor = static_cast<uint8_t>(f);
      }
    }
  }
  return {bestExponent, bestFactor};
}

template <typename T>
SamplingResult runOne(const std::string& datasetName, const Vector<T>& values) {
  const std::span<const T> logical{values.data(), values.size()};

  // Pre-touch the input span so both samplers race under the same cache
  // state on the first timed pass. Without this the first-run sampler wins
  // trivially on cold caches.
  volatile double sum = 0;
  for (uint32_t i = 0; i < values.size(); ++i) {
    sum = sum + values[i];
  }
  (void)sum;

  std::vector<T> sampleLegacy;
  std::vector<T> sampleChunked;

  // Warm-up passes discarded.
  gatherLegacyStride<T>(logical, sampleLegacy);
  gatherChunkedStride<T>(logical, sampleChunked);

  constexpr int kIters = 7;
  double legacyBest = std::numeric_limits<double>::infinity();
  double chunkedBest = std::numeric_limits<double>::infinity();
  for (int i = 0; i < kIters; ++i) {
    const auto t0 = std::chrono::steady_clock::now();
    gatherLegacyStride<T>(logical, sampleLegacy);
    const auto t1 = std::chrono::steady_clock::now();
    legacyBest = std::min(
        legacyBest,
        std::chrono::duration<double, std::nano>(t1 - t0).count());
  }
  for (int i = 0; i < kIters; ++i) {
    const auto t0 = std::chrono::steady_clock::now();
    gatherChunkedStride<T>(logical, sampleChunked);
    const auto t1 = std::chrono::steady_clock::now();
    chunkedBest = std::min(
        chunkedBest,
        std::chrono::duration<double, std::nano>(t1 - t0).count());
  }

  // Score each sample through the shared scoreCombination -> selector path.
  const std::span<const T> legacySpan{
      sampleLegacy.data(), sampleLegacy.size()};
  const std::span<const T> chunkedSpan{
      sampleChunked.data(), sampleChunked.size()};

  return SamplingResult{
      .dataset = datasetName,
      .dtype = std::is_same_v<T, float> ? "float" : "double",
      .rows = static_cast<uint32_t>(values.size()),
      .legacyGatherNs = legacyBest,
      .chunkedGatherNs = chunkedBest,
      .legacyPick = pickBySizeFromSample<T>(legacySpan),
      .chunkedPick = pickBySizeFromSample<T>(chunkedSpan),
  };
}

void printHeader() {
  {
    std::ostringstream oss;
    oss << "| " << std::left << std::setw(28) << "dataset" << " | "
        << std::setw(6) << "dtype" << " | " << std::right << std::setw(9)
        << "rows" << " | legacy us | chunked us |  speedup | "
        << "legacy (e,f) | chunked (e,f) | agree |";
    LOG(INFO) << oss.str();
  }
  {
    std::ostringstream oss;
    oss << "|" << std::string(30, '-') << "|" << std::string(8, '-') << "|"
        << std::string(11, '-') << "|" << std::string(11, '-') << "|"
        << std::string(12, '-') << "|" << std::string(10, '-') << "|"
        << std::string(15, '-') << "|" << std::string(16, '-') << "|"
        << std::string(7, '-') << "|";
    LOG(INFO) << oss.str();
  }
}

void printRow(const SamplingResult& r) {
  const double speedup =
      r.chunkedGatherNs > 0 ? r.legacyGatherNs / r.chunkedGatherNs : 0.0;
  const bool agree = r.legacyPick == r.chunkedPick;
  std::ostringstream oss;
  oss << "| " << std::left << std::setw(28) << r.dataset << " | "
      << std::setw(6) << r.dtype << " | " << std::right << std::setw(9)
      << r.rows << " | " << std::setw(9) << std::fixed << std::setprecision(2)
      << (r.legacyGatherNs / 1000.0) << " | " << std::setw(10)
      << (r.chunkedGatherNs / 1000.0) << " | " << std::setw(6)
      << std::setprecision(2) << speedup << "x | (" << std::setw(2)
      << int(r.legacyPick.first) << "," << std::setw(2)
      << int(r.legacyPick.second) << ")       | (" << std::setw(2)
      << int(r.chunkedPick.first) << "," << std::setw(2)
      << int(r.chunkedPick.second) << ")        | " << std::setw(5)
      << (agree ? "yes" : "DIFF") << " |";
  LOG(INFO) << oss.str();
}

// -----------------------------------------------------------------------------
// gather + selector A/B (matches prod findBestExponentFactorBySize shape)
// -----------------------------------------------------------------------------

struct SelectorResult {
  std::string dataset;
  std::string dtype;
  uint32_t rows;
  double legacyUs; // wall us for one gather+pick pass
  double chunkedUs;
  std::pair<uint8_t, uint8_t> legacyPick;
  std::pair<uint8_t, uint8_t> chunkedPick;
  std::pair<uint8_t, uint8_t> prodPick; // ALPEncoding::findBestExponentFactorBySize
  bool driftGuardOk; // chunked pick == prod pick
};

template <typename T>
SelectorResult runSelectorOne(
    const std::string& datasetName,
    const Vector<T>& values) {
  using Alp = ALPEncoding<T>;
  const facebook::nimble::Encoding::Options options{};
  const std::span<const T> logical{values.data(), values.size()};

  // Pre-touch to normalize cache state on the first timed pass. Same
  // rationale as the gather-only path.
  volatile double sum = 0;
  for (uint32_t i = 0; i < values.size(); ++i) {
    sum = sum + values[i];
  }
  (void)sum;

  auto legacySweep = [&]() -> std::pair<double, std::pair<uint8_t, uint8_t>> {
    std::vector<T> sample;
    const auto t0 = std::chrono::steady_clock::now();
    gatherLegacyStride<T>(logical, sample);
    const auto pick = pickBySizeFromSample<T>(
        std::span<const T>{sample.data(), sample.size()});
    const auto t1 = std::chrono::steady_clock::now();
    return {
        std::chrono::duration<double, std::micro>(t1 - t0).count(), pick};
  };
  auto chunkedSweep = [&]() -> std::pair<double, std::pair<uint8_t, uint8_t>> {
    std::vector<T> sample;
    const auto t0 = std::chrono::steady_clock::now();
    gatherChunkedStride<T>(logical, sample);
    const auto pick = pickBySizeFromSample<T>(
        std::span<const T>{sample.data(), sample.size()});
    const auto t1 = std::chrono::steady_clock::now();
    return {
        std::chrono::duration<double, std::micro>(t1 - t0).count(), pick};
  };

  // Warm-up passes discarded.
  (void)legacySweep();
  (void)chunkedSweep();

  constexpr int kIters = 5;
  double legacyBest = std::numeric_limits<double>::infinity();
  double chunkedBest = std::numeric_limits<double>::infinity();
  std::pair<uint8_t, uint8_t> legacyPick{0, 0};
  std::pair<uint8_t, uint8_t> chunkedPick{0, 0};
  for (int i = 0; i < kIters; ++i) {
    const auto [us, pick] = legacySweep();
    if (us < legacyBest) {
      legacyBest = us;
    }
    legacyPick = pick;
  }
  for (int i = 0; i < kIters; ++i) {
    const auto [us, pick] = chunkedSweep();
    if (us < chunkedBest) {
      chunkedBest = us;
    }
    chunkedPick = pick;
  }

  // Drift guard: the local chunked-stride sampler + local size selector must
  // pick the same (e, f) as prod findBestExponentFactorBySize on the same
  // input. Prod uses the same chunked-stride layout under the hood, so this
  // catches any accidental divergence in the local reimplementation.
  const auto prodPick = Alp::findBestExponentFactorBySize(logical, options);

  return SelectorResult{
      .dataset = datasetName,
      .dtype = std::is_same_v<T, float> ? "float" : "double",
      .rows = static_cast<uint32_t>(values.size()),
      .legacyUs = legacyBest,
      .chunkedUs = chunkedBest,
      .legacyPick = legacyPick,
      .chunkedPick = chunkedPick,
      .prodPick = prodPick,
      .driftGuardOk = (chunkedPick == prodPick),
  };
}

void printSelectorHeader() {
  {
    std::ostringstream oss;
    oss << "| " << std::left << std::setw(28) << "dataset" << " | "
        << std::setw(6) << "dtype" << " | " << std::right << std::setw(9)
        << "rows" << " | legacy us | chunked us |  speedup | "
        << "chunked (e,f) |  prod (e,f) | drift |";
    LOG(INFO) << oss.str();
  }
  {
    std::ostringstream oss;
    oss << "|" << std::string(30, '-') << "|" << std::string(8, '-') << "|"
        << std::string(11, '-') << "|" << std::string(11, '-') << "|"
        << std::string(12, '-') << "|" << std::string(10, '-') << "|"
        << std::string(16, '-') << "|" << std::string(14, '-') << "|"
        << std::string(7, '-') << "|";
    LOG(INFO) << oss.str();
  }
}

void printSelectorRow(const SelectorResult& r) {
  const double speedup = r.chunkedUs > 0 ? r.legacyUs / r.chunkedUs : 0.0;
  const char* drift = r.driftGuardOk ? "ok" : "DRIFT";
  std::ostringstream oss;
  oss << "| " << std::left << std::setw(28) << r.dataset << " | "
      << std::setw(6) << r.dtype << " | " << std::right << std::setw(9)
      << r.rows << " | " << std::setw(9) << std::fixed << std::setprecision(2)
      << r.legacyUs << " | " << std::setw(10) << r.chunkedUs << " | "
      << std::setw(6) << std::setprecision(2) << speedup << "x | ("
      << std::setw(2) << int(r.chunkedPick.first) << "," << std::setw(2)
      << int(r.chunkedPick.second) << ")        | (" << std::setw(2)
      << int(r.prodPick.first) << "," << std::setw(2) << int(r.prodPick.second)
      << ")      | " << std::setw(5) << drift << " |";
  LOG(INFO) << oss.str();
}

// The datasets used across both tables. Kept in one place so the two tables
// operate on identical inputs — otherwise a caller could easily compare rows
// generated from different data.
template <typename T>
std::vector<std::pair<std::string, Vector<T>>> makeDatasets() {
  std::vector<std::pair<std::string, Vector<T>>> out;
  // Two sizes span L2 to well beyond LLC on typical servers, showing the
  // prefetch win widen with input length.
  for (uint32_t n : {1u << 20, 1u << 23}) {
    out.emplace_back("two-decimal uniform", makeTwoDecimalUniform<T>(n));
    out.emplace_back("sensor + 2% outliers", makeSensorLike<T>(n, 0.02));
    out.emplace_back("sensor + 30% outliers", makeSensorLike<T>(n, 0.30));
    out.emplace_back(
        "clustered outliers (rare)",
        makeClusteredOutliers<T>(n, /*every=*/16'384, /*len=*/64));
  }
  return out;
}

template <typename T>
void runAll(const std::vector<std::pair<std::string, Vector<T>>>& datasets) {
  for (const auto& [name, values] : datasets) {
    printRow(runOne<T>(name, values));
  }
}

template <typename T>
void runAllSelector(
    const std::vector<std::pair<std::string, Vector<T>>>& datasets) {
  for (const auto& [name, values] : datasets) {
    printSelectorRow(runSelectorOne<T>(name, values));
  }
}

} // namespace

int main(int argc, char** argv) {
  folly::Init init{&argc, &argv};
  facebook::velox::memory::MemoryManager::initialize({});

  LOG(INFO) << "=== ALP sampler A/B: legacy strided-singleton vs "
            << "chunked-stride ===";
  LOG(INFO) << "sample size: " << kSampleSize << " values in either layout; "
            << "chunks: " << kSamplingChunks << ".";

  auto doubleDatasets = makeDatasets<double>();
  auto floatDatasets = makeDatasets<float>();

  LOG(INFO) << "-- Table 1: gather stage in isolation (min of 7 iters) --";
  printHeader();
  runAll<double>(doubleDatasets);
  runAll<float>(floatDatasets);

  LOG(INFO) << "Legend (Table 1): legacy us / chunked us = wall-clock "
            << "microseconds for the gather stage alone (min of 7).  "
            << "speedup = legacy / chunked.  (e, f) = size-based selector's "
            << "pick given each sample.  agree = whether legacy and "
            << "chunked-stride picked the same (e, f).  A 'DIFF' row is a "
            << "sampling-coverage diagnostic — legitimate on clustered "
            << "outliers, but should be rare on uniform / lightly-noisy "
            << "inputs.";

  LOG(INFO) << "-- Table 2: gather + selector together (min of 5 iters), "
            << "matches ALPEncoding::findBestExponentFactorBySize --";
  printSelectorHeader();
  runAllSelector<double>(doubleDatasets);
  runAllSelector<float>(floatDatasets);

  LOG(INFO) << "Legend (Table 2): legacy us / chunked us = wall-clock "
            << "microseconds for one full sweep of "
            << "(gatherStride + pickBySizeFromSample) — the substantive work "
            << "of findBestExponentFactorBySize.  speedup = legacy / "
            << "chunked, and captures the chunked-stride commit's "
            << "encode-wall attribution on selector-heavy paths.  chunked "
            << "(e, f) is the local sampler+selector pick; prod (e, f) is "
            << "ALPEncoding<T>::findBestExponentFactorBySize(logical, "
            << "options).  drift = whether the two agree — 'DRIFT' means "
            << "the local reimplementation diverged from prod on this "
            << "input, which should never happen and indicates a bug in "
            << "this benchmark, not in prod.";
  return 0;
}
