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

// ALP (exponent, factor) selection A/B benchmark.
//
// Compares two selection strategies over a small catalogue of representative
// float/double distributions and reports the actual encoded bytes each one
// produces.
//
//   count strategy: ALPEncoding<T>::findBestExponentFactorByCount(sample)
//   size  strategy: ALPEncoding<T>::findBestExponentFactorBySize(sample,
//   options)
//
// For each dataset the benchmark:
//   1. samples the first kSampleSize values to pick (e, f) via each strategy;
//   2. re-encodes the *full* dataset with `encodeWithExponentFactor` using each
//      chosen (e, f), with realNestedSelection=true so the nested uint64 stream
//      is dispatched to the real cost-based factory (Trivial / FixedBitWidth /
//      PFOR / SimdForBitpack) exactly like production;
//   3. reports the two encoded sizes plus the count-vs-size delta.
//
// The build target is a plain executable (not folly-benchmark); it prints a
// single markdown-friendly table.

#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "velox/common/memory/Memory.h"
#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/common/Vector.h"
#include "velox/dwio/nimble/encodings/ALPEncoding.h"
#include "velox/dwio/nimble/encodings/tests/TestUtils.h"

namespace {

using facebook::nimble::ALPEncoding;
using facebook::nimble::Buffer;
using facebook::nimble::CompressionType;
using facebook::nimble::Vector;
using facebook::nimble::test::Encoder;

std::shared_ptr<facebook::velox::memory::MemoryPool>& benchPool() {
  static auto pool =
      facebook::velox::memory::memoryManager()->addLeafPool("alp_ab_benchmark");
  return pool;
}

// -----------------------------------------------------------------------------
// Datasets
// -----------------------------------------------------------------------------
//
// Each generator returns a Vector<T> whose distribution stresses a different
// (representation-rate, integer-domain) trade-off. All generators use a fixed
// seed so numbers are reproducible run-to-run.

constexpr uint64_t kSeed = 0x51ED0FF1CEULL;
constexpr uint32_t kDefaultRows = 65'536;

template <typename T>
Vector<T> makeEmpty(uint32_t n) {
  Vector<T> v{benchPool().get()};
  v.reserve(n);
  return v;
}

// (1) All integers — every (e, f) with e >= f can represent everything;
//     count-based selection stops on the very first combination while
//     size-based selection walks the tie-break diagonal to (maxExponent,
//     maxFactor). Should be strictly neutral on real bytes if the estimator
//     matches reality.
template <typename T>
Vector<T> makeIntegers(uint32_t n) {
  auto v = makeEmpty<T>(n);
  std::mt19937_64 rng(kSeed);
  std::uniform_int_distribution<int32_t> dist(-1'000'000, 1'000'000);
  for (uint32_t i = 0; i < n; ++i) {
    v.push_back(static_cast<T>(dist(rng)));
  }
  return v;
}

// (2) Uniform 2-decimal values in [0, 10'000). Precisely representable at
//     (e=2, f=0); no exceptions in either strategy. Baseline check that size
//     does not regress on clean data.
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

// (3) Sensor-like: mostly 3-decimal values in [0, 100), 2% high-precision
//     outliers with 6 decimals and large magnitude. Count picks low e to
//     minimize exceptions; size picks high (e, f) to shrink FOR bit-width.
template <typename T>
Vector<T> makeSensorLike(uint32_t n, double outlierRate = 0.02) {
  auto v = makeEmpty<T>(n);
  std::mt19937_64 rng(kSeed);
  std::uniform_int_distribution<int32_t> clean(0, 99'999); // /1000 => 0..99.999
  std::uniform_real_distribution<double> pOut(0.0, 1.0);
  std::uniform_int_distribution<int64_t> outlier(
      5'000'000'000LL, 9'999'999'999LL); // /1e6 => 5000..9999.999999
  for (uint32_t i = 0; i < n; ++i) {
    if (pOut(rng) < outlierRate) {
      v.push_back(static_cast<T>(outlier(rng)) / static_cast<T>(1'000'000));
    } else {
      v.push_back(static_cast<T>(clean(rng)) / static_cast<T>(1'000));
    }
  }
  return v;
}

// (4) Financial prices: 2-decimal values clustered around a mean (log-normal-
//     ish, but simple uniform + shift). Realistic "clean money" case.
template <typename T>
Vector<T> makePrices(uint32_t n) {
  auto v = makeEmpty<T>(n);
  std::mt19937_64 rng(kSeed);
  std::normal_distribution<double> dist(500.0, 120.0);
  for (uint32_t i = 0; i < n; ++i) {
    // 2-decimal by construction: llround*0.01 keeps ALP exact at e=2.
    double d = dist(rng);
    if (d < 0) {
      d = -d;
    }
    int64_t cents = static_cast<int64_t>(std::llround(d * 100.0));
    v.push_back(static_cast<T>(cents) / static_cast<T>(100));
  }
  return v;
}

// (5) Millisecond timestamps: seconds-since-epoch + random ms. 3-decimal by
//     construction, ~10-digit integer domain — FOR bit-width is what dominates.
template <typename T>
Vector<T> makeTimestampsMs(uint32_t n) {
  auto v = makeEmpty<T>(n);
  std::mt19937_64 rng(kSeed);
  std::uniform_int_distribution<int64_t> secs(1'700'000'000LL, 1'800'000'000LL);
  std::uniform_int_distribution<int32_t> millis(0, 999);
  for (uint32_t i = 0; i < n; ++i) {
    int64_t ms = secs(rng) * 1'000LL + millis(rng);
    v.push_back(static_cast<T>(ms) / static_cast<T>(1'000));
  }
  return v;
}

// (6) Low cardinality — only 8 distinct 4-decimal values, drawn uniformly.
//     Tests the case where nested Dictionary / RLE would beat FixedBitWidth,
//     and both strategies pick the same (e, f).
template <typename T>
Vector<T> makeLowCardinality(uint32_t n) {
  auto v = makeEmpty<T>(n);
  const std::vector<double> palette = {
      0.1234, 1.5000, 2.7182, 3.1415, 42.0000, 100.9999, 999.0001, 12.3456};
  std::mt19937_64 rng(kSeed);
  std::uniform_int_distribution<uint32_t> pick(0, palette.size() - 1);
  for (uint32_t i = 0; i < n; ++i) {
    v.push_back(static_cast<T>(palette[pick(rng)]));
  }
  return v;
}

// (7) High-precision heavy: 30% high-precision exceptions on top of a clean
//     2-decimal base. Deliberately worse than sensor-like — stress test for
//     "does size still win when exceptions are common?"
template <typename T>
Vector<T> makeMixedPrecisionHeavy(uint32_t n) {
  return makeSensorLike<T>(n, /*outlierRate=*/0.30);
}

// (8) Essentially all exceptions: random doubles with no clean decimal form.
//     Both strategies should degrade similarly (no ALP win possible).
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
// A/B runner
// -----------------------------------------------------------------------------

struct AbResult {
  std::string dataset;
  std::string dtype;
  uint32_t rows;
  std::pair<uint8_t, uint8_t> countPick;
  std::pair<uint8_t, uint8_t> sizePick;
  uint64_t countBytes;
  uint64_t sizeBytes;
  double countMicros;
  double sizeMicros;
};

template <typename T>
AbResult runOne(const std::string& datasetName, const Vector<T>& values) {
  const facebook::nimble::Encoding::Options options{};
  const std::span<const T> logical{values.data(), values.size()};

  const auto tCount0 = std::chrono::steady_clock::now();
  const auto countPick = ALPEncoding<T>::findBestExponentFactorByCount(logical);
  const auto tCount1 = std::chrono::steady_clock::now();

  const auto tSize0 = std::chrono::steady_clock::now();
  const auto sizePick =
      ALPEncoding<T>::findBestExponentFactorBySize(logical, options);
  const auto tSize1 = std::chrono::steady_clock::now();

  // Encode the full dataset with each pick, using the real nested selection
  // path so the uint64 integer stream is dispatched to the production factory.
  Buffer bufCount{*benchPool()};
  const auto encCount = Encoder<ALPEncoding<T>>::encodeWithExponentFactor(
      bufCount,
      values,
      countPick.first,
      countPick.second,
      CompressionType::Uncompressed,
      options,
      /*realNestedSelection=*/true);

  Buffer bufSize{*benchPool()};
  const auto encSize = Encoder<ALPEncoding<T>>::encodeWithExponentFactor(
      bufSize,
      values,
      sizePick.first,
      sizePick.second,
      CompressionType::Uncompressed,
      options,
      /*realNestedSelection=*/true);

  AbResult r{
      .dataset = datasetName,
      .dtype = std::is_same_v<T, float> ? "float" : "double",
      .rows = static_cast<uint32_t>(values.size()),
      .countPick = countPick,
      .sizePick = sizePick,
      .countBytes = encCount.size(),
      .sizeBytes = encSize.size(),
      .countMicros =
          std::chrono::duration<double, std::micro>(tCount1 - tCount0).count(),
      .sizeMicros =
          std::chrono::duration<double, std::micro>(tSize1 - tSize0).count(),
  };
  return r;
}

void printRow(const AbResult& r) {
  const double delta = r.countBytes == 0
      ? 0.0
      : (static_cast<double>(r.sizeBytes) - static_cast<double>(r.countBytes)) /
          static_cast<double>(r.countBytes) * 100.0;
  std::cout << "| " << std::left << std::setw(28) << r.dataset << " | "
            << std::setw(6) << r.dtype << " | " << std::right << std::setw(7)
            << r.rows << " | (" << std::setw(2) << int(r.countPick.first) << ","
            << std::setw(2) << int(r.countPick.second) << ") | "
            << std::setw(10) << r.countBytes << " | (" << std::setw(2)
            << int(r.sizePick.first) << "," << std::setw(2)
            << int(r.sizePick.second) << ") | " << std::setw(10) << r.sizeBytes
            << " | " << std::fixed << std::setprecision(2) << std::setw(7)
            << delta << "% | " << std::setw(8) << std::setprecision(1)
            << r.countMicros << " | " << std::setw(8) << r.sizeMicros << " |\n";
}

void printHeader() {
  std::cout << "| " << std::left << std::setw(28) << "dataset" << " | "
            << std::setw(6) << "dtype" << " | " << std::right << std::setw(7)
            << "rows" << " | count (e, f) | count bytes | "
            << "size (e, f) | size bytes  |   delta | "
            << "count us | size us  |\n";
  std::cout << "|" << std::string(30, '-') << "|" << std::string(8, '-') << "|"
            << std::string(9, '-') << "|" << std::string(14, '-') << "|"
            << std::string(12, '-') << "|" << std::string(13, '-') << "|"
            << std::string(13, '-') << "|" << std::string(9, '-') << "|"
            << std::string(10, '-') << "|" << std::string(10, '-') << "|\n";
}

template <typename T>
void runAll() {
  const uint32_t n = kDefaultRows;
  std::vector<AbResult> rows;
  rows.push_back(runOne<T>("integers", makeIntegers<T>(n)));
  rows.push_back(runOne<T>("two-decimal uniform", makeTwoDecimalUniform<T>(n)));
  rows.push_back(runOne<T>("prices (2dp gaussian)", makePrices<T>(n)));
  rows.push_back(runOne<T>("timestamps (ms)", makeTimestampsMs<T>(n)));
  rows.push_back(runOne<T>("low cardinality (8)", makeLowCardinality<T>(n)));
  rows.push_back(runOne<T>("sensor + 2% outliers", makeSensorLike<T>(n)));
  rows.push_back(
      runOne<T>("mixed precision 30%", makeMixedPrecisionHeavy<T>(n)));
  rows.push_back(runOne<T>("chaotic (all exceptions)", makeChaotic<T>(n)));
  for (const auto& r : rows) {
    printRow(r);
  }
}

} // namespace

int main() {
  facebook::velox::memory::MemoryManager::initialize({});

  std::cout
      << "\n=== ALP (exponent, factor) selection A/B, "
      << "count-based vs size-based ===\n"
      << "rows per dataset: " << kDefaultRows
      << "; nested encoding: production factory (realNestedSelection=true)\n\n";

  printHeader();
  runAll<double>();
  runAll<float>();

  std::cout << "\nLegend: (e, f) = chosen exponent/factor.  "
            << "delta = (sizeBytes - countBytes) / countBytes.  "
            << "count us / size us: wall-clock of the selector alone "
            << "(single sample = kSampleSize rows).\n";
  return 0;
}
