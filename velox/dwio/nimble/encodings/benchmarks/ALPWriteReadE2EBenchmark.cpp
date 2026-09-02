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

// ALP end-to-end write/read benchmark.
//
// For each dataset x dtype, encodes the full column with the production ALP
// path (size-based (exponent, factor) selection, real cost-based nested factory
// for the uint64 integer stream) and then round-trips through `materialize` to
// reconstruct the original values. Reports:
//
//   * bytes:       encoded size (compression ratio vs sizeof(T) * rows)
//   * write wall:  wall-clock of encode() alone, us / megavalue
//   * write cpu:   CPU time of encode() alone (getrusage), us / megavalue
//   * read wall:   wall-clock of materialize() alone, us / megavalue
//   * read cpu:    CPU time of materialize() alone, us / megavalue
//   * rss delta:   process max RSS delta observed across encode+decode
//   * correctness: element-wise NimbleCompare::equals over the full column
//
// The benchmark also runs an A/B pair (count-based vs size-based selector) so
// regressions in the selector's output on either write or read side are
// visible in the same table.
//
// Not a folly-benchmark; runs once and prints a markdown table via glog
// `LOG(INFO)`.

#include <folly/init/Init.h>
#include <glog/logging.h>
#include <sys/resource.h>
#include <sys/time.h>

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
#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/common/Vector.h"
#include "velox/dwio/nimble/common/tests/NimbleCompare.h"
#include "velox/dwio/nimble/encodings/ALPEncoding.h"
#include "velox/dwio/nimble/encodings/tests/TestUtils.h"

namespace {

using facebook::nimble::ALPEncoding;
using facebook::nimble::Buffer;
using facebook::nimble::CompressionType;
using facebook::nimble::NimbleCompare;
using facebook::nimble::Vector;
using facebook::nimble::test::Encoder;

std::shared_ptr<facebook::velox::memory::MemoryPool>& benchPool() {
  static auto pool = facebook::velox::memory::memoryManager()->addLeafPool(
      "alp_e2e_benchmark");
  return pool;
}

// -----------------------------------------------------------------------------
// RSS + CPU time probes
// -----------------------------------------------------------------------------

// Peak resident set size in KiB observed by the current process. Read via
// getrusage(RUSAGE_SELF).ru_maxrss (kilobytes on Linux).
uint64_t peakRssKiB() {
  struct rusage ru{};
  ::getrusage(RUSAGE_SELF, &ru);
  return static_cast<uint64_t>(ru.ru_maxrss);
}

// User + system CPU time in microseconds consumed by the process so far.
uint64_t cpuMicros() {
  struct rusage ru{};
  ::getrusage(RUSAGE_SELF, &ru);
  auto toMicros = [](const struct timeval& tv) -> uint64_t {
    return static_cast<uint64_t>(tv.tv_sec) * 1'000'000ULL +
        static_cast<uint64_t>(tv.tv_usec);
  };
  return toMicros(ru.ru_utime) + toMicros(ru.ru_stime);
}

struct Timer {
  std::chrono::steady_clock::time_point wallStart;
  uint64_t cpuStart;

  void start() {
    wallStart = std::chrono::steady_clock::now();
    cpuStart = cpuMicros();
  }
  // Returns (wallMicros, cpuMicros) since start().
  std::pair<double, double> stop() const {
    const auto wallEnd = std::chrono::steady_clock::now();
    const uint64_t cpuEnd = cpuMicros();
    return {
        std::chrono::duration<double, std::micro>(wallEnd - wallStart).count(),
        static_cast<double>(cpuEnd - cpuStart)};
  }
};

// -----------------------------------------------------------------------------
// Datasets (identical shape/seed to ALPSelectionABBenchmark, so numbers can be
// cross-referenced directly).
// -----------------------------------------------------------------------------

constexpr uint64_t kSeed = 0x51ED0FF1CEULL;
constexpr uint32_t kDefaultRows = 65'536;

template <typename T>
Vector<T> makeEmpty(uint32_t n) {
  Vector<T> v{benchPool().get()};
  v.reserve(n);
  return v;
}

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
Vector<T> makeSensorLike(uint32_t n, double outlierRate = 0.02) {
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

template <typename T>
Vector<T> makePrices(uint32_t n) {
  auto v = makeEmpty<T>(n);
  std::mt19937_64 rng(kSeed);
  std::normal_distribution<double> dist(500.0, 120.0);
  for (uint32_t i = 0; i < n; ++i) {
    double d = dist(rng);
    if (d < 0) {
      d = -d;
    }
    int64_t cents = static_cast<int64_t>(std::llround(d * 100.0));
    v.push_back(static_cast<T>(cents) / static_cast<T>(100));
  }
  return v;
}

template <typename T>
Vector<T> makeTimestampsMs(uint32_t n) {
  auto v = makeEmpty<T>(n);
  std::mt19937_64 rng(kSeed);
  std::uniform_int_distribution<int64_t> secs(1'700'000'000LL, 1'800'000'000LL);
  std::uniform_int_distribution<int32_t> millis(0, 999);
  for (uint32_t i = 0; i < n; ++i) {
    int64_t ms = secs(rng) * 1000LL + millis(rng);
    v.push_back(static_cast<T>(ms) / static_cast<T>(1'000));
  }
  return v;
}

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

template <typename T>
Vector<T> makeMixedPrecisionHeavy(uint32_t n) {
  return makeSensorLike<T>(n, /*outlierRate=*/0.30);
}

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
// End-to-end runner
// -----------------------------------------------------------------------------

enum class Strategy { Count, Size };

struct RunStats {
  uint64_t encodedBytes = 0;
  double writeWallUs = 0;
  double writeCpuUs = 0;
  double readWallUs = 0;
  double readCpuUs = 0;
  int64_t rssDeltaKiB = 0;
  bool correct = true;
  std::pair<uint8_t, uint8_t> pick{0, 0};
};

template <typename T>
RunStats runOne(
    const Vector<T>& values,
    Strategy strategy,
    uint32_t warmupIters,
    uint32_t measureIters) {
  const facebook::nimble::Encoding::Options options{};
  const std::span<const T> logical{values.data(), values.size()};

  // Selector runs once (chunk-level in real code). Same pick used by every
  // measurement iteration.
  std::pair<uint8_t, uint8_t> pick;
  if (strategy == Strategy::Count) {
    pick = ALPEncoding<T>::findBestExponentFactorByCount(logical);
  } else {
    pick = ALPEncoding<T>::findBestExponentFactorBySize(logical, options);
  }

  const auto rssBefore = peakRssKiB();

  // Warm-up: primes allocator caches and instruction cache so measured iters
  // are representative. Results not recorded.
  for (uint32_t i = 0; i < warmupIters; ++i) {
    Buffer buf{*benchPool()};
    const auto encoded = Encoder<ALPEncoding<T>>::encodeWithExponentFactor(
        buf,
        values,
        pick.first,
        pick.second,
        CompressionType::Uncompressed,
        options,
        /*realNestedSelection=*/true);
    // Round-trip once to catch build breakage during warm-up.
    auto encoding = std::make_unique<ALPEncoding<T>>(
        *benchPool(),
        encoded,
        [](uint32_t) -> void* { return nullptr; },
        options);
    Vector<T> out{benchPool().get(), values.size()};
    encoding->materialize(values.size(), out.data());
  }

  RunStats agg;
  agg.pick = pick;

  for (uint32_t iter = 0; iter < measureIters; ++iter) {
    Buffer buf{*benchPool()};

    // --- write side --------------------------------------------------------
    Timer wt;
    wt.start();
    const auto encoded = Encoder<ALPEncoding<T>>::encodeWithExponentFactor(
        buf,
        values,
        pick.first,
        pick.second,
        CompressionType::Uncompressed,
        options,
        /*realNestedSelection=*/true);
    const auto [wallW, cpuW] = wt.stop();
    agg.writeWallUs += wallW;
    agg.writeCpuUs += cpuW;
    if (iter == 0) {
      agg.encodedBytes = encoded.size();
    }

    // --- read side ---------------------------------------------------------
    // Build the Encoding out of the just-emitted bytes and materialize the
    // full column. Both the constructor and materialize() are measured
    // together, matching what a real reader pays per chunk.
    Timer rt;
    rt.start();
    auto encoding = std::make_unique<ALPEncoding<T>>(
        *benchPool(),
        encoded,
        [](uint32_t) -> void* { return nullptr; },
        options);
    Vector<T> out{benchPool().get(), values.size()};
    encoding->materialize(values.size(), out.data());
    const auto [wallR, cpuR] = rt.stop();
    agg.readWallUs += wallR;
    agg.readCpuUs += cpuR;

    // Correctness check on the first iteration only (cheap safety net).
    if (iter == 0) {
      for (uint32_t j = 0; j < values.size(); ++j) {
        if (!NimbleCompare<T>::equals(out[j], values[j])) {
          agg.correct = false;
          break;
        }
      }
    }
  }

  agg.writeWallUs /= measureIters;
  agg.writeCpuUs /= measureIters;
  agg.readWallUs /= measureIters;
  agg.readCpuUs /= measureIters;
  agg.rssDeltaKiB =
      static_cast<int64_t>(peakRssKiB()) - static_cast<int64_t>(rssBefore);
  return agg;
}

// -----------------------------------------------------------------------------
// Output formatting
// -----------------------------------------------------------------------------

void printHeader() {
  {
    std::ostringstream oss;
    oss << "| " << std::left << std::setw(28) << "dataset" << " | "
        << std::setw(6) << "dtype" << " | " << std::setw(6) << "strat" << " | "
        << std::right << std::setw(6) << "(e, f)" << " | " << std::setw(9)
        << "bytes" << " | " << std::setw(7) << "ratio" << " | " << std::setw(9)
        << "w-wall" << " | " << std::setw(9) << "w-cpu" << " | " << std::setw(9)
        << "r-wall" << " | " << std::setw(9) << "r-cpu" << " | " << std::setw(8)
        << "rss dK" << " | " << std::setw(3) << "ok" << " |";
    LOG(INFO) << oss.str();
  }
  {
    std::ostringstream oss;
    oss << "|" << std::string(30, '-') << "|" << std::string(8, '-') << "|"
        << std::string(8, '-') << "|" << std::string(8, '-') << "|"
        << std::string(11, '-') << "|" << std::string(9, '-') << "|"
        << std::string(11, '-') << "|" << std::string(11, '-') << "|"
        << std::string(11, '-') << "|" << std::string(11, '-') << "|"
        << std::string(10, '-') << "|" << std::string(5, '-') << "|";
    LOG(INFO) << oss.str();
  }
}

template <typename T>
void printRow(
    const std::string& dataset,
    Strategy strategy,
    const RunStats& s,
    uint32_t rows) {
  const double megaVals = static_cast<double>(rows) / 1'000'000.0;
  const double rawBytes = static_cast<double>(rows) * sizeof(T);
  const double ratio = rawBytes == 0
      ? 0.0
      : static_cast<double>(s.encodedBytes) / rawBytes * 100.0;
  std::ostringstream oss;
  oss << "| " << std::left << std::setw(28) << dataset << " | " << std::setw(6)
      << (std::is_same_v<T, float> ? "float" : "double") << " | " << std::setw(6)
      << (strategy == Strategy::Count ? "count" : "size") << " | "
      << std::right << "(" << std::setw(2) << int(s.pick.first) << ","
      << std::setw(2) << int(s.pick.second) << ")" << " | " << std::setw(9)
      << s.encodedBytes << " | " << std::fixed << std::setprecision(2)
      << std::setw(6) << ratio << "% | " << std::setw(7) << std::setprecision(1)
      << (s.writeWallUs / megaVals) << "us | " << std::setw(7)
      << (s.writeCpuUs / megaVals) << "us | " << std::setw(7)
      << (s.readWallUs / megaVals) << "us | " << std::setw(7)
      << (s.readCpuUs / megaVals) << "us | " << std::setw(8) << s.rssDeltaKiB
      << " | " << std::setw(3) << (s.correct ? "OK" : "BAD") << " |";
  LOG(INFO) << oss.str();
}

template <typename T>
void runAll(uint32_t warmupIters, uint32_t measureIters) {
  const uint32_t n = kDefaultRows;
  struct Dataset {
    std::string name;
    Vector<T> data;
  };
  std::vector<Dataset> datasets;
  datasets.push_back({"integers", makeIntegers<T>(n)});
  datasets.push_back({"two-decimal uniform", makeTwoDecimalUniform<T>(n)});
  datasets.push_back({"prices (2dp gaussian)", makePrices<T>(n)});
  datasets.push_back({"timestamps (ms)", makeTimestampsMs<T>(n)});
  datasets.push_back({"low cardinality (8)", makeLowCardinality<T>(n)});
  datasets.push_back({"sensor + 2% outliers", makeSensorLike<T>(n)});
  datasets.push_back({"mixed precision 30%", makeMixedPrecisionHeavy<T>(n)});
  datasets.push_back({"chaotic (all exceptions)", makeChaotic<T>(n)});

  for (const auto& d : datasets) {
    const auto cnt =
        runOne<T>(d.data, Strategy::Count, warmupIters, measureIters);
    printRow<T>(d.name, Strategy::Count, cnt, n);
    const auto sz =
        runOne<T>(d.data, Strategy::Size, warmupIters, measureIters);
    printRow<T>(d.name, Strategy::Size, sz, n);
  }
}

} // namespace

int main(int argc, char** argv) {
  folly::Init init{&argc, &argv};
  facebook::velox::memory::MemoryManager::initialize({});

  // Iteration counts kept small by default so the benchmark finishes in a few
  // seconds; measure over ~5 iters after 2 warmups so noise averages out but
  // wall-clock is not dominated by benchmark overhead itself.
  uint32_t warmupIters = 2;
  uint32_t measureIters = 5;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--warmup" && i + 1 < argc) {
      warmupIters = static_cast<uint32_t>(std::stoi(argv[++i]));
    } else if (arg == "--iters" && i + 1 < argc) {
      measureIters = static_cast<uint32_t>(std::stoi(argv[++i]));
    }
  }

  LOG(INFO) << "=== ALP end-to-end write/read benchmark ===";
  LOG(INFO) << "rows / dataset: " << kDefaultRows
            << ", warmup iters: " << warmupIters
            << ", measure iters: " << measureIters;
  LOG(INFO) << "nested encoding: production factory "
               "(realNestedSelection=true)";
  LOG(INFO) << "write/read times: microseconds per megavalue "
               "(averaged over measure iters)";
  LOG(INFO) << "ratio: encoded / (rows * sizeof(T)), lower is better";
  LOG(INFO) << "rss dK: peak-RSS delta in KiB across encode+decode";

  printHeader();
  runAll<double>(warmupIters, measureIters);
  runAll<float>(warmupIters, measureIters);

  LOG(INFO) << "Legend: strat=count uses findBestExponentFactorByCount, "
            << "strat=size uses findBestExponentFactorBySize.";
  LOG(INFO) << "Compare rows pair-wise (same dataset+dtype) to see the size-"
            << "based selector's effect on read/write paths and encoded size.";
  return 0;
}
