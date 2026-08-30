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

// Public-API-only ALP encode/decode driver that compiles on both the baseline
// (pre-`feat/alp_enhance`, count-based selection + scalar transform +
// strided-singleton sampling + old estimator) and on the current branch
// (size-based selection + SIMD batchTransform + chunked-stride sampling +
// unified estimator). Run the same binary on each tree and diff the output
// row-by-row for the merged-collection A/B comparison.
//
// For each dataset x dtype it forces ALP as the outer encoding via
// ManualEncodingSelectionPolicy (candidates = [ALP, Trivial, FixedBitWidth],
// nested policy sees the default cost-based set) and reports:
//
//   * bytes       encoded size
//   * ratio       encoded / (rows * sizeof(T)), lower is better
//   * w-wall      encode wall-clock us / megavalue (averaged over --iters)
//   * w-cpu       encode CPU time  us / megavalue
//   * r-wall      decode wall-clock us / megavalue
//   * r-cpu       decode CPU time  us / megavalue
//   * ok          NimbleCompare::equals element-wise round-trip check

#include <folly/init/Init.h>
#include <glog/logging.h>
#include <sys/resource.h>
#include <sys/time.h>

#include <chrono>
#include <cstdint>
#include <functional>
#include <iomanip>
#include <memory>
#include <random>
#include <span>
#include <sstream>
#include <string>
#include <vector>

#include "velox/common/memory/Memory.h"
#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/common/Vector.h"
#include "velox/dwio/nimble/common/tests/NimbleCompare.h"
#include "velox/dwio/nimble/encodings/common/Encoding.h"
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelectionPolicy.h"

namespace {

using facebook::nimble::Buffer;
using facebook::nimble::EncodingFactory;
using facebook::nimble::EncodingType;
using facebook::nimble::ManualEncodingSelectionPolicy;
using facebook::nimble::NimbleCompare;
using facebook::nimble::Vector;

std::shared_ptr<facebook::velox::memory::MemoryPool>& benchPool() {
  static auto pool = facebook::velox::memory::memoryManager()->addLeafPool(
      "alp_baseline_vs_branch");
  return pool;
}

uint64_t peakRssKiB() {
  struct rusage ru{};
  ::getrusage(RUSAGE_SELF, &ru);
  return static_cast<uint64_t>(ru.ru_maxrss);
}

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
  uint64_t cpuStart{0};

  void start() {
    wallStart = std::chrono::steady_clock::now();
    cpuStart = cpuMicros();
  }

  std::pair<double, double> stop() const {
    const auto wallEnd = std::chrono::steady_clock::now();
    const uint64_t cpuEnd = cpuMicros();
    return {
        std::chrono::duration<double, std::micro>(wallEnd - wallStart).count(),
        static_cast<double>(cpuEnd - cpuStart)};
  }
};

// -----------------------------------------------------------------------------
// Datasets. Same set the selection/E2E benchmarks use.
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
    int64_t ms = secs(rng) * 1'000LL + millis(rng);
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
// Force ALP as outer encoding via ManualEncodingSelectionPolicy.
// -----------------------------------------------------------------------------

template <typename T>
std::unique_ptr<ManualEncodingSelectionPolicy<T>> makeAlpForcingPolicy() {
  std::vector<std::pair<EncodingType, float>> factors{
      {EncodingType::ALP, 1.0},
      {EncodingType::Trivial, 1.0},
      {EncodingType::FixedBitWidth, 1.0},
  };
  return std::make_unique<ManualEncodingSelectionPolicy<T>>(
      std::move(factors),
      /*compressionOptions=*/std::nullopt,
      /*identifier=*/std::nullopt);
}

struct RunStats {
  uint64_t encodedBytes{0};
  double writeWallUs{0};
  double writeCpuUs{0};
  double readWallUs{0};
  double readCpuUs{0};
  int64_t rssDeltaKiB{0};
  bool correct{true};
};

template <typename T>
RunStats runOne(
    const Vector<T>& values,
    uint32_t warmupIters,
    uint32_t measureIters) {
  const facebook::nimble::Encoding::Options options{};
  const std::span<const T> logical{values.data(), values.size()};

  const auto rssBefore = peakRssKiB();

  for (uint32_t i = 0; i < warmupIters; ++i) {
    Buffer buf{*benchPool()};
    auto policy = makeAlpForcingPolicy<T>();
    const auto encoded =
        EncodingFactory::encode<T>(std::move(policy), logical, buf, options);
    auto encoding = EncodingFactory(options).create(
        *benchPool(), encoded, [](uint32_t) { return nullptr; });
    Vector<T> out{benchPool().get(), values.size()};
    encoding->materialize(values.size(), out.data());
  }

  RunStats agg;

  for (uint32_t iter = 0; iter < measureIters; ++iter) {
    Buffer buf{*benchPool()};
    auto policy = makeAlpForcingPolicy<T>();

    Timer wt;
    wt.start();
    const auto encoded =
        EncodingFactory::encode<T>(std::move(policy), logical, buf, options);
    const auto [wallW, cpuW] = wt.stop();
    agg.writeWallUs += wallW;
    agg.writeCpuUs += cpuW;
    if (iter == 0) {
      agg.encodedBytes = encoded.size();
    }

    Timer rt;
    rt.start();
    auto encoding = EncodingFactory(options).create(
        *benchPool(), encoded, [](uint32_t) { return nullptr; });
    Vector<T> out{benchPool().get(), values.size()};
    encoding->materialize(values.size(), out.data());
    const auto [wallR, cpuR] = rt.stop();
    agg.readWallUs += wallR;
    agg.readCpuUs += cpuR;

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
// Output
// -----------------------------------------------------------------------------

void printHeader() {
  {
    std::ostringstream oss;
    oss << "| " << std::left << std::setw(28) << "dataset" << " | "
        << std::setw(6) << "dtype" << " | " << std::right << std::setw(10)
        << "bytes" << " | " << std::setw(6) << "ratio" << " | " << std::setw(9)
        << "w-wall" << " | " << std::setw(9) << "w-cpu" << " | " << std::setw(9)
        << "r-wall" << " | " << std::setw(9) << "r-cpu" << " | " << std::setw(3)
        << "ok" << " |";
    LOG(INFO) << oss.str();
  }
  {
    std::ostringstream oss;
    oss << "|" << std::string(30, '-') << "|" << std::string(8, '-') << "|"
        << std::string(12, '-') << "|" << std::string(8, '-') << "|"
        << std::string(11, '-') << "|" << std::string(11, '-') << "|"
        << std::string(11, '-') << "|" << std::string(11, '-') << "|"
        << std::string(5, '-') << "|";
    LOG(INFO) << oss.str();
  }
}

template <typename T>
void printRow(const std::string& dataset, const RunStats& s, uint32_t rows) {
  const double megaVals = static_cast<double>(rows) / 1'000'000.0;
  const double rawBytes = static_cast<double>(rows) * sizeof(T);
  const double ratio = rawBytes == 0
      ? 0.0
      : static_cast<double>(s.encodedBytes) / rawBytes * 100.0;
  std::ostringstream oss;
  oss << "| " << std::left << std::setw(28) << dataset << " | "
      << std::setw(6) << (std::is_same_v<T, float> ? "float" : "double")
      << " | " << std::right << std::setw(10) << s.encodedBytes << " | "
      << std::fixed << std::setprecision(2) << std::setw(5) << ratio << "% | "
      << std::setw(7) << std::setprecision(1) << (s.writeWallUs / megaVals)
      << "us | " << std::setw(7) << (s.writeCpuUs / megaVals) << "us | "
      << std::setw(7) << (s.readWallUs / megaVals) << "us | " << std::setw(7)
      << (s.readCpuUs / megaVals) << "us | " << std::setw(3)
      << (s.correct ? "OK" : "BAD") << " |";
  LOG(INFO) << oss.str();
}

template <typename T>
struct Dataset {
  std::string name;
  std::function<Vector<T>(uint32_t)> maker;
};

template <typename T>
void runAll(uint32_t warmupIters, uint32_t measureIters) {
  const uint32_t n = kDefaultRows;
  std::vector<Dataset<T>> datasets{
      {"integers", &makeIntegers<T>},
      {"two-decimal uniform", &makeTwoDecimalUniform<T>},
      {"prices (2dp gaussian)", &makePrices<T>},
      {"timestamps (ms)", &makeTimestampsMs<T>},
      {"low cardinality (8)", &makeLowCardinality<T>},
      {"sensor + 2% outliers", [](uint32_t k) { return makeSensorLike<T>(k); }},
      {"mixed precision 30%", &makeMixedPrecisionHeavy<T>},
      {"chaotic (all exceptions)", &makeChaotic<T>},
  };
  for (const auto& d : datasets) {
    auto data = d.maker(n);
    const auto s = runOne<T>(data, warmupIters, measureIters);
    printRow<T>(d.name, s, n);
  }
}

} // namespace

int main(int argc, char** argv) {
  folly::Init init{&argc, &argv};
  facebook::velox::memory::MemoryManager::initialize({});

  uint32_t warmupIters = 3;
  uint32_t measureIters = 10;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--warmup" && i + 1 < argc) {
      warmupIters = static_cast<uint32_t>(std::stoi(argv[++i]));
    } else if (arg == "--iters" && i + 1 < argc) {
      measureIters = static_cast<uint32_t>(std::stoi(argv[++i]));
    }
  }

  LOG(INFO) << "=== ALP encode/decode public-API benchmark ===";
  LOG(INFO) << "rows / dataset: " << kDefaultRows
            << ", warmup iters: " << warmupIters
            << ", measure iters: " << measureIters;
  LOG(INFO) << "outer policy: ManualEncodingSelectionPolicy "
               "[ALP, Trivial, FixedBitWidth]";
  LOG(INFO) << "times: microseconds per megavalue "
               "(averaged over measure iters)";
  LOG(INFO) << "ratio: encoded / (rows * sizeof(T)), lower is better";

  printHeader();
  runAll<double>(warmupIters, measureIters);
  runAll<float>(warmupIters, measureIters);
  return 0;
}
