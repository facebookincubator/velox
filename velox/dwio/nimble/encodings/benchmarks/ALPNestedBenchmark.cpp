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

// Nested-ALP mini-benchmark.
//
// After the ALP vectorization change (batchTransform in
// encodeWithExponentFactor + scoreCombination), the two nested paths that
// pipe floating-point data into ALP -- Dictionary alphabet and RLE run values
// -- should already flow through the vectorized inner loop:
//
//   * DictionaryEncoding<T>::encodeAlphabet (floating-point branch)
//       -> EncodingFactory::encode<T>
//       -> ALPEncoding<T>::encode
//       -> encodeWithExponentFactor  (batchTransform-driven)
//
//   * RLEEncoding<T>::estimateRunValuesSize (floating-point branch)
//       -> detail::nestedAlpSize
//       -> ALPEncoding<T>::estimateSize
//       -> findBestExponentFactorBySize/scoreCombination (batchTransform)
//
// This benchmark drives the real EncodingFactory with
// `allowNestedAlpSelection = {false, true}` on Dictionary-friendly and
// RLE-friendly float/double datasets and reports encode/decode wall/cpu
// microseconds per megavalue, encoded bytes, compression ratio, RSS delta,
// and a round-trip correctness bit. Intended as evidence that the
// vectorization already covers the nested-ALP paths and as a baseline for
// detecting future regressions there.
//
// Not a folly-benchmark; runs once and prints a markdown table.

#include <sys/resource.h>
#include <sys/time.h>

#include <chrono>
#include <cstdint>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "velox/common/memory/Memory.h"
#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/common/Vector.h"
#include "velox/dwio/nimble/common/tests/NimbleCompare.h"
#include "velox/dwio/nimble/encodings/common/Encoding.h"
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelectionPolicy.h"
#include "velox/dwio/nimble/tools/EncodingUtilities.h"

namespace {

using facebook::nimble::Buffer;
using facebook::nimble::DataType;
using facebook::nimble::EncodingFactory;
using facebook::nimble::EncodingType;
using facebook::nimble::ManualEncodingSelectionPolicy;
using facebook::nimble::NimbleCompare;
using facebook::nimble::TypeTraits;
using facebook::nimble::Vector;

std::shared_ptr<facebook::velox::memory::MemoryPool>& benchPool() {
  static auto pool = facebook::velox::memory::memoryManager()->addLeafPool(
      "alp_nested_benchmark");
  return pool;
}

// -----------------------------------------------------------------------------
// RSS + CPU time probes (same primitives as ALPWriteReadE2EBenchmark so
// numbers are directly comparable).
// -----------------------------------------------------------------------------

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
// Datasets. Each dataset is either "dictionary-friendly" (few unique values
// spread over a big column) or "RLE-friendly" (long runs of the same value).
// -----------------------------------------------------------------------------

constexpr uint64_t kSeed = 0x51ED0FF1CEULL;
constexpr uint32_t kDefaultRows = 65'536;

template <typename T>
Vector<T> makeEmpty(uint32_t n) {
  Vector<T> v{benchPool().get()};
  v.reserve(n);
  return v;
}

// Dictionary-friendly: small palette of ALP-representable decimal values
// scattered uniformly. Alphabet is tiny so Dictionary wins the outer selection;
// its alphabet is the interesting path for nested ALP.
template <typename T>
Vector<T> makeDictSmallPalette(uint32_t n) {
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

// Dictionary-friendly: two-decimal prices with medium cardinality
// (~256 distinct values). ALP handles the alphabet well; Trivial/FixedBitWidth
// on the raw column would eat sizeof(T) per row.
template <typename T>
Vector<T> makeDictTwoDecimalPrices(uint32_t n) {
  auto v = makeEmpty<T>(n);
  std::vector<double> palette;
  palette.reserve(256);
  for (int cents = 0; cents < 25'600; cents += 100) {
    palette.push_back(cents / 100.0);
  }
  std::mt19937_64 rng(kSeed);
  std::uniform_int_distribution<uint32_t> pick(0, palette.size() - 1);
  for (uint32_t i = 0; i < n; ++i) {
    v.push_back(static_cast<T>(palette[pick(rng)]));
  }
  return v;
}

// RLE-friendly: long runs of the same two-decimal price. runCount is tiny,
// runValues is the interesting stream for nested ALP.
template <typename T>
Vector<T> makeRleTwoDecimalRuns(uint32_t n) {
  auto v = makeEmpty<T>(n);
  std::mt19937_64 rng(kSeed);
  std::uniform_int_distribution<uint32_t> runLen(64, 256);
  std::uniform_int_distribution<int32_t> price(0, 999'999);
  uint32_t i = 0;
  while (i < n) {
    const uint32_t rl = std::min(runLen(rng), n - i);
    const T val = static_cast<T>(price(rng)) / static_cast<T>(100);
    for (uint32_t j = 0; j < rl; ++j) {
      v.push_back(val);
    }
    i += rl;
  }
  return v;
}

// RLE-friendly: shorter runs with a wider distinct-run count so RLE + nested
// ALP is meaningfully exercised for the run-values stream.
template <typename T>
Vector<T> makeRleShortRuns(uint32_t n) {
  auto v = makeEmpty<T>(n);
  std::mt19937_64 rng(kSeed);
  std::uniform_int_distribution<uint32_t> runLen(8, 32);
  std::uniform_int_distribution<int32_t> price(0, 99'999);
  uint32_t i = 0;
  while (i < n) {
    const uint32_t rl = std::min(runLen(rng), n - i);
    const T val = static_cast<T>(price(rng)) / static_cast<T>(1'000);
    for (uint32_t j = 0; j < rl; ++j) {
      v.push_back(val);
    }
    i += rl;
  }
  return v;
}

// -----------------------------------------------------------------------------
// Encoding tree inspection: after encode(), walk the serialized bytes and
// note whether an ALP node appears anywhere in the tree. If Dict/RLE didn't
// nest into ALP, the "path=" column in the report will show that.
// -----------------------------------------------------------------------------

struct EncodedShape {
  EncodingType outer{EncodingType::Trivial};
  bool hasAlp{false};
};

EncodedShape inspect(std::string_view encoded) {
  EncodedShape shape;
  bool first = true;
  facebook::nimble::tools::traverseEncodings(
      encoded,
      [&](auto encodingType,
          auto /* dataType */,
          auto /* level */,
          auto /* index */,
          auto /* nestedEncodingName */,
          auto /* properties */) {
        if (first) {
          shape.outer = encodingType;
          first = false;
        }
        if (encodingType == EncodingType::ALP) {
          shape.hasAlp = true;
        }
        return true;
      });
  return shape;
}

// -----------------------------------------------------------------------------
// End-to-end runner.
// -----------------------------------------------------------------------------

// Which candidates the outer policy considers. Keep this list narrow so the
// selector's choice is unambiguous per dataset shape:
//   * Dict candidates include Dictionary + fallbacks;
//   * RLE candidates include RLE + fallbacks.
// The nested policy still sees the full default set (via
// createNestedPolicy), so ALP is reachable there when allowNestedAlpSelection
// is on.
enum class OuterPick { Dictionary, Rle };

template <typename T>
std::unique_ptr<ManualEncodingSelectionPolicy<T>> makeOuterPolicy(
    OuterPick outer) {
  std::vector<std::pair<EncodingType, float>> factors;
  switch (outer) {
    case OuterPick::Dictionary:
      factors = {
          {EncodingType::Dictionary, 1.0},
          {EncodingType::Trivial, 1.0},
          {EncodingType::FixedBitWidth, 1.0},
      };
      break;
    case OuterPick::Rle:
      factors = {
          {EncodingType::RLE, 1.0},
          {EncodingType::Trivial, 1.0},
          {EncodingType::FixedBitWidth, 1.0},
      };
      break;
  }
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
  EncodedShape shape;
};

template <typename T>
RunStats runOne(
    const Vector<T>& values,
    OuterPick outer,
    bool allowNestedAlp,
    uint32_t warmupIters,
    uint32_t measureIters) {
  const facebook::nimble::Encoding::Options options{
      .useVarintRowCount = false,
      .fixedBitWidthUseExactBits = true,
      .allowNestedAlpSelection = allowNestedAlp};

  const std::span<const T> logical{values.data(), values.size()};

  const auto rssBefore = peakRssKiB();

  // Warm-up: primes allocator caches; results discarded.
  for (uint32_t i = 0; i < warmupIters; ++i) {
    Buffer buf{*benchPool()};
    auto policy = makeOuterPolicy<T>(outer);
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
    auto policy = makeOuterPolicy<T>(outer);

    Timer wt;
    wt.start();
    const auto encoded =
        EncodingFactory::encode<T>(std::move(policy), logical, buf, options);
    const auto [wallW, cpuW] = wt.stop();
    agg.writeWallUs += wallW;
    agg.writeCpuUs += cpuW;
    if (iter == 0) {
      agg.encodedBytes = encoded.size();
      agg.shape = inspect(encoded);
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
// Output formatting
// -----------------------------------------------------------------------------

const char* outerLabel(OuterPick outer) {
  switch (outer) {
    case OuterPick::Dictionary:
      return "dict";
    case OuterPick::Rle:
      return "rle";
  }
  return "?";
}

const char* encodingLabel(EncodingType t) {
  switch (t) {
    case EncodingType::Trivial:
      return "Trivial";
    case EncodingType::FixedBitWidth:
      return "FixedBW";
    case EncodingType::Dictionary:
      return "Dict";
    case EncodingType::RLE:
      return "RLE";
    case EncodingType::ALP:
      return "ALP";
    default:
      return "other";
  }
}

void printHeader() {
  std::cout << "\n"
            << "| " << std::left << std::setw(24) << "dataset" << " | "
            << std::setw(6) << "dtype" << " | " << std::setw(6) << "outer"
            << " | " << std::setw(4) << "alp?" << " | " << std::setw(10)
            << "path" << " | " << std::right << std::setw(9) << "bytes"
            << " | " << std::setw(7) << "ratio" << " | " << std::setw(9)
            << "w-wall" << " | " << std::setw(9) << "w-cpu" << " | "
            << std::setw(9) << "r-wall" << " | " << std::setw(9) << "r-cpu"
            << " | " << std::setw(8) << "rss dK" << " | " << std::setw(3)
            << "ok" << " |\n";
  std::cout << "|" << std::string(26, '-') << "|" << std::string(8, '-') << "|"
            << std::string(8, '-') << "|" << std::string(6, '-') << "|"
            << std::string(12, '-') << "|" << std::string(11, '-') << "|"
            << std::string(9, '-') << "|" << std::string(11, '-') << "|"
            << std::string(11, '-') << "|" << std::string(11, '-') << "|"
            << std::string(11, '-') << "|" << std::string(10, '-') << "|"
            << std::string(5, '-') << "|\n";
}

template <typename T>
void printRow(
    const std::string& dataset,
    OuterPick outer,
    bool allowNestedAlp,
    const RunStats& s,
    uint32_t rows) {
  const double megaVals = static_cast<double>(rows) / 1'000'000.0;
  const double rawBytes = static_cast<double>(rows) * sizeof(T);
  const double ratio = rawBytes == 0
      ? 0.0
      : static_cast<double>(s.encodedBytes) / rawBytes * 100.0;
  std::string path = encodingLabel(s.shape.outer);
  if (s.shape.hasAlp) {
    path += "+ALP";
  }
  std::cout << "| " << std::left << std::setw(24) << dataset << " | "
            << std::setw(6) << (std::is_same_v<T, float> ? "float" : "double")
            << " | " << std::setw(6) << outerLabel(outer) << " | "
            << std::setw(4) << (allowNestedAlp ? "on" : "off") << " | "
            << std::setw(10) << path << " | " << std::right << std::setw(9)
            << s.encodedBytes << " | " << std::fixed << std::setprecision(2)
            << std::setw(6) << ratio << "% | " << std::setw(7)
            << std::setprecision(1) << (s.writeWallUs / megaVals) << "us | "
            << std::setw(7) << (s.writeCpuUs / megaVals) << "us | "
            << std::setw(7) << (s.readWallUs / megaVals) << "us | "
            << std::setw(7) << (s.readCpuUs / megaVals) << "us | "
            << std::setw(8) << s.rssDeltaKiB << " | " << std::setw(3)
            << (s.correct ? "OK" : "BAD") << " |\n";
}

template <typename T>
struct Dataset {
  std::string name;
  OuterPick outer;
  std::function<Vector<T>(uint32_t)> maker;
};

template <typename T>
void runAll(uint32_t warmupIters, uint32_t measureIters) {
  const uint32_t n = kDefaultRows;
  std::vector<Dataset<T>> datasets{
      {"dict small palette (8)",
       OuterPick::Dictionary,
       &makeDictSmallPalette<T>},
      {"dict two-decimal (256)",
       OuterPick::Dictionary,
       &makeDictTwoDecimalPrices<T>},
      {"rle long runs (2dp)", OuterPick::Rle, &makeRleTwoDecimalRuns<T>},
      {"rle short runs (3dp)", OuterPick::Rle, &makeRleShortRuns<T>},
  };

  for (const auto& d : datasets) {
    auto data = d.maker(n);
    for (bool nested : {false, true}) {
      const auto s =
          runOne<T>(data, d.outer, nested, warmupIters, measureIters);
      printRow<T>(d.name, d.outer, nested, s, n);
    }
  }
}

} // namespace

int main(int argc, char** argv) {
  facebook::velox::memory::MemoryManager::initialize({});

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

  std::cout << "\n=== ALP nested-selection mini-benchmark ===\n"
            << "rows / dataset: " << kDefaultRows
            << ", warmup iters: " << warmupIters
            << ", measure iters: " << measureIters
            << "\nalp? column: allowNestedAlpSelection value\n"
            << "path column: outer encoding chosen by the selector, "
               "+ALP if any node in the tree is ALP\n"
            << "write/read times: microseconds per megavalue "
               "(averaged over measure iters)\n"
            << "ratio: encoded / (rows * sizeof(T)), lower is better\n"
            << "rss dK: peak-RSS delta in KiB across encode+decode\n";

  printHeader();
  runAll<double>(warmupIters, measureIters);
  runAll<float>(warmupIters, measureIters);

  std::cout << "\nRead each dataset as a pair (alp?=off vs alp?=on). "
            << "On matching path=<outer>+ALP the nested path was taken; "
            << "compare bytes / ratio and w-wall / r-wall to see the "
            << "vectorized nested-ALP cost profile.\n";
  return 0;
}
