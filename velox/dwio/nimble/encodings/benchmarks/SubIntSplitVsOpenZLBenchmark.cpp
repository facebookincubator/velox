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

// Comparison driver: the structural SubIntSplit encoding (with tuned section
// candidates + entropy) vs the OpenZL general-purpose compressor, Zstd, and
// FixedBitWidth, across many integer data characteristics.
//
// Two modes:
//   (default)  folly microbenchmarks -- encode + decode wall-clock, one size.
//   --csv      robust sweep: for every (pattern, size, method) it averages
//              compression ratio and encode/decode throughput over --trials
//              independent fixed seeds and reports mean and standard deviation.
//              This feeds the evaluation tables/plots.
//
// All generators are seeded deterministically, so every run is reproducible and
// trials differ only by an explicit per-trial seed.

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "folly/Benchmark.h"
#include "folly/init/Init.h"
#include "velox/common/memory/Memory.h"
#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/common/Vector.h"
#include "velox/dwio/nimble/compression/Compression.h"
#include "velox/dwio/nimble/compression/CompressionPolicy.h"
#include "velox/dwio/nimble/encodings/FixedBitWidthEncoding.h"
#include "velox/dwio/nimble/encodings/SubIntSplitEncoding.h"
#include "velox/dwio/nimble/encodings/benchmarks/BenchmarkUtils.h"
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelection.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelectionPolicy.h"
#include "velox/dwio/nimble/encodings/selection/Statistics.h"

using namespace facebook::nimble;
using namespace facebook::nimble::benchmarks;

DEFINE_bool(
    csv,
    false,
    "Emit the robust CSV sweep instead of microbenchmarks.");
DEFINE_int32(
    trials,
    5,
    "Independent seeds averaged per (pattern, size, method).");
DEFINE_int64(
    only_size,
    0,
    "If >0, run the CSV sweep for only this element count (e.g. 100000000).");
DEFINE_bool(
    layout,
    false,
    "Print the SubIntSplit section layout (bit ranges + chosen encoding) "
    "for each pattern instead of running microbenchmarks.");

namespace {

using T = uint64_t;
constexpr DataType kDataType = DataType::Uint64;
constexpr int kBitWidth = 64;

// ---------------------------------------------------------------------------
// Seeded data generators, parameterized by element count and trial seed.
// ---------------------------------------------------------------------------

// Uniformly random full-width values (incompressible worst case).
Vector<T> genRandom(uint32_t n, uint64_t seed) {
  std::mt19937_64 rng{seed};
  Vector<T> d{benchmarkPool().get()};
  d.resize(n);
  for (uint32_t i = 0; i < n; ++i) {
    d[i] = rng();
  }
  return d;
}

// Values that fit in `bitWidth` bits, uniformly random.
Vector<T> genNarrow(int bitWidth, uint32_t n, uint64_t seed) {
  std::mt19937_64 rng{seed};
  Vector<T> d{benchmarkPool().get()};
  d.resize(n);
  const T mask = bitWidth >= 64 ? ~T{0} : ((T{1} << bitWidth) - 1);
  for (uint32_t i = 0; i < n; ++i) {
    d[i] = rng() & mask;
  }
  return d;
}

// Constant high prefix, uniformly random low `lowBits`.
Vector<T> genBitStruct(int lowBits, uint32_t n, uint64_t seed) {
  std::mt19937_64 rng{seed};
  Vector<T> d{benchmarkPool().get()};
  d.resize(n);
  const T prefix = 0x1234567800000000ULL;
  const T lowMask = (T{1} << lowBits) - 1;
  for (uint32_t i = 0; i < n; ++i) {
    d[i] = prefix | (rng() & lowMask);
  }
  return d;
}

// Snowflake-style IDs: 41-bit ms timestamp (slowly increasing), 10-bit machine
// id (constant), 12-bit per-ms sequence counter.
Vector<T> genSnowflake(uint32_t n, uint64_t seed) {
  std::mt19937_64 rng{seed};
  Vector<T> d{benchmarkPool().get()};
  d.resize(n);
  const T machineId = 501;
  T ts = 1700000000000ULL & ((T{1} << 41) - 1);
  T seq = 0;
  for (uint32_t i = 0; i < n; ++i) {
    if ((rng() % 8) == 0) {
      ++ts;
      seq = 0;
    } else {
      seq = (seq + 1) & 0xFFF;
    }
    d[i] = (ts << 22) | (machineId << 12) | seq;
  }
  return d;
}

// Snowflake variant with a handful of distinct machine ids (low-cardinality
// middle field) -- a harder multi-field case.
Vector<T> genSnowflakeMultiDc(uint32_t n, uint64_t seed) {
  std::mt19937_64 rng{seed};
  Vector<T> d{benchmarkPool().get()};
  d.resize(n);
  T ts = 1700000000000ULL & ((T{1} << 41) - 1);
  T seq = 0;
  for (uint32_t i = 0; i < n; ++i) {
    if ((rng() % 8) == 0) {
      ++ts;
      seq = 0;
    } else {
      seq = (seq + 1) & 0xFFF;
    }
    const T machineId = 500 + (rng() % 8);
    d[i] = (ts << 22) | (machineId << 12) | seq;
  }
  return d;
}

// Monotonically increasing millisecond timestamps with small jitter.
Vector<T> genTimestampMs(uint32_t n, uint64_t seed) {
  std::mt19937_64 rng{seed};
  Vector<T> d{benchmarkPool().get()};
  d.resize(n);
  T now = 1700000000000ULL;
  for (uint32_t i = 0; i < n; ++i) {
    now += rng() % 5;
    d[i] = now;
  }
  return d;
}

// Sorted random values (monotone, non-uniform gaps).
Vector<T> genSorted(uint32_t n, uint64_t seed) {
  std::mt19937_64 rng{seed};
  Vector<T> d{benchmarkPool().get()};
  d.resize(n);
  T cur = 0;
  for (uint32_t i = 0; i < n; ++i) {
    cur += rng() % 1024;
    d[i] = cur;
  }
  return d;
}

// Dense sequential ids starting at an offset (0,1,2,...).
Vector<T> genSequential(uint32_t n, uint64_t seed) {
  std::mt19937_64 rng{seed};
  const T base = rng();
  Vector<T> d{benchmarkPool().get()};
  d.resize(n);
  for (uint32_t i = 0; i < n; ++i) {
    d[i] = base + i;
  }
  return d;
}

// Low cardinality: values drawn uniformly from [0, cardinality).
Vector<T> genLowCard(uint32_t cardinality, uint32_t n, uint64_t seed) {
  std::mt19937_64 rng{seed};
  Vector<T> d{benchmarkPool().get()};
  d.resize(n);
  for (uint32_t i = 0; i < n; ++i) {
    d[i] = rng() % cardinality;
  }
  return d;
}

// 95% one dominant value, 5% random outliers.
Vector<T> genMainlyConstant(uint32_t n, uint64_t seed) {
  std::mt19937_64 rng{seed};
  Vector<T> d{benchmarkPool().get()};
  d.resize(n);
  for (uint32_t i = 0; i < n; ++i) {
    d[i] = (rng() % 100 < 95) ? T{42} : rng();
  }
  return d;
}

// ~99% zero, 1% random (very sparse).
Vector<T> genMostlyZero(uint32_t n, uint64_t seed) {
  std::mt19937_64 rng{seed};
  Vector<T> d{benchmarkPool().get()};
  d.resize(n);
  for (uint32_t i = 0; i < n; ++i) {
    d[i] = (rng() % 100 == 0) ? rng() : T{0};
  }
  return d;
}

// Skewed (Zipfian-ish) over `cardinality` values: frequent values get small
// codes. Approximates a power-law by exponentiating a uniform.
Vector<T> genZipfian(uint32_t cardinality, uint32_t n, uint64_t seed) {
  std::mt19937_64 rng{seed};
  std::uniform_real_distribution<double> unif{0.0, 1.0};
  Vector<T> d{benchmarkPool().get()};
  d.resize(n);
  for (uint32_t i = 0; i < n; ++i) {
    const double u = unif(rng);
    d[i] = static_cast<T>(cardinality * std::pow(u, 3.0)) % cardinality;
  }
  return d;
}

// Packed multi-entropy field: 8-bit constant, 16-bit low-card, 20-bit counter,
// 20-bit random -- exercises several sections at once.
Vector<T> genMultiField(uint32_t n, uint64_t seed) {
  std::mt19937_64 rng{seed};
  Vector<T> d{benchmarkPool().get()};
  d.resize(n);
  const T constField = 0xAB;
  T counter = 0;
  for (uint32_t i = 0; i < n; ++i) {
    const T lowCard = rng() % 64;
    const T rnd = rng() & 0xFFFFF; // 20 bits
    counter = (counter + 1) & 0xFFFFF; // 20-bit wrapping counter
    d[i] = (constField << 56) | (lowCard << 40) | (counter << 20) | rnd;
  }
  return d;
}

struct Pattern {
  std::string name;
  std::function<Vector<T>(uint32_t, uint64_t)> gen;
};

std::vector<Pattern> patterns() {
  return {
      {"Random", [](uint32_t n, uint64_t s) { return genRandom(n, s); }},
      {"Narrow8", [](uint32_t n, uint64_t s) { return genNarrow(8, n, s); }},
      {"Narrow16", [](uint32_t n, uint64_t s) { return genNarrow(16, n, s); }},
      {"Narrow32", [](uint32_t n, uint64_t s) { return genNarrow(32, n, s); }},
      {"Narrow48", [](uint32_t n, uint64_t s) { return genNarrow(48, n, s); }},
      {"BitStruct16",
       [](uint32_t n, uint64_t s) { return genBitStruct(16, n, s); }},
      {"BitStruct32",
       [](uint32_t n, uint64_t s) { return genBitStruct(32, n, s); }},
      {"Snowflake", [](uint32_t n, uint64_t s) { return genSnowflake(n, s); }},
      {"SnowflakeMultiDc",
       [](uint32_t n, uint64_t s) { return genSnowflakeMultiDc(n, s); }},
      {"TimestampMs",
       [](uint32_t n, uint64_t s) { return genTimestampMs(n, s); }},
      {"Sorted", [](uint32_t n, uint64_t s) { return genSorted(n, s); }},
      {"Sequential",
       [](uint32_t n, uint64_t s) { return genSequential(n, s); }},
      {"LowCard16",
       [](uint32_t n, uint64_t s) { return genLowCard(16, n, s); }},
      {"LowCard256",
       [](uint32_t n, uint64_t s) { return genLowCard(256, n, s); }},
      {"LowCard1024",
       [](uint32_t n, uint64_t s) { return genLowCard(1024, n, s); }},
      {"LowCard64K",
       [](uint32_t n, uint64_t s) { return genLowCard(65536, n, s); }},
      {"MainlyConstant",
       [](uint32_t n, uint64_t s) { return genMainlyConstant(n, s); }},
      {"MostlyZero",
       [](uint32_t n, uint64_t s) { return genMostlyZero(n, s); }},
      {"Zipfian4096",
       [](uint32_t n, uint64_t s) { return genZipfian(4096, n, s); }},
      {"MultiField",
       [](uint32_t n, uint64_t s) { return genMultiField(n, s); }},
  };
}

// ---------------------------------------------------------------------------
// Methods.
// ---------------------------------------------------------------------------

struct Encoded {
  std::string bytes;
  bool decodable;
};

class ForcePolicy : public CompressionPolicy {
 public:
  explicit ForcePolicy(CompressionType type) : type_{type} {}
  CompressionConfig config() const override {
    CompressionConfig config;
    config.compressionType = type_;
    config.minCompressionSize = 0;
    config.parameters.zstd.compressionLevel = 3;
    config.parameters.openzl.compressionLevel = 6;
    config.parameters.openzl.decompressionLevel = 3;
    config.parameters.openzl.formatVersion = 25;
    return config;
  }
  bool shouldAccept(CompressionType, uint64_t, uint64_t) const override {
    return true;
  }

 private:
  const CompressionType type_;
};

struct Method {
  std::string name;
  std::function<Encoded(const Vector<T>&)> encode;
  std::function<void(const std::string&, uint32_t n)> decode;
};

Encoded compressWith(CompressionType type, const Vector<T>& data) {
  auto& pool = benchmarkPool();
  ForcePolicy policy{type};
  std::string_view raw{
      reinterpret_cast<const char*>(data.data()), data.size() * sizeof(T)};
  auto result = Compression::compress(*pool, raw, kDataType, kBitWidth, policy);
  if (result.buffer.has_value() &&
      result.compressionType != CompressionType::Uncompressed) {
    return {std::string{result.buffer->data(), result.buffer->size()}, true};
  }
  return {std::string{raw}, false};
}

void decompressWith(CompressionType type, const std::string& compressed) {
  auto& pool = benchmarkPool();
  auto buffer = Compression::uncompress(
      *pool, type, kDataType, compressed, nullptr, nullptr);
  folly::doNotOptimizeAway(buffer);
}

std::vector<std::pair<EncodingType, float>> tunedReadFactors() {
  auto factors =
      ManualEncodingSelectionPolicyFactory::defaultEncodingReadFactors();
  factors.emplace_back(EncodingType::Delta, 1.0f);
  factors.emplace_back(EncodingType::FOR, 1.0f);
  factors.emplace_back(EncodingType::BlockBitPacking, 1.0f);
  return factors;
}

Encoded encodeSubIntSplitWith(
    const Vector<T>& data,
    std::vector<std::pair<EncodingType, float>> readFactors,
    CompressionType compressionType) {
  auto& pool = benchmarkPool();
  Buffer buffer{*pool};
  std::span<const uint64_t> values{data.data(), data.size()};
  std::optional<CompressionOptions> compressionOptions;
  if (compressionType != CompressionType::Uncompressed) {
    CompressionOptions options;
    options.compressionType = compressionType;
    options.compressionAcceptRatio = 1.0f;
    options.zstdMinCompressionSize = 0;
    options.openzlMinCompressionSize = 0;
    compressionOptions = options;
  }
  ManualEncodingSelectionPolicyFactory factory{
      std::move(readFactors), compressionOptions};
  EncodingSelectionResult result{.encodingType = EncodingType::SubIntSplit};
  // SubIntSplit::encode never reads the parent selection's statistics (it draws
  // its own sample and each section computes fresh statistics inside
  // encodeNested), so we skip the expensive full-stream Statistics pass here.
  // This matches the production replay path, where SubIntSplit is applied from
  // a captured layout without recomputing outer statistics.
  EncodingSelection<uint64_t> selection{
      std::move(result),
      Statistics<uint64_t>::create(values.subspan(0, 1)),
      factory.createPolicy(DataType::Uint64)};
  auto encoded =
      SubIntSplitEncoding<uint64_t>::encode(selection, values, buffer, {});
  return {std::string{encoded.data(), encoded.size()}, true};
}

void decodeNimble(const std::string& encoded, uint32_t n) {
  auto& pool = benchmarkPool();
  std::vector<T> out(n);
  auto enc = EncodingFactory{}.create(*pool, encoded, nullFactory());
  enc->materialize(n, out.data());
  folly::doNotOptimizeAway(out);
}

// SubIntSplit is not wired into EncodingFactory dispatch, so decode it by
// constructing the encoding directly. This is driver-only: the production
// EncodingFactory is unchanged. Nested (possibly Zstd/OpenZL-compressed)
// sections are still built through the constructor's internal factory.
void decodeSubIntSplit(const std::string& encoded, uint32_t n) {
  auto& pool = benchmarkPool();
  std::vector<T> out(n);
  SubIntSplitEncoding<T> enc{*pool, encoded, nullFactory()};
  enc.materialize(n, out.data());
  folly::doNotOptimizeAway(out);
}

std::vector<Method> makeMethods() {
  std::vector<Method> methods;
  methods.push_back(
      {"SubIntSplit",
       [](const Vector<T>& d) {
         return encodeSubIntSplitWith(
             d,
             ManualEncodingSelectionPolicyFactory::defaultEncodingReadFactors(),
             CompressionType::Uncompressed);
       },
       decodeSubIntSplit});
  methods.push_back(
      {"SubIntSplitTuned",
       [](const Vector<T>& d) {
         return encodeSubIntSplitWith(
             d, tunedReadFactors(), CompressionType::Zstd);
       },
       decodeSubIntSplit});
  methods.push_back(
      {"SubIntSplitOpenZL",
       [](const Vector<T>& d) {
         return encodeSubIntSplitWith(
             d, tunedReadFactors(), CompressionType::OpenZL);
       },
       decodeSubIntSplit});
  methods.push_back(
      {"OpenZL",
       [](const Vector<T>& d) {
         return compressWith(CompressionType::OpenZL, d);
       },
       [](const std::string& c, uint32_t) {
         decompressWith(CompressionType::OpenZL, c);
       }});
  methods.push_back(
      {"Zstd",
       [](const Vector<T>& d) {
         return compressWith(CompressionType::Zstd, d);
       },
       [](const std::string& c, uint32_t) {
         decompressWith(CompressionType::Zstd, c);
       }});
  methods.push_back(
      {"FixedBitWidth",
       [](const Vector<T>& d) {
         return Encoded{
             encodeData<FixedBitWidthEncoding<T>>(
                 EncodingType::FixedBitWidth, d),
             true};
       },
       decodeNimble});
  return methods;
}

// ---------------------------------------------------------------------------
// Robust CSV sweep.
// ---------------------------------------------------------------------------

double timedThroughputMbPerSec(
    const std::function<void()>& fn,
    uint64_t rawBytes,
    int iters) {
  fn(); // warmup
  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < iters; ++i) {
    fn();
  }
  auto end = std::chrono::steady_clock::now();
  const double seconds =
      std::chrono::duration<double>(end - start).count() / iters;
  return (static_cast<double>(rawBytes) / (1024.0 * 1024.0)) / seconds;
}

struct Stat {
  double mean{0.0};
  double stddev{0.0};
};

Stat summarize(const std::vector<double>& xs) {
  if (xs.empty()) {
    return {};
  }
  double sum = 0.0;
  for (const double x : xs) {
    sum += x;
  }
  const double mean = sum / xs.size();
  double var = 0.0;
  for (const double x : xs) {
    var += (x - mean) * (x - mean);
  }
  return {mean, std::sqrt(var / xs.size())};
}

// Per-element-count run configuration. Very large sizes use a representative
// pattern subset with fewer trials/iterations so the sweep stays tractable
// (100M elements is 800 MB raw per dataset; the full matrix would run for
// hours). The subset spans the win/tie/loss spectrum seen at smaller sizes.
struct SizeSpec {
  uint64_t size;
  int trials;
  int encodeIters;
  int decodeIters;
  bool subset;
};

bool inBigSubset(const std::string& name) {
  static const std::vector<std::string> kSubset = {
      "Random",
      "BitStruct16",
      "Snowflake",
      "TimestampMs",
      "Sequential",
      "LowCard1024",
      "MainlyConstant",
      "MultiField"};
  return std::find(kSubset.begin(), kSubset.end(), name) != kSubset.end();
}

void emitCsv(int trials, int64_t onlySize) {
  std::vector<SizeSpec> specs = {
      {1'000'000ULL, trials, 8, 80, false},
      {10'000'000ULL, std::min(trials, 3), 3, 20, false},
      {100'000'000ULL, std::min(trials, 2), 1, 3, true},
  };
  if (onlySize > 0) {
    std::vector<SizeSpec> filtered;
    for (const auto& s : specs) {
      if (static_cast<int64_t>(s.size) == onlySize) {
        filtered.push_back(s);
      }
    }
    specs = filtered;
  }

  const auto pats = patterns();
  const auto methods = makeMethods();

  std::cout << "pattern,size,method,trials,ratio_mean,ratio_stddev,"
               "encode_mb_s_mean,decode_mb_s_mean\n";
  for (const auto& spec : specs) {
    const uint64_t rawBytes = spec.size * sizeof(T);
    for (const auto& pat : pats) {
      if (spec.subset && !inBigSubset(pat.name)) {
        continue;
      }
      // Per-method accumulators across trials.
      std::vector<std::vector<double>> ratios(methods.size());
      std::vector<std::vector<double>> encMbs(methods.size());
      std::vector<std::vector<double>> decMbs(methods.size());
      for (int t = 0; t < spec.trials; ++t) {
        const uint64_t seed = 0x9e3779b97f4a7c15ULL * (t + 1) + spec.size;
        // Generate the dataset once per trial and reuse across all methods.
        const Vector<T> data = pat.gen(static_cast<uint32_t>(spec.size), seed);
        for (size_t m = 0; m < methods.size(); ++m) {
          const auto& method = methods[m];
          const Encoded encoded = method.encode(data);
          ratios[m].push_back(
              static_cast<double>(rawBytes) /
              static_cast<double>(encoded.bytes.size()));
          encMbs[m].push_back(timedThroughputMbPerSec(
              [&] { folly::doNotOptimizeAway(method.encode(data)); },
              rawBytes,
              spec.encodeIters));
          if (encoded.decodable) {
            decMbs[m].push_back(timedThroughputMbPerSec(
                [&] { method.decode(encoded.bytes, data.size()); },
                rawBytes,
                spec.decodeIters));
          }
        }
      }
      for (size_t m = 0; m < methods.size(); ++m) {
        const Stat ratio = summarize(ratios[m]);
        const Stat enc = summarize(encMbs[m]);
        const Stat dec = summarize(decMbs[m]);
        std::cout << pat.name << "," << spec.size << "," << methods[m].name
                  << "," << spec.trials << "," << ratio.mean << ","
                  << ratio.stddev << "," << enc.mean << ",";
        if (!decMbs[m].empty()) {
          std::cout << dec.mean;
        }
        std::cout << "\n";
      }
    }
  }
}

// Print, for each pattern, the section layout the tuned SubIntSplit encoder
// chose: the bit range of each section and the encoding selected for it (via
// the encoding's own debugString, which recurses into each section).
void emitLayout() {
  auto& pool = benchmarkPool();
  for (const auto& pat : patterns()) {
    const uint64_t seed = 0x9e3779b97f4a7c15ULL + 1'000'000ULL;
    const Vector<T> data = pat.gen(1'000'000u, seed);
    const Encoded encoded =
        encodeSubIntSplitWith(data, tunedReadFactors(), CompressionType::Zstd);
    const double ratio = static_cast<double>(data.size() * sizeof(T)) /
        static_cast<double>(encoded.bytes.size());
    auto enc = EncodingFactory{}.create(*pool, encoded.bytes, nullFactory());
    std::cout << "===== " << pat.name << " (SubIntSplitTuned ratio " << ratio
              << ") =====\n"
              << enc->debugString(0) << "\n";
  }
}

// ---------------------------------------------------------------------------
// folly microbenchmark mode (single size/seed, subset of methods).
// ---------------------------------------------------------------------------

std::vector<std::pair<std::string, Vector<T>>>& defaultDatasets() {
  static std::vector<std::pair<std::string, Vector<T>>> ds = [] {
    std::vector<std::pair<std::string, Vector<T>>> v;
    for (const auto& p : patterns()) {
      v.emplace_back(p.name, p.gen(kNumElements, 0x1234u));
    }
    return v;
  }();
  return ds;
}

std::vector<Method>& methodsSingleton() {
  static std::vector<Method> m = makeMethods();
  return m;
}

void registerBenchmarks() {
  for (auto& [name, data] : defaultDatasets()) {
    for (auto& method : methodsSingleton()) {
      folly::addBenchmark(
          __FILE__,
          method.name + "_Encode_" + name,
          [&method, &data](unsigned iters) {
            while (iters--) {
              folly::doNotOptimizeAway(method.encode(data));
            }
            return 1;
          });
      auto encoded = std::make_shared<Encoded>(method.encode(data));
      if (!encoded->decodable) {
        continue;
      }
      const uint32_t n = data.size();
      folly::addBenchmark(
          __FILE__,
          method.name + "_Decode_" + name,
          [&method, encoded, n](unsigned iters) {
            while (iters--) {
              method.decode(encoded->bytes, n);
            }
            return 1;
          });
    }
  }
}

} // namespace

int main(int argc, char** argv) {
  folly::Init init(&argc, &argv);
  facebook::velox::memory::MemoryManager::initialize({});

  if (FLAGS_layout) {
    emitLayout();
    return 0;
  }

  if (FLAGS_csv) {
    emitCsv(FLAGS_trials, FLAGS_only_size);
    return 0;
  }

  registerBenchmarks();
  folly::runBenchmarks();
  return 0;
}
