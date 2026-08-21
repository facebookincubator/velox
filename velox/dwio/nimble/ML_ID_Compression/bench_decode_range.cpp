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

#ifdef NIMBLE_ENABLE_EXPERIMENTAL_ENCODINGS

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <gflags/gflags.h>

#include "velox/dwio/nimble/ML_ID_Compression/BenchCommon.h"
#include "velox/dwio/nimble/ML_ID_Compression/CachePolicy.h"
#include "velox/dwio/nimble/ML_ID_Compression/MeasureLoop.h"

DEFINE_int32(grid, 16, "Grid resolution per axis; ~grid^2/2 cells in triangle");
DEFINE_string(cache_state, "hot", "hot | cold-payload | cold-all");
DEFINE_bool(validate, false, "Round-trip check before measuring");
DEFINE_bool(dry_run, false, "Print sweep plan and exit");

namespace facebook::nimble::mlidc {
namespace {

using Elem = int64_t;
constexpr size_t kElemSize = sizeof(Elem);

struct Cell { size_t a{}; size_t b{}; };

Cell resolveCell(double aFrac, double bFrac, size_t n) {
  Cell c;
  c.a = static_cast<size_t>(std::llround(aFrac * static_cast<double>(n)));
  c.b = std::max<size_t>(
      1, static_cast<size_t>(std::llround(bFrac * static_cast<double>(n))));
  if (c.a >= n) c.a = n - 1;
  if (c.a + c.b > n) c.b = n - c.a;
  return c;
}

} // namespace
} // namespace facebook::nimble::mlidc

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  facebook::velox::memory::MemoryManager::initialize({});
  using namespace facebook::nimble::mlidc;

  const uint32_t n = static_cast<uint32_t>(FLAGS_mlidc_rows);
  const size_t iters = static_cast<size_t>(FLAGS_mlidc_iters);
  const uint64_t seed = static_cast<uint64_t>(FLAGS_mlidc_seed);
  const size_t grid = static_cast<size_t>(std::max(1, FLAGS_grid));

  CacheState cacheState{};
  if (!parseCacheState(FLAGS_cache_state, cacheState)) {
    std::cerr << "Unknown --cache_state: " << FLAGS_cache_state
              << " (expected hot|cold-payload|cold-all)\n";
    return 1;
  }

  std::vector<double> aFracs, bFracs;
  for (size_t i = 0; i < grid; ++i)
    aFracs.push_back(static_cast<double>(i) / static_cast<double>(grid));
  for (size_t j = 1; j <= grid; ++j)
    bFracs.push_back(static_cast<double>(j) / static_cast<double>(grid));
  size_t cellCount = 0;
  for (double a : aFracs)
    for (double b : bFracs)
      if (a + b <= 1.0 + 1e-9) ++cellCount;

  auto encoders = buildDefaultEncoders<Elem>();
  auto datasets = defaultInt64Datasets<Elem>();
  const CacheTopology topo = CacheTopology::detect();

  CachePolicy policy;
  policy.state = cacheState;
  try {
    CacheController(policy, topo);
  } catch (const std::exception& e) {
    std::cerr << "ERROR: " << e.what() << "\n";
    return 1;
  }

  std::cout << "bench_decode_range: " << encoders.size() << " encoders x "
            << datasets.size() << " datasets, N=" << n << ", grid=" << grid
            << " (" << cellCount << " cells), iters=" << iters
            << ", cache=" << cacheStateName(cacheState) << "\n  "
            << topo.describe() << "\n\n";

  if (FLAGS_dry_run) {
    std::cout << "Encoders:\n";
    for (const auto& e : encoders)
      std::cout << "  " << e.name << " [" << e.family << "]\n";
    std::cout << "\nDatasets:\n";
    for (const auto& d : datasets) std::cout << "  " << d.name << "\n";
    return 0;
  }

  std::vector<std::string> csvColumns = {
      "driver",      "dataset",     "encoding",       "family",
      "variant",     "is_sequential", "fast_skip",    "random_access",
      "N",           "seed",        "cache_state",    "evict_method",
      "evict_ns",    "payload_bytes", "compression_ratio",
      "iterations",  "warmup",      "contract",       "A_frac",
      "B_frac",      "A",           "B",              "time_ns",
      "time_p90_ns", "time_min_ns", "elem_Meps",      "input_MBps",
      "skipped"};
  std::string csvPath = FLAGS_mlidc_output_csv.empty()
      ? "bench_decode_range.csv"
      : FLAGS_mlidc_output_csv;
  CsvResultWriter csv(csvPath, csvColumns);
  if (!FLAGS_mlidc_output_manifest.empty())
    writeRunManifest(FLAGS_mlidc_output_manifest);

  std::vector<Elem> sink(n, Elem{});
  int validateFailures = 0;
  MeasureSpec spec;
  spec.iterations = iters;
  spec.warmup = 2;

  auto writeSkipRow = [&](const std::string& ds, const std::string& enc) {
    csv.beginRow();
    csv.set("driver", "bench_decode_range");
    csv.set("dataset", ds);
    csv.set("encoding", enc);
    csv.set("skipped", int64_t{1});
    csv.endRow();
  };

  for (const auto& ds : datasets) {
    std::cout << "== Dataset: " << ds.name << " ==\n";
    auto data = ds.generate(n, seed);
    const size_t rawBytes = static_cast<size_t>(n) * kElemSize;

    for (const auto& enc : encoders) {
      facebook::nimble::Encoding::Options opts;
      std::unique_ptr<NimbleBenchTargetBase<Elem>> target;
      bool skipped = false;
      try {
        target = enc.factory(data, opts);
      } catch (const std::exception& ex) {
        std::cerr << "  [SKIP] " << enc.name << ": " << ex.what() << "\n";
        skipped = true;
      }
      if (skipped) { writeSkipRow(ds.name, enc.name); continue; }

      const size_t payloadBytes = target->payloadSize();
      const double ratio = rawBytes > 0
          ? static_cast<double>(payloadBytes) / static_cast<double>(rawBytes)
          : 0.0;

      if (FLAGS_validate && enc.variant != "fpe_noindex") {
        bool ok = true;
        std::vector<Elem> check;
        for (double aFrac : aFracs) {
          Cell c = resolveCell(aFrac, bFracs.front(), n);
          check.assign(c.b, Elem{});
          target->materializeRange(
              static_cast<uint32_t>(c.a), static_cast<uint32_t>(c.b),
              check.data());
          for (size_t i = 0; i < c.b && ok; ++i)
            ok = check[i] == data[c.a + i];
          if (!ok) break;
        }
        if (!ok) {
          std::cerr << "  [VALIDATE FAIL] " << enc.name << " / " << ds.name
                    << "\n";
          ++validateFailures;
          writeSkipRow(ds.name, enc.name);
          continue;
        }
      }

      CachePolicy cellPolicy;
      cellPolicy.state = cacheState;
      CacheController controller(cellPolicy, topo);
      auto bufs = target->internalBuffers();
      EvictionTargets targets;
      if (!bufs.empty()) targets.payload = bufs[0];
      targets.sink = std::span<std::byte>(
          reinterpret_cast<std::byte*>(sink.data()),
          static_cast<size_t>(n) * kElemSize);
      if (bufs.size() > 1)
        targets.codecInternal.assign(bufs.begin() + 1, bufs.end());

      for (double aFrac : aFracs) {
        for (double bFrac : bFracs) {
          if (aFrac + bFrac > 1.0 + 1e-9) continue;
          const Cell c = resolveCell(aFrac, bFrac, n);

          auto result = measure(spec, controller, targets, [&]() {
            target->materializeRange(
                static_cast<uint32_t>(c.a), static_cast<uint32_t>(c.b),
                sink.data());
          });

          const double timeNs = static_cast<double>(result.time.median_ns);
          const double elemMeps =
              timeNs > 0.0 ? static_cast<double>(c.b) / timeNs * 1e3 : 0.0;
          const double inputBytes = enc.isSequential
              ? static_cast<double>(payloadBytes)
              : static_cast<double>(c.b) * static_cast<double>(payloadBytes) /
                  static_cast<double>(n);
          const double inputMBps =
              timeNs > 0.0 ? inputBytes / timeNs * 1e3 : 0.0;

          csv.beginRow();
          csv.set("driver", "bench_decode_range");
          csv.set("dataset", ds.name);
          csv.set("encoding", enc.name);
          csv.set("family", enc.family);
          csv.set("variant", enc.variant);
          csv.set("is_sequential", enc.isSequential ? int64_t{1} : int64_t{0});
          csv.set("fast_skip", enc.fastSkip ? int64_t{1} : int64_t{0});
          csv.set("random_access", enc.randomAccess ? int64_t{1} : int64_t{0});
          csv.set("N", static_cast<int64_t>(n));
          csv.set("seed", static_cast<int64_t>(seed));
          csv.set("cache_state",
              std::string(cacheStateName(controller.effectivePolicy().state)));
          csv.set("evict_method", std::string(
              evictMethodName(controller.effectivePolicy().method)));
          csv.set("evict_ns", result.evict.median_ns);
          csv.set("payload_bytes", static_cast<int64_t>(payloadBytes));
          csv.set("compression_ratio", ratio);
          csv.set("iterations", static_cast<int64_t>(iters));
          csv.set("warmup", static_cast<int64_t>(spec.warmup));
          csv.set("contract", std::string("range_into"));
          csv.set("A_frac", aFrac);
          csv.set("B_frac", bFrac);
          csv.set("A", static_cast<int64_t>(c.a));
          csv.set("B", static_cast<int64_t>(c.b));
          csv.set("time_ns", result.time.median_ns);
          csv.set("time_p90_ns", result.time.p90_ns);
          csv.set("time_min_ns", result.time.min_ns);
          csv.set("elem_Meps", elemMeps);
          csv.set("input_MBps", inputMBps);
          csv.set("skipped", int64_t{0});
          csv.endRow();
        }
      }
      csv.flush();
      std::cout << "  " << enc.name << ": " << payloadBytes << " B, "
                << cellCount << " cells swept\n";
    }
  }

  std::cout << "\nResults written to: " << csvPath << "\n";
  if (validateFailures > 0) {
    std::cerr << validateFailures << " validation failure(s)\n";
    return 2;
  }
  return 0;
}

#else

#include <iostream>
int main() {
  std::cerr
      << "bench_decode_range requires NIMBLE_ENABLE_EXPERIMENTAL_ENCODINGS\n";
  return 1;
}

#endif
