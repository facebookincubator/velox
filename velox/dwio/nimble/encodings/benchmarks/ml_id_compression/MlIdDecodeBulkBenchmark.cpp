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
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <gflags/gflags.h>

#include "velox/dwio/nimble/encodings/benchmarks/ml_id_compression/BenchCommon.h"
#include "velox/dwio/nimble/encodings/benchmarks/ml_id_compression/CachePolicy.h"
#include "velox/dwio/nimble/encodings/benchmarks/ml_id_compression/DriverSweep.h"
#include "velox/dwio/nimble/encodings/benchmarks/ml_id_compression/ElemType.h"
#include "velox/dwio/nimble/encodings/benchmarks/ml_id_compression/MeasureLoop.h"

DEFINE_string(cache_state, "hot", "hot | cold-payload | cold-all");
DEFINE_bool(validate, false, "Round-trip check before measuring");
DEFINE_bool(dry_run, false, "Print sweep plan and exit");

constexpr std::string_view kDriver = "bench_decode_bulk";

namespace facebook::nimble::mlidc {
namespace {

// The whole driver body, templated on the element type. main() picks the
// type from --mlidc_dtype and dispatches here.
template <typename Elem>
int runBenchmark() {
  constexpr size_t kElemSize = sizeof(Elem);

  const uint32_t n = static_cast<uint32_t>(FLAGS_mlidc_rows);
  const size_t iters = static_cast<size_t>(FLAGS_mlidc_iters);
  const uint64_t seed = static_cast<uint64_t>(FLAGS_mlidc_seed);

  CacheState cacheState{};
  if (!parseCacheState(FLAGS_cache_state, cacheState)) {
    std::cerr << "Unknown --cache_state: " << FLAGS_cache_state
              << " (expected hot|cold-payload|cold-all)\n";
    return 1;
  }

  auto contextOrNull =
      makeSweepContext<Elem>(/*withOpenZL=*/true, cacheState, n);
  if (!contextOrNull.has_value()) {
    return 1;
  }
  const auto& context = *contextOrNull;

  std::cout << "bench_decode_bulk: " << context.encoders.size()
            << " encoders x " << context.datasets.size() << " datasets, N=" << n
            << ", iters=" << iters << ", cache=" << cacheStateName(cacheState)
            << "\n  " << context.topology.describe() << "\n\n";

  if (FLAGS_dry_run) {
    std::cout << "Encoders:\n";
    for (const auto& e : context.encoders)
      std::cout << "  " << e.name << " [" << e.family << "]\n";
    std::cout << "\nDatasets:\n";
    for (const auto& d : context.datasets)
      std::cout << "  " << d.name << "\n";
    return 0;
  }

  std::vector<std::string> csvColumns = {
      "driver",        "dtype",       "dataset",       "encoding",
      "family",        "variant",     "is_sequential", "fast_skip",
      "random_access", "N",           "seed",          "cache_state",
      "evict_method",  "evict_ns",    "payload_bytes", "compression_ratio",
      "iterations",    "warmup",      "time_ns",       "time_p90_ns",
      "time_min_ns",   "decode_Meps", "decode_MBps",   "skipped"};

  std::string csvPath = FLAGS_mlidc_output_csv.empty() ? "bench_decode_bulk.csv"
                                                       : FLAGS_mlidc_output_csv;
  CsvResultWriter csv(csvPath, csvColumns);

  if (!FLAGS_mlidc_output_manifest.empty()) {
    writeRunManifest(FLAGS_mlidc_output_manifest);
  }

  std::vector<Elem> sink(n, Elem{});
  int validateFailures = 0;

  MeasureSpec spec;
  spec.iterations = iters;
  spec.warmup = 2;

  for (const auto& ds : context.datasets) {
    std::cout << "== Dataset: " << ds.name << " ==\n";
    auto data = ds.generate(n, seed);
    const size_t rawBytes = static_cast<size_t>(n) * kElemSize;

    for (const auto& enc : context.encoders) {
      auto target = makeTargetOrSkip<Elem>(enc, data, csv, kDriver, ds.name);
      if (target == nullptr) {
        continue;
      }

      const size_t payloadBytes = target->payloadSize();
      const double ratio = rawBytes > 0
          ? static_cast<double>(payloadBytes) / static_cast<double>(rawBytes)
          : 0.0;

      if (FLAGS_validate && enc.variant != "fpe_noindex") {
        std::vector<Elem> check(n);
        target->materializeAll(check.data(), n);
        bool ok = true;
        for (uint32_t i = 0; i < n; ++i) {
          if (check[i] != data[i]) {
            ok = false;
            break;
          }
        }
        if (!ok) {
          std::cerr << "  [VALIDATE FAIL] " << enc.name << " / " << ds.name
                    << "\n";
          ++validateFailures;
          writeSkipRow<Elem>(csv, kDriver, ds.name, enc);
          continue;
        }
      }

      auto cell = makeCellCache<Elem>(
          context.cacheState,
          context.topology,
          *target,
          std::span<std::byte>(
              reinterpret_cast<std::byte*>(sink.data()),
              static_cast<size_t>(n) * kElemSize));

      auto result = measure(spec, cell.controller, cell.targets, [&]() {
        target->materializeAll(sink.data(), n);
      });

      const double timeNs = static_cast<double>(result.time.median_ns);
      const double meps =
          timeNs > 0.0 ? static_cast<double>(n) / timeNs * 1e3 : 0.0;
      const double mbps = timeNs > 0.0
          ? static_cast<double>(n * kElemSize) / timeNs * 1e3
          : 0.0;

      std::cout << "  " << enc.name << ": " << payloadBytes << " B, "
                << std::fixed << std::setprecision(1) << meps << " Melem/s\n";

      csv.beginRow();
      setIdentityColumns<Elem>(csv, kDriver, ds.name, enc);
      csv.set("fast_skip", enc.fastSkip ? int64_t{1} : int64_t{0});
      csv.set("random_access", enc.randomAccess ? int64_t{1} : int64_t{0});
      csv.set("N", static_cast<int64_t>(n));
      csv.set("seed", static_cast<int64_t>(seed));
      setCacheColumns(csv, cell.controller, result);
      setPayloadColumns(csv, payloadBytes, context.rawBytes());
      setMeasureColumns(csv, spec);
      setTimingColumns(csv, result);
      csv.set("decode_Meps", meps);
      csv.set("decode_MBps", mbps);
      csv.set("skipped", int64_t{0});
      csv.endRow();
      csv.flush();
    }
  }

  std::cout << "\nResults written to: " << csvPath << "\n";

  if (validateFailures > 0) {
    std::cerr << validateFailures << " validation failure(s)\n";
    return 2;
  }
  return 0;
}

} // namespace
} // namespace facebook::nimble::mlidc

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  facebook::velox::memory::MemoryManager::initialize({});
  using namespace facebook::nimble::mlidc;
  return dispatchElemType(
      parseElemDataType(FLAGS_mlidc_dtype),
      [&]<typename T>() { return runBenchmark<T>(); });
}

#else

#include <iostream>
int main() {
  std::cerr
      << "bench_decode_bulk requires NIMBLE_ENABLE_EXPERIMENTAL_ENCODINGS\n";
  return 1;
}

#endif
