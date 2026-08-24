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
#include <array>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <gflags/gflags.h>

#include "velox/dwio/nimble/encodings/benchmarks/ml_id_compression/BenchCommon.h"
#include "velox/dwio/nimble/encodings/benchmarks/ml_id_compression/CachePolicy.h"
#include "velox/dwio/nimble/encodings/benchmarks/ml_id_compression/MeasureLoop.h"
#include "velox/dwio/nimble/encodings/benchmarks/ml_id_compression/PointTraceGen.h"

DEFINE_int32(probes, 65536, "Number of point lookups per measurement");
DEFINE_string(cache_state, "hot", "hot | cold-payload | cold-all");
DEFINE_bool(validate, false, "Round-trip check before measuring");
DEFINE_bool(dry_run, false, "Print sweep plan and exit");

namespace facebook::nimble::mlidc {
namespace {

using Elem = int64_t;
constexpr size_t kElemSize = sizeof(Elem);

} // namespace
} // namespace facebook::nimble::mlidc

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  facebook::velox::memory::MemoryManager::initialize({});
  using namespace facebook::nimble::mlidc;

  const uint32_t n = static_cast<uint32_t>(FLAGS_mlidc_rows);
  const size_t iters = static_cast<size_t>(FLAGS_mlidc_iters);
  const uint64_t seed = static_cast<uint64_t>(FLAGS_mlidc_seed);
  const size_t probes = static_cast<size_t>(std::max(1, FLAGS_probes));

  CacheState cacheState{};
  if (!parseCacheState(FLAGS_cache_state, cacheState)) {
    std::cerr << "Unknown --cache_state: " << FLAGS_cache_state
              << " (expected hot|cold-payload|cold-all)\n";
    return 1;
  }

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

  std::cout << "bench_decode_point: " << encoders.size() << " encoders x "
            << datasets.size() << " datasets, N=" << n
            << ", probes=" << probes << ", iters=" << iters
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
      "driver",          "dataset",         "encoding",        "family",
      "variant",         "is_sequential",   "N",               "seed",
      "cache_state",     "evict_method",    "payload_bytes",   "compression_ratio",
      "iterations",      "warmup",          "probes",          "distinct_probes",
      "distinct_fraction", "time_ns",       "time_p90_ns",     "time_min_ns",
      "ns_per_probe",    "Mprobes_ps",      "clock_overhead_ns", "emulated_point_read",
      "skipped"};
  std::string csvPath = FLAGS_mlidc_output_csv.empty()
      ? "bench_decode_point.csv"
      : FLAGS_mlidc_output_csv;
  CsvResultWriter csv(csvPath, csvColumns);
  if (!FLAGS_mlidc_output_manifest.empty())
    writeRunManifest(FLAGS_mlidc_output_manifest);

  const TimingSummary clockOverhead = measureClockOverhead(1000);

  PointTraceParams traceParams;
  traceParams.streamLength = n;
  traceParams.probes = probes;
  traceParams.seed = seed;
  traceParams.ascending = false;
  const PointTrace trace = buildPointTrace(traceParams);

  Elem sink{};
  int validateFailures = 0;
  MeasureSpec spec;
  spec.iterations = iters;
  spec.warmup = 2;

  auto writeSkipRow = [&](const std::string& ds, const std::string& enc) {
    csv.beginRow();
    csv.set("driver", "bench_decode_point");
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
        Elem check{};
        for (size_t idx : trace.indices) {
          target->materializeRange(static_cast<uint32_t>(idx), 1, &check);
          ok = check == data[idx];
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
          reinterpret_cast<std::byte*>(&sink), kElemSize);
      if (bufs.size() > 1)
        targets.codecInternal.assign(bufs.begin() + 1, bufs.end());

      auto result = measure(spec, controller, targets, [&]() {
        for (size_t idx : trace.indices)
          target->materializeRange(static_cast<uint32_t>(idx), 1, &sink);
      });

      const double timeNs = static_cast<double>(result.time.median_ns);
      const double nsPerProbe =
          probes > 0 ? timeNs / static_cast<double>(probes) : 0.0;
      const double mProbesPerSec = timeNs > 0.0
          ? static_cast<double>(probes) / timeNs * 1e3
          : 0.0;

      csv.beginRow();
      csv.set("driver", "bench_decode_point");
      csv.set("dataset", ds.name);
      csv.set("encoding", enc.name);
      csv.set("family", enc.family);
      csv.set("variant", enc.variant);
      csv.set("is_sequential", enc.isSequential ? int64_t{1} : int64_t{0});
      csv.set("N", static_cast<int64_t>(n));
      csv.set("seed", static_cast<int64_t>(seed));
      csv.set("cache_state",
          std::string(cacheStateName(controller.effectivePolicy().state)));
      csv.set("evict_method", std::string(
          evictMethodName(controller.effectivePolicy().method)));
      csv.set("payload_bytes", static_cast<int64_t>(payloadBytes));
      csv.set("compression_ratio", ratio);
      csv.set("iterations", static_cast<int64_t>(iters));
      csv.set("warmup", static_cast<int64_t>(spec.warmup));
      csv.set("probes", static_cast<int64_t>(probes));
      csv.set("distinct_probes", static_cast<int64_t>(trace.distinctIndices));
      csv.set("distinct_fraction", trace.distinctFraction);
      csv.set("time_ns", result.time.median_ns);
      csv.set("time_p90_ns", result.time.p90_ns);
      csv.set("time_min_ns", result.time.min_ns);
      csv.set("ns_per_probe", nsPerProbe);
      csv.set("Mprobes_ps", mProbesPerSec);
      csv.set("clock_overhead_ns", clockOverhead.median_ns);
      csv.set("emulated_point_read", int64_t{1});
      csv.set("skipped", int64_t{0});
      csv.endRow();
      csv.flush();
      std::cout << "  " << enc.name << ": " << payloadBytes << " B, "
                << nsPerProbe << " ns/probe\n";
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
      << "bench_decode_point requires NIMBLE_ENABLE_EXPERIMENTAL_ENCODINGS\n";
  return 1;
}

#endif
