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

#include <array>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <gflags/gflags.h>

#include "velox/dwio/nimble/ML_ID_Compression/BenchCommon.h"
#include "velox/dwio/nimble/ML_ID_Compression/OpenZLBenchTarget.h"
#include "velox/dwio/nimble/ML_ID_Compression/CachePolicy.h"
#include "velox/dwio/nimble/ML_ID_Compression/MeasureLoop.h"

DEFINE_bool(validate, false, "Round-trip check after encoding");
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

  auto encoders = buildDefaultEncoders<Elem>();
  encoders.push_back(buildOpenZLEncoder<Elem>());
  auto datasets = defaultInt64Datasets<Elem>();

  const CacheTopology topo = CacheTopology::detect();

  std::cout << "bench_encode: " << encoders.size() << " encoders x "
            << datasets.size() << " datasets, N=" << n
            << ", iters=" << iters << "\n  " << topo.describe() << "\n\n";

  if (FLAGS_dry_run) {
    std::cout << "Encoders:\n";
    for (const auto& e : encoders)
      std::cout << "  " << e.name << " [" << e.family << "]\n";
    std::cout << "\nDatasets:\n";
    for (const auto& d : datasets)
      std::cout << "  " << d.name << "\n";
    return 0;
  }

  std::vector<std::string> csvColumns = {
      "driver",     "dataset",       "encoding",    "family",
      "variant",    "is_sequential", "N",           "seed",
      "payload_bytes", "compression_ratio", "iterations", "warmup",
      "time_ns",    "time_p90_ns",   "time_min_ns", "encode_Meps",
      "encode_MBps", "skipped"};

  std::string csvPath =
      FLAGS_mlidc_output_csv.empty() ? "bench_encode.csv" : FLAGS_mlidc_output_csv;
  CsvResultWriter csv(csvPath, csvColumns);

  if (!FLAGS_mlidc_output_manifest.empty()) {
    writeRunManifest(FLAGS_mlidc_output_manifest);
  }

  int validateFailures = 0;

  MeasureSpec spec;
  spec.iterations = iters;
  spec.warmup = 2;

  CachePolicy hotPolicy;
  hotPolicy.state = CacheState::Hot;
  EvictionTargets emptyTargets;

  for (const auto& ds : datasets) {
    std::cout << "== Dataset: " << ds.name << " ==\n";
    auto data = ds.generate(n, seed);
    const size_t rawBytes = static_cast<size_t>(n) * kElemSize;

    for (const auto& enc : encoders) {
      facebook::nimble::Encoding::Options opts;
      std::unique_ptr<NimbleBenchTargetBase<Elem>> target;

      CacheController controller(hotPolicy, topo);

      try {
        auto result = measure(spec, controller, emptyTargets, [&]() {
          target = enc.factory(data, opts);
        });

        const size_t payloadBytes = target->payloadSize();
        const double ratio = rawBytes > 0
            ? static_cast<double>(payloadBytes) / static_cast<double>(rawBytes)
            : 0.0;

        if (FLAGS_validate && enc.variant != "fpe_noindex") {
          std::vector<Elem> check(n);
          target->materializeAll(check.data(), n);
          for (uint32_t i = 0; i < n; ++i) {
            if (check[i] != data[i]) {
              throw std::runtime_error("round-trip mismatch");
            }
          }
        }

        const double timeNs = static_cast<double>(result.time.median_ns);
        const double meps =
            timeNs > 0.0 ? static_cast<double>(n) / timeNs * 1e3 : 0.0;
        const double mbps = timeNs > 0.0
            ? static_cast<double>(rawBytes) / timeNs * 1e3
            : 0.0;

        std::cout << "  " << enc.name << ": " << payloadBytes << " B, "
                  << std::fixed << std::setprecision(1) << meps
                  << " Melem/s\n";

        csv.beginRow();
        csv.set("driver", "bench_encode");
        csv.set("dataset", ds.name);
        csv.set("encoding", enc.name);
        csv.set("family", enc.family);
        csv.set("variant", enc.variant);
        csv.set("is_sequential", enc.isSequential ? int64_t{1} : int64_t{0});
        csv.set("N", static_cast<int64_t>(n));
        csv.set("seed", static_cast<int64_t>(seed));
        csv.set("payload_bytes", static_cast<int64_t>(payloadBytes));
        csv.set("compression_ratio", ratio);
        csv.set("iterations", static_cast<int64_t>(iters));
        csv.set("warmup", static_cast<int64_t>(spec.warmup));
        csv.set("time_ns", result.time.median_ns);
        csv.set("time_p90_ns", result.time.p90_ns);
        csv.set("time_min_ns", result.time.min_ns);
        csv.set("encode_Meps", meps);
        csv.set("encode_MBps", mbps);
        csv.set("skipped", int64_t{0});
        csv.endRow();
      } catch (const std::exception& ex) {
        std::cerr << "  [SKIP] " << enc.name << ": " << ex.what() << "\n";
        if (FLAGS_validate) {
          ++validateFailures;
        }
        csv.beginRow();
        csv.set("driver", "bench_encode");
        csv.set("dataset", ds.name);
        csv.set("encoding", enc.name);
        csv.set("skipped", int64_t{1});
        csv.endRow();
      }
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

#else

#include <iostream>
int main() {
  std::cerr << "bench_encode requires NIMBLE_ENABLE_EXPERIMENTAL_ENCODINGS\n";
  return 1;
}

#endif
