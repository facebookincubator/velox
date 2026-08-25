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

#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <gflags/gflags.h>

#include "velox/dwio/nimble/encodings/benchmarks/ml_id_compression/BenchCommon.h"
#include "velox/dwio/nimble/encodings/benchmarks/ml_id_compression/DriverSweep.h"

DEFINE_bool(validate, false, "Round-trip check after each encode");
DEFINE_bool(dry_run, false, "Print sweep plan and exit");

namespace facebook::nimble::mlidc {
namespace {

using Elem = int64_t;
constexpr size_t kElemSize = sizeof(Elem);

} // namespace
} // namespace facebook::nimble::mlidc

constexpr std::string_view kDriver = "bench_compression";

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  facebook::velox::memory::MemoryManager::initialize({});
  using namespace facebook::nimble::mlidc;

  const uint32_t n = static_cast<uint32_t>(FLAGS_mlidc_rows);
  const uint64_t seed = static_cast<uint64_t>(FLAGS_mlidc_seed);

  // No cache sweep here, so the state is fixed at hot.
  auto contextOrNull =
      makeSweepContext<Elem>(/*withOpenZL=*/true, CacheState::Hot, n);
  if (!contextOrNull.has_value()) {
    return 1;
  }
  const auto& context = *contextOrNull;

  std::cout << "bench_compression: " << context.encoders.size() << " encoders x "
            << context.datasets.size() << " datasets, N=" << n << "\n\n";

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
      "driver",     "dataset",    "encoding",   "family",
      "variant",    "is_sequential", "N",        "seed",
      "payload_bytes", "raw_bytes", "compression_ratio",
      "bits_per_elem", "skipped"};
  std::string csvPath = FLAGS_mlidc_output_csv.empty()
      ? "bench_compression.csv"
      : FLAGS_mlidc_output_csv;
  CsvResultWriter csv(csvPath, csvColumns);
  if (!FLAGS_mlidc_output_manifest.empty()) {
    writeRunManifest(FLAGS_mlidc_output_manifest);
  }

  int validateFailures = 0;

  for (const auto& ds : context.datasets) {
    std::cout << "== Dataset: " << ds.name << " ==\n";
    auto data = ds.generate(n, seed);
    const size_t rawBytes = static_cast<size_t>(n) * kElemSize;

    for (const auto& enc : context.encoders) {
      auto target = makeTargetOrSkip<Elem>(enc, data, csv, kDriver, ds.name);
      if (target == nullptr) {
        continue;
      }

      if (FLAGS_mlidc_dump_encoding) {
        auto tree = target->describe();
        if (!tree.empty()) {
          std::cout << "  --- " << enc.name << " encoding tree ---\n"
                    << tree << "\n";
        }
      }

      const size_t payloadBytes = target->payloadSize();
      const double bpe = n > 0
          ? static_cast<double>(payloadBytes) * 8.0 / static_cast<double>(n)
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
          csv.beginRow();
          csv.set("driver", "bench_compression");
          csv.set("dataset", ds.name);
          csv.set("encoding", enc.name);
          csv.set("skipped", int64_t{1});
          csv.endRow();
          continue;
        }
      }

      std::cout << "  " << enc.name << ": " << payloadBytes << " B, "
                << std::fixed << std::setprecision(2) << bpe << " bpe\n";

      csv.beginRow();
      setIdentityColumns<Elem>(csv, kDriver, ds.name, enc);
      csv.set("N", static_cast<int64_t>(n));
      csv.set("seed", static_cast<int64_t>(seed));
      setPayloadColumns(csv, payloadBytes, context.rawBytes());
      csv.set("raw_bytes", static_cast<int64_t>(rawBytes));
      csv.set("bits_per_elem", bpe);
      csv.set("skipped", int64_t{0});
      csv.endRow();
    }
    csv.flush();
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
      << "bench_compression requires NIMBLE_ENABLE_EXPERIMENTAL_ENCODINGS\n";
  return 1;
}

#endif
