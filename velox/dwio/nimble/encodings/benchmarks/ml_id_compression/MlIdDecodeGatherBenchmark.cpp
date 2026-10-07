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
#include <iostream>
#include <string>
#include <vector>

#include <gflags/gflags.h>

#include "velox/dwio/nimble/encodings/benchmarks/ml_id_compression/Axes.h"
#include "velox/dwio/nimble/encodings/benchmarks/ml_id_compression/BenchCommon.h"
#include "velox/dwio/nimble/encodings/benchmarks/ml_id_compression/CachePolicy.h"
#include "velox/dwio/nimble/encodings/benchmarks/ml_id_compression/DriverSweep.h"
#include "velox/dwio/nimble/encodings/benchmarks/ml_id_compression/ElemType.h"
#include "velox/dwio/nimble/encodings/benchmarks/ml_id_compression/GatherTraceGen.h"
#include "velox/dwio/nimble/encodings/benchmarks/ml_id_compression/MeasureLoop.h"

DEFINE_int32(selectivity_steps, 8, "Steps in the selectivity axis");
DEFINE_int32(run_length_steps, 6, "Steps in the run-length axis");
DEFINE_string(cache_state, "hot", "hot | cold-payload | cold-all");
DEFINE_bool(validate, false, "Round-trip check before measuring");
DEFINE_bool(dry_run, false, "Print sweep plan and exit");

namespace facebook::nimble::mlidc {
namespace {

std::vector<std::pair<uint32_t, uint32_t>> toRanges(const GatherTrace& t) {
  std::vector<std::pair<uint32_t, uint32_t>> ranges;
  ranges.reserve(t.ranges.size());
  for (const auto& r : t.ranges)
    ranges.emplace_back(
        static_cast<uint32_t>(r.begin), static_cast<uint32_t>(r.size()));
  return ranges;
}

} // namespace
} // namespace facebook::nimble::mlidc

constexpr std::string_view kDriver = "bench_decode_gather";

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

  const auto selectivityAxis =
      linSpaced(0.05, 1.0, static_cast<size_t>(FLAGS_selectivity_steps));
  const auto runLengthAxis = logSpaced(
      1,
      std::max<size_t>(1, n / 4),
      static_cast<size_t>(FLAGS_run_length_steps));

  auto contextOrNull =
      makeSweepContext<Elem>(/*withOpenZL=*/true, cacheState, n);
  if (!contextOrNull.has_value()) {
    return 1;
  }
  const auto& context = *contextOrNull;

  std::cout << "bench_decode_gather: " << context.encoders.size()
            << " encoders x " << context.datasets.size() << " datasets, N=" << n
            << ", selectivity_steps=" << selectivityAxis.size()
            << ", run_length_steps=" << runLengthAxis.size()
            << ", iters=" << iters
            << ", cache=" << cacheStateName(context.cacheState) << "\n  "
            << context.topology.describe() << "\n\n";

  if (FLAGS_dry_run) {
    std::cout << "Encoders:\n";
    for (const auto& e : context.encoders)
      std::cout << "  " << e.name << "\n";
    std::cout << "Datasets:\n";
    for (const auto& d : context.datasets)
      std::cout << "  " << d.name << "\n";
    return 0;
  }

  std::vector<std::string> csvColumns = {
      "driver",
      "dtype",
      "dataset",
      "encoding",
      "family",
      "variant",
      "is_sequential",
      "N",
      "seed",
      "cache_state",
      "evict_method",
      "evict_ns",
      "payload_bytes",
      "compression_ratio",
      "iterations",
      "warmup",
      "selectivity",
      "run_length",
      "selectivity_achieved",
      "range_count",
      "selected_rows",
      "gap_model",
      "time_ns",
      "time_p90_ns",
      "time_min_ns",
      "gather_Meps",
      "skipped"};
  std::string csvPath = FLAGS_mlidc_output_csv.empty()
      ? "bench_decode_gather.csv"
      : FLAGS_mlidc_output_csv;
  CsvResultWriter csv(csvPath, csvColumns);
  if (!FLAGS_mlidc_output_manifest.empty())
    writeRunManifest(FLAGS_mlidc_output_manifest);

  std::vector<Elem> sink(n, Elem{});
  int validateFailures = 0;
  MeasureSpec spec;
  spec.iterations = iters;
  spec.warmup = 2;

  for (const auto& ds : context.datasets) {
    auto data = ds.generate(n, seed);

    for (const auto& enc : context.encoders) {
      auto target = makeTargetOrSkip<Elem>(enc, data, csv, kDriver, ds.name);
      if (target == nullptr) {
        continue;
      }

      // Block codecs decompress everything per read; cap their iterations so
      // the sweep finishes in reasonable time.
      const MeasureSpec encSpec = specFor(
          spec,
          enc.wholePayloadCodec,
          static_cast<size_t>(FLAGS_mlidc_block_codec_iters));

      const size_t payloadBytes = target->payloadSize();

      if (FLAGS_validate && enc.variant != "fpe_noindex") {
        GatherAccessParams vp{
            .start = 0,
            .span = n,
            .selectivity = 0.3,
            .runLength = 4,
            .gapModel = GapModel::UniformDeterministic,
            .seed = seed};
        auto trace = buildGatherTrace(n, vp);
        auto ranges = toRanges(trace);
        std::vector<Elem> check(trace.selectedRows);
        target->skipThenMaterialize(ranges, check.data());
        bool ok = true;
        size_t idx = 0;
        for (const auto& r : trace.ranges)
          for (size_t i = r.begin; i < r.end; ++i, ++idx)
            if (check[idx] != data[i])
              ok = false;
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

      for (double sigma : selectivityAxis) {
        for (size_t rl : runLengthAxis) {
          GatherAccessParams p{
              .start = 0,
              .span = n,
              .selectivity = sigma,
              .runLength = rl,
              .gapModel = GapModel::UniformDeterministic,
              .seed = seed};
          auto trace = buildGatherTrace(n, p);
          if (trace.selectedRows == 0)
            continue;
          auto ranges = toRanges(trace);

          auto result = measure(encSpec, cell.controller, cell.targets, [&]() {
            target->skipThenMaterialize(ranges, sink.data());
          });

          const double timeNs = static_cast<double>(result.time.median_ns);
          const double meps = timeNs > 0.0
              ? static_cast<double>(trace.selectedRows) / timeNs * 1e3
              : 0.0;

          csv.beginRow();
          setIdentityColumns<Elem>(csv, kDriver, ds.name, enc);
          csv.set("N", static_cast<int64_t>(n));
          csv.set("seed", static_cast<int64_t>(seed));
          setCacheColumns(csv, cell.controller, result);
          setPayloadColumns(csv, payloadBytes, context.rawBytes());
          setMeasureColumns(csv, encSpec);
          csv.set("selectivity", sigma);
          csv.set("run_length", static_cast<int64_t>(rl));
          csv.set("selectivity_achieved", trace.selectivityAchieved);
          csv.set("range_count", static_cast<int64_t>(trace.rangeCount));
          csv.set("selected_rows", static_cast<int64_t>(trace.selectedRows));
          csv.set("gap_model", std::string(gapModelName(p.gapModel)));
          setTimingColumns(csv, result);
          csv.set("gather_Meps", meps);
          csv.set("skipped", int64_t{0});
          csv.endRow();
        }
      }
      csv.flush();
    }
  }

  std::cout << "Results written to: " << csvPath << "\n";
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
      << "bench_decode_gather requires NIMBLE_ENABLE_EXPERIMENTAL_ENCODINGS\n";
  return 1;
}

#endif
