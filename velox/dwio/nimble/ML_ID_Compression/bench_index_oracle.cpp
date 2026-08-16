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

// bench_index_oracle: sweeps the 4 FrequencyPartitionEncoding index types
// (NoIndex, PerTierBitmaps, TierTagArray, EliasFano) across datasets and
// reports payload bytes, bulk decode throughput, point-lookup latency, the
// Pareto frontier over (bytes, point_ns), and a scalarised oracle
// J = bytes + lambda * point_ns for a log-spaced lambda sweep.

#ifdef NIMBLE_ENABLE_EXPERIMENTAL_ENCODINGS

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include <gflags/gflags.h>

#include "velox/dwio/nimble/ML_ID_Compression/BenchCommon.h"
#include "velox/dwio/nimble/ML_ID_Compression/CachePolicy.h"
#include "velox/dwio/nimble/ML_ID_Compression/MeasureLoop.h"
#include "velox/dwio/nimble/ML_ID_Compression/PointTraceGen.h"

#include "velox/dwio/nimble/encodings/FrequencyPartitionEncoding.h"

DEFINE_int32(probes, 16384, "Number of point lookups per measurement");
DEFINE_string(cache_state, "hot", "hot | cold-payload | cold-all");
DEFINE_bool(validate, false, "Round-trip check before measuring");
DEFINE_bool(dry_run, false, "Print sweep plan and exit");

namespace facebook::nimble::mlidc {
namespace {

using Elem = int64_t;
constexpr size_t kElemSize = sizeof(Elem);
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

struct IndexTypeEntry {
  std::string name;
  uint8_t idx;
  bool randomAccess;
};

std::vector<IndexTypeEntry> indexTypes = {
    {"NoIndex", 0, false},
    {"PerTierBitmaps", 1, true},
    {"TierTagArray", 2, true},
    {"EliasFano", 3, true},
};

std::vector<double> buildLambdaSweep() {
  std::vector<double> lambdas;
  for (int e = -6; e <= 2; ++e) {
    lambdas.push_back(std::pow(10.0, e));
  }
  return lambdas;
}

// One measured cell: (dataset, index_type) pair.
struct Cell {
  std::string indexTypeName;
  uint8_t idx{};
  bool randomAccess{};
  bool viable{};
  bool skipped{};
  size_t payloadBytes{};
  double bulkNs{kNaN};
  double pointNs{kNaN}; // ns per probe; NaN if not viable
  bool onFrontier{false};
};

// A point is Pareto-optimal (over viable cells) when no other viable cell has
// bytes <= this and pointNs <= this, with at least one strict inequality.
void computeParetoFrontier(std::vector<Cell>& cells) {
  for (auto& c : cells) {
    c.onFrontier = false;
    if (!c.viable) {
      continue;
    }
    bool dominated = false;
    for (const auto& other : cells) {
      if (!other.viable || &other == &c) {
        continue;
      }
      const bool notWorseBytes = other.payloadBytes <= c.payloadBytes;
      const bool notWorseLatency = other.pointNs <= c.pointNs;
      const bool strictlyBetter =
          (other.payloadBytes < c.payloadBytes) || (other.pointNs < c.pointNs);
      if (notWorseBytes && notWorseLatency && strictlyBetter) {
        dominated = true;
        break;
      }
    }
    c.onFrontier = !dominated;
  }
}

} // namespace
} // namespace facebook::nimble::mlidc

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  facebook::velox::memory::MemoryManager::initialize({});
  using namespace facebook::nimble::mlidc;
  using facebook::nimble::FrequencyPartitionEncoding;

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

  auto datasets = defaultInt64Datasets<Elem>();
  const std::vector<double> lambdas = buildLambdaSweep();
  const CacheTopology topo = CacheTopology::detect();

  CachePolicy policy;
  policy.state = cacheState;
  try {
    CacheController(policy, topo);
  } catch (const std::exception& e) {
    std::cerr << "ERROR: " << e.what() << "\n";
    return 1;
  }

  std::cout << "bench_index_oracle: " << indexTypes.size()
             << " index types x " << datasets.size() << " datasets, N=" << n
             << ", probes=" << probes << ", iters=" << iters
             << ", cache=" << cacheStateName(cacheState) << "\n  "
             << topo.describe() << "\n\n";

  if (FLAGS_dry_run) {
    std::cout << "Index types:\n";
    for (const auto& it : indexTypes) {
      std::cout << "  " << it.name << " (idx=" << static_cast<int>(it.idx)
                 << ", random_access=" << it.randomAccess << ")\n";
    }
    std::cout << "\nDatasets:\n";
    for (const auto& d : datasets) {
      std::cout << "  " << d.name << "\n";
    }
    std::cout << "\nLambdas:\n";
    for (double l : lambdas) {
      std::cout << "  " << l << "\n";
    }
    return 0;
  }

  std::vector<std::string> csvColumns = {
      "driver",         "dataset",           "index_type",   "N",
      "seed",           "cache_state",       "payload_bytes", "random_access",
      "viable",         "bulk_ns",           "point_ns",     "probes",
      "on_pareto_frontier", "lambda_ns_per_byte", "objective_J", "oracle_pick",
      "regret_bytes",   "regret_ns",         "skipped"};
  std::string csvPath = FLAGS_mlidc_output_csv.empty()
      ? "bench_index_oracle.csv"
      : FLAGS_mlidc_output_csv;
  CsvResultWriter csv(csvPath, csvColumns);
  if (!FLAGS_mlidc_output_manifest.empty()) {
    writeRunManifest(FLAGS_mlidc_output_manifest);
  }

  Elem sink{};
  std::vector<Elem> bulkSink(n, Elem{});
  int validateFailures = 0;

  MeasureSpec spec;
  spec.iterations = iters;
  spec.warmup = 2;

  PointTraceParams traceParams;
  traceParams.streamLength = n;
  traceParams.probes = probes;
  traceParams.seed = seed;
  traceParams.ascending = false;
  const PointTrace trace = buildPointTrace(traceParams);

  for (const auto& ds : datasets) {
    std::cout << "== Dataset: " << ds.name << " ==\n";
    auto data = ds.generate(n, seed);

    std::vector<Cell> cells;
    cells.reserve(indexTypes.size());

    for (const auto& it : indexTypes) {
      Cell cell;
      cell.indexTypeName = it.name;
      cell.idx = it.idx;
      cell.randomAccess = it.randomAccess;

      std::unique_ptr<NimbleBenchTargetBase<Elem>> target;
      bool skipped = false;
      try {
        auto impl = std::make_unique<
            NimbleBenchTargetImpl<FrequencyPartitionEncoding<Elem>>>();
        facebook::nimble::Encoding::Options o;
        o.frequencyPartitionIndex = it.idx;
        impl->target.encode(data, o);
        target = std::unique_ptr<NimbleBenchTargetBase<Elem>>(std::move(impl));
      } catch (const std::exception& ex) {
        std::cerr << "  [SKIP] " << it.name << ": " << ex.what() << "\n";
        skipped = true;
      }

      if (skipped) {
        cell.skipped = true;
        cells.push_back(cell);
        csv.beginRow();
        csv.set("driver", "bench_index_oracle");
        csv.set("dataset", ds.name);
        csv.set("index_type", it.name);
        csv.set("skipped", int64_t{1});
        csv.endRow();
        continue;
      }

      cell.payloadBytes = target->payloadSize();

      // Validation: RA index types only. NoIndex cannot be validated (no
      // original-order recovery).
      if (FLAGS_validate && it.idx >= 1) {
        bool ok = true;
        Elem check{};
        for (size_t idx : trace.indices) {
          target->materializeRange(static_cast<uint32_t>(idx), 1, &check);
          if (check != data[idx]) {
            ok = false;
            break;
          }
        }
        if (!ok) {
          std::cerr << "  [VALIDATE FAIL] " << it.name << " / " << ds.name
                     << "\n";
          ++validateFailures;
          cell.skipped = true;
          cells.push_back(cell);
          csv.beginRow();
          csv.set("driver", "bench_index_oracle");
          csv.set("dataset", ds.name);
          csv.set("index_type", it.name);
          csv.set("skipped", int64_t{1});
          csv.endRow();
          continue;
        }
      }

      // Bulk decode measurement.
      {
        CachePolicy cellPolicy;
        cellPolicy.state = cacheState;
        CacheController controller(cellPolicy, topo);
        auto bufs = target->internalBuffers();
        EvictionTargets targets;
        if (!bufs.empty()) {
          targets.payload = bufs[0];
        }
        targets.sink = std::span<std::byte>(
            reinterpret_cast<std::byte*>(bulkSink.data()),
            static_cast<size_t>(n) * kElemSize);
        if (bufs.size() > 1) {
          targets.codecInternal.assign(bufs.begin() + 1, bufs.end());
        }
        auto result = measure(spec, controller, targets, [&]() {
          target->materializeAll(bulkSink.data(), n);
        });
        cell.bulkNs = static_cast<double>(result.time.median_ns);
      }

      // Point-lookup measurement: only for RA index types.
      cell.viable = it.randomAccess;
      if (cell.viable) {
        CachePolicy cellPolicy;
        cellPolicy.state = cacheState;
        CacheController controller(cellPolicy, topo);
        auto bufs = target->internalBuffers();
        EvictionTargets targets;
        if (!bufs.empty()) {
          targets.payload = bufs[0];
        }
        targets.sink = std::span<std::byte>(
            reinterpret_cast<std::byte*>(&sink), kElemSize);
        if (bufs.size() > 1) {
          targets.codecInternal.assign(bufs.begin() + 1, bufs.end());
        }
        auto result = measure(spec, controller, targets, [&]() {
          for (size_t idx : trace.indices) {
            target->materializeRange(static_cast<uint32_t>(idx), 1, &sink);
          }
        });
        const double timeNs = static_cast<double>(result.time.median_ns);
        cell.pointNs = probes > 0 ? timeNs / static_cast<double>(probes) : 0.0;
      } else {
        cell.pointNs = kNaN;
      }

      cells.push_back(cell);
      std::cout << "  " << it.name << ": " << cell.payloadBytes << " B, "
                 << (cell.viable ? std::to_string(cell.pointNs) : "n/a")
                 << " ns/probe\n";
    }

    computeParetoFrontier(cells);

    // Compute best bytes / best latency among viable cells (for regret).
    size_t bestBytes = std::numeric_limits<size_t>::max();
    double bestPointNs = std::numeric_limits<double>::max();
    for (const auto& c : cells) {
      if (c.skipped) {
        continue;
      }
      bestBytes = std::min(bestBytes, c.payloadBytes);
      if (c.viable) {
        bestPointNs = std::min(bestPointNs, c.pointNs);
      }
    }

    for (const auto& c : cells) {
      if (c.skipped) {
        continue;
      }

      const int64_t regretBytes = bestBytes != std::numeric_limits<size_t>::max()
          ? static_cast<int64_t>(c.payloadBytes) - static_cast<int64_t>(bestBytes)
          : 0;
      const double regretNs = (c.viable && bestPointNs != std::numeric_limits<double>::max())
          ? c.pointNs - bestPointNs
          : kNaN;

      for (double lambda : lambdas) {
        // Oracle pick over the whole cell set for this lambda: argmin J
        // among viable cells (point lookups require RA); if no viable cell
        // exists, the byte-only baseline (NoIndex) is the pick.
        std::string oraclePick;
        double bestJ = std::numeric_limits<double>::max();
        bool anyViable = false;
        for (const auto& other : cells) {
          if (other.skipped || !other.viable) {
            continue;
          }
          anyViable = true;
          const double j = static_cast<double>(other.payloadBytes) +
              lambda * other.pointNs;
          if (j < bestJ) {
            bestJ = j;
            oraclePick = other.indexTypeName;
          }
        }
        if (!anyViable) {
          // Fall back to smallest payload (e.g. NoIndex-only viable set).
          size_t minB = std::numeric_limits<size_t>::max();
          for (const auto& other : cells) {
            if (other.skipped) {
              continue;
            }
            if (other.payloadBytes < minB) {
              minB = other.payloadBytes;
              oraclePick = other.indexTypeName;
            }
          }
        }

        const double j = c.viable
            ? static_cast<double>(c.payloadBytes) + lambda * c.pointNs
            : static_cast<double>(c.payloadBytes);

        csv.beginRow();
        csv.set("driver", "bench_index_oracle");
        csv.set("dataset", ds.name);
        csv.set("index_type", c.indexTypeName);
        csv.set("N", static_cast<int64_t>(n));
        csv.set("seed", static_cast<int64_t>(seed));
        csv.set("cache_state", std::string(cacheStateName(cacheState)));
        csv.set("payload_bytes", static_cast<int64_t>(c.payloadBytes));
        csv.set("random_access", c.randomAccess ? int64_t{1} : int64_t{0});
        csv.set("viable", c.viable ? int64_t{1} : int64_t{0});
        csv.set("bulk_ns", c.bulkNs);
        csv.set("point_ns", c.pointNs);
        csv.set("probes", static_cast<int64_t>(probes));
        csv.set("on_pareto_frontier", c.onFrontier ? int64_t{1} : int64_t{0});
        csv.set("lambda_ns_per_byte", lambda);
        csv.set("objective_J", j);
        csv.set("oracle_pick", oraclePick);
        csv.set("regret_bytes", regretBytes);
        csv.set("regret_ns", regretNs);
        csv.set("skipped", int64_t{0});
        csv.endRow();
      }
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
      << "bench_index_oracle requires NIMBLE_ENABLE_EXPERIMENTAL_ENCODINGS\n";
  return 1;
}

#endif
