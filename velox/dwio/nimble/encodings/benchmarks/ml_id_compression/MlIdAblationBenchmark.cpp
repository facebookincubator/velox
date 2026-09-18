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

// bench_ablation: progressive encoding-set restriction for SIS sections.
//
// SubIntSplit's random-access property is the MIN over its sections. One
// sequential-access section encoding costs the WHOLE encoding its RA property.
// This driver varies the allowed encoding set along a ladder and reports, per
// rung, both the estimated compression and the access-class consequences.

#ifdef NIMBLE_ENABLE_EXPERIMENTAL_ENCODINGS

#include <cstdint>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <gflags/gflags.h>

#include "velox/dwio/nimble/encodings/SubIntSplitSampler.h"
#include "velox/dwio/nimble/encodings/benchmarks/ml_id_compression/AblationPolicy.h"
#include "velox/dwio/nimble/encodings/benchmarks/ml_id_compression/BenchCommon.h"
#include "velox/dwio/nimble/encodings/benchmarks/ml_id_compression/ElemType.h"

DEFINE_bool(dry_run, false, "Print sweep plan and exit");

namespace facebook::nimble::mlidc {
namespace {

using namespace facebook::nimble::detail::subintsplit;

std::string encodingTypeName(EncodingType t) {
  switch (t) {
    case EncodingType::Trivial:
      return "Trivial";
    case EncodingType::FixedBitWidth:
      return "FixedBitWidth";
    case EncodingType::Constant:
      return "Constant";
    case EncodingType::Dictionary:
      return "Dictionary";
    case EncodingType::MainlyConstant:
      return "MainlyConstant";
    case EncodingType::RLE:
      return "RLE";
    case EncodingType::Varint:
      return "Varint";
    default:
      return "Other";
  }
}

std::string formatPlan(const std::vector<SegmentPlan>& segments) {
  std::ostringstream oss;
  for (size_t i = 0; i < segments.size(); ++i) {
    if (i > 0)
      oss << ";";
    oss << segments[i].bitStart << "-" << segments[i].bitEnd << ":"
        << encodingTypeName(segments[i].encoding);
  }
  return oss.str();
}

} // namespace
} // namespace facebook::nimble::mlidc

namespace facebook::nimble::mlidc {
namespace {

// The whole driver body, templated on the element type. main() picks the
// type from --mlidc_dtype and dispatches here.
template <typename Elem>
int runBenchmark() {
  using namespace facebook::nimble::detail::subintsplit;

  const uint32_t n = static_cast<uint32_t>(FLAGS_mlidc_rows);
  const uint64_t seed = static_cast<uint64_t>(FLAGS_mlidc_seed);

  auto datasets = defaultDatasets<Elem>();
  auto ladder = combinedLadder();

  std::cout << "bench_ablation: " << ladder.size() << " rungs x "
            << datasets.size() << " datasets, N=" << n << "\n\n";

  if (FLAGS_dry_run) {
    std::cout << "Ladder:\n";
    for (size_t i = 0; i < ladder.size(); ++i) {
      std::cout << "  [" << i << "] " << ladder[i].name << " ("
                << ladder[i].allowed.size() << " encodings, worst="
                << accessClassName(ladder[i].worstAllowed) << ")\n";
    }
    std::cout << "\nDatasets:\n";
    for (const auto& d : datasets)
      std::cout << "  " << d.name << "\n";
    return 0;
  }

  std::vector<std::string> csvColumns = {
      "driver",
      "dtype",
      "dataset",
      "N",
      "seed",
      "sample_size",
      "rung_name",
      "rung_index",
      "segment_count",
      "total_est_bits",
      "bits_per_elem_est",
      "all_sections_ra",
      "worst_access_class",
      "cost_model_consistent",
      "segment_plan",
      "skipped"};
  std::string csvPath = FLAGS_mlidc_output_csv.empty() ? "bench_ablation.csv"
                                                       : FLAGS_mlidc_output_csv;
  CsvResultWriter csv(csvPath, csvColumns);
  if (!FLAGS_mlidc_output_manifest.empty()) {
    writeRunManifest(FLAGS_mlidc_output_manifest);
  }

  SamplerConfig samplerCfg = defaultSamplerConfig();
  SelectorConfig selectorCfg = defaultSelectorConfig();

  for (const auto& ds : datasets) {
    std::cout << "== Dataset: " << ds.name << " ==\n";
    auto data = ds.generate(n, seed);

    // Sample the physical bit pattern, not the logical value. sampleIntoU64
    // names its parameter `physicalType` (SubIntSplitSampler.h:57), and that is
    // what SubIntSplit itself splits into bit ranges. Passing the logical type
    // is bit-preserving for the integer types but would be a *value*
    // conversion for float and double, which would analyse the wrong bits.
    // Same view the encodings take of their input (TestUtils.h:298-301).
    using Phys = typename TypeTraits<Elem>::physicalType;
    auto physical =
        std::span<const Phys>(reinterpret_cast<const Phys*>(data.data()), n);

    std::vector<uint64_t> samples;
    sampleIntoU64(physical, samples, samplerCfg);

    if (samples.empty()) {
      std::cerr << "  [SKIP] empty sample\n";
      continue;
    }

    for (size_t ri = 0; ri < ladder.size(); ++ri) {
      const auto& rung = ladder[ri];

      SelectorResult result;
      bool skipped = false;
      try {
        result = detail_ablation::selectSplitsRestricted(
            samples, 64, n, selectorCfg, rung.allowed);
      } catch (const std::exception& ex) {
        std::cerr << "  [SKIP] " << rung.name << ": " << ex.what() << "\n";
        skipped = true;
      }

      if (skipped) {
        csv.beginRow();
        csv.set("driver", "bench_ablation");
        csv.set("dtype", elemTypeName<Elem>());
        csv.set("dataset", ds.name);
        csv.set("rung_name", rung.name);
        csv.set("rung_index", static_cast<int64_t>(ri));
        csv.set("skipped", int64_t{1});
        csv.endRow();
        continue;
      }

      const double bpe = n > 0 ? result.totalCost / static_cast<double>(n) : 0;

      AccessClass worst = AccessClass::PureRA;
      bool allRA = true;
      for (const auto& seg : result.segments) {
        AccessClass ac = accessClassOf(seg.encoding);
        if (ac > worst)
          worst = ac;
        if (ac != AccessClass::PureRA)
          allRA = false;
      }

      std::cout << "  " << rung.name << ": " << result.segments.size()
                << " segments, " << std::fixed << std::setprecision(2) << bpe
                << " bpe, allRA=" << allRA << "\n";

      csv.beginRow();
      csv.set("driver", "bench_ablation");
      csv.set("dtype", elemTypeName<Elem>());
      csv.set("dataset", ds.name);
      csv.set("N", static_cast<int64_t>(n));
      csv.set("seed", static_cast<int64_t>(seed));
      csv.set("sample_size", static_cast<int64_t>(samples.size()));
      csv.set("rung_name", rung.name);
      csv.set("rung_index", static_cast<int64_t>(ri));
      csv.set("segment_count", static_cast<int64_t>(result.segments.size()));
      csv.set("total_est_bits", result.totalCost);
      csv.set("bits_per_elem_est", bpe);
      csv.set("all_sections_ra", allRA ? int64_t{1} : int64_t{0});
      csv.set("worst_access_class", std::string(accessClassName(worst)));
      csv.set(
          "cost_model_consistent",
          rung.costModelConsistent ? int64_t{1} : int64_t{0});
      csv.set("segment_plan", formatPlan(result.segments));
      csv.set("skipped", int64_t{0});
      csv.endRow();
    }
    csv.flush();
  }

  std::cout << "\nResults written to: " << csvPath << "\n";
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
  std::cerr << "bench_ablation requires NIMBLE_ENABLE_EXPERIMENTAL_ENCODINGS\n";
  return 1;
}

#endif
