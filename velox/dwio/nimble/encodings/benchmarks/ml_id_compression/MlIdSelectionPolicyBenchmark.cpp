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

// bench_selection_policy: measures whether the bit-flip-probability
// top-level policies (see SubIntSplitTopLevelPolicy.h for scope) would
// correctly predict "SubIntSplit wins" if wired into production selection.
//
// For every dataset column this driver: computes the best cost among the
// real, unmodified candidate encodings via
// EncodingSizeEstimation<T>::estimateSize() (mirroring
// ManualEncodingSelectionPolicy::select()'s own comparison, without touching
// it), computes SubIntSplit's real cost via the unconstrained
// estimateSubIntSplitSize() (ground truth for "would SubIntSplit have
// won"), and compares that ground truth against each gate's prediction
// (precision/recall/F1) and each gate's own wall-clock cost.

#ifdef NIMBLE_ENABLE_EXPERIMENTAL_ENCODINGS

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include <gflags/gflags.h>

#include "velox/dwio/nimble/encodings/SubIntSplitEstimator.h"
#include "velox/dwio/nimble/encodings/benchmarks/ml_id_compression/BenchCommon.h"
#include "velox/dwio/nimble/encodings/benchmarks/ml_id_compression/ElemType.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelectionPolicy.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSizeEstimation.h"
#include "velox/dwio/nimble/encodings/selection/Statistics.h"

namespace {
// Single source of truth for the gflag defaults below: mirrors
// TopLevelPolicyConfig's own default member initializers.
constexpr facebook::nimble::detail::subintsplit::TopLevelPolicyConfig
    kDefaultPolicyConfig{};
} // namespace

DEFINE_double(
    mlidsp_variance_threshold,
    kDefaultPolicyConfig.varianceGateThreshold,
    "TopLevelPolicyConfig::varianceGateThreshold override");
DEFINE_double(
    mlidsp_gradient_multiplier,
    kDefaultPolicyConfig.gradientStdDevMultiplier,
    "TopLevelPolicyConfig::gradientStdDevMultiplier override");
DEFINE_int32(
    mlidsp_min_gradient_boundaries,
    kDefaultPolicyConfig.minGradientBoundaries,
    "TopLevelPolicyConfig::minGradientBoundaries override");
DEFINE_double(
    mlidsp_min_gradient_magnitude,
    kDefaultPolicyConfig.minGradientMagnitude,
    "TopLevelPolicyConfig::minGradientMagnitude override");
DEFINE_int32(
    mlidsp_timing_iters,
    5,
    "Repeats per timed operation; the minimum elapsed time is reported");
DEFINE_bool(dry_run, false, "Print sweep plan and exit");

namespace facebook::nimble::mlidc {
namespace {

using namespace facebook::nimble::detail;
using namespace facebook::nimble::detail::subintsplit;

// Runs `fn` `iterations` times and returns the minimum elapsed wall time, in
// nanoseconds -- the minimum, rather than mean or median, is the standard
// choice for microbenchmarking a short, deterministic operation: every
// slower repeat is attributable to scheduling or cache noise on top of the
// true cost, never a faster true cost hiding beneath it.
template <typename Fn>
int64_t timedNs(int iterations, Fn&& fn) {
  auto best = std::chrono::steady_clock::duration::max();
  for (int i = 0; i < iterations; ++i) {
    const auto start = std::chrono::steady_clock::now();
    fn();
    const auto elapsed = std::chrono::steady_clock::now() - start;
    if (elapsed < best) {
      best = elapsed;
    }
  }
  return std::chrono::duration_cast<std::chrono::nanoseconds>(best).count();
}

struct ColumnResult {
  std::string dataset;
  double bestOtherCostBytes{std::numeric_limits<double>::infinity()};
  EncodingType bestOtherType{EncodingType::Trivial};
  double sisCostBytes{std::numeric_limits<double>::infinity()};
  bool sisWouldWin{false};
  bool varianceGatePredictsWorthTrying{false};
  bool gradientGatePredictsWorthTrying{false};
  double gateVariance{0.0};
  size_t numBoundaries{0};
  // Cost of computing BitFlipProfile from the raw column, shared by both
  // gates -- neither gate can run without it.
  int64_t profileComputeNs{0};
  // Incremental cost of each gate's own decision, given an already-computed
  // profile. Each gate's real standalone cost is profileComputeNs plus its
  // own field here.
  int64_t varianceGateNs{0};
  int64_t gradientGateNs{0};
};

// Best cost among the real (unmodified) candidate encodings, mirroring
// ManualEncodingSelectionPolicy::select()'s own estimatedSize * readFactor
// comparison -- computed here, not by calling into that class, so this
// driver never touches production selection code.
template <typename Elem>
std::pair<double, EncodingType> bestOtherCandidateCost(
    std::span<const typename TypeTraits<Elem>::physicalType> values,
    const Statistics<typename TypeTraits<Elem>::physicalType>& statistics,
    const Encoding::Options& options) {
  double bestCost = std::numeric_limits<double>::infinity();
  EncodingType bestType = EncodingType::Trivial;
  for (const auto& [type, factor] :
       ManualEncodingSelectionPolicyFactory::defaultEncodingReadFactors()) {
    const auto estimate = EncodingSizeEstimation<Elem>::estimateSize(
        type, values, statistics, options);
    if (!estimate.has_value()) {
      continue;
    }
    const double cost = static_cast<double>(estimate.value()) * factor;
    if (cost < bestCost) {
      bestCost = cost;
      bestType = type;
    }
  }
  return {bestCost, bestType};
}

template <typename Elem>
ColumnResult evaluateColumn(
    const std::string& datasetName,
    std::span<const typename TypeTraits<Elem>::physicalType> physical,
    const TopLevelPolicyConfig& policyConfig) {
  using Phys = typename TypeTraits<Elem>::physicalType;

  ColumnResult result;
  result.dataset = datasetName;

  const auto statistics = Statistics<Phys>::create(physical);
  const Encoding::Options options{};

  std::tie(result.bestOtherCostBytes, result.bestOtherType) =
      bestOtherCandidateCost<Elem>(physical, statistics, options);

  // Ground truth: unconstrained DP, gate bypassed -- "what SubIntSplit would
  // really cost if we always tried it."
  const auto groundTruth = estimateSubIntSplitSize<Phys>(
      physical,
      policyConfig,
      defaultSamplerConfig(),
      defaultSelectorConfig(),
      /*applyGate=*/false);
  result.gateVariance = groundTruth.profile.variance;
  result.sisCostBytes = groundTruth.estimatedBytes.has_value()
      ? static_cast<double>(groundTruth.estimatedBytes.value())
      : std::numeric_limits<double>::infinity();
  result.sisWouldWin = result.sisCostBytes < result.bestOtherCostBytes;
  result.numBoundaries =
      bitFlipGradientBoundaries(groundTruth.profile, policyConfig).size();

  // Timed separately from the ground-truth call above: this measures what
  // each gate would really cost if wired into a live
  // selection path -- profile computed fresh from the raw column, then each
  // gate's own decision given that profile. computeBitFlipProfile()
  // recomputes the identical profile groundTruth.profile already holds
  // (deterministic), traded here for an honest, isolated timing.
  BitFlipProfile timedProfile;
  result.profileComputeNs = timedNs(FLAGS_mlidsp_timing_iters, [&] {
    timedProfile = computeBitFlipProfile<Phys>(physical);
  });
  result.varianceGateNs = timedNs(FLAGS_mlidsp_timing_iters, [&] {
    result.varianceGatePredictsWorthTrying =
        bitFlipVarianceGate(timedProfile, policyConfig);
  });
  result.gradientGateNs = timedNs(FLAGS_mlidsp_timing_iters, [&] {
    result.gradientGatePredictsWorthTrying =
        bitFlipGradientGate(timedProfile, policyConfig);
  });

  return result;
}

// The whole driver body, templated on the element type. main() picks the
// type from --mlidc_dtype and dispatches here.
template <typename Elem>
int runBenchmark() {
  using Phys = typename TypeTraits<Elem>::physicalType;
  static_assert(isIntegralType<Phys>());

  const uint32_t n = static_cast<uint32_t>(FLAGS_mlidc_rows);
  const uint64_t seed = static_cast<uint64_t>(FLAGS_mlidc_seed);

  auto datasets = defaultDatasets<Elem>();

  TopLevelPolicyConfig policyConfig;
  policyConfig.varianceGateThreshold = FLAGS_mlidsp_variance_threshold;
  policyConfig.gradientStdDevMultiplier = FLAGS_mlidsp_gradient_multiplier;
  policyConfig.minGradientBoundaries = FLAGS_mlidsp_min_gradient_boundaries;
  policyConfig.minGradientMagnitude = FLAGS_mlidsp_min_gradient_magnitude;

  std::cout << "bench_selection_policy: " << datasets.size()
            << " datasets, N=" << n << "\n\n";

  if (FLAGS_dry_run) {
    std::cout << "Datasets:\n";
    for (const auto& d : datasets) {
      std::cout << "  " << d.name << "\n";
    }
    return 0;
  }

  std::vector<std::string> csvColumns = {
      "driver",
      "dtype",
      "dataset",
      "N",
      "seed",
      "best_other_encoding",
      "best_other_cost_bytes",
      "sis_cost_bytes",
      "sis_would_win",
      "variance_gate_predicts_worth_trying",
      "variance_gate_correct",
      "gate_variance",
      "gradient_gate_predicts_worth_trying",
      "gradient_gate_correct",
      "num_boundaries",
      "num_bits",
      "profile_compute_ns",
      "variance_gate_ns",
      "gradient_gate_ns",
  };
  std::string csvPath = FLAGS_mlidc_output_csv.empty()
      ? "bench_selection_policy.csv"
      : FLAGS_mlidc_output_csv;
  CsvResultWriter csv(csvPath, csvColumns);
  if (!FLAGS_mlidc_output_manifest.empty()) {
    writeRunManifest(FLAGS_mlidc_output_manifest);
  }

  // One confusion matrix per gate -- they are independent standalone
  // policies over the same ground truth, not stages of one pipeline.
  struct ConfusionMatrix {
    int64_t truePositive{0};
    int64_t falsePositive{0};
    int64_t trueNegative{0};
    int64_t falseNegative{0};

    void record(bool predicted, bool actual) {
      if (predicted && actual) {
        ++truePositive;
      } else if (predicted && !actual) {
        ++falsePositive;
      } else if (!predicted && actual) {
        ++falseNegative;
      } else {
        ++trueNegative;
      }
    }

    void print(std::ostream& out, const char* label) const {
      const int64_t total =
          truePositive + falsePositive + trueNegative + falseNegative;
      const double precision = (truePositive + falsePositive) > 0
          ? static_cast<double>(truePositive) / (truePositive + falsePositive)
          : std::numeric_limits<double>::quiet_NaN();
      const double recall = (truePositive + falseNegative) > 0
          ? static_cast<double>(truePositive) / (truePositive + falseNegative)
          : std::numeric_limits<double>::quiet_NaN();
      const double f1 = (precision + recall) > 0
          ? 2.0 * precision * recall / (precision + recall)
          : std::numeric_limits<double>::quiet_NaN();
      out << total << " columns [" << label << "]: TP=" << truePositive
          << " FP=" << falsePositive << " TN=" << trueNegative
          << " FN=" << falseNegative << std::fixed << std::setprecision(3)
          << "  precision=" << precision << " recall=" << recall << " f1=" << f1
          << "\n";
    }
  };
  ConfusionMatrix varianceMatrix;
  ConfusionMatrix gradientMatrix;

  for (const auto& ds : datasets) {
    std::cout << "== Dataset: " << ds.name << " ==\n";
    auto data = ds.generate(n, seed);
    auto physical = std::span<const Phys>(
        reinterpret_cast<const Phys*>(data.data()), data.size());

    const auto result = evaluateColumn<Elem>(ds.name, physical, policyConfig);

    const bool varianceCorrect =
        result.varianceGatePredictsWorthTrying == result.sisWouldWin;
    const bool gradientCorrect =
        result.gradientGatePredictsWorthTrying == result.sisWouldWin;
    varianceMatrix.record(
        result.varianceGatePredictsWorthTrying, result.sisWouldWin);
    gradientMatrix.record(
        result.gradientGatePredictsWorthTrying, result.sisWouldWin);

    std::cout << "  SIS: " << result.sisCostBytes << " B, best other ("
              << toString(result.bestOtherType)
              << "): " << result.bestOtherCostBytes
              << " B, SIS would win: " << (result.sisWouldWin ? "yes" : "no")
              << ", variance gate: "
              << (result.varianceGatePredictsWorthTrying ? "yes" : "no")
              << (varianceCorrect ? "" : " [WRONG]") << ", gradient gate: "
              << (result.gradientGatePredictsWorthTrying ? "yes" : "no")
              << (gradientCorrect ? "" : " [WRONG]") << "\n";

    csv.beginRow();
    csv.set("driver", std::string("bench_selection_policy"));
    csv.set("dtype", elemTypeName<Elem>());
    csv.set("dataset", ds.name);
    csv.set("N", static_cast<int64_t>(n));
    csv.set("seed", static_cast<int64_t>(seed));
    csv.set("best_other_encoding", toString(result.bestOtherType));
    csv.set("best_other_cost_bytes", result.bestOtherCostBytes);
    csv.set("sis_cost_bytes", result.sisCostBytes);
    csv.set("sis_would_win", static_cast<int64_t>(result.sisWouldWin ? 1 : 0));
    csv.set(
        "variance_gate_predicts_worth_trying",
        static_cast<int64_t>(result.varianceGatePredictsWorthTrying ? 1 : 0));
    csv.set(
        "variance_gate_correct", static_cast<int64_t>(varianceCorrect ? 1 : 0));
    csv.set("gate_variance", result.gateVariance);
    csv.set(
        "gradient_gate_predicts_worth_trying",
        static_cast<int64_t>(result.gradientGatePredictsWorthTrying ? 1 : 0));
    csv.set(
        "gradient_gate_correct", static_cast<int64_t>(gradientCorrect ? 1 : 0));
    csv.set("num_boundaries", static_cast<int64_t>(result.numBoundaries));
    csv.set("num_bits", static_cast<int64_t>(sizeof(Phys) * 8));
    csv.set("profile_compute_ns", result.profileComputeNs);
    csv.set("variance_gate_ns", result.varianceGateNs);
    csv.set("gradient_gate_ns", result.gradientGateNs);
    csv.endRow();
  }
  csv.flush();

  std::cout << "\n";
  varianceMatrix.print(std::cout, "variance gate");
  gradientMatrix.print(std::cout, "gradient gate");
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
  std::cerr << "bench_selection_policy requires "
               "NIMBLE_ENABLE_EXPERIMENTAL_ENCODINGS\n";
  return 1;
}

#endif
