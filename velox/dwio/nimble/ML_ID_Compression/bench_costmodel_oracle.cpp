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

// bench_costmodel_oracle: validates SubIntSplit's cost-model-driven DP
// selector against an "oracle" that actually encodes each candidate bit
// range with every candidate encoding and measures real byte counts.
//
// For every dataset this driver: samples the stream, builds an oracle grid
// (measured bytes per [l..r] range per encoding) and a cost-model grid
// (bestCostBits() estimates for the same ranges), runs both AutoSIS's
// selectSplits() DP and a simple unconstrained oracle DP, and reports
// per-cell agreement (top-1 accuracy, Spearman rho, mean |rel err|) plus
// plan-level regret between the cost model's choice and the oracle optimum.

#ifdef NIMBLE_ENABLE_EXPERIMENTAL_ENCODINGS

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>
#include <span>
#include <string>
#include <vector>

#include <gflags/gflags.h>

#include "velox/dwio/nimble/ML_ID_Compression/BenchCommon.h"

#include "velox/dwio/nimble/encodings/ConstantEncoding.h"
#include "velox/dwio/nimble/encodings/DictionaryEncoding.h"
#include "velox/dwio/nimble/encodings/FixedBitWidthEncoding.h"
#include "velox/dwio/nimble/encodings/MainlyConstantEncoding.h"
#include "velox/dwio/nimble/encodings/RLEEncoding.h"
#include "velox/dwio/nimble/encodings/SubIntSplitCostModels.h"
#include "velox/dwio/nimble/encodings/SubIntSplitMetrics.h"
#include "velox/dwio/nimble/encodings/SubIntSplitSampler.h"
#include "velox/dwio/nimble/encodings/SubIntSplitSelector.h"
#include "velox/dwio/nimble/encodings/TrivialEncoding.h"
#include "velox/dwio/nimble/encodings/VarintEncoding.h"

DEFINE_bool(validate, false, "Sanity-check oracle encode calls do not throw");
DEFINE_bool(dry_run, false, "Print sweep plan and exit");

namespace facebook::nimble::mlidc {
namespace {

using namespace facebook::nimble::detail::subintsplit;

using Elem = int64_t;
constexpr int kBits = 64;

struct CandidateEncoding {
  std::string name;
  EncodingType type;
};

// Try to encode `sectionData` with EncodingT; return byte count, or SIZE_MAX
// on failure (throws, e.g. Constant on non-constant data).
template <typename EncodingT>
size_t tryEncode(const Vector<Elem>& sectionData) {
  try {
    auto& pool = benchmarks::benchmarkPool();
    Buffer buf{*pool};
    facebook::nimble::Encoding::Options opts;
    auto encoded = test::Encoder<EncodingT>::encode(
        buf, sectionData, CompressionType::Uncompressed, opts);
    return encoded.size();
  } catch (...) {
    return std::numeric_limits<size_t>::max();
  }
}

// Dispatch oracle encode by EncodingType (only the 7 candidates we track).
size_t oracleEncodeBytes(EncodingType type, const Vector<Elem>& sectionData) {
  switch (type) {
    case EncodingType::Trivial:
      return tryEncode<TrivialEncoding<Elem>>(sectionData);
    case EncodingType::FixedBitWidth:
      return tryEncode<FixedBitWidthEncoding<Elem>>(sectionData);
    case EncodingType::Constant:
      return tryEncode<ConstantEncoding<Elem>>(sectionData);
    case EncodingType::MainlyConstant:
      return tryEncode<MainlyConstantEncoding<Elem>>(sectionData);
    case EncodingType::Dictionary:
      return tryEncode<DictionaryEncoding<Elem>>(sectionData);
    case EncodingType::RLE:
      return tryEncode<RLEEncoding<Elem>>(sectionData);
    case EncodingType::Varint:
      return tryEncode<VarintEncoding<Elem>>(sectionData);
    default:
      return std::numeric_limits<size_t>::max();
  }
}

std::vector<CandidateEncoding> candidateEncodings() {
  return {
      {"Trivial", EncodingType::Trivial},
      {"FixedBitWidth", EncodingType::FixedBitWidth},
      {"Constant", EncodingType::Constant},
      {"MainlyConstant", EncodingType::MainlyConstant},
      {"Dictionary", EncodingType::Dictionary},
      {"RLE", EncodingType::RLE},
      {"Varint", EncodingType::Varint},
  };
}

struct OracleResult {
  size_t bytes{std::numeric_limits<size_t>::max()};
};

struct OracleCell {
  size_t bestBytes{std::numeric_limits<size_t>::max()};
  EncodingType bestEncoding{EncodingType::Trivial};
  std::vector<OracleResult> results; // parallel to candidateEncodings()
};

struct ModelCell {
  double bestBits{std::numeric_limits<double>::infinity()};
  EncodingType bestEncoding{EncodingType::Trivial};
  std::vector<double> estBits; // parallel to candidateEncodings(), per-encoding
};

// Spearman rank correlation between two equal-length rank vectors (1-based
// dense ranks are fine; ties broken by encounter order, matching the
// playground's approach).
double spearmanRho(
    const std::vector<double>& a,
    const std::vector<double>& b) {
  const size_t n = a.size();
  if (n < 3) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  auto rankOf = [n](const std::vector<double>& v) {
    std::vector<size_t> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(
        idx.begin(), idx.end(), [&](size_t i, size_t j) { return v[i] < v[j]; });
    std::vector<double> rank(n);
    for (size_t r = 0; r < n; ++r) {
      rank[idx[r]] = static_cast<double>(r);
    }
    return rank;
  };
  auto ra = rankOf(a);
  auto rb = rankOf(b);
  double sumSqDiff = 0.0;
  for (size_t i = 0; i < n; ++i) {
    const double d = ra[i] - rb[i];
    sumSqDiff += d * d;
  }
  const double nd = static_cast<double>(n);
  return 1.0 - (6.0 * sumSqDiff) / (nd * (nd * nd - 1.0));
}

// Simple unconstrained oracle DP over the measured grid: minimises total
// measured bytes, no split penalty.
struct OracleSegment {
  int bitStart{0};
  int bitEnd{0};
  EncodingType encoding{EncodingType::Trivial};
  size_t bytes{0};
};

struct OracleDpResult {
  std::vector<OracleSegment> segments;
  size_t totalBytes{0};
};

OracleDpResult oracleDp(
    const std::vector<std::vector<OracleCell>>& grid,
    int sz) {
  std::vector<double> dp(sz + 1, std::numeric_limits<double>::infinity());
  std::vector<int> prev(sz + 1, -1);
  dp[0] = 0.0;
  for (int i = 1; i <= sz; ++i) {
    for (int j = 0; j < i; ++j) {
      const auto& cell = grid[j][i - 1];
      if (cell.bestBytes == std::numeric_limits<size_t>::max()) {
        continue;
      }
      const double candidate = dp[j] + static_cast<double>(cell.bestBytes);
      if (candidate < dp[i]) {
        dp[i] = candidate;
        prev[i] = j;
      }
    }
  }
  OracleDpResult result;
  if (!std::isfinite(dp[sz])) {
    return result;
  }
  result.totalBytes = static_cast<size_t>(dp[sz]);
  int idx = sz;
  while (idx > 0) {
    const int start = prev[idx];
    if (start < 0) {
      break;
    }
    const auto& cell = grid[start][idx - 1];
    result.segments.push_back(
        {start, idx - 1, cell.bestEncoding, cell.bestBytes});
    idx = start;
  }
  std::reverse(result.segments.begin(), result.segments.end());
  return result;
}

} // namespace
} // namespace facebook::nimble::mlidc

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  facebook::velox::memory::MemoryManager::initialize({});

  using namespace facebook::nimble;
  using namespace facebook::nimble::mlidc;
  using namespace facebook::nimble::detail::subintsplit;

  const uint32_t n = static_cast<uint32_t>(FLAGS_mlidc_rows);
  const uint64_t seed = static_cast<uint64_t>(FLAGS_mlidc_seed);

  auto datasets = defaultInt64Datasets<Elem>();
  auto candidates = candidateEncodings();

  std::cout << "bench_costmodel_oracle: " << datasets.size()
            << " datasets, N=" << n << ", kBits=" << kBits << "\n\n";

  if (FLAGS_dry_run) {
    std::cout << "Datasets:\n";
    for (const auto& d : datasets) {
      std::cout << "  " << d.name << "\n";
    }
    std::cout << "Candidate encodings:\n";
    for (const auto& c : candidates) {
      std::cout << "  " << c.name << "\n";
    }
    return 0;
  }

  std::vector<std::string> csvColumns = {
      "driver",
      "dataset",
      "N",
      "seed",
      "sample_size",
      "min_segment_width",
      "l",
      "r",
      "width",
      "encoding",
      "has_cost_model",
      "est_bits",
      "actual_bytes",
      "actual_bits_per_elem",
      "rel_err",
      "model_rank",
      "actual_rank",
      "is_model_pick",
      "is_oracle_pick",
      "plan_type",
      "plan_segment_count",
      "plan_total_sample_bytes",
      "top1_accuracy",
      "spearman_rho",
      "mean_abs_rel_err",
      "regret_sample_bytes",
      "skipped"};

  std::string csvPath = FLAGS_mlidc_output_csv.empty()
      ? "bench_costmodel_oracle.csv"
      : FLAGS_mlidc_output_csv;
  CsvResultWriter csv(csvPath, csvColumns);

  if (!FLAGS_mlidc_output_manifest.empty()) {
    writeRunManifest(FLAGS_mlidc_output_manifest);
  }

  SamplerConfig samplerCfg = defaultSamplerConfig();
  SelectorConfig selectorCfg = defaultSelectorConfig();
  const MetricFlags requiredFlags = allCostModelRequiredFlags();

  for (const auto& ds : datasets) {
    std::cout << "== Dataset: " << ds.name << " ==\n";
    auto data = ds.generate(n, seed);

    std::vector<uint64_t> samples;
    sampleIntoU64(
        std::span<const Elem>(data.data(), data.size()), samples, samplerCfg);
    const size_t sampleSize = samples.size();
    if (sampleSize == 0) {
      std::cerr << "  [SKIP] empty sample\n";
      continue;
    }

    // -----------------------------------------------------------------
    // Build oracle grid + cost model grid over [l..r], 0 <= l <= r < kBits.
    // -----------------------------------------------------------------
    std::vector<std::vector<OracleCell>> oracleGrid(
        kBits, std::vector<OracleCell>(kBits));
    std::vector<std::vector<ModelCell>> modelGrid(
        kBits, std::vector<ModelCell>(kBits));

    MetricCollector collector;
    BitRangeExtractor extractor(samples);
    auto& pool = benchmarks::benchmarkPool();

    int cellCount = 0;
    int agreeCount = 0;
    double spearmanSum = 0.0;
    int spearmanCount = 0;
    double relErrSum = 0.0;
    int relErrCount = 0;

    for (int l = 0; l < kBits; ++l) {
      extractor.reset(l);
      for (int r = l; r < kBits; ++r) {
        extractor.extend(r);
        const std::vector<uint64_t>& sectionU64 = extractor.values();
        const int width = r - l + 1;

        // Convert to Vector<int64_t> for oracle encoding.
        Vector<Elem> sectionData{pool.get()};
        sectionData.resize(sectionU64.size());
        for (size_t i = 0; i < sectionU64.size(); ++i) {
          sectionData[i] = static_cast<Elem>(sectionU64[i]);
        }

        // Cost model metrics + per-encoding estimates.
        const SegmentMetrics metrics = collector.compute(sectionU64, requiredFlags);
        EncodingType modelBestEnc = EncodingType::Trivial;
        const double modelBestBits =
            bestCostBits(metrics, sampleSize, width, modelBestEnc);

        ModelCell& mc = modelGrid[l][r];
        mc.bestBits = modelBestBits;
        mc.bestEncoding = modelBestEnc;
        mc.estBits.resize(candidates.size());
        for (size_t ci = 0; ci < candidates.size(); ++ci) {
          double bits;
          switch (candidates[ci].type) {
            case EncodingType::Trivial:
              bits = trivialCostBits(metrics, sampleSize, width);
              break;
            case EncodingType::FixedBitWidth:
              bits = fixedBitWidthCostBits(metrics, sampleSize, width);
              break;
            case EncodingType::Constant:
              bits = constantCostBits(metrics, sampleSize, width);
              break;
            case EncodingType::MainlyConstant:
              bits = mainlyConstantCostBits(metrics, sampleSize, width);
              break;
            case EncodingType::Dictionary:
              bits = dictionaryCostBits(metrics, sampleSize, width);
              break;
            case EncodingType::RLE:
              bits = rleCostBits(metrics, sampleSize, width);
              break;
            case EncodingType::Varint:
              bits = varintCostBits(metrics, sampleSize, width);
              break;
            default:
              bits = std::numeric_limits<double>::infinity();
          }
          mc.estBits[ci] = bits;
        }

        // Oracle: actually encode with each candidate, measure bytes.
        OracleCell& oc = oracleGrid[l][r];
        oc.results.resize(candidates.size());
        for (size_t ci = 0; ci < candidates.size(); ++ci) {
          const size_t bytes =
              oracleEncodeBytes(candidates[ci].type, sectionData);
          oc.results[ci].bytes = bytes;
          if (bytes < oc.bestBytes) {
            oc.bestBytes = bytes;
            oc.bestEncoding = candidates[ci].type;
          }
        }

        // Per-cell comparisons.
        ++cellCount;
        if (oc.bestBytes != std::numeric_limits<size_t>::max() &&
            oc.bestEncoding == mc.bestEncoding) {
          ++agreeCount;
        }

        // Rank vectors over usable candidates (finite model estimate AND
        // successful oracle encode) for Spearman rho.
        std::vector<double> modelVals;
        std::vector<double> actualVals;
        for (size_t ci = 0; ci < candidates.size(); ++ci) {
          const bool modelUsable = std::isfinite(mc.estBits[ci]);
          const bool actualUsable =
              oc.results[ci].bytes != std::numeric_limits<size_t>::max();
          if (modelUsable && actualUsable) {
            modelVals.push_back(mc.estBits[ci]);
            actualVals.push_back(static_cast<double>(oc.results[ci].bytes));
          }
        }
        double rho = std::numeric_limits<double>::quiet_NaN();
        if (modelVals.size() >= 3) {
          rho = spearmanRho(modelVals, actualVals);
          if (std::isfinite(rho)) {
            spearmanSum += rho;
            ++spearmanCount;
          }
        }

        // Compute ranks for CSV emission (1 = best/lowest).
        std::vector<size_t> modelOrder(candidates.size());
        std::iota(modelOrder.begin(), modelOrder.end(), 0);
        std::sort(modelOrder.begin(), modelOrder.end(), [&](size_t a, size_t b) {
          return mc.estBits[a] < mc.estBits[b];
        });
        std::vector<int> modelRank(candidates.size(), -1);
        for (size_t rk = 0; rk < modelOrder.size(); ++rk) {
          modelRank[modelOrder[rk]] = static_cast<int>(rk) + 1;
        }

        std::vector<size_t> actualOrder(candidates.size());
        std::iota(actualOrder.begin(), actualOrder.end(), 0);
        std::sort(
            actualOrder.begin(), actualOrder.end(), [&](size_t a, size_t b) {
              return oc.results[a].bytes < oc.results[b].bytes;
            });
        std::vector<int> actualRank(candidates.size(), -1);
        for (size_t rk = 0; rk < actualOrder.size(); ++rk) {
          actualRank[actualOrder[rk]] = static_cast<int>(rk) + 1;
        }

        for (size_t ci = 0; ci < candidates.size(); ++ci) {
          const bool hasCostModel = std::isfinite(mc.estBits[ci]);
          const bool oracleOk =
              oc.results[ci].bytes != std::numeric_limits<size_t>::max();

          csv.beginRow();
          csv.set("driver", "bench_costmodel_oracle");
          csv.set("dataset", ds.name);
          csv.set("N", static_cast<int64_t>(n));
          csv.set("seed", static_cast<int64_t>(seed));
          csv.set("sample_size", static_cast<int64_t>(sampleSize));
          csv.set(
              "min_segment_width",
              static_cast<int64_t>(selectorCfg.minSegmentWidth));
          csv.set("l", static_cast<int64_t>(l));
          csv.set("r", static_cast<int64_t>(r));
          csv.set("width", static_cast<int64_t>(width));
          csv.set("encoding", candidates[ci].name);
          csv.set("has_cost_model", hasCostModel ? int64_t{1} : int64_t{0});
          if (hasCostModel) {
            csv.set("est_bits", mc.estBits[ci]);
          }
          if (oracleOk) {
            csv.set(
                "actual_bytes", static_cast<int64_t>(oc.results[ci].bytes));
            const double bitsPerElem = sampleSize > 0
                ? static_cast<double>(oc.results[ci].bytes) * 8.0 /
                    static_cast<double>(sampleSize)
                : 0.0;
            csv.set("actual_bits_per_elem", bitsPerElem);
            if (hasCostModel && oc.results[ci].bytes > 0) {
              const double relErr =
                  (mc.estBits[ci] -
                   static_cast<double>(oc.results[ci].bytes) * 8.0) /
                  (static_cast<double>(oc.results[ci].bytes) * 8.0);
              csv.set("rel_err", relErr);
              relErrSum += std::fabs(relErr);
              ++relErrCount;
            }
          }
          if (modelRank[ci] > 0) {
            csv.set("model_rank", static_cast<int64_t>(modelRank[ci]));
          }
          if (actualRank[ci] > 0) {
            csv.set("actual_rank", static_cast<int64_t>(actualRank[ci]));
          }
          csv.set(
              "is_model_pick",
              (hasCostModel && candidates[ci].type == mc.bestEncoding)
                  ? int64_t{1}
                  : int64_t{0});
          csv.set(
              "is_oracle_pick",
              (oracleOk && candidates[ci].type == oc.bestEncoding)
                  ? int64_t{1}
                  : int64_t{0});
          if (std::isfinite(rho) && ci == 0) {
            csv.set("spearman_rho", rho);
          }
          csv.set("skipped", int64_t{0});
          csv.endRow();
        }
      }
    }
    csv.flush();

    const double top1Accuracy =
        cellCount > 0 ? static_cast<double>(agreeCount) / cellCount : 0.0;
    const double meanRho =
        spearmanCount > 0 ? spearmanSum / spearmanCount : 0.0;
    const double meanAbsRelErr =
        relErrCount > 0 ? relErrSum / relErrCount : 0.0;

    std::cout << "  top1_accuracy=" << top1Accuracy
              << " mean_spearman_rho=" << meanRho
              << " mean_abs_rel_err=" << meanAbsRelErr << "\n";

    // -----------------------------------------------------------------
    // Plan comparison: AutoSIS DP (cost-model driven) vs oracle DP.
    // -----------------------------------------------------------------
    SelectorResult autoResult =
        selectSplits(samples, kBits, sampleSize, selectorCfg);
    OracleDpResult oracleResult = oracleDp(oracleGrid, kBits);

    // Regret: sum over AutoSIS's chosen segments of (measured bytes for the
    // AutoSIS pick) minus (oracle's best bytes for that same [l..r] range).
    size_t autoTotalSampleBytes = 0;
    size_t regretBytes = 0;
    for (const auto& seg : autoResult.segments) {
      const auto& cell = oracleGrid[seg.bitStart][seg.bitEnd];
      size_t autoBytesForPick = std::numeric_limits<size_t>::max();
      for (size_t ci = 0; ci < candidates.size(); ++ci) {
        if (candidates[ci].type == seg.encoding) {
          autoBytesForPick = cell.results[ci].bytes;
          break;
        }
      }
      if (autoBytesForPick == std::numeric_limits<size_t>::max()) {
        continue;
      }
      autoTotalSampleBytes += autoBytesForPick;
      if (cell.bestBytes != std::numeric_limits<size_t>::max() &&
          autoBytesForPick > cell.bestBytes) {
        regretBytes += (autoBytesForPick - cell.bestBytes);
      }

      csv.beginRow();
      csv.set("driver", "bench_costmodel_oracle");
      csv.set("dataset", ds.name);
      csv.set("N", static_cast<int64_t>(n));
      csv.set("seed", static_cast<int64_t>(seed));
      csv.set("sample_size", static_cast<int64_t>(sampleSize));
      csv.set("l", static_cast<int64_t>(seg.bitStart));
      csv.set("r", static_cast<int64_t>(seg.bitEnd));
      csv.set("width", static_cast<int64_t>(seg.bitEnd - seg.bitStart + 1));
      for (const auto& c : candidates) {
        if (c.type == seg.encoding) {
          csv.set("encoding", c.name);
          break;
        }
      }
      csv.set("actual_bytes", static_cast<int64_t>(autoBytesForPick));
      csv.set("plan_type", "autosis");
      csv.set(
          "plan_segment_count",
          static_cast<int64_t>(autoResult.segments.size()));
      csv.set("skipped", int64_t{0});
      csv.endRow();
    }

    for (const auto& seg : oracleResult.segments) {
      csv.beginRow();
      csv.set("driver", "bench_costmodel_oracle");
      csv.set("dataset", ds.name);
      csv.set("N", static_cast<int64_t>(n));
      csv.set("seed", static_cast<int64_t>(seed));
      csv.set("sample_size", static_cast<int64_t>(sampleSize));
      csv.set("l", static_cast<int64_t>(seg.bitStart));
      csv.set("r", static_cast<int64_t>(seg.bitEnd));
      csv.set("width", static_cast<int64_t>(seg.bitEnd - seg.bitStart + 1));
      for (const auto& c : candidates) {
        if (c.type == seg.encoding) {
          csv.set("encoding", c.name);
          break;
        }
      }
      csv.set("actual_bytes", static_cast<int64_t>(seg.bytes));
      csv.set("plan_type", "oracle");
      csv.set(
          "plan_segment_count",
          static_cast<int64_t>(oracleResult.segments.size()));
      csv.set("skipped", int64_t{0});
      csv.endRow();
    }

    // Summary row for this dataset.
    csv.beginRow();
    csv.set("driver", "bench_costmodel_oracle");
    csv.set("dataset", ds.name);
    csv.set("N", static_cast<int64_t>(n));
    csv.set("seed", static_cast<int64_t>(seed));
    csv.set("sample_size", static_cast<int64_t>(sampleSize));
    csv.set(
        "min_segment_width", static_cast<int64_t>(selectorCfg.minSegmentWidth));
    csv.set("plan_type", "summary");
    csv.set(
        "plan_total_sample_bytes", static_cast<int64_t>(autoTotalSampleBytes));
    csv.set("top1_accuracy", top1Accuracy);
    csv.set("spearman_rho", meanRho);
    csv.set("mean_abs_rel_err", meanAbsRelErr);
    csv.set("regret_sample_bytes", static_cast<int64_t>(regretBytes));
    csv.set("skipped", int64_t{0});
    csv.endRow();
    csv.flush();

    std::cout << "  AutoSIS plan: " << autoResult.segments.size()
              << " segments, sample_bytes=" << autoTotalSampleBytes << "\n";
    std::cout << "  Oracle plan:  " << oracleResult.segments.size()
              << " segments, sample_bytes=" << oracleResult.totalBytes
              << "  regret=" << regretBytes << "\n";
  }

  std::cout << "\nResults written to: " << csvPath << "\n";
  return 0;
}

#else

#include <iostream>
int main() {
  std::cerr
      << "bench_costmodel_oracle requires NIMBLE_ENABLE_EXPERIMENTAL_ENCODINGS\n";
  return 1;
}

#endif
