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
// (bestCostBitsRestricted() estimates for the same ranges), runs both AutoSIS's
// DP and a simple unconstrained oracle DP, and reports per-cell agreement
// (top-1 accuracy, Spearman rho, mean |rel err|) plus regret between the cost
// model's choice and the oracle optimum.
//
// Three properties the numbers depend on, all of them things this driver got
// wrong before and which made its output flattering rather than merely
// incomplete:
//
//  - The oracle encodes under the writer's section options, obtained from
//    sectionEncodingOptions(). Under default options FixedBitWidth rounds to a
//    byte boundary and the writer does not, so the ground truth was wrong for
//    the single most-selected encoding in the system.
//  - The model and the oracle score the same inventory. The oracle used to
//    offer seven encodings against a model minimising over fifteen, so
//    "top-1 accuracy" was reporting the difference between two lists.
//  - Huffman follows what production ships (withdrawn) rather than what
//    bestCostBits() hardcodes (allowed); --allow_huffman measures the other
//    configuration. Its bytes are still measured either way, so the size cost
//    of withdrawing it stays readable from the CSV.
//
//  - Sections are narrowed to the storage type the writer gives them before
//    being measured. Measuring at the column's element width charged Trivial
//    eight bytes per value on a 3-bit section where the writer spends one.
//
// What the sweep reports, and what it cannot:
//
// --sample_sizes runs the whole analysis at each size in turn, grid included,
// so the series is a convergence series and its last point is a grid measured
// over the whole column. At that point the sampler's windows tile contiguously
// -- the sample is the column, in order, with no window seams -- so the oracle
// is optimal over the DP's search space outright rather than chosen on a
// sample. A gap that survives to that point is model error. A gap that closes
// as the size grows was the instrument sampling badly, and the smaller-sample
// numbers were overstating the model's fault.
//
// Four plans are scored at every size: the model's DP and the oracle's DP,
// each at sample scale and at the column's real row count. Each is encoded
// over the whole column for real payload bytes and bits per element, because
// the driver exists to audit a sample-derived estimate and cannot itself
// deliver a sample-derived verdict. Both scales are kept because the
// difference between them is the evidence for per-section overhead being
// weighted at sample scale. At the whole-column point the two scales coincide
// by construction -- cost_scale is 1 -- so the two pairs of plans should agree
// exactly there, and it is a bug if they do not.
//
// Three quantities, and they bound different things:
//
//   oracle_cell_sum_bytes   sum over a plan's ranges of the smallest bytes any
//                           one encoding reached on that range, measured on its
//                           own. What the inventory could deliver if selection
//                           were perfect. Minimised by the oracle_dp plan.
//   plan_total_sample_bytes the same sum, but for the encoding each plan names.
//   full_column_bytes       the assembled plan encoded over the whole column by
//                           the writer. What is actually delivered.
//
// The first is NOT a lower bound on the last, and oracle_dp is not a bound on
// what a plan can encode to. A cell records the minimum over every candidate,
// while an assembled section is encoded with whatever nested selection picks --
// estimateSize times readFactor, not smallest measured bytes. So a plan chosen
// on cell minima has been optimised for an encoder that does not turn up. The
// sharpest statement of that: SimdForBitpack wins more cells than any other
// encoding and selection picks it zero times, because FixedBitWidth's estimate
// omits the seven-byte FixedBitArray slop its encode always writes while the
// two are otherwise byte-identical. The assembled encode also carries a
// SubIntSplit header and section directory no cell sum contains.
//
// oracle_dp_selection exists for that reason. It minimises the bytes of the
// candidate selection will really choose, so it is the plan a model plan should
// be held against, and the gap between it and oracle_dp is the cost of
// selection's mis-estimates -- the quantity this branch exists to reduce.
//
// A prediction worth checking rather than remembering: correcting the
// FixedBitWidth slop should shrink the gap between oracle_dp and
// oracle_dp_selection, because it removes the largest single case where argmin
// bytes and argmin estimate-times-read-factor disagree. If the gap survives
// that fix, there is a second source and it has not been found yet.
//
// So compare model against oracle on full_column_bytes, which is like-for-like,
// and read full_column_vs_cell_sum as the size of the objective mismatch. A
// model plan encoding smaller than the oracle_dp plan is not a contradiction
// and not evidence of a bug; it means the model's boundaries happened to suit
// the encoder the writer actually runs.
//
// What no sample size reveals: whether the true optimum is a partition neither
// DP can express. Both search contiguous bit ranges of at least
// min_segment_width, so a better partition outside that shape leaves them
// wrong together. The search_space column says what was searched, so "oracle"
// is not read as "optimal".
//
// The seam confound, which the sweep is partly built to expose: at any size
// below the whole column the sampler draws windows at a stride, and the metric
// walk counts the pair spanning two windows as an ordinary adjacent pair. On a
// monotone column each such pair spans a stride's worth of rows, and the
// delta-family models size a packed array to a global maximum over them, so a
// handful of artefacts set the width charged to every value. Sample size
// changes how many there are, which can look exactly like statistical
// convergence. max_delta and sample_seam_count are emitted so the two can be
// told apart, and per-encoding oracle win counts are emitted at every size so
// a family that takes no cells when sampled and many at full column is visible
// directly.
//
// One parity gap remains and is deliberate: with --nested_selection, a
// candidate's per-cell sub-streams are chosen by the default read factors
// rather than by the SubIntSplit-augmented list a real section's children see
// (EncodingSelectionPolicy.h's parentEncodingType == SubIntSplit block). That
// affects grandchildren of a per-cell measurement only; the full-column plan
// encodes go through the real policy and do not have it.

#ifdef NIMBLE_ENABLE_EXPERIMENTAL_ENCODINGS

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <span>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

#include <gflags/gflags.h>

#include "velox/dwio/nimble/encodings/benchmarks/ml_id_compression/BenchCommon.h"
#include "velox/dwio/nimble/encodings/benchmarks/ml_id_compression/ElemType.h"

#include "velox/dwio/nimble/encodings/BlockBitPackingEncoding.h"
#include "velox/dwio/nimble/encodings/ConstantEncoding.h"
#include "velox/dwio/nimble/encodings/DeltaBlockEncoding.h"
#include "velox/dwio/nimble/encodings/DeltaEncoding.h"
#include "velox/dwio/nimble/encodings/DictionaryEncoding.h"
#include "velox/dwio/nimble/encodings/FixedBitWidthEncoding.h"
#include "velox/dwio/nimble/encodings/ForEncoding.h"
#include "velox/dwio/nimble/encodings/FrequencyPartitionEncoding.h"
#include "velox/dwio/nimble/encodings/HuffmanEncoding.h"
#include "velox/dwio/nimble/encodings/MainlyConstantEncoding.h"
#include "velox/dwio/nimble/encodings/PFOREncoding.h"
#include "velox/dwio/nimble/encodings/RLEEncoding.h"
#include "velox/dwio/nimble/encodings/SimdForBitpackEncoding.h"
#include "velox/dwio/nimble/encodings/SubIntSplitEncoding.h"
#include "velox/dwio/nimble/encodings/TrivialEncoding.h"
#include "velox/dwio/nimble/encodings/VarintEncoding.h"
#include "velox/dwio/nimble/encodings/common/EncodingLayout.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelection.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelectionPolicy.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSizeEstimation.h"
#include "velox/dwio/nimble/encodings/subintsplit/CostModel.h"
#include "velox/dwio/nimble/encodings/subintsplit/Sampler.h"
#include "velox/dwio/nimble/encodings/subintsplit/SectionMetrics.h"
#include "velox/dwio/nimble/encodings/subintsplit/SplitBoundaries.h"
#include "velox/dwio/nimble/encodings/subintsplit/SplitSelector.h"
#include "velox/dwio/nimble/encodings/subintsplit/TopLevelPolicy.h"

DEFINE_int32(
    hybrid_shortlist,
    0,
    "When positive, also scores a hybrid plan: the k cheapest segmentations "
    "under the planner's models and under selection's estimators, plus the "
    "writer's own plan, re-scored with selection's estimators on a larger "
    "sample (--hybrid_rescore_samples), cheapest estimate kept.");
DEFINE_int64(
    hybrid_rescore_samples,
    65536,
    "Rows sampled for re-scoring the hybrid shortlist. Only ranges appearing "
    "in shortlisted plans are costed at this size.");
DEFINE_bool(
    hybrid_cheap_shortlist,
    false,
    "Build the hybrid shortlist without costing every range with selection's "
    "estimators: the planner's models, the bit-flip boundaries costed by the "
    "models, and the writer's plan. The estimator grid is what a writer could "
    "not afford, so this is the shortlist a production planner would have.");
DEFINE_bool(
    hybrid_refine_bitflip_splits,
    false,
    "Restrict the hybrid refinement's split moves to bit-flip gradient "
    "boundaries instead of every interior bit. Splits dominate refinement "
    "cost.");
DEFINE_bool(validate, false, "Sanity-check oracle encode calls do not throw");
DEFINE_bool(dry_run, false, "Print sweep plan and exit");
DEFINE_bool(
    allow_huffman,
    false,
    "Whether a bit range may be costed and measured as Huffman. False is what "
    "production ships (Encoding::Options::subIntSplitAllowHuffman); pass true "
    "to measure the withdrawn configuration.");
DEFINE_bool(
    allow_delta_block,
    false,
    "Whether a bit range may be costed and measured as DeltaBlock. False is "
    "what production ships (Encoding::Options::subIntSplitAllowDeltaBlock); "
    "pass true to measure the withdrawn configuration. Costing DeltaBlock "
    "walks the sample per grid cell, so this flag moves encode time as well as "
    "the plans chosen.");
DEFINE_bool(
    ignore_writer_mismatch,
    false,
    "Continue after the writer-reproduction check fails. That check compares a "
    "pinned plan's encoded bytes against the same plan derived by the encoder "
    "itself, and a mismatch means every full-column number in the run measures "
    "something other than the plan it names.");
DEFINE_string(
    sample_sizes,
    "2048,8192,32768,100000",
    "Comma-separated sample sizes to sweep, smallest first. 0 means the whole "
    "column, which is the sweep run to its limit rather than a separate mode: "
    "at that size the sampler's windows tile contiguously, so the sample is "
    "the column in order. Cost grows linearly in sample size and the whole-"
    "column entry dominates a sweep, so narrow --mlidc_datasets before asking "
    "for it.");
DEFINE_bool(
    nested_selection,
    true,
    "Whether the oracle picks a candidate's sub-stream encodings by real "
    "cost-based selection, as the writer does. False forces every sub-stream "
    "to Trivial, which is the test harness default and overstates the measured "
    "size of every encoding that has children.");

namespace facebook::nimble::mlidc {
namespace {

using namespace facebook::nimble::subintsplit;

struct CandidateEncoding {
  std::string name;
  EncodingType type;
};

// Whether EncodingFactory::encode can instantiate `type` for a section whose
// storage type is `storageBytes` wide.
//
// Taken from the guards in EncodingFactory::encode, which is the function the
// writer dispatches every nested section through: an encoding that function
// refuses for a type is one no section of that width can be given. A section's
// storage type is always an unsigned integer (SubIntSplitEncoding narrows to
// uint8/16/32/64), so every guard there about bool, strings and floating point
// is satisfied by construction and only a width guard can bite. Varint carries
// the only one -- `sizeof(physicalType) == 4 || sizeof(T) == 8`; the rest are
// guarded on numeric or integral, not on size.
//
// This cannot be derived automatically. The constraints are static_asserts and
// if-constexpr branches, and neither is visible to a type trait: a failing
// static_assert is a hard error, not a substitution failure, so there is
// nothing to detect. What is guaranteed instead is that one predicate feeds
// both the oracle's inventory and the model's, so the two cannot come apart
// even if this falls behind EncodingFactory.
inline bool encodingAvailableAtWidth(EncodingType type, int storageBytes) {
  if (type == EncodingType::Varint) {
    return storageBytes == 4 || storageBytes == 8;
  }
  return true;
}

// The candidates a SubIntSplit section is chosen from, and their read factors.
//
// A section sits under a SubIntSplit parent, so this is exactly what the writer
// hands one: the default factors with the parent type removed and the
// SubIntSplit-specific integer encodings added. Taken from
// nestedEncodingReadFactors rather than restated, so it cannot drift from the
// writer's own list.
//
// Measuring a section standalone does not give it this list by itself. The
// policy chain of a standalone encode has the candidate encoding as its parent,
// never SubIntSplit, so its children would be offered SubIntSplit -- which a
// real section's children cannot be, because SubIntSplit is already gone one
// level up. Seeding the top-level policy here is what puts a standalone section
// in the position a real one occupies.
inline const std::vector<std::pair<EncodingType, float>>&
sectionCandidateReadFactors() {
  static const std::vector<std::pair<EncodingType, float>> kFactors =
      nestedEncodingReadFactors(
          ManualEncodingSelectionPolicyFactory::defaultEncodingReadFactors(),
          EncodingType::SubIntSplit);
  return kFactors;
}

// The read factor selection would weigh `type` by, or nullopt when selection
// would not consider it for a section at all.
inline std::optional<float> sectionReadFactor(EncodingType type) {
  for (const auto& [candidate, factor] : sectionCandidateReadFactors()) {
    if (candidate == type) {
      return factor;
    }
  }
  return std::nullopt;
}

// Try to encode `sectionData` with EncodingT under `options`; return byte
// count, or SIZE_MAX on failure (throws, e.g. Constant on non-constant data).
//
// `options` is the writer's section options, not a default-constructed set.
// A default set rounds FixedBitWidth up to a byte boundary while the writer
// packs a section at its exact bit width, so measuring under defaults made the
// ground truth wrong for the most-selected encoding in the system: a 12-bit
// section measured 16 bits/value against a model that correctly said 12.
//
// Sub-stream encodings are chosen by real cost-based selection, as the writer
// chooses them, from the list a section really gets. Forcing them to Trivial
// charges Dictionary four bytes per index where the writer bit-packs them, and
// does the same to every candidate with children.
template <typename EncodingT, typename Storage>
size_t tryEncode(
    const Vector<Storage>& sectionData,
    const facebook::nimble::Encoding::Options& options) {
  try {
    auto& pool = benchmarks::benchmarkPool();
    Buffer buf{*pool};
    if (!FLAGS_nested_selection) {
      // Diagnostic only: every sub-stream forced to Trivial, which is what the
      // Encoder helper does by default and what this driver used to measure.
      return test::Encoder<EncodingT>::encode(
                 buf,
                 sectionData,
                 CompressionType::Uncompressed,
                 options,
                 /*realNestedSelection=*/false)
          .size();
    }
    const auto values =
        std::span<const Storage>(sectionData.data(), sectionData.size());
    EncodingSelection<Storage> selection{
        {.encodingType = test::EncodingTypeTraits<EncodingT>::encodingType},
        Statistics<Storage>::create(values),
        ManualEncodingSelectionPolicyFactory{
            sectionCandidateReadFactors(), std::nullopt}
            .createPolicy(TypeTraits<Storage>::dataType)};
    return EncodingT::encode(selection, values, buf, options).size();
  } catch (...) {
    return std::numeric_limits<size_t>::max();
  }
}

// Dispatch oracle encode by EncodingType, over every encoding the split
// planner's cost models score.
//
// `Storage` is the section's storage type, not the column's element type. The
// writer narrows every section to the smallest unsigned integer that holds its
// bit width before encoding it (SubIntSplitEncoding.h's storage-width switch),
// so a 3-bit section of an int64 column is encoded as uint8_t and costs one
// byte per value under Trivial, not eight. Measuring at the column's width
// instead inflated every encoding whose payload scales with sizeof(T): Trivial
// by up to 8x, which is the whole of its apparent -50% error.
//
// Narrowing also removes the floating point special case. A section is a bit
// pattern held in an unsigned integer whatever the column's type, so every
// encoding applies to every section and the model and the oracle score the
// same fifteen throughout.
template <typename Storage>
size_t oracleEncodeBytes(
    EncodingType type,
    const Vector<Storage>& sectionData,
    const facebook::nimble::Encoding::Options& options) {
  switch (type) {
    case EncodingType::Trivial:
      return tryEncode<TrivialEncoding<Storage>, Storage>(sectionData, options);
    case EncodingType::FixedBitWidth:
      return tryEncode<FixedBitWidthEncoding<Storage>, Storage>(
          sectionData, options);
    case EncodingType::Constant:
      return tryEncode<ConstantEncoding<Storage>, Storage>(
          sectionData, options);
    case EncodingType::MainlyConstant:
      return tryEncode<MainlyConstantEncoding<Storage>, Storage>(
          sectionData, options);
    case EncodingType::Dictionary:
      return tryEncode<DictionaryEncoding<Storage>, Storage>(
          sectionData, options);
    case EncodingType::RLE:
      return tryEncode<RLEEncoding<Storage>, Storage>(sectionData, options);
    case EncodingType::Varint:
      // Guarded, not merely skipped at runtime: VarintEncoding static_asserts
      // on its type width, so naming it for a narrow section is a compile
      // error rather than an incompatible encoding. Mirrors the if-constexpr
      // in EncodingFactory::encode's Varint case.
      if constexpr (sizeof(Storage) == 4 || sizeof(Storage) == 8) {
        return tryEncode<VarintEncoding<Storage>, Storage>(
            sectionData, options);
      } else {
        return std::numeric_limits<size_t>::max();
      }
    case EncodingType::SimdForBitpack:
      return tryEncode<SimdForBitpackEncoding<Storage>, Storage>(
          sectionData, options);
    case EncodingType::PFOR:
      return tryEncode<PFOREncoding<Storage>, Storage>(sectionData, options);
    case EncodingType::BlockBitPacking:
      return tryEncode<BlockBitPackingEncoding<Storage>, Storage>(
          sectionData, options);
    case EncodingType::Delta:
      return tryEncode<DeltaEncoding<Storage>, Storage>(sectionData, options);
    case EncodingType::FOR:
      return tryEncode<ForEncoding<Storage>, Storage>(sectionData, options);
    case EncodingType::FrequencyPartition:
      return tryEncode<FrequencyPartitionEncoding<Storage>, Storage>(
          sectionData, options);
    case EncodingType::Huffman:
      return tryEncode<HuffmanEncoding<Storage>, Storage>(sectionData, options);
    case EncodingType::DeltaBlock:
      return tryEncode<DeltaBlockEncoding<Storage>, Storage>(
          sectionData, options);
    default:
      return std::numeric_limits<size_t>::max();
  }
}

// Narrows one bit-range slice to the storage type the writer would give it and
// hands the result to `fn`. Mirrors SubIntSplitEncoding's storage-width switch,
// including the plain static_cast it narrows with: the slice is already masked
// to its bit range, so the cast cannot lose a bit.
template <typename Fn>
auto withNarrowedSection(
    int width,
    const std::vector<uint64_t>& sectionU64,
    velox::memory::MemoryPool& pool,
    Fn&& fn) {
  const auto build = [&]<typename Storage>(std::type_identity<Storage>) {
    Vector<Storage> narrowed{&pool};
    narrowed.resize(sectionU64.size());
    for (size_t i = 0; i < sectionU64.size(); ++i) {
      narrowed[i] = static_cast<Storage>(sectionU64[i]);
    }
    return fn(narrowed);
  };
  switch (storageWidthBits(width)) {
    case 8:
      return build(std::type_identity<uint8_t>{});
    case 16:
      return build(std::type_identity<uint16_t>{});
    case 32:
      return build(std::type_identity<uint32_t>{});
    default:
      return build(std::type_identity<uint64_t>{});
  }
}

// The sample sizes to sweep, smallest first, deduplicated. A 0 entry means
// the whole column and is kept last however it was written, since it is the
// limit of the series rather than a point in it.
std::vector<size_t> parseSampleSizes(const std::string& spec) {
  std::vector<size_t> sizes;
  bool wantsFullColumn = false;
  std::string field;
  std::istringstream stream{spec};
  while (std::getline(stream, field, ',')) {
    const auto begin = field.find_first_not_of(" \t");
    if (begin == std::string::npos) {
      continue;
    }
    const auto value = std::stoull(field.substr(begin));
    if (value == 0) {
      wantsFullColumn = true;
      continue;
    }
    sizes.push_back(static_cast<size_t>(value));
  }
  std::sort(sizes.begin(), sizes.end());
  sizes.erase(std::unique(sizes.begin(), sizes.end()), sizes.end());
  if (wantsFullColumn) {
    sizes.push_back(0);
  }
  return sizes;
}

// Every encoding bestCostBitsRestricted scores, so that "the model's pick" and
// "the oracle's pick" range over the same inventory.
//
// They did not before: the oracle offered seven encodings while the model
// minimised over fifteen, so every range the model awarded to one of the other
// eight counted as a disagreement whatever the model had estimated. The
// reported top-1 accuracy was measuring the gap between the two lists at least
// as much as it was measuring the model.
inline std::vector<CandidateEncoding> candidateEncodings() {
  return {
      {"Trivial", EncodingType::Trivial},
      {"FixedBitWidth", EncodingType::FixedBitWidth},
      {"Constant", EncodingType::Constant},
      {"MainlyConstant", EncodingType::MainlyConstant},
      {"Dictionary", EncodingType::Dictionary},
      {"RLE", EncodingType::RLE},
      {"Varint", EncodingType::Varint},
      {"SimdForBitpack", EncodingType::SimdForBitpack},
      {"PFOR", EncodingType::PFOR},
      {"BlockBitPacking", EncodingType::BlockBitPacking},
      {"Delta", EncodingType::Delta},
      {"FOR", EncodingType::FOR},
      {"FrequencyPartition", EncodingType::FrequencyPartition},
      {"Huffman", EncodingType::Huffman},
      {"DeltaBlock", EncodingType::DeltaBlock},
  };
}

struct OracleResult {
  // What this encoding really costs on this range.
  size_t bytes{std::numeric_limits<size_t>::max()};
  // What EncodingSizeEstimation told selection it would cost, before the read
  // factor. The pair is the whole diagnosis: an encoding is chosen on the
  // estimate and paid for in the bytes, so wherever the two diverge selection
  // is buying something other than what it was quoted.
  size_t estimateBytes{std::numeric_limits<size_t>::max()};
};

struct OracleCell {
  // Smallest bytes any one candidate reached on this range.
  size_t bestBytes{std::numeric_limits<size_t>::max()};
  EncodingType bestEncoding{EncodingType::Trivial};
  // Bytes of the candidate nested selection would actually pick for this range
  // -- argmin of estimateSize times readFactor, not argmin of measured bytes.
  //
  // These two differ, and the difference is the point of measuring both. A plan
  // chosen on bestBytes has been optimised for an encoder that does not turn
  // up: SimdForBitpack wins more cells than any other encoding and selection
  // picks it zero times, because FixedBitWidth's estimate omits the seven-byte
  // FixedBitArray slop its encode always writes while the two are otherwise
  // byte-identical. bestBytes is what the inventory could deliver under perfect
  // selection; selectionBytes is what the writer will deliver.
  size_t selectionBytes{std::numeric_limits<size_t>::max()};
  EncodingType selectionEncoding{EncodingType::Trivial};
  // What selection was quoted for the candidate it picks. Unlike the two byte
  // counts above, a planner could know this without encoding anything, so a
  // DP over it is a plan the writer could really compute: the planner costing
  // ranges with the estimators selection uses instead of its own models.
  size_t selectionEstimateBytes{std::numeric_limits<size_t>::max()};
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
double spearmanRho(const std::vector<double>& a, const std::vector<double>& b) {
  const size_t n = a.size();
  if (n < 3) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  auto rankOf = [n](const std::vector<double>& v) {
    std::vector<size_t> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&](size_t i, size_t j) {
      return v[i] < v[j];
    });
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

// Oracle DP over the measured grid, minimising the same objective the model's
// DP minimises.
//
// `costScale` is fullCount/sampleSize, the factor selectSplitsImpl applies to
// a per-sample cell cost, and `splitPenaltyBytes` is the model's split penalty
// converted from bits. Both are needed for the two plans to be comparable: the
// oracle used to run with no penalty at all while the model ran with 10 bits,
// so part of every measured plan difference was the two DPs optimising
// different things rather than the model being wrong. Scaling matters for the
// same reason -- a cell's fixed header is charged once per segment whatever
// the row count, so the scale is what decides how many segments either DP is
// willing to pay for.
struct OracleSegment {
  int bitStart{0};
  int bitEnd{0};
  EncodingType encoding{EncodingType::Trivial};
  size_t bytes{0};
};

struct OracleDpResult {
  std::vector<OracleSegment> sections;
  size_t totalBytes{0};
};

/// Which per-cell cost the oracle DP minimises.
enum class OracleObjective {
  /// The smallest bytes any candidate reached. Optimal for an inventory whose
  /// selection is perfect, which is not the selection the writer runs.
  kCellMinimum,
  /// The bytes of the candidate selection will actually choose. Optimal for
  /// what the writer delivers, and so the plan to hold a model plan against.
  kSelectionRealistic,
  /// Selection's estimate for the candidate it picks. Not an oracle: it uses
  /// no measured bytes, so it is the plan a planner costing ranges with
  /// EncodingSizeEstimation instead of SubIntSplitCostModels would choose.
  kSelectionEstimate,
};

OracleDpResult oracleDp(
    const std::vector<std::vector<OracleCell>>& grid,
    int sz,
    double costScale,
    double splitPenaltyBytes,
    OracleObjective objective = OracleObjective::kCellMinimum) {
  const auto cellBytes = [objective](const OracleCell& cell) {
    switch (objective) {
      case OracleObjective::kCellMinimum:
        return cell.bestBytes;
      case OracleObjective::kSelectionRealistic:
        return cell.selectionBytes;
      case OracleObjective::kSelectionEstimate:
        return cell.selectionEstimateBytes;
    }
    return cell.bestBytes;
  };
  const auto cellEncoding = [objective](const OracleCell& cell) {
    return objective == OracleObjective::kCellMinimum ? cell.bestEncoding
                                                      : cell.selectionEncoding;
  };
  std::vector<double> dp(sz + 1, std::numeric_limits<double>::infinity());
  std::vector<int> prev(sz + 1, -1);
  dp[0] = 0.0;
  for (int i = 1; i <= sz; ++i) {
    for (int j = 0; j < i; ++j) {
      const auto& cell = grid[j][i - 1];
      if (cellBytes(cell) == std::numeric_limits<size_t>::max()) {
        continue;
      }
      const double splitCost = (j == 0) ? 0.0 : splitPenaltyBytes;
      const double candidate =
          dp[j] + static_cast<double>(cellBytes(cell)) * costScale + splitCost;
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
  int idx = sz;
  while (idx > 0) {
    const int start = prev[idx];
    if (start < 0) {
      break;
    }
    const auto& cell = grid[start][idx - 1];
    result.sections.push_back(
        {start, idx - 1, cellEncoding(cell), cellBytes(cell)});
    // Reported unscaled, so the number stays a count of bytes the oracle
    // actually measured rather than a projection of them.
    result.totalBytes += cellBytes(cell);
    idx = start;
  }
  std::reverse(result.sections.begin(), result.sections.end());
  return result;
}

// The preserve-mode config that pins a SubIntSplit encode to `segments`.
inline EncodingLayout::Config planConfigFor(
    const std::vector<SectionPlan>& segments) {
  return EncodingLayout::Config{{
      {std::string(kSplitModeConfigKey), std::string(kSplitModePreserve)},
      {std::string(kSplitBoundariesConfigKey),
       serializeSplitBoundaries(segments)},
  }};
}

// Encodes the whole column and returns the real encoded byte count, or nullopt
// if the encode throws. An empty `segments` lets the encoder derive its own
// split, which is what the writer does; a non-empty one pins it to that plan.
//
// The point of the driver is to audit a sample-derived estimate, so the audit
// cannot itself be sample-derived: this is what the column costs, not what a
// prefix of it costs scaled up.
//
// The plan is applied, never re-derived. SubIntSplitEncoding's preserve mode
// takes the boundaries from the selection config and skips its own sampler and
// DP entirely, so what is measured is the given plan and not a second run of
// the planner. Section encodings are still chosen by real nested selection,
// which is the point -- the plan fixes where the splits fall and the writer
// decides the rest.
//
// Every part of the encode except the plan comes from encodeWithCompression,
// the same function the compression driver encodes through. This is what makes
// the two numbers comparable at all: policy, compression wiring, statistics and
// options are then shared by construction rather than by two call sites
// happening to agree. An earlier version of this built its own
// EncodingSelection with a ManualEncodingSelectionPolicy at the top, which gave
// each section the SubIntSplit-augmented candidate list while the driver's
// policy gives it only the default eight, so the two were encoding different
// things and neither number could be checked against the other.
template <typename Elem>
std::optional<size_t> encodeColumn(
    const Vector<Elem>& column,
    const std::vector<SectionPlan>& segments,
    const facebook::nimble::Encoding::Options& columnOptions) {
  if (column.empty()) {
    return std::nullopt;
  }
  try {
    auto& pool = benchmarks::benchmarkPool();
    Buffer buffer{*pool};
    const auto encoded = encodeWithCompression<SubIntSplitEncoding<Elem>, Elem>(
        buffer,
        column,
        parseCompressionType(FLAGS_mlidc_substream_compression),
        columnOptions,
        /*realNestedSelection=*/true,
        segments.empty() ? EncodingLayout::Config{} : planConfigFor(segments));
    return encoded.size();
  } catch (...) {
    return std::nullopt;
  }
}

// The plan SubIntSplitEncoding derives for itself in recompute mode.
//
// Mirrors its own recompute branch exactly -- the default sampler config, the
// column's row count as the DP's fullCount, the caller's allowed set, and
// allowHuffman taken from the options -- so that pinning an encode to this plan
// and letting the encoder derive its own must produce identical bytes. That
// equality is what the writer-reproduction check in runBenchmark asserts.
template <typename Phys>
std::vector<SectionPlan> writerDerivedPlan(
    std::span<const Phys> physical,
    int kBits,
    const facebook::nimble::Encoding::Options& columnOptions) {
  std::vector<uint64_t> writerSample;
  sampleIntoU64<Phys>(physical, writerSample, defaultSamplerConfig());
  auto writerCfg = defaultSelectorConfig();
  writerCfg.allowHuffman = columnOptions.subIntSplitAllowHuffman;
  writerCfg.allowDeltaBlock = columnOptions.subIntSplitAllowDeltaBlock;
  return selectSplitsRestricted(
             writerSample,
             kBits,
             physical.size(),
             columnOptions.subIntSplitAllowedEncodings,
             writerCfg)
      .sections;
}

// The oracle's segments as SectionPlan, so both plans reach encodeColumn and
// serializeSplitBoundaries the same way.
inline std::vector<SectionPlan> toSegmentPlans(
    const std::vector<OracleSegment>& segments) {
  std::vector<SectionPlan> plans;
  plans.reserve(segments.size());
  for (const auto& segment : segments) {
    SectionPlan plan;
    plan.bitStart = segment.bitStart;
    plan.bitEnd = segment.bitEnd;
    plan.encoding = segment.encoding;
    plans.push_back(plan);
  }
  return plans;
}

// A segmentation of the bit positions as inclusive [bitStart, bitEnd] ranges,
// lowest bit first.
using RangePlan = std::vector<std::pair<int, int>>;

// The k cheapest segmentations of [0, sz) under `rangeCost`, cheapest first,
// charging `splitPenalty` per boundary as selectSplitsImpl does.
//
// The writer's DP keeps one predecessor per position and trusts its argmin.
// On the grid that argmin names the truly cheapest encoding for a range a
// fifth of the time, so the plan it returns is a guess worth checking rather
// than an answer. Keeping k predecessors per position is what lets a planner
// hand a shortlist to a more expensive scorer. Ranges whose cost is not finite
// are never used.
template <typename RangeCost>
std::vector<RangePlan> kBestSegmentations(
    int sz,
    size_t k,
    double splitPenalty,
    RangeCost&& rangeCost) {
  struct Entry {
    double cost;
    int prevPosition;
    size_t prevRank;
  };
  std::vector<std::vector<Entry>> best(sz + 1);
  best[0].push_back({0.0, -1, 0});
  for (int end = 1; end <= sz; ++end) {
    std::vector<Entry> candidates;
    for (int start = 0; start < end; ++start) {
      const double range = rangeCost(start, end - 1);
      if (!std::isfinite(range)) {
        continue;
      }
      const double penalty = start == 0 ? 0.0 : splitPenalty;
      for (size_t rank = 0; rank < best[start].size(); ++rank) {
        candidates.push_back(
            {best[start][rank].cost + range + penalty, start, rank});
      }
    }
    const size_t keep = std::min(k, candidates.size());
    std::partial_sort(
        candidates.begin(),
        candidates.begin() + keep,
        candidates.end(),
        [](const Entry& a, const Entry& b) { return a.cost < b.cost; });
    candidates.resize(keep);
    best[end] = std::move(candidates);
  }
  std::vector<RangePlan> plans;
  for (size_t rank = 0; rank < best[sz].size(); ++rank) {
    RangePlan plan;
    int position = sz;
    size_t atRank = rank;
    while (position > 0) {
      const Entry& entry = best[position][atRank];
      plan.emplace_back(entry.prevPosition, position - 1);
      position = entry.prevPosition;
      atRank = entry.prevRank;
    }
    std::reverse(plan.begin(), plan.end());
    plans.push_back(std::move(plan));
  }
  return plans;
}

} // namespace
} // namespace facebook::nimble::mlidc

namespace facebook::nimble::mlidc {
namespace {

// The whole driver body, templated on the element type. main() picks the
// type from --mlidc_dtype and dispatches here.
template <typename Elem>
int runBenchmark() {
  // SubIntSplit splits the physical bit pattern, so the bit grid is as wide as
  // the physical type: 32 bits for the 4-byte types, not a fixed 64.
  using Phys = typename TypeTraits<Elem>::physicalType;
  constexpr int kBits = sizeof(Phys) * 8;

  const uint32_t n = static_cast<uint32_t>(FLAGS_mlidc_rows);
  const uint64_t seed = static_cast<uint64_t>(FLAGS_mlidc_seed);

  auto datasets = defaultDatasets<Elem>();
  auto candidates = candidateEncodings();

  // The options a column is written under, and the options a section of it is
  // encoded under, the latter taken from the writer's own derivation rather
  // than restated here so the two cannot drift. Per-cell measurements use the
  // section options; a full-column plan encode gets the column options and
  // derives its own.
  facebook::nimble::Encoding::Options columnOptions;
  columnOptions.subIntSplitAllowHuffman = FLAGS_allow_huffman;
  columnOptions.subIntSplitAllowDeltaBlock = FLAGS_allow_delta_block;
  const facebook::nimble::Encoding::Options sectionOptions =
      sectionEncodingOptions(columnOptions);

  // The inventory both the model and the oracle score. Built from the
  // candidate list so the two are matched by construction, minus Huffman
  // unless it was asked for: production withdraws Huffman from the planner
  // (Encoding::Options::subIntSplitAllowHuffman), and an oracle that keeps
  // scoring it would charge the model for passing over a range Huffman wins
  // when passing over it is exactly what production does.
  AllowedEncodings allowed;
  for (const auto& candidate : candidates) {
    if (candidate.type == EncodingType::Huffman && !FLAGS_allow_huffman) {
      continue;
    }
    allowed.insert(candidate.type);
  }

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
    std::cout << "Sample sizes:\n";
    for (const size_t size : parseSampleSizes(FLAGS_sample_sizes)) {
      std::cout << "  " << (size == 0 ? n : static_cast<uint32_t>(size))
                << (size == 0 ? " (whole column)" : "") << "\n";
    }
    return 0;
  }

  std::vector<std::string> csvColumns = {
      "driver",
      "dtype",
      "dataset",
      "N",
      "seed",
      "sample_size",
      "min_segment_width",
      "l",
      "r",
      "width",
      "encoding",
      "selection_est_bytes",
      "selection_est_ratio",
      "selection_encoding",
      "selection_bytes",
      "planner_encoding",
      "planner_cell_bits",
      "planner_selection_agree",
      "planner_cell_ratio",
      "planner_agreement_rate",
      "planner_cell_ratio_median",
      "available_at_width",
      "max_delta",
      "sample_seam_count",
      "oracle_win_count",
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
      "oracle_cell_sum_bytes",
      "full_column_vs_cell_sum",
      "plan_unresolved_segments",
      "full_column_bytes",
      "full_column_rows",
      "full_column_bits_per_elem",
      "writer_recompute_bytes",
      "writer_preserve_bytes",
      "writer_reproduced",
      "cost_scale",
      "split_penalty_bits",
      "search_space",
      "unavailable_model_picks",
      "allow_huffman",
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
  const std::vector<size_t> sampleSizes = parseSampleSizes(FLAGS_sample_sizes);
  NIMBLE_CHECK(
      !sampleSizes.empty(), "No usable sample sizes: {}", FLAGS_sample_sizes);
  SelectorConfig selectorCfg = defaultSelectorConfig();
  selectorCfg.allowHuffman = FLAGS_allow_huffman;
  selectorCfg.allowDeltaBlock = FLAGS_allow_delta_block;
  const MetricFlags requiredFlags = allCostModelRequiredFlags();

  for (const auto& ds : datasets) {
    std::cout << "== Dataset: " << ds.name << " ==\n";
    auto data = ds.generate(n, seed);

    // Sample the physical bit pattern; see the note in
    // MlIdAblationBenchmark.cpp. Bit-range analysis is only meaningful over
    // the bits the encoding actually splits.
    auto physical = std::span<const Phys>(
        reinterpret_cast<const Phys*>(data.data()), data.size());

    // Before anything is measured: check that pinning an encode to a plan
    // reproduces the encode that derives the same plan for itself.
    //
    // Everything this driver reports about a plan rests on encodeColumn
    // measuring the plan it was handed. If preserve mode were not taking the
    // boundaries, or they did not round-trip through their string form, or
    // writerDerivedPlan had drifted from the encoder's own recompute branch,
    // every full-column number would still look plausible and would mean
    // nothing. So the equality is asserted rather than assumed, and it is an
    // equality on bytes with no tolerance: the same plan through the same
    // encoder is the same output, or the check has found something.
    //
    // This instrument has produced plausible wrong numbers three times. The
    // point of the check being here is that a fourth announces itself rather
    // than waiting to be noticed.
    const auto writerPlan =
        writerDerivedPlan<Phys>(physical, kBits, columnOptions);
    const auto recomputeBytes =
        encodeColumn<Elem>(data, std::vector<SectionPlan>{}, columnOptions);
    const auto preserveBytes =
        encodeColumn<Elem>(data, writerPlan, columnOptions);
    const bool writerReproduced = recomputeBytes.has_value() &&
        preserveBytes.has_value() &&
        recomputeBytes.value() == preserveBytes.value();

    csv.beginRow();
    csv.set("driver", "bench_costmodel_oracle");
    csv.set("dtype", elemTypeName<Elem>());
    csv.set("dataset", ds.name);
    csv.set("N", static_cast<int64_t>(n));
    csv.set("seed", static_cast<int64_t>(seed));
    csv.set("plan_type", "writer_reproduction");
    csv.set("plan_segment_count", static_cast<int64_t>(writerPlan.size()));
    csv.set("full_column_rows", static_cast<int64_t>(data.size()));
    if (recomputeBytes.has_value()) {
      csv.set(
          "writer_recompute_bytes",
          static_cast<int64_t>(recomputeBytes.value()));
    }
    if (preserveBytes.has_value()) {
      csv.set(
          "writer_preserve_bytes", static_cast<int64_t>(preserveBytes.value()));
    }
    csv.set("writer_reproduced", writerReproduced ? int64_t{1} : int64_t{0});
    csv.set("skipped", writerReproduced ? int64_t{0} : int64_t{1});
    csv.endRow();
    csv.flush();

    if (!writerReproduced) {
      std::cerr << "  [FAIL] scoring a pinned plan does not reproduce the "
                   "writer on "
                << ds.name << ": recompute="
                << (recomputeBytes.has_value()
                        ? std::to_string(recomputeBytes.value())
                        : std::string("n/a"))
                << " preserve="
                << (preserveBytes.has_value()
                        ? std::to_string(preserveBytes.value())
                        : std::string("n/a"))
                << " -- every full-column number below would be measuring "
                   "something other than the plan it names\n";
      if (!FLAGS_ignore_writer_mismatch) {
        std::cerr
            << "  Pass --ignore_writer_mismatch to record the run anyway.\n";
        return 1;
      }
    } else {
      std::cout << "  writer reproduction ok: " << recomputeBytes.value()
                << " B over " << writerPlan.size() << " segments ("
                << (static_cast<double>(recomputeBytes.value()) * 8.0 /
                    static_cast<double>(data.size()))
                << " bits/elem)\n";
    }

    // Every sample size asked for, so a run produces a convergence series
    // rather than a point. What the series answers: whether the model's plan
    // stops changing as the sample grows, whether the oracle's does, and
    // whether the gap between them closes. A gap that survives to the whole
    // column is model error; one that shrinks toward zero was the instrument
    // sampling badly, and the smaller-sample numbers were overstating the
    // model's fault.
    //
    // What the series cannot answer: whether either DP's search space holds
    // the true optimum. Both search contiguous bit ranges subject to
    // min_segment_width, so a partition neither can express leaves them wrong
    // together by a margin no sample size reveals. The oracle is optimal
    // within that search space and the output says so rather than calling it
    // optimal.
    for (const size_t sampleTarget : sampleSizes) {
      samplerCfg.maxSamples = (sampleTarget == 0) ? n : sampleTarget;
      std::vector<uint64_t> samples;
      sampleIntoU64(physical, samples, samplerCfg);
      const size_t sampleSize = samples.size();
      if (sampleSize == 0) {
        std::cerr << "  [SKIP] empty sample\n";
        continue;
      }

      // How many window boundaries the sampler leaves in the sample.
      //
      // sampleIntoU64 draws contiguous windows at blockStride, and the metric
      // walk counts the pair spanning two windows as an ordinary adjacent
      // pair. Those pairs are artefacts: one spans blockStride rows of the
      // real column. Their number changes with the sample size, so a
      // delta-family metric can move across the sweep for a reason that is
      // not statistical, and a convergence that is really seams thinning out
      // would read as sampling error going away. Emitted alongside max_delta
      // so the two can be checked against each other before any such
      // convergence is believed. Zero once the windows tile, which is what
      // the whole-column entry does.
      const size_t sampleBlocks = samplerCfg.blockSize > 0
          ? std::max<size_t>(1, sampleSize / samplerCfg.blockSize)
          : 1;
      const size_t blockStride = std::max<size_t>(1, n / sampleBlocks);
      const size_t sampleSeams =
          (samplerCfg.blockSize > 0 && blockStride > samplerCfg.blockSize)
          ? sampleBlocks - 1
          : 0;

      std::cout << "  -- sample_size=" << sampleSize << " seams=" << sampleSeams
                << "\n";

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
      // Times the model named an encoding the writer could not have given this
      // section. Counted rather than assumed: the claim that the model never
      // prices an unavailable encoding rests on varintCostBits' own width gate
      // being at least as strict as EncodingFactory's, and a claim like that is
      // worth a counter rather than a comment.
      int unavailablePickCount = 0;
      // Cells each encoding was the smallest measured on, so the trend across
      // sweep sizes is readable without post-processing. The delta family is
      // the reason: a sampled grid counts window seams as real adjacencies, and
      // delta-family models read a global maximum over them, so an encoding
      // taking no cells at 2048 and many at full column is the seam distortion
      // lifting rather than the data changing.
      std::vector<int> oracleWinCounts(candidates.size(), 0);
      // How far the split planner's own cost models sit from what the writer
      // delivers for the same bit range.
      //
      // The planner minimises its per-cell cost to choose boundaries, and the
      // writer then encodes each section with whatever nested selection picks.
      // Those are two independent sets of models: the planner's are hand-rolled
      // in SubIntSplitCostModels.h, selection's are the encodings' own
      // estimateSize. Where the planner's cell cost tracks what selection
      // delivers, its models are earning their place. Where it does not, the
      // boundaries are being chosen against a cost nothing pays, and the
      // question stops being which model to correct and becomes whether the
      // planner should score ranges with its own models at all.
      // Time spent building Statistics and calling EncodingSizeEstimation over
      // the grid: what costing ranges with the estimators would add to the
      // writer's planning, since the encodes around it are oracle-only work.
      uint64_t estimatorCostingNanos = 0;
      // The same for the planner's own models, for comparison.
      uint64_t costModelNanos = 0;
      int plannerAgreeCount = 0;
      int plannerComparableCells = 0;
      std::vector<double> plannerCellRatios;
      plannerCellRatios.reserve(static_cast<size_t>(kBits) * kBits);
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
          // The width the writer would narrow this section to, which is what
          // decides the inventory available to it.
          const int storageBytes = storageWidthBits(width) / 8;

          // Cost model metrics + per-encoding estimates.
          const auto costModelStart = std::chrono::steady_clock::now();
          const SectionMetrics metrics =
              collector.compute(sectionU64, requiredFlags);
          EncodingType modelBestEnc = EncodingType::Trivial;
          const double modelBestBits = bestCostBitsRestricted(
              metrics,
              sampleSize,
              n,
              width,
              sectionU64,
              allowed,
              FLAGS_allow_huffman,
              FLAGS_allow_delta_block,
              modelBestEnc);
          costModelNanos += static_cast<uint64_t>(
              std::chrono::duration_cast<std::chrono::nanoseconds>(
                  std::chrono::steady_clock::now() - costModelStart)
                  .count());

          ModelCell& mc = modelGrid[l][r];
          mc.bestBits = modelBestBits;
          mc.bestEncoding = modelBestEnc;
          mc.estBits.resize(candidates.size());
          // The viability gates bestCostBitsRestricted applies before it
          // considers an encoding at all. Repeated here so a per-encoding
          // estimate agrees with the pick taken from the same models: without
          // them a gated-out Dictionary would report the lowest est_bits in the
          // row while is_model_pick stayed zero, which reads as a bug in the
          // driver rather than as the model declining to offer it.
          const bool dictionaryViable = metrics.uniqueCount > 0 &&
              estimatedStreamUniqueCount(metrics, sampleSize, width, n) <
                  static_cast<double>(n) / 2.0;
          const bool frequencyPartitionViable = metrics.uniqueCount > 0 &&
              !metrics.uniqueCountCapped && metrics.uniqueCount <= 1024;
          const bool huffmanViable = FLAGS_allow_huffman &&
              metrics.uniqueCount > 0 && !metrics.uniqueCountCapped &&
              metrics.uniqueCount <= HuffmanEncoding<uint64_t>::kMaxSymbols;
          constexpr double kUnavailable =
              std::numeric_limits<double>::infinity();
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
                bits = dictionaryViable
                    ? dictionaryCostBits(metrics, sampleSize, n, width)
                    : kUnavailable;
                break;
              case EncodingType::RLE:
                bits = rleCostBits(metrics, sampleSize, width);
                break;
              case EncodingType::Varint:
                bits = varintCostBits(metrics, sampleSize, width);
                break;
              case EncodingType::SimdForBitpack:
                bits = simdForBitpackCostBits(metrics, sampleSize, width);
                break;
              case EncodingType::PFOR:
                bits = pforCostBits(metrics, sampleSize, width);
                break;
              case EncodingType::BlockBitPacking:
                bits = blockBitPackingCostBits(sectionU64, sampleSize);
                break;
              case EncodingType::Delta:
                bits = deltaCostBits(metrics, sampleSize, width);
                break;
              case EncodingType::FOR:
                bits = forCostBits(metrics, sampleSize, width);
                break;
              case EncodingType::FrequencyPartition:
                bits = frequencyPartitionViable
                    ? frequencyPartitionCostBits(metrics, sampleSize, width)
                    : kUnavailable;
                break;
              case EncodingType::Huffman:
                bits = huffmanViable ? huffmanCostBits(sectionU64, sampleSize)
                                     : kUnavailable;
                break;
              case EncodingType::DeltaBlock:
                bits = deltaBlockCostBits(sectionU64, sampleSize);
                break;
              default:
                bits = kUnavailable;
            }
            mc.estBits[ci] = bits;
          }

          // Oracle: actually encode with each candidate, measure bytes.
          OracleCell& oc = oracleGrid[l][r];
          oc.results.resize(candidates.size());
          withNarrowedSection(
              width, sectionU64, *pool, [&](const auto& sectionData) {
                using Storage = std::decay_t<decltype(sectionData.data()[0])>;
                const auto values = std::span<const Storage>(
                    sectionData.data(), sectionData.size());
                const auto timed = [&estimatorCostingNanos](auto&& fn) {
                  const auto start = std::chrono::steady_clock::now();
                  auto result = fn();
                  estimatorCostingNanos += static_cast<uint64_t>(
                      std::chrono::duration_cast<std::chrono::nanoseconds>(
                          std::chrono::steady_clock::now() - start)
                          .count());
                  return result;
                };
                // Selection reads its estimates off Statistics, so build it
                // once for the range rather than per candidate. The lazy
                // aggregates it computes are charged to whichever estimate
                // first asks for them, all inside the timed region.
                const auto statistics =
                    timed([&] { return Statistics<Storage>::create(values); });
                // FixedBitWidth's estimate for this range, which is what
                // effectiveReadFactor needs to decide whether Trivial's
                // discount is earned. Computed once, as select() does.
                const auto fixedBitWidthEstimate =
                    encodingAvailableAtWidth(
                        EncodingType::FixedBitWidth, storageBytes)
                    ? timed([&] {
                        return facebook::nimble::detail::
                            EncodingSizeEstimation<Storage>::estimateSize(
                                EncodingType::FixedBitWidth,
                                values,
                                statistics,
                                sectionOptions);
                      })
                    : std::optional<uint64_t>{};
                double bestSelectionCost =
                    std::numeric_limits<double>::infinity();

                for (size_t ci = 0; ci < candidates.size(); ++ci) {
                  if (!encodingAvailableAtWidth(
                          candidates[ci].type, storageBytes)) {
                    continue;
                  }
                  const size_t bytes = oracleEncodeBytes(
                      candidates[ci].type, sectionData, sectionOptions);
                  // Measured for every candidate, so the size an encoding would
                  // have reached stays in the CSV even where the inventory
                  // withholds it -- that number is what the cost of withholding
                  // it is read from.
                  oc.results[ci].bytes = bytes;
                  if (bytes < oc.bestBytes &&
                      allowed.count(candidates[ci].type) > 0) {
                    oc.bestBytes = bytes;
                    oc.bestEncoding = candidates[ci].type;
                  }

                  // What nested selection would pick for this range: the same
                  // estimateSize times readFactor comparison
                  // ManualEncodingSelectionPolicy::select runs, over the same
                  // candidate list a section is offered. No extra encode -- the
                  // bytes are the ones just measured, and only the argmin
                  // changes.
                  const auto readFactor =
                      sectionReadFactor(candidates[ci].type);
                  if (!readFactor.has_value() ||
                      bytes == std::numeric_limits<size_t>::max()) {
                    continue;
                  }
                  const auto estimate = timed([&] {
                    return facebook::nimble::detail::
                        EncodingSizeEstimation<Storage>::estimateSize(
                            candidates[ci].type,
                            values,
                            statistics,
                            sectionOptions);
                  });
                  if (!estimate.has_value()) {
                    continue;
                  }
                  oc.results[ci].estimateBytes = estimate.value();
                  // Weighted through effectiveReadFactor rather than by the
                  // table value directly, so this tracks select()'s rule for
                  // withholding Trivial's discount instead of keeping a second
                  // copy of it. Without that, selection_encoding disagrees with
                  // the writer on exactly the cells where the rule fires --
                  // which are the cells anyone reads this column to study.
                  const double selectionCost =
                      static_cast<double>(estimate.value()) *
                      static_cast<double>(effectiveReadFactor(
                          candidates[ci].type,
                          readFactor.value(),
                          estimate.value(),
                          fixedBitWidthEstimate));
                  if (selectionCost < bestSelectionCost) {
                    bestSelectionCost = selectionCost;
                    oc.selectionBytes = bytes;
                    oc.selectionEncoding = candidates[ci].type;
                    oc.selectionEstimateBytes = estimate.value();
                  }
                }
              });

          // Per-cell comparisons.
          ++cellCount;
          if (!encodingAvailableAtWidth(mc.bestEncoding, storageBytes)) {
            ++unavailablePickCount;
          }
          for (size_t ci = 0; ci < candidates.size(); ++ci) {
            if (oc.bestBytes != std::numeric_limits<size_t>::max() &&
                candidates[ci].type == oc.bestEncoding) {
              ++oracleWinCounts[ci];
              break;
            }
          }
          if (oc.bestBytes != std::numeric_limits<size_t>::max() &&
              oc.bestEncoding == mc.bestEncoding) {
            ++agreeCount;
          }

          // Planner against selection, on the same range.
          double plannerCellRatio = std::numeric_limits<double>::quiet_NaN();
          const bool plannerComparable =
              oc.selectionBytes != std::numeric_limits<size_t>::max() &&
              std::isfinite(mc.bestBits) && mc.bestBits > 0.0;
          if (plannerComparable) {
            ++plannerComparableCells;
            if (mc.bestEncoding == oc.selectionEncoding) {
              ++plannerAgreeCount;
            }
            // Actual over quoted, the same convention as the per-encoding
            // ratios: above one means the planner under-quoted the range.
            plannerCellRatio =
                static_cast<double>(oc.selectionBytes) * 8.0 / mc.bestBits;
            plannerCellRatios.push_back(plannerCellRatio);
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
          std::sort(
              modelOrder.begin(), modelOrder.end(), [&](size_t a, size_t b) {
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
            csv.set("dtype", elemTypeName<Elem>());
            csv.set("dataset", ds.name);
            csv.set("N", static_cast<int64_t>(n));
            csv.set("seed", static_cast<int64_t>(seed));
            csv.set("sample_size", static_cast<int64_t>(sampleSize));
            csv.set(
                "min_segment_width",
                static_cast<int64_t>(selectorCfg.minSectionWidth));
            csv.set("l", static_cast<int64_t>(l));
            csv.set("r", static_cast<int64_t>(r));
            csv.set("width", static_cast<int64_t>(width));
            csv.set("encoding", candidates[ci].name);
            csv.set(
                "available_at_width",
                encodingAvailableAtWidth(candidates[ci].type, storageBytes)
                    ? int64_t{1}
                    : int64_t{0});
            // The largest rising step the delta family sizes its packed array
            // to. At the whole-column point this is the real maximum adjacent
            // delta; at a sampled point it also counts the pairs that span two
            // sampling windows, which are artefacts. Comparing the two across
            // the sweep is the measurement of how much the sampler inflates it.
            csv.set("max_delta", static_cast<int64_t>(metrics.maxDelta));
            csv.set("sample_seam_count", static_cast<int64_t>(sampleSeams));
            csv.set("has_cost_model", hasCostModel ? int64_t{1} : int64_t{0});
            if (hasCostModel) {
              csv.set("est_bits", mc.estBits[ci]);
            }
            if (oracleOk) {
              csv.set(
                  "actual_bytes", static_cast<int64_t>(oc.results[ci].bytes));
              // What nested selection was quoted for this encoding on this
              // range, and how far off it was. selection_est_ratio above 1
              // means the encoding costs more than selection was told, which is
              // the direction that lets a mispriced encoding win a section and
              // then blow it up.
              if (oc.results[ci].estimateBytes !=
                  std::numeric_limits<size_t>::max()) {
                csv.set(
                    "selection_est_bytes",
                    static_cast<int64_t>(oc.results[ci].estimateBytes));
                if (oc.results[ci].estimateBytes > 0) {
                  csv.set(
                      "selection_est_ratio",
                      static_cast<double>(oc.results[ci].bytes) /
                          static_cast<double>(oc.results[ci].estimateBytes));
                }
              }
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
            if (ci == 0) {
              for (const auto& candidate : candidates) {
                if (candidate.type == mc.bestEncoding) {
                  csv.set("planner_encoding", candidate.name);
                  break;
                }
              }
              if (std::isfinite(mc.bestBits)) {
                csv.set("planner_cell_bits", mc.bestBits);
              }
              if (plannerComparable) {
                csv.set(
                    "planner_selection_agree",
                    mc.bestEncoding == oc.selectionEncoding ? int64_t{1}
                                                            : int64_t{0});
                csv.set("planner_cell_ratio", plannerCellRatio);
              }
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

      // Median rather than mean: these ratios have a heavy right tail, and a
      // mean over them says more about the worst cell than about the typical
      // one. The per-encoding figures this is read beside are medians for the
      // same reason.
      double plannerRatioMedian = std::numeric_limits<double>::quiet_NaN();
      if (!plannerCellRatios.empty()) {
        const size_t middle = plannerCellRatios.size() / 2;
        std::nth_element(
            plannerCellRatios.begin(),
            plannerCellRatios.begin() + middle,
            plannerCellRatios.end());
        plannerRatioMedian = plannerCellRatios[middle];
      }
      const double plannerAgreementRate = plannerComparableCells > 0
          ? static_cast<double>(plannerAgreeCount) /
              static_cast<double>(plannerComparableCells)
          : 0.0;

      std::cout << "  top1_accuracy=" << top1Accuracy
                << " mean_spearman_rho=" << meanRho
                << " mean_abs_rel_err=" << meanAbsRelErr << "\n";
      std::cout << "  planner_vs_selection: agree=" << plannerAgreementRate
                << " cell_ratio_median=" << plannerRatioMedian << " over "
                << plannerComparableCells << " cells\n";

      // -----------------------------------------------------------------
      // Plan comparison.
      //
      // Four plans are scored, not two. The model's DP is run twice: once with
      // fullCount == sampleSize, which is what this driver used to do, and once
      // at the column's real row count, which is what production does. The two
      // are not the same plan. A cell's fixed per-section header is charged
      // once per segment whatever the row count, so at sample scale it weighs
      // fullCount/sampleSize times more against the per-value terms than it
      // really does -- and the number of segments is exactly what that tradeoff
      // decides. Reporting only the sample-scale plan measured a configuration
      // production never runs, and hid the one defect known to move segment
      // counts. Both are kept because the difference between them is the
      // evidence for that defect.
      //
      // The oracle DP is run at both scales too, so each model plan has an
      // oracle at its own scale to be read against.
      // -----------------------------------------------------------------
      const double fullCostScale =
          static_cast<double>(n) / static_cast<double>(sampleSize);
      const double splitPenaltyBytes = selectorCfg.splitPenalty / 8.0;

      const SelectorResult autoSampleScale = selectSplitsRestricted(
          samples, kBits, sampleSize, allowed, selectorCfg);
      const SelectorResult autoFullScale =
          selectSplitsRestricted(samples, kBits, n, allowed, selectorCfg);
      const OracleDpResult oracleSampleScale =
          oracleDp(oracleGrid, kBits, /*costScale=*/1.0, splitPenaltyBytes);
      const OracleDpResult oracleFullScale =
          oracleDp(oracleGrid, kBits, fullCostScale, splitPenaltyBytes);
      // The oracle for the cost the writer actually delivers. oracle_dp above
      // minimises the smallest bytes any candidate reached, which no assembled
      // plan receives, so it is a bound on the inventory rather than on any
      // plan. This one minimises the bytes of the candidate selection will pick
      // and is therefore the plan a model plan should be held against on
      // full_column_bytes.
      const OracleDpResult selectionOracle = oracleDp(
          oracleGrid,
          kBits,
          fullCostScale,
          splitPenaltyBytes,
          OracleObjective::kSelectionRealistic);
      // The planner costing ranges with selection's estimators. Uses no
      // measured bytes, so unlike the two oracles it is a plan the writer could
      // compute; scored on full_column_bytes like the model plans.
      const OracleDpResult estimatorDp = oracleDp(
          oracleGrid,
          kBits,
          fullCostScale,
          splitPenaltyBytes,
          OracleObjective::kSelectionEstimate);
      std::cout << "  costing_time: cost_models="
                << static_cast<double>(costModelNanos) / 1e6
                << " ms, estimators="
                << static_cast<double>(estimatorCostingNanos) / 1e6
                << " ms over " << kBits * (kBits + 1) / 2 << " ranges\n";

      // Scores one plan: its measured bytes on the sample, its regret against
      // the best each of its own ranges could have reached, and what the whole
      // column encodes to under it.
      const auto scorePlan = [&](const std::string& planType,
                                 const std::vector<SectionPlan>& segments) {
        size_t planSampleBytes = 0;
        size_t cellBestSum = 0;
        size_t regretBytes = 0;
        size_t unresolvedSegments = 0;

        for (const auto& seg : segments) {
          const auto& cell = oracleGrid[seg.bitStart][seg.bitEnd];
          size_t bytesForPick = std::numeric_limits<size_t>::max();
          for (size_t ci = 0; ci < candidates.size(); ++ci) {
            if (candidates[ci].type == seg.encoding) {
              bytesForPick = cell.results[ci].bytes;
              break;
            }
          }
          // A segment whose measured bytes cannot be obtained is counted, not
          // skipped. Skipping it left the plan total summing a subset of the
          // plan while the oracle total summed all of its own, so a plan could
          // look cheaper than the oracle by having had segments dropped out of
          // it. A nonzero count says outright that the total is not comparable.
          const bool resolved =
              bytesForPick != std::numeric_limits<size_t>::max();
          if (resolved) {
            planSampleBytes += bytesForPick;
            if (cell.bestBytes != std::numeric_limits<size_t>::max()) {
              cellBestSum += cell.bestBytes;
              if (bytesForPick > cell.bestBytes) {
                regretBytes += (bytesForPick - cell.bestBytes);
              }
            }
          } else {
            ++unresolvedSegments;
          }

          csv.beginRow();
          csv.set("driver", "bench_costmodel_oracle");
          csv.set("dtype", elemTypeName<Elem>());
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
          if (resolved) {
            csv.set("actual_bytes", static_cast<int64_t>(bytesForPick));
          }
          // What nested selection actually gives this section, beside what it
          // was quoted for it. For the autosis plan these rows are the
          // production per-section breakdown: the encoding each section lands
          // on, what it costs, and what selection was told it would cost.
          for (const auto& c : candidates) {
            if (c.type == cell.selectionEncoding) {
              csv.set("selection_encoding", c.name);
              break;
            }
          }
          if (cell.selectionBytes != std::numeric_limits<size_t>::max()) {
            csv.set(
                "selection_bytes", static_cast<int64_t>(cell.selectionBytes));
          }
          for (size_t ci = 0; ci < candidates.size(); ++ci) {
            if (candidates[ci].type != cell.selectionEncoding) {
              continue;
            }
            const auto estimated = cell.results[ci].estimateBytes;
            if (estimated != std::numeric_limits<size_t>::max()) {
              csv.set("selection_est_bytes", static_cast<int64_t>(estimated));
              if (estimated > 0 &&
                  cell.selectionBytes != std::numeric_limits<size_t>::max()) {
                csv.set(
                    "selection_est_ratio",
                    static_cast<double>(cell.selectionBytes) /
                        static_cast<double>(estimated));
              }
            }
            break;
          }
          csv.set("plan_type", planType);
          csv.set("plan_segment_count", static_cast<int64_t>(segments.size()));
          csv.set("skipped", resolved ? int64_t{0} : int64_t{1});
          csv.endRow();
        }

        const auto fullColumnBytes =
            encodeColumn<Elem>(data, segments, columnOptions);

        csv.beginRow();
        csv.set("driver", "bench_costmodel_oracle");
        csv.set("dtype", elemTypeName<Elem>());
        csv.set("dataset", ds.name);
        csv.set("N", static_cast<int64_t>(n));
        csv.set("seed", static_cast<int64_t>(seed));
        csv.set("sample_size", static_cast<int64_t>(sampleSize));
        csv.set("plan_type", planType + "_summary");
        csv.set("plan_segment_count", static_cast<int64_t>(segments.size()));
        csv.set(
            "plan_total_sample_bytes", static_cast<int64_t>(planSampleBytes));
        // The sum over this plan's ranges of the smallest bytes any single
        // encoding reached on each range measured on its own.
        //
        // This is what oracleDp minimises, and it is NOT a bound on
        // full_column_bytes. Two different things separate them, both in the
        // same direction. The assembled encode charges a SubIntSplit header and
        // a section directory that no cell sum contains, so it is larger. And a
        // cell records the minimum over every candidate, while an assembled
        // section is encoded with whatever nested selection picks --
        // estimateSize times readFactor, not smallest measured bytes -- so a
        // plan gets the encoding selection names, never the one the cell
        // recorded. SimdForBitpack is the sharp case: it wins the most cells of
        // any encoding and selection never picks it, so a plan optimised on
        // cell minima is optimised for an encoder that will not turn up.
        //
        // That is why a model plan can encode smaller than the oracle plan
        // without anything being wrong: the oracle's boundaries are optimal for
        // the cell sum, and the cell sum is not the cost the writer delivers.
        // Compare model against oracle on full_column_bytes, which is
        // like-for-like; do not read the difference between these two columns
        // as regret.
        csv.set("oracle_cell_sum_bytes", static_cast<int64_t>(cellBestSum));
        csv.set(
            "plan_unresolved_segments",
            static_cast<int64_t>(unresolvedSegments));
        csv.set("regret_sample_bytes", static_cast<int64_t>(regretBytes));
        csv.set("cost_scale", fullCostScale);
        csv.set("split_penalty_bits", selectorCfg.splitPenalty);
        csv.set("allow_huffman", FLAGS_allow_huffman ? int64_t{1} : int64_t{0});
        if (fullColumnBytes.has_value()) {
          csv.set(
              "full_column_bytes",
              static_cast<int64_t>(fullColumnBytes.value()));
          // Divided by the rows actually encoded, not the rows asked for. A
          // file-backed column can be shorter than --mlidc_rows, and dividing
          // by the request would quietly scale every bits/elem in the file.
          csv.set("full_column_rows", static_cast<int64_t>(data.size()));
          csv.set(
              "full_column_bits_per_elem",
              static_cast<double>(fullColumnBytes.value()) * 8.0 /
                  static_cast<double>(data.size()));
          // How far the assembled encode sits from the cell sum this plan was
          // scored on, with the cell sum lifted to the column's row count so
          // the two are in the same units. Above 1 is the normal direction --
          // header plus selection not picking the cell winner. Emitted rather
          // than left to be inferred, because its size is the size of the
          // objective mismatch between the two grids.
          if (cellBestSum > 0) {
            csv.set(
                "full_column_vs_cell_sum",
                static_cast<double>(fullColumnBytes.value()) /
                    (static_cast<double>(cellBestSum) * fullCostScale));
          }
        }
        csv.set("skipped", int64_t{0});
        csv.endRow();

        std::cout << "  " << planType << ": " << segments.size()
                  << " segments, sample_bytes=" << planSampleBytes
                  << ", regret=" << regretBytes << ", full_column_bytes=";
        if (fullColumnBytes.has_value()) {
          std::cout << fullColumnBytes.value() << " ("
                    << (static_cast<double>(fullColumnBytes.value()) * 8.0 /
                        static_cast<double>(data.size()))
                    << " bits/elem)";
        } else {
          std::cout << "n/a";
        }
        std::cout << "\n";
        if (unresolvedSegments > 0) {
          std::cout
              << "  [WARN] " << planType << " sample_bytes excludes "
              << unresolvedSegments
              << " segment(s) whose encoding could not be measured; it is "
                 "not comparable with the other plans\n";
        }
      };

      scorePlan("autosis_sample_scale", autoSampleScale.sections);
      scorePlan("autosis", autoFullScale.sections);
      scorePlan(
          "oracle_dp_sample_scale", toSegmentPlans(oracleSampleScale.sections));
      scorePlan("oracle_dp", toSegmentPlans(oracleFullScale.sections));
      scorePlan(
          "oracle_dp_selection", toSegmentPlans(selectionOracle.sections));
      // An empty plan would make encodeColumn fall back to the writer's own
      // plan and score shipped bytes under this name, so refuse it loudly.
      NIMBLE_CHECK(
          !estimatorDp.sections.empty(),
          "estimator_dp found no plan: {}",
          ds.name);
      scorePlan("estimator_dp", toSegmentPlans(estimatorDp.sections));

      // The hybrid planner: shortlist cheaply at this sample, then re-score
      // only the shortlisted ranges with selection's estimators on a larger
      // sample. Costing every range at the larger size is what the grid shows
      // is accurate and what is too slow to ship; a shortlist bounds it.
      if (FLAGS_hybrid_shortlist > 0) {
        const size_t k = static_cast<size_t>(FLAGS_hybrid_shortlist);
        const double splitPenaltyBits = selectorCfg.splitPenalty;
        std::vector<std::pair<std::string, RangePlan>> shortlist;
        const auto addPlans = [&](const std::string& source,
                                  std::vector<RangePlan> plans) {
          for (auto& plan : plans) {
            shortlist.emplace_back(source, std::move(plan));
          }
        };
        addPlans(
            "models",
            kBestSegmentations(kBits, k, splitPenaltyBits, [&](int l, int r) {
              return modelGrid[l][r].bestBits * fullCostScale;
            }));
        if (!FLAGS_hybrid_cheap_shortlist) {
          addPlans(
              "estimators",
              kBestSegmentations(kBits, k, splitPenaltyBits, [&](int l, int r) {
                const size_t bytes = oracleGrid[l][r].selectionEstimateBytes;
                return bytes == std::numeric_limits<size_t>::max()
                    ? std::numeric_limits<double>::infinity()
                    : static_cast<double>(bytes) * 8.0 * fullCostScale;
              }));
        }
        std::vector<bool> isCut;
        // Bit-flip hints as a shortlist source rather than a constraint.
        // Restricting the whole DP to the profile's gradient boundaries was
        // measured at 65% mean regret, so here they only nominate plans: the
        // k cheapest under the estimators that cut nowhere else.
        {
          const auto profileStatistics = Statistics<uint64_t>::create(
              std::span<const uint64_t>(samples.data(), samples.size()));
          isCut.assign(kBits + 1, false);
          isCut[0] = true;
          isCut[kBits] = true;
          for (const int boundary : bitFlipGradientBoundaries(
                   profileStatistics.bitFlipProfile(),
                   TopLevelPolicyConfig{})) {
            if (boundary > 0 && boundary < kBits) {
              isCut[boundary] = true;
            }
          }
          addPlans(
              "bitflip",
              kBestSegmentations(kBits, k, splitPenaltyBits, [&](int l, int r) {
                if (!isCut[l] || !isCut[r + 1]) {
                  return std::numeric_limits<double>::infinity();
                }
                if (FLAGS_hybrid_cheap_shortlist) {
                  return modelGrid[l][r].bestBits * fullCostScale;
                }
                const size_t bytes = oracleGrid[l][r].selectionEstimateBytes;
                return bytes == std::numeric_limits<size_t>::max()
                    ? std::numeric_limits<double>::infinity()
                    : static_cast<double>(bytes) * 8.0 * fullCostScale;
              }));
        }
        RangePlan writerRanges;
        for (const auto& segment : writerPlan) {
          writerRanges.emplace_back(segment.bitStart, segment.bitEnd);
        }
        shortlist.emplace_back("writer", std::move(writerRanges));

        const auto rescoreStart = std::chrono::steady_clock::now();
        SamplerConfig rescoreCfg = samplerCfg;
        rescoreCfg.maxSamples = std::min<size_t>(
            n, static_cast<size_t>(FLAGS_hybrid_rescore_samples));
        std::vector<uint64_t> rescoreSamples;
        sampleIntoU64(physical, rescoreSamples, rescoreCfg);
        const double rescoreScale =
            static_cast<double>(n) / static_cast<double>(rescoreSamples.size());

        // Selection's estimate, in bytes, for the candidate it would pick on
        // one range of the larger sample, with the encoding it names. Cached,
        // since shortlisted plans share most of their ranges.
        std::map<std::pair<int, int>, std::pair<double, EncodingType>> rescored;
        const auto rescoreRange = [&](int l, int r) {
          const auto key = std::make_pair(l, r);
          if (const auto it = rescored.find(key); it != rescored.end()) {
            return it->second;
          }
          const int width = r - l + 1;
          const uint64_t mask =
              width >= 64 ? ~uint64_t{0} : ((uint64_t{1} << width) - 1);
          std::vector<uint64_t> sectionU64(rescoreSamples.size());
          for (size_t i = 0; i < rescoreSamples.size(); ++i) {
            sectionU64[i] = (rescoreSamples[i] >> l) & mask;
          }
          const int storageBytes = storageWidthBits(width) / 8;
          std::pair<double, EncodingType> result{
              std::numeric_limits<double>::infinity(), EncodingType::Trivial};
          withNarrowedSection(
              width, sectionU64, *pool, [&](const auto& sectionData) {
                using Storage = std::decay_t<decltype(sectionData.data()[0])>;
                const auto values = std::span<const Storage>(
                    sectionData.data(), sectionData.size());
                const auto statistics = Statistics<Storage>::create(values);
                const auto fixedBitWidthEstimate = facebook::nimble::detail::
                    EncodingSizeEstimation<Storage>::estimateSize(
                        EncodingType::FixedBitWidth,
                        values,
                        statistics,
                        sectionOptions);
                double bestSelectionCost =
                    std::numeric_limits<double>::infinity();
                for (const auto& candidate : candidates) {
                  if (allowed.count(candidate.type) == 0 ||
                      !encodingAvailableAtWidth(candidate.type, storageBytes)) {
                    continue;
                  }
                  const auto readFactor = sectionReadFactor(candidate.type);
                  if (!readFactor.has_value()) {
                    continue;
                  }
                  const auto estimate = facebook::nimble::detail::
                      EncodingSizeEstimation<Storage>::estimateSize(
                          candidate.type, values, statistics, sectionOptions);
                  if (!estimate.has_value()) {
                    continue;
                  }
                  const double selectionCost =
                      static_cast<double>(estimate.value()) *
                      static_cast<double>(effectiveReadFactor(
                          candidate.type,
                          readFactor.value(),
                          estimate.value(),
                          fixedBitWidthEstimate));
                  if (selectionCost < bestSelectionCost) {
                    bestSelectionCost = selectionCost;
                    result = {
                        static_cast<double>(estimate.value()), candidate.type};
                  }
                }
              });
          rescored.emplace(key, result);
          return result;
        };

        double bestPlanBits = std::numeric_limits<double>::infinity();
        size_t bestIndex = 0;
        for (size_t p = 0; p < shortlist.size(); ++p) {
          double planBits = splitPenaltyBits *
              static_cast<double>(shortlist[p].second.size() - 1);
          for (const auto& [l, r] : shortlist[p].second) {
            planBits += rescoreRange(l, r).first * 8.0 * rescoreScale;
          }
          if (planBits < bestPlanBits) {
            bestPlanBits = planBits;
            bestIndex = p;
          }
        }
        const double rescoreMs =
            static_cast<double>(
                std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::steady_clock::now() - rescoreStart)
                    .count()) /
            1e3;

        const auto toPlan = [&](const RangePlan& ranges) {
          std::vector<SectionPlan> plan;
          for (const auto& [l, r] : ranges) {
            SectionPlan segment;
            segment.bitStart = l;
            segment.bitEnd = r;
            segment.encoding = rescoreRange(l, r).second;
            plan.push_back(segment);
          }
          return plan;
        };
        const size_t rangesAfterShortlist = rescored.size();
        std::cout << "  hybrid: shortlist=" << shortlist.size()
                  << " plans, distinct_ranges=" << rangesAfterShortlist
                  << ", rescore_rows=" << rescoreSamples.size()
                  << ", rescore_ms=" << rescoreMs
                  << ", picked=" << shortlist[bestIndex].first << "\n";
        scorePlan("hybrid", toPlan(shortlist[bestIndex].second));

        // Local refinement of the picked plan under the same re-scoring. The
        // plans the shortlist misses mostly differ from its best by where a
        // boundary sits, not by how many there are, and a shortlist drawn from
        // the small sample's grid cannot see which neighbour is right. Each
        // move is priced from the range cache, so it costs only the ranges it
        // introduces. First improvement wins; stops when no move improves.
        const auto refineStart = std::chrono::steady_clock::now();
        const auto planCost = [&](const RangePlan& ranges) {
          double bits =
              splitPenaltyBits * static_cast<double>(ranges.size() - 1);
          for (const auto& [l, r] : ranges) {
            bits += rescoreRange(l, r).first * 8.0 * rescoreScale;
          }
          return bits;
        };
        RangePlan refined = shortlist[bestIndex].second;
        double refinedBits = planCost(refined);
        int moves = 0;
        for (bool improved = true; improved && moves < 64;) {
          improved = false;
          std::vector<RangePlan> neighbours;
          for (size_t i = 0; i + 1 < refined.size(); ++i) {
            // Shift the boundary between segments i and i + 1.
            for (int delta : {-3, -2, -1, 1, 2, 3}) {
              const int cut = refined[i].second + delta;
              if (cut < refined[i].first || cut >= refined[i + 1].second) {
                continue;
              }
              RangePlan moved = refined;
              moved[i].second = cut;
              moved[i + 1].first = cut + 1;
              neighbours.push_back(std::move(moved));
            }
            // Merge segments i and i + 1.
            RangePlan merged = refined;
            merged[i].second = merged[i + 1].second;
            merged.erase(merged.begin() + i + 1);
            neighbours.push_back(std::move(merged));
          }
          // Split a segment in two at any interior bit.
          for (size_t i = 0; i < refined.size(); ++i) {
            for (int cut = refined[i].first; cut < refined[i].second; ++cut) {
              if (FLAGS_hybrid_refine_bitflip_splits && !isCut[cut + 1]) {
                continue;
              }
              RangePlan split = refined;
              split[i].second = cut;
              split.insert(split.begin() + i + 1, {cut + 1, refined[i].second});
              neighbours.push_back(std::move(split));
            }
          }
          for (auto& neighbour : neighbours) {
            const double bits = planCost(neighbour);
            if (bits < refinedBits) {
              refinedBits = bits;
              refined = std::move(neighbour);
              improved = true;
              ++moves;
              break;
            }
          }
        }
        const double refineMs =
            static_cast<double>(
                std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::steady_clock::now() - refineStart)
                    .count()) /
            1e3;
        std::cout << "  hybrid_refined: moves=" << moves
                  << ", new_ranges=" << rescored.size() - rangesAfterShortlist
                  << ", refine_ms=" << refineMs << "\n";
        scorePlan("hybrid_refined", toPlan(refined));
      }

      // Per-cell accuracy summary for this dataset. Kept separate from the plan
      // rows: it describes the models, not any one plan.
      csv.beginRow();
      csv.set("driver", "bench_costmodel_oracle");
      csv.set("dtype", elemTypeName<Elem>());
      csv.set("dataset", ds.name);
      csv.set("N", static_cast<int64_t>(n));
      csv.set("seed", static_cast<int64_t>(seed));
      csv.set("sample_size", static_cast<int64_t>(sampleSize));
      csv.set(
          "min_segment_width",
          static_cast<int64_t>(selectorCfg.minSectionWidth));
      csv.set("plan_type", "summary");
      csv.set("sample_seam_count", static_cast<int64_t>(sampleSeams));
      // The oracle is optimal over contiguous bit ranges of at least
      // min_segment_width, which is the only partition shape either DP can
      // express. A partition outside that shape leaves both wrong together by
      // a margin no sample size reveals, so the column says what was searched
      // rather than letting "oracle" be read as "optimal".
      csv.set(
          "search_space",
          "contiguous_ranges_min_width_" +
              std::to_string(selectorCfg.minSectionWidth));
      csv.set("cost_scale", fullCostScale);
      csv.set("split_penalty_bits", selectorCfg.splitPenalty);
      csv.set("allow_huffman", FLAGS_allow_huffman ? int64_t{1} : int64_t{0});
      csv.set("top1_accuracy", top1Accuracy);
      csv.set("spearman_rho", meanRho);
      csv.set("mean_abs_rel_err", meanAbsRelErr);
      csv.set("planner_agreement_rate", plannerAgreementRate);
      if (std::isfinite(plannerRatioMedian)) {
        csv.set("planner_cell_ratio_median", plannerRatioMedian);
      }
      csv.set(
          "unavailable_model_picks",
          static_cast<int64_t>(unavailablePickCount));
      csv.set("skipped", int64_t{0});
      csv.endRow();

      // One row per encoding carrying how many cells it was smallest on at
      // this sample size, so the per-encoding trend across the sweep reads off
      // the CSV directly instead of being recovered from is_oracle_pick.
      for (size_t ci = 0; ci < candidates.size(); ++ci) {
        csv.beginRow();
        csv.set("driver", "bench_costmodel_oracle");
        csv.set("dtype", elemTypeName<Elem>());
        csv.set("dataset", ds.name);
        csv.set("N", static_cast<int64_t>(n));
        csv.set("seed", static_cast<int64_t>(seed));
        csv.set("sample_size", static_cast<int64_t>(sampleSize));
        csv.set("sample_seam_count", static_cast<int64_t>(sampleSeams));
        csv.set("plan_type", "oracle_wins");
        csv.set("encoding", candidates[ci].name);
        csv.set("oracle_win_count", static_cast<int64_t>(oracleWinCounts[ci]));
        csv.set("skipped", int64_t{0});
        csv.endRow();
      }
      csv.flush();

      std::cout << "  oracle wins:";
      for (size_t ci = 0; ci < candidates.size(); ++ci) {
        if (oracleWinCounts[ci] > 0) {
          std::cout << " " << candidates[ci].name << "=" << oracleWinCounts[ci];
        }
      }
      std::cout << "\n";
      if (unavailablePickCount > 0) {
        std::cout << "  [WARN] model named an encoding unavailable at the "
                     "section's storage width on "
                  << unavailablePickCount << " cell(s)\n";
      }
    }
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
  std::cerr
      << "bench_costmodel_oracle requires NIMBLE_ENABLE_EXPERIMENTAL_ENCODINGS\n";
  return 1;
}

#endif
