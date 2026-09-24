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

#pragma once

#ifdef NIMBLE_ENABLE_EXPERIMENTAL_ENCODINGS

#include <algorithm>
#include <iostream>
#include <memory>
#include <optional>
#include <span>
#include <sstream>
#include <string>
#include <vector>

#include "velox/dwio/nimble/encodings/benchmarks/ml_id_compression/BenchCommon.h"
#include "velox/dwio/nimble/encodings/benchmarks/ml_id_compression/BlockCodecTarget.h"
#include "velox/dwio/nimble/encodings/benchmarks/ml_id_compression/CachePolicy.h"
#include "velox/dwio/nimble/encodings/benchmarks/ml_id_compression/ElemType.h"
#include "velox/dwio/nimble/encodings/benchmarks/ml_id_compression/MeasureLoop.h"
#include "velox/dwio/nimble/encodings/benchmarks/ml_id_compression/OpenZLBenchTarget.h"
#include "velox/dwio/nimble/encodings/benchmarks/ml_id_compression/ResultWriter.h"

// Scaffolding shared by the sweep drivers. Every driver walks the same shape:
// build the encoder and dataset suites, loop over both, encode, set up the
// cache for one cell, measure, then write a row. Only the measurement and the
// driver-specific columns differ, and those stay in the driver.

DECLARE_string(mlidc_input_order);

namespace facebook::nimble::mlidc {

/// Overrides the upstream SubIntSplit switches named in `features`, a
/// comma-separated list where a name sets its switch on and no-<name> sets it
/// off. Throws on a name it does not know.
inline void applyUpstreamFeatures(
    std::string_view features,
    Encoding::Options& options) {
  size_t start = 0;
  while (start < features.size()) {
    const size_t end = std::min(features.find(',', start), features.size());
    std::string_view name = features.substr(start, end - start);
    start = end + 1;
    if (name.empty()) {
      continue;
    }
    const bool on = !name.starts_with("no-");
    if (!on) {
      name.remove_prefix(3);
    }
    if (name == "delta") {
      options.subIntSplitDeltaPreTransform = on;
    } else if (name == "trim") {
      options.subIntSplitTrimConstantPlanes = on;
    } else if (name == "prune") {
      options.subIntSplitBoundaryPruneThreshold =
          on ? subintsplit::kBoundaryPruneThreshold : 0.0;
    } else if (name == "fold") {
      options.subIntSplitFoldConstantSections = on;
    } else if (name == "passthrough") {
      options.subIntSplitPassThrough = on;
    } else if (name == "visitorblock") {
      options.subIntSplitVisitorBlockBuffer = on;
    } else if (name == "huffmandeep") {
      options.huffmanPriceLengthLimited = on;
    } else {
      NIMBLE_UNSUPPORTED("Unknown upstream SubIntSplit feature: {}", name);
    }
  }
}

/// Holds the encoder and dataset suites a sweep driver walks, with the cache
/// topology the measurements run against.
///
/// Deliberately does not carry rows, iterations or seed: drivers read those
/// from flags before building the suites, and several use them while computing
/// their own axes, so folding them in here would force an ordering that does
/// not suit every driver.
template <typename T>
struct SweepContext {
  std::vector<EncoderEntry<T>> encoders;
  std::vector<DatasetEntry<T>> datasets;
  CacheTopology topology;
  /// Cache state the measurements run under, as parsed by the driver.
  CacheState cacheState{};
  /// Rows per dataset instance, kept so rawBytes() has a single definition.
  uint32_t rows{0};

  /// Raw bytes of one dataset instance, the denominator for compression ratio.
  size_t rawBytes() const {
    return static_cast<size_t>(rows) * sizeof(T);
  }
};

/// Builds the suites and validates the cache policy, or returns nullopt after
/// reporting why.
///
/// withOpenZL adds the block codec, which only the drivers that can host one
/// pass true for.
template <typename T>
std::optional<SweepContext<T>>
makeSweepContext(bool withOpenZL, CacheState cacheState, uint32_t rows) {
  SweepContext<T> context;
  context.cacheState = cacheState;
  context.rows = rows;
  context.encoders = buildDefaultEncoders<T>();
  // Zstd as an arm in its own right, in fixed-size blocks; needs no OpenZL.
  for (auto& entry : buildZstdBlockEncoders<T>()) {
    context.encoders.push_back(std::move(entry));
  }
  // The same blocks, kept once decoded, so decoding "one block" does not
  // decompress the entire column the way a whole-payload inner would.
  for (auto& entry : buildZstdBlockEncoders<T>()) {
    if (entry.variant == "block-" + std::to_string(kLazyBlockElementCount)) {
      context.encoders.push_back(
          withBlockLazyMaterialization<T>(
              std::move(entry), kLazyBlockElementCount));
    }
  }
  // Zstd over the whole column, and the same bytes materialized once on
  // first access: the amortisation question for a blackbox codec.
  context.encoders.push_back(buildZstdWholeEncoder<T>());
  context.encoders.push_back(
      withMaterializedAccess<T>(buildZstdWholeEncoder<T>()));
  if (withOpenZL) {
    // Serves partial reads by decompressing the whole column, which is the
    // comparison the decode drivers exist to make.
    context.encoders.push_back(buildOpenZLEncoder<T>());
    // The same frame, decompressed once and then served from memory, since
    // otherwise a blackbox codec has only one way to answer a probe.
    context.encoders.push_back(
        withMaterializedAccess<T>(buildOpenZLEncoder<T>()));
    // The same codec deployed the way a columnar format deploys one,
    // separating the codec from the granularity it is shipped at.
    for (auto& entry : buildOpenZLBlockEncoders<T>()) {
      context.encoders.push_back(std::move(entry));
    }
    // Holds only the blocks a workload actually reached, unlike
    // openzl/auto+materialize which holds the whole decoded column.
    for (auto& entry : buildOpenZLBlockEncoders<T>()) {
      if (entry.variant == "block-" + std::to_string(kLazyBlockElementCount)) {
        context.encoders.push_back(
            withBlockLazyMaterialization<T>(
                std::move(entry), kLazyBlockElementCount));
      }
    }
  }
  // Applied last so it can select an OpenZL entry too. Comma-separated names
  // matched against the CSV's encoding column; an unknown name is an error.
  if (!FLAGS_mlidc_encoders.empty()) {
    std::vector<EncoderEntry<T>> filtered;
    std::stringstream names(FLAGS_mlidc_encoders);
    std::string want;
    while (std::getline(names, want, ',')) {
      if (want.empty()) {
        continue;
      }
      auto it = std::find_if(
          context.encoders.begin(),
          context.encoders.end(),
          [&](const auto& entry) { return entry.name == want; });
      NIMBLE_USER_CHECK(
          it != context.encoders.end(), "Unknown encoder name: {}", want);
      filtered.push_back(std::move(*it));
    }
    context.encoders = std::move(filtered);
  }
  context.datasets = defaultDatasets<T>();
  context.topology = CacheTopology::detect();

  // Construct a controller up front so an unsupported eviction policy is
  // reported before any measuring rather than part way through a sweep.
  CachePolicy policy;
  policy.state = cacheState;
  try {
    CacheController(policy, context.topology);
  } catch (const std::exception& e) {
    std::cerr << "ERROR: " << e.what() << "\n";
    return std::nullopt;
  }
  return context;
}

/// Sets the columns identifying which encoder and dataset produced a row.
/// Every driver writes these, in this order, including on skip rows.
template <typename T>
void setIdentityColumns(
    CsvResultWriter& csv,
    std::string_view driver,
    const std::string& dataset,
    const EncoderEntry<T>& encoder) {
  csv.set("driver", std::string(driver));
  csv.set("dtype", elemTypeName<T>());
  csv.set("dataset", dataset);
  csv.set("encoding", encoder.name);
  csv.set("family", encoder.family);
  csv.set("variant", encoder.variant);
  csv.set("inventory", encoder.inventory);
  csv.set("transform", encoder.transform);
  csv.set("input_order", FLAGS_mlidc_input_order);
  csv.set("is_sequential", encoder.isSequential ? int64_t{1} : int64_t{0});
}

/// Writes the row that records an encoder declining a dataset.
///
/// Sets only the three columns needed to identify the pair, not the full
/// identity set: an encoder that never encoded has no meaningful family or
/// variant to report, and the remaining columns are left empty, which the CSV
/// writer renders as null.
template <typename T>
void writeSkipRow(
    CsvResultWriter& csv,
    std::string_view driver,
    const std::string& dataset,
    const EncoderEntry<T>& encoder) {
  csv.beginRow();
  csv.set("driver", std::string(driver));
  csv.set("dtype", elemTypeName<T>());
  csv.set("dataset", dataset);
  csv.set("encoding", encoder.name);
  csv.set("skipped", int64_t{1});
  csv.endRow();
}

/// Encodes data with one encoder, or returns nullptr after writing a skip row.
///
/// An encoder that cannot represent a dataset throws from its factory, which is
/// an expected outcome rather than an error: Constant on non-constant data, for
/// instance. The sweep records it and moves on.
template <typename T>
std::unique_ptr<NimbleBenchTargetBase<T>> makeTargetOrSkip(
    const EncoderEntry<T>& encoder,
    const Vector<T>& data,
    CsvResultWriter& csv,
    std::string_view driver,
    const std::string& dataset) {
  facebook::nimble::Encoding::Options options;
  // Withdraws FrequencyPartition from the encodings the split planner may
  // cost a section against. This moves the boundaries the DP picks, not just
  // which encoding a section names, since the DP minimises over section cost.
  if (FLAGS_mlidc_sis_withdraw_frequency_partition) {
    options.subIntSplitAllowedEncodings = {
        facebook::nimble::EncodingType::Trivial,
        facebook::nimble::EncodingType::RLE,
        facebook::nimble::EncodingType::Dictionary,
        facebook::nimble::EncodingType::FixedBitWidth,
        facebook::nimble::EncodingType::Varint,
        facebook::nimble::EncodingType::Delta,
        facebook::nimble::EncodingType::Constant,
        facebook::nimble::EncodingType::MainlyConstant,
        facebook::nimble::EncodingType::PFOR,
        facebook::nimble::EncodingType::SimdForBitpack,
        facebook::nimble::EncodingType::BlockBitPacking,
        facebook::nimble::EncodingType::FOR,
        facebook::nimble::EncodingType::Huffman,
        facebook::nimble::EncodingType::DeltaBlock,
    };
  }
  // Zero by default, so every driver's selection is unchanged unless the flag
  // is set; reaches every arm without threading an argument through each
  // factory.
  options.subIntSplitDecodeWeight = FLAGS_mlidc_sis_decode_weight;
  options.subIntSplitDecodeAccessPattern =
      static_cast<uint8_t>(FLAGS_mlidc_sis_decode_access_pattern);
  options.subIntSplitDecodeReadPath =
      static_cast<uint8_t>(FLAGS_mlidc_sis_decode_read_path);
  options.subIntSplitMaxSizeRegression = FLAGS_mlidc_sis_max_size_regression;
  options.subIntSplitAdmission =
      static_cast<uint8_t>(FLAGS_mlidc_sis_admission);
  options.subIntSplitAdmissionForces = FLAGS_mlidc_sis_admission_forces;
  options.subIntSplitRowFrame = FLAGS_mlidc_sis_row_frame;
  options.subIntSplitEstimateCompressionGuard =
      FLAGS_mlidc_sis_estimate_compression_guard;
  options.subIntSplitEstimateBitFlipScreen =
      FLAGS_mlidc_sis_estimate_bitflip_screen;
  applyUpstreamFeatures(FLAGS_mlidc_sis_upstream_features, options);
  // Which arm is being built is known here, so the encode cache reads it from
  // here rather than threading it through every encode signature.
  setCacheContext(encoder.name);
  try {
    auto target = encoder.factory(data, options);
    clearCacheContext();
    return target;
  } catch (const std::exception& ex) {
    clearCacheContext();
    std::cerr << "  [SKIP] " << encoder.name << ": " << ex.what() << "\n";
    writeSkipRow(csv, driver, dataset, encoder);
    return nullptr;
  }
}

/// Owns the cache controller and the buffer spans it evicts for one cell.
struct CellCache {
  CacheController controller;
  EvictionTargets targets;
};

/// Builds the cache controller and eviction targets for one measurement cell.
///
/// The first internal buffer is the encoded payload; anything after it is codec
/// scratch that must also be evicted for a cold measurement to mean anything.
template <typename T>
CellCache makeCellCache(
    CacheState cacheState,
    const CacheTopology& topology,
    const NimbleBenchTargetBase<T>& target,
    std::span<std::byte> sink) {
  CachePolicy policy;
  policy.state = cacheState;
  CellCache cell{CacheController(policy, topology), EvictionTargets{}};

  auto buffers = target.internalBuffers();
  if (!buffers.empty()) {
    cell.targets.payload = buffers[0];
    if (buffers.size() > 1) {
      cell.targets.codecInternal.assign(buffers.begin() + 1, buffers.end());
    }
  }
  cell.targets.sink = sink;
  return cell;
}

/// Sets the columns describing the cache state a measurement ran under.
inline void setCacheColumns(
    CsvResultWriter& csv,
    const CacheController& controller,
    const MeasureResult& result) {
  csv.set(
      "cache_state",
      std::string(cacheStateName(controller.effectivePolicy().state)));
  csv.set(
      "evict_method",
      std::string(evictMethodName(controller.effectivePolicy().method)));
  csv.set("evict_ns", result.evict.median_ns);
}

/// Sets the encoded size and its ratio against the raw column.
inline void
setPayloadColumns(CsvResultWriter& csv, size_t payloadBytes, size_t rawBytes) {
  csv.set("payload_bytes", static_cast<int64_t>(payloadBytes));
  csv.set(
      "compression_ratio",
      rawBytes > 0
          ? static_cast<double>(payloadBytes) / static_cast<double>(rawBytes)
          : 0.0);
}

/// Sets the repetition counts a cell actually ran, which differ from the
/// requested values for whole-payload codecs.
inline void setMeasureColumns(CsvResultWriter& csv, const MeasureSpec& spec) {
  csv.set("iterations", static_cast<int64_t>(spec.iterations));
  csv.set("warmup", static_cast<int64_t>(spec.warmup));
}

/// Sets the timing columns every measured driver reports.
inline void setTimingColumns(
    CsvResultWriter& csv,
    const MeasureResult& result) {
  csv.set("time_ns", result.time.median_ns);
  csv.set("time_p90_ns", result.time.p90_ns);
  csv.set("time_min_ns", result.time.min_ns);
}

/// Adds the columns that carry a target's read path and the two halves of
/// its cost: what building the access structure cost, and what a read cost
/// once it was built.
inline void appendAccessColumns(std::vector<std::string>& columns) {
  columns.push_back("read_path");
  columns.push_back("builds_access_structure");
  columns.push_back("build_ns");
  columns.push_back("time_incl_build_ns");
  columns.push_back("resident_bytes");
}

/// Measures what building the target's access structure costs, and leaves it
/// built.
///
/// Returns zeros for a target with nothing to build. That is the honest answer
/// rather than a missing one: a cursor arm's build cost really is nothing, and
/// an amortisation plot that joins the two needs the zero in order to draw the
/// flat line the cursor arm makes.
template <typename T>
MeasureResult measureAccessStructureBuild(
    const MeasureSpec& spec,
    CacheController& controller,
    const EvictionTargets& targets,
    NimbleBenchTargetBase<T>& target) {
  if (!target.buildsAccessStructure()) {
    return MeasureResult{};
  }
  auto result = measure(spec, controller, targets, [&]() {
    target.discardAccessStructure();
    target.buildAccessStructure();
  });
  // Left built on purpose: the per-read measurement that follows excludes
  // construction and must not find the structure discarded.
  target.buildAccessStructure();
  return result;
}

/// Sets the access-path columns.
///
/// buildNs is what measureAccessStructureBuild() reported and timeNs the
/// measured read time that excludes it, so the row carries both and their sum
/// and a reader never has to guess which of the two a number is.
///
/// resident_bytes is sampled here, which is after that row's reads rather than
/// after its build. That ordering is the point: an arm that materialises
/// lazily has no final footprint until a workload has touched it, so sampling
/// at construction would report every lazy arm at its compressed size and
/// erase the axis. It also means the column answers the question the time
/// columns cannot -- what an arm is holding in order to go that fast.
template <typename T>
void setAccessColumns(
    CsvResultWriter& csv,
    const NimbleBenchTargetBase<T>& target,
    int64_t buildNs,
    int64_t timeNs) {
  csv.set("read_path", std::string(readPathName(target.readPath())));
  csv.set(
      "builds_access_structure",
      target.buildsAccessStructure() ? int64_t{1} : int64_t{0});
  csv.set("build_ns", buildNs);
  csv.set("time_incl_build_ns", timeNs + buildNs);
  csv.set("resident_bytes", static_cast<int64_t>(target.residentBytes()));
}

} // namespace facebook::nimble::mlidc

#endif // NIMBLE_ENABLE_EXPERIMENTAL_ENCODINGS
