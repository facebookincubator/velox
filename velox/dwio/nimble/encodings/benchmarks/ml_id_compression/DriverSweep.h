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

#include <iostream>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "velox/dwio/nimble/encodings/benchmarks/ml_id_compression/BenchCommon.h"
#include "velox/dwio/nimble/encodings/benchmarks/ml_id_compression/ElemType.h"
#include "velox/dwio/nimble/encodings/benchmarks/ml_id_compression/CachePolicy.h"
#include "velox/dwio/nimble/encodings/benchmarks/ml_id_compression/MeasureLoop.h"
#include "velox/dwio/nimble/encodings/benchmarks/ml_id_compression/OpenZLBenchTarget.h"
#include "velox/dwio/nimble/encodings/benchmarks/ml_id_compression/ResultWriter.h"

// Scaffolding shared by the sweep drivers. Every driver walks the same shape:
// build the encoder and dataset suites, loop over both, encode, set up the
// cache for one cell, measure, then write a row. Only the measurement and the
// driver-specific columns differ, and those stay in the driver.

namespace facebook::nimble::mlidc {

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
std::optional<SweepContext<T>> makeSweepContext(
    bool withOpenZL,
    CacheState cacheState,
    uint32_t rows) {
  SweepContext<T> context;
  context.cacheState = cacheState;
  context.rows = rows;
  context.encoders = buildDefaultEncoders<T>();
  if (withOpenZL) {
    // Serves partial reads by decompressing the whole column, which is the
    // comparison the decode drivers exist to make.
    context.encoders.push_back(buildOpenZLEncoder<T>());
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
  try {
    return encoder.factory(data, options);
  } catch (const std::exception& ex) {
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
inline void setPayloadColumns(
    CsvResultWriter& csv,
    size_t payloadBytes,
    size_t rawBytes) {
  csv.set("payload_bytes", static_cast<int64_t>(payloadBytes));
  csv.set(
      "compression_ratio",
      rawBytes > 0 ? static_cast<double>(payloadBytes) /
              static_cast<double>(rawBytes)
                   : 0.0);
}

/// Sets the repetition counts a cell actually ran, which differ from the
/// requested values for whole-payload codecs.
inline void setMeasureColumns(CsvResultWriter& csv, const MeasureSpec& spec) {
  csv.set("iterations", static_cast<int64_t>(spec.iterations));
  csv.set("warmup", static_cast<int64_t>(spec.warmup));
}

/// Sets the timing columns every measured driver reports.
inline void setTimingColumns(CsvResultWriter& csv, const MeasureResult& result) {
  csv.set("time_ns", result.time.median_ns);
  csv.set("time_p90_ns", result.time.p90_ns);
  csv.set("time_min_ns", result.time.min_ns);
}

} // namespace facebook::nimble::mlidc

#endif // NIMBLE_ENABLE_EXPERIMENTAL_ENCODINGS
