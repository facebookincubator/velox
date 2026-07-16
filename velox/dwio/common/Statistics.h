/*
 * Copyright (c) Facebook, Inc. and its affiliates.
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

#include <fmt/format.h>
#include <folly/Hash.h>
#include <folly/Synchronized.h>
#include <folly/container/F14Map.h>
#include <type_traits>
#include <utility>
#include "velox/common/time/CpuWallTimer.h"
#include "velox/common/time/Timer.h"
#include "velox/dwio/common/Options.h"
#include "velox/dwio/common/TypeWithId.h"
#include "velox/dwio/common/UnitLoader.h"

#include "velox/common/base/Exceptions.h"
#include "velox/common/base/RuntimeMetrics.h"
#include "velox/common/io/IoStatistics.h"
#include "velox/type/Type.h"

namespace facebook::velox::dwio::common {

/// Provides a common base for writer version information used when
/// interpreting metadata.
///
/// Enables format-independent function signatures while allowing each format
/// implementation to downcast to its specific metadata context.
struct StatsContext {
  virtual ~StatsContext() = default;
};

/// Encodes either an integer or string flat-map key.
struct KeyInfo {
  explicit KeyInfo(int64_t intKey)
      : intKey{std::make_optional<int64_t>(intKey)} {}
  explicit KeyInfo(const std::string& bytesKey)
      : bytesKey{std::make_optional<std::string>(bytesKey)} {}

  bool operator==(const KeyInfo& other) const;

  std::string toString() const;
  std::optional<int64_t> intKey;
  std::optional<std::string> bytesKey;

 private:
  KeyInfo() {}
};

/// Hashes a `KeyInfo` using the active key variant.
struct KeyInfoHash {
  KeyInfoHash() = default;

  size_t operator()(const KeyInfo& keyInfo) const;
};

/// Statistics that are available for all types of columns.
class ColumnStatistics {
 public:
  ColumnStatistics(
      std::optional<uint64_t> valueCount,
      std::optional<bool> hasNull,
      std::optional<uint64_t> rawSize,
      std::optional<uint64_t> size,
      std::optional<int64_t> numDistinct = std::nullopt)
      : valueCount_(valueCount),
        hasNull_(hasNull),
        rawSize_(rawSize),
        size_(size),
        numDistinct_(numDistinct) {}

  virtual ~ColumnStatistics() = default;

  /// Get the number of values in this column. It will differ from the number
  /// of rows because of NULL values and repeated (list/map) values.
  std::optional<uint64_t> getNumberOfValues() const {
    return valueCount_;
  }

  /// Get whether column has null value.
  ///
  /// WARNING: Some writer implementation does not take ancestor nulls into
  /// account, so this value should not be trusted. Check whether
  /// `getNumberOfValues()` is smaller than the row group size for a more
  /// accurate signal.
  std::optional<bool> hasNull() const {
    return hasNull_;
  }

  /// Get uncompressed size of all data including child columns.
  std::optional<uint64_t> getRawSize() const {
    return rawSize_;
  }

  /// Get total length of all streams including child columns.
  std::optional<uint64_t> getSize() const {
    return size_;
  }

  /// Returns the number of distinct values when available.
  std::optional<uint64_t> numDistinct() const {
    return numDistinct_;
  }

  /// Sets the distinct-value count once when the writer provides it.
  void setNumDistinct(int64_t count);

  /// Returns true if there are no non-null values (value count is known to be
  /// zero).
  bool isAllNull() const;

  /// Return string representation of this stats object.
  virtual std::string toString() const;

 protected:
  ColumnStatistics() {}

  std::optional<uint64_t> valueCount_;
  std::optional<bool> hasNull_;
  std::optional<uint64_t> rawSize_;
  std::optional<uint64_t> size_;
  std::optional<uint64_t> numDistinct_;
};

/// Statistics for binary columns.
class BinaryColumnStatistics : public virtual ColumnStatistics {
 public:
  BinaryColumnStatistics(
      std::optional<uint64_t> valueCount,
      std::optional<bool> hasNull,
      std::optional<uint64_t> rawSize,
      std::optional<uint64_t> size,
      std::optional<uint64_t> length)
      : ColumnStatistics(valueCount, hasNull, rawSize, size), length_(length) {}

  BinaryColumnStatistics(
      const ColumnStatistics& colStats,
      std::optional<uint64_t> length)
      : ColumnStatistics(colStats), length_(length) {}

  ~BinaryColumnStatistics() override = default;

  /// Get optional total length.
  std::optional<uint64_t> getTotalLength() const {
    return length_;
  }

  std::string toString() const override;

 protected:
  BinaryColumnStatistics() {}

  std::optional<uint64_t> length_;
};

/// Statistics for boolean columns.
class BooleanColumnStatistics : public virtual ColumnStatistics {
 public:
  BooleanColumnStatistics(
      std::optional<uint64_t> valueCount,
      std::optional<bool> hasNull,
      std::optional<uint64_t> rawSize,
      std::optional<uint64_t> size,
      std::optional<uint64_t> trueCount)
      : ColumnStatistics(valueCount, hasNull, rawSize, size),
        trueCount_(trueCount) {}

  BooleanColumnStatistics(
      const ColumnStatistics& colStats,
      std::optional<uint64_t> trueCount)
      : ColumnStatistics(colStats), trueCount_(trueCount) {}

  ~BooleanColumnStatistics() override = default;

  /// Get optional true count.
  std::optional<uint64_t> getTrueCount() const {
    return trueCount_;
  }

  /// Get optional false count.
  std::optional<uint64_t> getFalseCount() const;

  std::string toString() const override;

 protected:
  BooleanColumnStatistics() {}

  std::optional<uint64_t> trueCount_;
};

/// Statistics for float and double columns.
class DoubleColumnStatistics : public virtual ColumnStatistics {
 public:
  DoubleColumnStatistics(
      std::optional<uint64_t> valueCount,
      std::optional<bool> hasNull,
      std::optional<uint64_t> rawSize,
      std::optional<uint64_t> size,
      std::optional<double> min,
      std::optional<double> max,
      std::optional<double> sum)
      : ColumnStatistics(valueCount, hasNull, rawSize, size),
        min_(min),
        max_(max),
        sum_(sum) {}

  DoubleColumnStatistics(
      const ColumnStatistics& colStats,
      std::optional<double> min,
      std::optional<double> max,
      std::optional<double> sum)
      : ColumnStatistics(colStats), min_(min), max_(max), sum_(sum) {}

  ~DoubleColumnStatistics() override = default;

  /// Get optional smallest value in the column. Only defined if
  /// `getNumberOfValues()` is non-zero.
  std::optional<double> getMinimum() const {
    return min_;
  }

  /// Get optional largest value in the column. Only defined if
  /// `getNumberOfValues()` is non-zero.
  std::optional<double> getMaximum() const {
    return max_;
  }

  /// Get optional sum of the values in the column.
  std::optional<double> getSum() const {
    return sum_;
  }

  std::string toString() const override;

 protected:
  DoubleColumnStatistics() {}

  std::optional<double> min_;
  std::optional<double> max_;
  std::optional<double> sum_;
};

/// Statistics for all integer columns, such as byte, short, int, and long.
class IntegerColumnStatistics : public virtual ColumnStatistics {
 public:
  IntegerColumnStatistics(
      std::optional<uint64_t> valueCount,
      std::optional<bool> hasNull,
      std::optional<uint64_t> rawSize,
      std::optional<uint64_t> size,
      std::optional<int64_t> min,
      std::optional<int64_t> max,
      std::optional<int64_t> sum)
      : ColumnStatistics(valueCount, hasNull, rawSize, size),
        min_(min),
        max_(max),
        sum_(sum) {}

  IntegerColumnStatistics(
      const ColumnStatistics& colStats,
      std::optional<int64_t> min,
      std::optional<int64_t> max,
      std::optional<int64_t> sum)
      : ColumnStatistics(colStats), min_(min), max_(max), sum_(sum) {}

  ~IntegerColumnStatistics() override = default;

  /// Get optional smallest value in the column. Only defined if
  /// `getNumberOfValues()` is non-zero.
  std::optional<int64_t> getMinimum() const {
    return min_;
  }

  /// Get optional largest value in the column. Only defined if
  /// `getNumberOfValues()` is non-zero.
  std::optional<int64_t> getMaximum() const {
    return max_;
  }

  /// Get optional sum of the column. Only valid if `getNumberOfValues()` is
  /// non-zero and the sum does not overflow.
  std::optional<int64_t> getSum() const {
    return sum_;
  }

  std::string toString() const override;

 protected:
  IntegerColumnStatistics() {}

  std::optional<int64_t> min_;
  std::optional<int64_t> max_;
  std::optional<int64_t> sum_;
};

/**
 * Statistics for timestamp columns.
 */
class TimestampColumnStatistics : public virtual ColumnStatistics {
 public:
  TimestampColumnStatistics(
      std::optional<uint64_t> valueCount,
      std::optional<bool> hasNull,
      std::optional<uint64_t> rawSize,
      std::optional<uint64_t> size,
      std::optional<Timestamp> min,
      std::optional<Timestamp> max)
      : ColumnStatistics(valueCount, hasNull, rawSize, size),
        min_(min),
        max_(max) {}

  TimestampColumnStatistics(
      const ColumnStatistics& colStats,
      std::optional<Timestamp> min,
      std::optional<Timestamp> max)
      : ColumnStatistics(colStats), min_(min), max_(max) {}

  ~TimestampColumnStatistics() override = default;

  std::optional<Timestamp> getMinimum() const {
    return min_;
  }

  std::optional<Timestamp> getMaximum() const {
    return max_;
  }

  std::string toString() const override;

 protected:
  TimestampColumnStatistics() {}

  std::optional<Timestamp> min_;
  std::optional<Timestamp> max_;
};

/**
 * Statistics for string columns.
 */
class StringColumnStatistics : public virtual ColumnStatistics {
 public:
  StringColumnStatistics(
      std::optional<uint64_t> valueCount,
      std::optional<bool> hasNull,
      std::optional<uint64_t> rawSize,
      std::optional<uint64_t> size,
      std::optional<std::string> min,
      std::optional<std::string> max,
      std::optional<int64_t> length)
      : ColumnStatistics(valueCount, hasNull, rawSize, size),
        min_(min),
        max_(max),
        length_(length) {}

  StringColumnStatistics(
      const ColumnStatistics& colStats,
      std::optional<std::string> min,
      std::optional<std::string> max,
      std::optional<int64_t> length)
      : ColumnStatistics(colStats), min_(min), max_(max), length_(length) {}

  ~StringColumnStatistics() override = default;

  /// Get optional minimum value for the column.
  const std::optional<std::string>& getMinimum() const {
    return min_;
  }

  /// Get optional maximum value for the column.
  const std::optional<std::string>& getMaximum() const {
    return max_;
  }

  /// Get optional total length of all values.
  std::optional<uint64_t> getTotalLength() const {
    return length_;
  }

  std::string toString() const override;

 protected:
  StringColumnStatistics() {}

  std::optional<std::string> min_;
  std::optional<std::string> max_;
  std::optional<uint64_t> length_;
};

/// Statistics for (flat) map columns.
class MapColumnStatistics : public virtual ColumnStatistics {
 public:
  MapColumnStatistics(
      std::optional<uint64_t> valueCount,
      std::optional<bool> hasNull,
      std::optional<uint64_t> rawSize,
      std::optional<uint64_t> size,
      folly::F14FastMap<
          KeyInfo,
          std::unique_ptr<ColumnStatistics>,
          folly::transparent<KeyInfoHash>>&& entryStatistics)
      : ColumnStatistics(valueCount, hasNull, rawSize, size),
        entryStatistics_{std::move(entryStatistics)} {}

  ~MapColumnStatistics() override = default;

  const folly::F14FastMap<
      KeyInfo,
      std::unique_ptr<ColumnStatistics>,
      folly::transparent<KeyInfoHash>>&
  getEntryStatistics() const {
    return entryStatistics_;
  }

  std::string toString() const override;

 protected:
  MapColumnStatistics()
      : entryStatistics_{17, folly::transparent<KeyInfoHash>()} {}

  folly::F14FastMap<
      KeyInfo,
      std::unique_ptr<ColumnStatistics>,
      folly::transparent<KeyInfoHash>>
      entryStatistics_;
};

/// Exposes column statistics for a file or row group.
class Statistics {
 public:
  virtual ~Statistics() = default;

  /// Get the statistics of the given column.
  virtual const ColumnStatistics& getColumnStatistics(uint32_t colId) const = 0;

  /// Get the number of columns.
  virtual uint32_t getNumberOfColumns() const = 0;
};

/// Runs 'func' and records decompression CPU time if 'counter' is non-null.
template <typename F>
auto withDecompressStats(io::IoCounter* counter, F&& func)
    -> std::enable_if_t<!std::is_void_v<decltype(func())>, decltype(func())> {
  if (counter) {
    uint64_t cpuNanos = 0;
    auto result = [&] {
      NanosecondCPUTimer timer{&cpuNanos};
      return func();
    }();
    counter->increment(cpuNanos);
    return result;
  }
  return func();
}

template <typename F>
auto withDecompressStats(io::IoCounter* counter, F&& func)
    -> std::enable_if_t<std::is_void_v<decltype(func())>> {
  if (counter) {
    uint64_t cpuNanos = 0;
    {
      NanosecondCPUTimer timer{&cpuNanos};
      func();
    }
    counter->increment(cpuNanos);
    return;
  }
  func();
}

/// Per-column statistics counters. Wraps multiple IoCounter instances for
/// different types of measurements (decompression, encoding, etc.).
/// Can be used by any file format reader (DWRF, Nimble, Parquet, etc.).
struct DecodingStats {
  explicit DecodingStats(TypeKind type = TypeKind::INVALID) : typeKind(type) {}

  TypeKind typeKind;
  io::IoCounter decompressCPUTimeNanos;
  io::IoCounter decodeCPUTimeNanos;

  /// Merges stats from another DecodingStats instance.
  void merge(const DecodingStats& other);
};

/// Thread-safe collection of per-column decoding statistics keyed by nodeId.
/// Can be used by any file format reader (DWRF, Nimble, Parquet, etc.).
struct DecodingStatsSet {
  /// Gets or creates a DecodingStats for a column. Sets typeKind when
  /// creating.
  DecodingStats* getOrCreate(
      uint32_t nodeId,
      TypeKind typeKind = TypeKind::INVALID);

  /// Merges all column decoding statistics from another DecodingStatsSet
  /// instance.
  void mergeFrom(const DecodingStatsSet& other);

  /// Exports per-column metrics into the runtime metrics result map.
  void toRuntimeMetrics(
      std::unordered_map<std::string, RuntimeMetric>& result) const;

 private:
  folly::Synchronized<
      folly::F14FastMap<uint32_t, std::unique_ptr<DecodingStats>>>
      map_;
};

/// Collects runtime metrics produced while reading columns.
struct ColumnReaderStatistics {
  // Number of rows returned by string dictionary reader that is flattened
  // instead of keeping dictionary encoding.
  int64_t flattenStringDictionaryValues{0};

  // Total time spent in loading pages, in nanoseconds.
  io::IoCounter pageLoadTimeNs;

  // Per-column decoding statistics. Only populated when decoding stats
  // collection is enabled.
  std::optional<DecodingStatsSet> decodingStatsSet;

  /// Initializes column stats collection for the given schema if enabled in
  /// options. Recursively registers metrics for all columns in the type tree.
  void initColumnStatsCollection(
      const TypeWithId& schema,
      const RowReaderOptions& options);

  /// Merges all stats from another ColumnReaderStatistics instance.
  void mergeFrom(const ColumnReaderStatistics& other);

  /// Exports all metrics into the runtime metrics result map.
  void toRuntimeMetrics(
      std::unordered_map<std::string, RuntimeMetric>& result) const;

 private:
  void registerDecodingStatsImpl(const TypeWithId& node);
};

/// Aggregates runtime statistics collected while processing a split.
struct RuntimeStatistics {
  // Number of splits skipped based on statistics.
  int64_t skippedSplits{0};

  // Number of splits processed based on statistics.
  int64_t processedSplits{0};

  // Total bytes in splits skipped based on statistics.
  int64_t skippedSplitBytes{0};

  // Number of strides (row groups) skipped based on statistics.
  int64_t skippedStrides{0};

  // Number of strides (row groups) processed based on statistics.
  int64_t processedStrides{0};

  // Records extra bytes read past the ideal footer size.
  int64_t footerBufferOverread{0};

  // Records missing bytes relative to the ideal footer size.
  int64_t footerBufferUnderread{0};

  // Counts footer cache hits.
  int64_t footerCacheHit{0};

  // Counts stripes observed in the file.
  int64_t numStripes{0};

  // Estimated bytes reported to the memory pool for the deserialized
  // Parquet file footer, when the parquet reader's footer-memory
  // tracking path is engaged. Lets operators compare the estimate
  // against actual pool usage. 0 when the reader did not engage
  // tracking (e.g. footer below threshold or non-parquet format).
  int64_t parquetFooterEstimatedBytes{0};

  // Stores unit-loader runtime metrics.
  UnitLoaderStats unitLoaderStats;
  // Stores reader-side column runtime metrics.
  ColumnReaderStatistics columnReaderStats;

  // Exports collected counters as runtime metrics.
  std::unordered_map<std::string, RuntimeMetric> toRuntimeMetricMap() const;
};

} // namespace facebook::velox::dwio::common
