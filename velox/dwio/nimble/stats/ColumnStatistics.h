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

#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <span>
#include <string>

#include "velox/buffer/Buffer.h"
#include "velox/dwio/common/Statistics.h"
#include "velox/dwio/common/TypeWithId.h"
#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/common/SchemaReader.h"
#include "velox/dwio/nimble/common/Vector.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelectionPolicy.h"

namespace facebook::nimble {

enum class StatType : uint8_t {
  DEFAULT,
  INTEGRAL,
  FLOATING_POINT,
  STRING,
  DEDUPLICATED,
};

// Forward declarations for friend classes
class StatisticsCollector;
class IntegralStatisticsCollector;
class FloatingPointStatisticsCollector;
class StringStatisticsCollector;

// In memory column statistics representation. The typical
// usage pattern is for each field writer to hold their own
// pointer to the specific type of column statistics collector,
// and build the statistics during flush time, based on subrange
// of rows.
// The statistics objects can then either be serialized to an
// optional metadata section, or converted to other in memory
// representations for the ingestion engine. The conversion call
// site requires access to the file schema.
class ColumnStatistics {
 public:
  ColumnStatistics() = default;
  virtual ~ColumnStatistics() = default;
  ColumnStatistics(
      uint64_t valueCount,
      uint64_t nullCount,
      uint64_t logicalSize,
      uint64_t physicalSize);

  virtual uint64_t getValueCount() const;
  virtual uint64_t getNullCount() const;
  virtual uint64_t getLogicalSize() const;
  virtual uint64_t getPhysicalSize() const;

  virtual StatType getType() const;

  /// Converts this nimble ColumnStatistics to dwio::common::ColumnStatistics
  /// for use in file-level filter pushdown.
  std::unique_ptr<velox::dwio::common::ColumnStatistics> toCommonStatistics()
      const;

  template <typename T>
  T* as() {
    static_assert(std::is_base_of_v<ColumnStatistics, T>);
    return dynamic_cast<T*>(this);
  }

  template <typename T>
  const T* as() const {
    static_assert(std::is_base_of_v<ColumnStatistics, T>);
    return dynamic_cast<const T*>(this);
  }

 protected:
  friend class StatisticsCollector;

  uint64_t valueCount_{0};
  uint64_t nullCount_{0};
  uint64_t logicalSize_{0};
  std::atomic_uint64_t physicalSize_{0};
};

class StringStatistics : public ColumnStatistics {
 public:
  StringStatistics() = default;
  StringStatistics(
      uint64_t valueCount,
      uint64_t nullCount,
      uint64_t logicalSize,
      uint64_t physicalSize,
      std::optional<std::string> min,
      std::optional<std::string> max);

  std::optional<std::string> getMin() const;
  std::optional<std::string> getMax() const;

  StatType getType() const override;

 protected:
  friend class StringStatisticsCollector;

  std::optional<std::string> min_;
  std::optional<std::string> max_;
};

class IntegralStatistics : public ColumnStatistics {
 public:
  IntegralStatistics() = default;
  IntegralStatistics(
      uint64_t valueCount,
      uint64_t nullCount,
      uint64_t logicalSize,
      uint64_t physicalSize,
      std::optional<int64_t> min,
      std::optional<int64_t> max);

  std::optional<int64_t> getMin() const;
  std::optional<int64_t> getMax() const;

  StatType getType() const override;

 protected:
  friend class IntegralStatisticsCollector;

  std::optional<int64_t> min_;
  std::optional<int64_t> max_;
};

class FloatingPointStatistics : public ColumnStatistics {
 public:
  FloatingPointStatistics() = default;
  FloatingPointStatistics(
      uint64_t valueCount,
      uint64_t nullCount,
      uint64_t logicalSize,
      uint64_t physicalSize,
      std::optional<double> min,
      std::optional<double> max);

  std::optional<double> getMin() const;
  std::optional<double> getMax() const;

  StatType getType() const override;

 protected:
  friend class FloatingPointStatisticsCollector;

  std::optional<double> min_;
  std::optional<double> max_;
};

// Deduplicated column statistics wraps a base ColumnStatistics and adds
// deduplication-specific metrics. The base statistics is preserved to allow
// access to specialized statistics (e.g., IntegralStatistics with min/max).
// TODO: A prettier alternative would be to implement it with template, column
// statistics as the policy or base class.
// Alternatively, make deduplicated stats an optional field in column stats.
class DeduplicatedColumnStatistics : public virtual ColumnStatistics {
 public:
  DeduplicatedColumnStatistics() = default;
  DeduplicatedColumnStatistics(
      ColumnStatistics* baseStatistics,
      uint64_t dedupedCount,
      uint64_t dedupedLogicalSize);

  // Getters delegate to baseStatistics_ for base metrics
  uint64_t getValueCount() const override;
  uint64_t getNullCount() const override;
  uint64_t getLogicalSize() const override;
  uint64_t getPhysicalSize() const override;

  uint64_t getDedupedCount() const;
  uint64_t getDedupedLogicalSize() const;

  StatType getType() const override;

  // Access the wrapped base statistics (e.g., to get min/max for
  // IntegralStatistics)
  ColumnStatistics* getBaseStatistics();
  const ColumnStatistics* getBaseStatistics() const;

 protected:
  friend class DeduplicatedStatisticsCollector;

  ColumnStatistics* baseStatistics_;
  uint64_t dedupedCount_{0};
  uint64_t dedupedLogicalSize_{0};
};

// The statistics collectors are ways to abstract away different
// patterns to modify/update the stats. The current stats collector
// design is around each field writer making bulk updates at flush
// time (stripe or chunk). This design avoids dependencies of writes
// for nested columns, and avoids disrupting ingestion parallelism.
// When we implement more granular stats or index, we can then serialize
// or fork the current stats snap shot.
// We populate local stats and rely on a separate util to roll up
// when all local stats are complete.
class StatisticsCollector {
 public:
  // We do not create stats collectors recursively. Instead, we traverse
  // the schema tree and build the stats collector for each node. We can
  // then lay out the stats collectors in a vector in schema traversal order.
  static std::unique_ptr<StatisticsCollector> create(
      const std::shared_ptr<const velox::dwio::common::TypeWithId>& type);

  StatisticsCollector();
  virtual ~StatisticsCollector() = default;

  virtual uint64_t getValueCount() const;
  virtual uint64_t getNullCount() const;
  virtual uint64_t getLogicalSize() const;
  virtual uint64_t getPhysicalSize() const;
  virtual StatType getType() const;

  // Returns a view of the underlying ColumnStatistics.
  // Caller can cast to specific type using as<T>().
  virtual ColumnStatistics* getStatsView();
  virtual const ColumnStatistics* getStatsView() const;

  template <typename T>
  T* as() {
    static_assert(std::is_base_of_v<StatisticsCollector, T>);
    return dynamic_cast<T*>(this);
  }

  template <typename T>
  T* asChecked() {
    static_assert(std::is_base_of_v<StatisticsCollector, T>);
    auto* result = dynamic_cast<T*>(this);
    NIMBLE_CHECK_NOT_NULL(result, "Failed to cast StatisticsCollector");
    return result;
  }

  virtual bool shared() const {
    return false;
  }

  // Mutation methods for accumulating statistics.
  void addValues(std::span<bool> values);
  virtual void addCounts(uint64_t totalCount, uint64_t nullCount);
  // BE: if we want to go extra, we can have a complex/aggregate type
  // and limit the availability of this method.
  virtual void addLogicalSize(uint64_t logicalSize);
  virtual void addPhysicalSize(uint64_t physicalSize);

  virtual void merge(const StatisticsCollector& other);

 protected:
  std::unique_ptr<ColumnStatistics> stats_;
};

class IntegralStatisticsCollector : public StatisticsCollector {
 public:
  IntegralStatisticsCollector();

  template <typename T>
  void addValues(std::span<T> values);

  void merge(const StatisticsCollector& other) override;

 private:
  // Helper to access stats_ as IntegralStatistics
  IntegralStatistics* integralStats();
};

class FloatingPointStatisticsCollector : public StatisticsCollector {
 public:
  FloatingPointStatisticsCollector();

  template <typename T>
  void addValues(std::span<T> values);

  void merge(const StatisticsCollector& other) override;

 private:
  // Helper to access stats_ as FloatingPointStatistics
  FloatingPointStatistics* floatingPointStats();
};

class StringStatisticsCollector : public StatisticsCollector {
 public:
  StringStatisticsCollector();

  void addValues(std::span<std::string_view> values);
  void merge(const StatisticsCollector& other) override;

 private:
  // Helper to access stats_ as StringStatistics
  StringStatistics* stringStats();
};

// TODO: All ancestors of the deduplicated column will need the deduplicated
// stats. Cleanest is adding another finalization step to
// retroactively wrap.
class DeduplicatedStatisticsCollector : public StatisticsCollector {
 public:
  static std::unique_ptr<DeduplicatedStatisticsCollector> wrap(
      std::unique_ptr<StatisticsCollector> baseCollector);
  explicit DeduplicatedStatisticsCollector(
      std::unique_ptr<StatisticsCollector> baseCollector);

  ColumnStatistics* getStatsView() override;
  const ColumnStatistics* getStatsView() const override;

  void addCounts(uint64_t totalCount, uint64_t nullCount) override;
  void addLogicalSize(uint64_t logicalSize) override;
  void addPhysicalSize(uint64_t physicalSize) override;

  StatisticsCollector* getBaseCollector() const;
  void recordDeduplicatedStats(uint64_t count, uint64_t deduplicatedSize);

  void merge(const StatisticsCollector& other) override;

 private:
  std::unique_ptr<StatisticsCollector> baseCollector_;
  DeduplicatedColumnStatistics* dedupedStats_;
};

// Used for concurrent processing of stats covering multiple field writers.
// A prime use case for this is the flat map layout.
class SharedStatisticsCollector : public StatisticsCollector {
 public:
  static std::unique_ptr<SharedStatisticsCollector> wrap(
      std::unique_ptr<StatisticsCollector> baseCollector);
  explicit SharedStatisticsCollector(
      std::unique_ptr<StatisticsCollector> baseCollector);

  uint64_t getValueCount() const override;
  uint64_t getNullCount() const override;
  uint64_t getLogicalSize() const override;
  uint64_t getPhysicalSize() const override;
  StatType getType() const override;

  ColumnStatistics* getStatsView() override;
  const ColumnStatistics* getStatsView() const override;

  void addCounts(uint64_t totalCount, uint64_t nullCount) override;
  void addLogicalSize(uint64_t logicalSize) override;
  void addPhysicalSize(uint64_t physicalSize) override;

  bool shared() const override {
    return true;
  }

  // For thread unsafe read options
  ColumnStatistics* getBaseStatistics() const;

  void updateBaseCollector(
      std::function<void(StatisticsCollector*)> updateFunc);

  void merge(const StatisticsCollector& other) override;

 private:
  mutable std::mutex mutex_;
  std::unique_ptr<StatisticsCollector> baseCollector_;
};

} // namespace facebook::nimble
