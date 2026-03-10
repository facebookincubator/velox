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

#include "velox/common/base/Exceptions.h"
#include "velox/common/hyperloglog/SparseHll.h"
#include "velox/dwio/common/Statistics.h"
#include "velox/type/Type.h"

namespace facebook::velox::dwio::stats {

// Import column statistics types from dwio::common.
using common::BinaryColumnStatistics;
using common::BooleanColumnStatistics;
using common::ColumnStatistics;
using common::DoubleColumnStatistics;
using common::IntegerColumnStatistics;
using common::StringColumnStatistics;

/// Options for creating StatisticsBuilder instances.
struct StatisticsBuilderOptions {
  explicit StatisticsBuilderOptions(
      uint32_t stringLengthLimit,
      std::optional<uint64_t> initialSize = std::nullopt,
      bool countDistincts = false,
      HashStringAllocator* allocator = nullptr)
      : stringLengthLimit{stringLengthLimit},
        initialSize{initialSize},
        countDistincts(countDistincts),
        allocator(allocator) {}

  /// Maximum length of min/max string values to track. Strings longer than
  /// this limit are dropped from statistics.
  uint32_t stringLengthLimit;

  /// Initial value for the size statistic (total stream bytes). Nullopt means
  /// size tracking is disabled until ensureSize() is called.
  std::optional<uint64_t> initialSize;

  /// Whether to count approximate distinct values using HyperLogLog. Requires
  /// 'allocator' to be set.
  bool countDistincts{false};

  /// Allocator for HyperLogLog distinct counting. Required if 'countDistincts'
  /// is true.
  HashStringAllocator* allocator;

  /// Returns a copy with distinct counting disabled.
  StatisticsBuilderOptions dropNumDistinct() const {
    return StatisticsBuilderOptions(stringLengthLimit, initialSize);
  }
};

/// Base class for stats builder. Stats builder is used in writer and file merge
/// to collect and merge stats.
/// It can also be used for gathering stats in ad hoc sampling. In this case it
/// may also count distinct values if enabled in 'options'.
class StatisticsBuilder : public virtual ColumnStatistics {
 public:
  explicit StatisticsBuilder(const StatisticsBuilderOptions& options)
      : options_{options} {
    VELOX_CHECK(
        !options.countDistincts || options.allocator != nullptr,
        "allocator is required when countDistincts is true");
    init();
  }

  ~StatisticsBuilder() override = default;

  void setHasNull() {
    hasNull_ = true;
  }

  void increaseValueCount(uint64_t count = 1) {
    if (valueCount_.has_value()) {
      valueCount_.value() += count;
    }
  }

  void increaseRawSize(uint64_t rawSize) {
    if (rawSize_.has_value()) {
      rawSize_.value() += rawSize;
    }
  }

  void clearRawSize() {
    rawSize_.reset();
  }

  void ensureSize() {
    if (!size_.has_value()) {
      size_ = 0;
    }
  }

  void incrementSize(uint64_t size);

  template <typename T>
  void addHash(const T& data) {
    if (hll_) {
      hll_->insertHash(folly::hasher<T>()(data));
    }
  }

  int64_t cardinality() const {
    VELOX_CHECK_NOT_NULL(hll_);
    return hll_->cardinality();
  }

  /// Returns estimated number of distinct values if distinct counting is
  /// enabled, or std::nullopt otherwise.
  std::optional<int64_t> estimateNumDistinct() const {
    if (hll_) {
      return hll_->cardinality();
    }
    return std::nullopt;
  }

  /// Merges stats of same type. Used in writer to aggregate file level stats.
  virtual void merge(const ColumnStatistics& other, bool ignoreSize = false);

  /// Resets to initial state. Used where row index entry level stats is
  /// captured.
  virtual void reset() {
    init();
  }

  /// Builds a read-only ColumnStatistics snapshot. Typed stats (min/max/sum)
  /// are omitted when isAllNull(). String min/max are omitted when they exceed
  /// the string length limit.
  virtual std::unique_ptr<ColumnStatistics> build() const;

  /// Creates a StatisticsBuilder for the given type. For MAP type, creates a
  /// base StatisticsBuilder (not a MapStatisticsBuilder, which stays in DWRF).
  static std::unique_ptr<StatisticsBuilder> create(
      const Type& type,
      const StatisticsBuilderOptions& options);

  /// For the given type tree, creates a list of stat builders.
  static void createTree(
      std::vector<std::unique_ptr<StatisticsBuilder>>& statBuilders,
      const Type& type,
      const StatisticsBuilderOptions& options);

 private:
  void init() {
    valueCount_ = 0;
    hasNull_ = false;
    rawSize_ = 0;
    size_ = options_.initialSize;
    if (options_.countDistincts) {
      hll_ =
          std::make_shared<velox::common::hll::SparseHll<>>(options_.allocator);
    }
  }

 protected:
  StatisticsBuilderOptions options_;
  std::shared_ptr<velox::common::hll::SparseHll<>> hll_;
};

class BooleanStatisticsBuilder : public virtual StatisticsBuilder,
                                 public BooleanColumnStatistics {
 public:
  explicit BooleanStatisticsBuilder(const StatisticsBuilderOptions& options)
      : StatisticsBuilder{options.dropNumDistinct()} {
    init();
  }

  ~BooleanStatisticsBuilder() override = default;

  void addValues(bool value, uint64_t count = 1);

  std::unique_ptr<ColumnStatistics> build() const override;

  void merge(const ColumnStatistics& other, bool ignoreSize = false) override;

  void reset() override {
    StatisticsBuilder::reset();
    init();
  }

 private:
  void init() {
    trueCount_ = 0;
  }
};

class IntegerStatisticsBuilder : public virtual StatisticsBuilder,
                                 public IntegerColumnStatistics {
 public:
  explicit IntegerStatisticsBuilder(const StatisticsBuilderOptions& options)
      : StatisticsBuilder{options} {
    init();
  }

  ~IntegerStatisticsBuilder() override = default;

  void addValues(int64_t value, uint64_t count = 1);

  std::unique_ptr<ColumnStatistics> build() const override;

  void merge(const ColumnStatistics& other, bool ignoreSize = false) override;

  void reset() override {
    StatisticsBuilder::reset();
    init();
  }

 private:
  void init() {
    min_ = std::numeric_limits<int64_t>::max();
    max_ = std::numeric_limits<int64_t>::min();
    sum_ = 0;
  }
};

static_assert(
    std::numeric_limits<double>::has_infinity,
    "infinity not defined");

class DoubleStatisticsBuilder : public virtual StatisticsBuilder,
                                public DoubleColumnStatistics {
 public:
  explicit DoubleStatisticsBuilder(const StatisticsBuilderOptions& options)
      : StatisticsBuilder{options} {
    init();
  }

  ~DoubleStatisticsBuilder() override = default;

  void addValues(double value, uint64_t count = 1);

  std::unique_ptr<ColumnStatistics> build() const override;

  void merge(const ColumnStatistics& other, bool ignoreSize = false) override;

  void reset() override {
    StatisticsBuilder::reset();
    init();
  }

 private:
  void init() {
    min_ = std::numeric_limits<double>::infinity();
    max_ = -std::numeric_limits<double>::infinity();
    sum_ = 0;
  }

  void clear() {
    min_.reset();
    max_.reset();
    sum_.reset();
  }
};

class StringStatisticsBuilder : public virtual StatisticsBuilder,
                                public StringColumnStatistics {
 public:
  explicit StringStatisticsBuilder(const StatisticsBuilderOptions& options)
      : StatisticsBuilder{options}, lengthLimit_{options.stringLengthLimit} {
    init();
  }

  ~StringStatisticsBuilder() override = default;

  void addValues(std::string_view value, uint64_t count = 1);

  std::unique_ptr<ColumnStatistics> build() const override;

  void merge(const ColumnStatistics& other, bool ignoreSize = false) override;

  void reset() override {
    StatisticsBuilder::reset();
    init();
  }

 protected:
  uint32_t lengthLimit_;

  bool shouldKeep(const std::optional<std::string>& val) const {
    return val.has_value() && val.value().size() <= lengthLimit_;
  }

 private:
  void init() {
    min_.reset();
    max_.reset();
    length_ = 0;
  }
};

class BinaryStatisticsBuilder : public virtual StatisticsBuilder,
                                public BinaryColumnStatistics {
 public:
  explicit BinaryStatisticsBuilder(const StatisticsBuilderOptions& options)
      : StatisticsBuilder{options.dropNumDistinct()} {
    init();
  }

  ~BinaryStatisticsBuilder() override = default;

  void addValues(uint64_t length, uint64_t count = 1);

  std::unique_ptr<ColumnStatistics> build() const override;

  void merge(const ColumnStatistics& other, bool ignoreSize = false) override;

  void reset() override {
    StatisticsBuilder::reset();
    init();
  }

 private:
  void init() {
    length_ = 0;
  }
};

} // namespace facebook::velox::dwio::stats
