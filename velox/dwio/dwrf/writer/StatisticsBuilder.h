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

#include "velox/dwio/common/StatisticsBuilder.h"
#include "velox/dwio/dwrf/common/Config.h"
#include "velox/dwio/dwrf/common/Statistics.h"
#include "velox/dwio/dwrf/common/wrap/dwrf-proto-wrapper.h"

namespace facebook::velox::dwrf {

// Re-export common types into dwrf namespace for backward compatibility.
using dwio::stats::StatisticsBuilderOptions;

/// DWRF-specific StatisticsBuilder that adds proto serialization and
/// proto-based build() on top of the common StatisticsBuilder.
class StatisticsBuilder : public virtual dwio::stats::StatisticsBuilder {
 public:
  explicit StatisticsBuilder(const StatisticsBuilderOptions& options)
      : dwio::stats::StatisticsBuilder(options),
        arena_(std::make_unique<google::protobuf::Arena>()) {}

  ~StatisticsBuilder() override = default;

  /// Serializes statistics to a proto wrapper.
  virtual void toProto(ColumnStatisticsWriteWrapper& stats) const;

  /// Builds a read-only ColumnStatistics by round-tripping through proto.
  std::unique_ptr<dwio::common::ColumnStatistics> build() const override;

  /// Creates a DWRF-specific StatisticsBuilder for the given type. For MAP
  /// type, returns a MapStatisticsBuilder.
  static std::unique_ptr<StatisticsBuilder> create(
      const Type& type,
      const StatisticsBuilderOptions& options);

  /// For the given type tree, creates a list of DWRF stat builders.
  static void createTree(
      std::vector<std::unique_ptr<StatisticsBuilder>>& statBuilders,
      const Type& type,
      const StatisticsBuilderOptions& options);

  /// Creates StatisticsBuilderOptions from a DWRF Config.
  static StatisticsBuilderOptions optionsFromConfig(const Config& config) {
    return StatisticsBuilderOptions{config.get(Config::STRING_STATS_LIMIT)};
  }

 private:
  std::unique_ptr<google::protobuf::Arena> arena_;
};

class BooleanStatisticsBuilder : public StatisticsBuilder,
                                 public dwio::stats::BooleanStatisticsBuilder {
 public:
  explicit BooleanStatisticsBuilder(const StatisticsBuilderOptions& options)
      : dwio::stats::StatisticsBuilder{options.dropNumDistinct()},
        StatisticsBuilder{options.dropNumDistinct()},
        dwio::stats::BooleanStatisticsBuilder{options} {}

  ~BooleanStatisticsBuilder() override = default;

  std::unique_ptr<dwio::common::ColumnStatistics> build() const override {
    return StatisticsBuilder::build();
  }

  void toProto(ColumnStatisticsWriteWrapper& stats) const override;
};

class IntegerStatisticsBuilder : public StatisticsBuilder,
                                 public dwio::stats::IntegerStatisticsBuilder {
 public:
  explicit IntegerStatisticsBuilder(const StatisticsBuilderOptions& options)
      : dwio::stats::StatisticsBuilder{options},
        StatisticsBuilder{options},
        dwio::stats::IntegerStatisticsBuilder{options} {}

  ~IntegerStatisticsBuilder() override = default;

  std::unique_ptr<dwio::common::ColumnStatistics> build() const override {
    return StatisticsBuilder::build();
  }

  void toProto(ColumnStatisticsWriteWrapper& stats) const override;
};

class DoubleStatisticsBuilder : public StatisticsBuilder,
                                public dwio::stats::DoubleStatisticsBuilder {
 public:
  explicit DoubleStatisticsBuilder(const StatisticsBuilderOptions& options)
      : dwio::stats::StatisticsBuilder{options},
        StatisticsBuilder{options},
        dwio::stats::DoubleStatisticsBuilder{options} {}

  ~DoubleStatisticsBuilder() override = default;

  std::unique_ptr<dwio::common::ColumnStatistics> build() const override {
    return StatisticsBuilder::build();
  }

  void toProto(ColumnStatisticsWriteWrapper& stats) const override;
};

class StringStatisticsBuilder : public StatisticsBuilder,
                                public dwio::stats::StringStatisticsBuilder {
 public:
  explicit StringStatisticsBuilder(const StatisticsBuilderOptions& options)
      : dwio::stats::StatisticsBuilder{options},
        StatisticsBuilder{options},
        dwio::stats::StringStatisticsBuilder{options} {}

  ~StringStatisticsBuilder() override = default;

  std::unique_ptr<dwio::common::ColumnStatistics> build() const override {
    return StatisticsBuilder::build();
  }

  void toProto(ColumnStatisticsWriteWrapper& stats) const override;
};

class BinaryStatisticsBuilder : public StatisticsBuilder,
                                public dwio::stats::BinaryStatisticsBuilder {
 public:
  explicit BinaryStatisticsBuilder(const StatisticsBuilderOptions& options)
      : dwio::stats::StatisticsBuilder{options.dropNumDistinct()},
        StatisticsBuilder{options.dropNumDistinct()},
        dwio::stats::BinaryStatisticsBuilder{options} {}

  ~BinaryStatisticsBuilder() override = default;

  std::unique_ptr<dwio::common::ColumnStatistics> build() const override {
    return StatisticsBuilder::build();
  }

  void toProto(ColumnStatisticsWriteWrapper& stats) const override;
};

class MapStatisticsBuilder : public StatisticsBuilder,
                             public dwio::common::MapColumnStatistics {
 public:
  MapStatisticsBuilder(
      const Type& type,
      const StatisticsBuilderOptions& options)
      : dwio::stats::StatisticsBuilder{options},
        StatisticsBuilder{options},
        valueType_{type.as<velox::TypeKind::MAP>().valueType()} {
    init();
    hll_.reset();
  }

  ~MapStatisticsBuilder() override = default;

  void addValues(
      const dwrf::proto::KeyInfo& keyInfo,
      const StatisticsBuilder& stats);

  void incrementSize(const dwrf::proto::KeyInfo& keyInfo, uint64_t size);

  void merge(
      const dwio::common::ColumnStatistics& other,
      bool ignoreSize = false) override;

  void reset() override {
    StatisticsBuilder::reset();
    init();
  }

  void toProto(ColumnStatisticsWriteWrapper& stats) const override;

  /// Converts a proto KeyInfo to a dwio::common::KeyInfo.
  static dwio::common::KeyInfo constructKey(
      const dwrf::proto::KeyInfo& keyInfo);

 private:
  void init() {
    entryStatistics_.clear();
  }

  StatisticsBuilder& getKeyStats(const dwio::common::KeyInfo& keyInfo) {
    auto result = entryStatistics_.try_emplace(
        keyInfo, StatisticsBuilder::create(*valueType_, options_));
    return dynamic_cast<StatisticsBuilder&>(*result.first->second);
  }

  const TypePtr valueType_;
};
} // namespace facebook::velox::dwrf
