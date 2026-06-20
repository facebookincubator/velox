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

#include "velox/common/config/ConfigProperty.h"
#include "velox/core/QueryConfig.h"

namespace facebook::velox::functions::sparksql {

// Defines a traits struct (constName##Property), a key constant, and a typed
// accessor that reads the qualified key (kPrefix + key) from the wrapped
// QueryConfig. The 'key' is the unqualified property name
// (e.g., "legacy_date_formatter"); SparkQueryConfig::kPrefix is prepended at
// read time. Writers populating the QueryConfig map must qualify the key the
// same way (use SparkQueryConfig::qualify()).
#define VELOX_SPARK_CONFIG(                                      \
    constName, accessorName, keyStr, CppType, defaultVal, desc)  \
  struct constName##Property {                                   \
    using type = CppType;                                        \
    static constexpr const char* key = keyStr;                   \
    static constexpr auto defaultValue = defaultVal;             \
    static constexpr const char* description = desc;             \
  };                                                             \
  static constexpr const char* constName = keyStr;               \
  CppType accessorName() const {                                 \
    return config_.get<CppType>(qualify(constName), defaultVal); \
  }

/// Typed view over Spark-specific session properties stored in the shared
/// QueryConfig. Construct on demand: SparkQueryConfig{queryConfig}.
/// Holds a reference to the QueryConfig and must not outlive it.
///
/// Property names are declared unqualified (e.g., "legacy_date_formatter").
/// The shared QueryConfig storage namespace is disambiguated by kPrefix.
/// Read accessors prepend kPrefix when looking up values; writers populating
/// the QueryConfig map must do the same via qualify().
class SparkQueryConfig {
 public:
  explicit SparkQueryConfig(const core::QueryConfig& config)
      : config_{config} {}

  /// Storage-key prefix applied to all Spark properties in the shared
  /// QueryConfig map.
  ///
  /// This is hardcoded: the process can host only one Spark mount, and that
  /// mount must use this prefix. Making it configurable would require either
  /// (a) plumbing a per-query prefix into SparkQueryConfig, or (b) moving
  /// Spark properties to a per-package ConfigBase (as connectors do). Neither
  /// is feasible while simple functions receive only const QueryConfig& in
  /// their initialize() signature, with no channel for function-package
  /// context.
  static constexpr std::string_view kPrefix = "spark.";

  /// Returns the qualified storage key for the given unqualified property
  /// name. Use when populating a QueryConfig map with Spark properties.
  static std::string qualify(std::string_view name) {
    return std::string{kPrefix} + std::string{name};
  }

  /// If true, Spark function's behavior is ANSI-compliant.
  VELOX_SPARK_CONFIG(
      kAnsiEnabled,
      ansiEnabled,
      "ansi_enabled",
      bool,
      false,
      "Enable ANSI-compliant behavior for Spark functions.")

  /// Default number of expected items for the bloom filter.
  VELOX_SPARK_CONFIG(
      kBloomFilterExpectedNumItems,
      bloomFilterExpectedNumItems,
      "bloom_filter.expected_num_items",
      int64_t,
      1'000'000L,
      "Default number of expected items for the Spark bloom filter.")

  /// Default number of bits to use for the bloom filter.
  VELOX_SPARK_CONFIG(
      kBloomFilterNumBits,
      bloomFilterNumBits,
      "bloom_filter.num_bits",
      int64_t,
      8'388'608L,
      "Default number of bits for the Spark bloom filter.")

  /// Maximum number of bits to use for the bloom filter.
  VELOX_SPARK_CONFIG(
      kBloomFilterMaxNumBits,
      bloomFilterMaxNumBits,
      "bloom_filter.max_num_bits",
      int64_t,
      67'108'864L,
      "Maximum number of bits for the Spark bloom filter.")

  /// Maximum number of items to use for the bloom filter.
  VELOX_SPARK_CONFIG(
      kBloomFilterMaxNumItems,
      bloomFilterMaxNumItems,
      "bloom_filter.max_num_items",
      int64_t,
      4'000'000L,
      "Maximum number of items for the Spark bloom filter.")

  /// Current Spark partition id for the running task. Typically set by the
  /// caller before task execution; defaults to 0.
  VELOX_SPARK_CONFIG(
      kPartitionId,
      partitionId,
      "partition_id",
      int32_t,
      0,
      "Current Spark partition id for the running task. Defaults to 0.")

  /// If true, simple date formatter is used for time formatting and parsing.
  VELOX_SPARK_CONFIG(
      kLegacyDateFormatter,
      legacyDateFormatter,
      "legacy_date_formatter",
      bool,
      false,
      "Use simple date formatter instead of Joda for Spark.")

  /// If true, Spark statistical aggregation functions return NaN instead of
  /// NULL when dividing by zero.
  VELOX_SPARK_CONFIG(
      kLegacyStatisticalAggregate,
      legacyStatisticalAggregate,
      "legacy_statistical_aggregate",
      bool,
      false,
      "Return NaN instead of NULL for Spark statistical aggregation on divide-by-zero.")

  /// If true, ignore null fields when generating JSON string.
  VELOX_SPARK_CONFIG(
      kJsonIgnoreNullFields,
      jsonIgnoreNullFields,
      "json_ignore_null_fields",
      bool,
      true,
      "Ignore null fields when generating JSON string in Spark.")

  /// If true, collect_list aggregate function will ignore nulls in the input.
  VELOX_SPARK_CONFIG(
      kCollectListIgnoreNulls,
      collectListIgnoreNulls,
      "collect_list.ignore_nulls",
      bool,
      true,
      "If true, Spark collect_list() ignores nulls in the input.")

  /// Returns all registered Spark config properties. Property names are
  /// unqualified (no "spark." prefix).
  static const std::vector<config::ConfigProperty>& registeredProperties();

 private:
  const core::QueryConfig& config_;
};

#undef VELOX_SPARK_CONFIG

} // namespace facebook::velox::functions::sparksql
