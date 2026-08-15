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

namespace facebook::velox::functions::prestosql {

// Defines a traits struct (constName##Property), a key constant, and a typed
// accessor that reads the qualified key (kPrefix + key) from the wrapped
// QueryConfig. Mirrors SparkQueryConfig's macro pattern.
#define VELOX_PRESTO_CONFIG(                                     \
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

/// Typed view over Presto-specific session properties stored in the shared
/// QueryConfig. Construct on demand: PrestoQueryConfig{queryConfig}.
/// Holds a reference to the QueryConfig and must not outlive it.
///
/// Property names are declared unqualified (e.g., "array_agg.ignore_nulls").
/// The shared QueryConfig storage namespace is disambiguated by kPrefix.
/// Read accessors prepend kPrefix when looking up values; writers populating
/// the QueryConfig map must do the same via qualify().
class PrestoQueryConfig {
 public:
  explicit PrestoQueryConfig(const core::QueryConfig& config)
      : config_{config} {}

  /// Storage-key prefix applied to all Presto properties in the shared
  /// QueryConfig map. Does not include the trailing dot; qualify() inserts
  /// the dot when building the qualified key.
  ///
  /// Hardcoded for the same reason as SparkQueryConfig::kPrefix: simple
  /// functions receive only const QueryConfig& in initialize() and have no
  /// channel for per-package config, so the process can host only one Presto
  /// mount and that mount must use this prefix.
  static constexpr std::string_view kPrefix = "presto";

  /// Returns the qualified storage key for the given unqualified property
  /// name. Use when populating a QueryConfig map with Presto properties.
  static std::string qualify(std::string_view name) {
    return std::string{kPrefix} + "." + std::string{name};
  }

  /// If true, array_agg() aggregation function ignores nulls in the input.
  VELOX_PRESTO_CONFIG(
      kArrayAggIgnoreNulls,
      arrayAggIgnoreNulls,
      "array_agg.ignore_nulls",
      bool,
      false,
      "If true, array_agg() ignores nulls in the input.")

  /// Returns all registered Presto config properties. Property names are
  /// unqualified (no "presto." prefix).
  static const std::vector<config::ConfigProperty>& registeredProperties();

 private:
  const core::QueryConfig& config_;
};

#undef VELOX_PRESTO_CONFIG

} // namespace facebook::velox::functions::prestosql
