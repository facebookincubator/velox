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

#include <folly/Singleton.h>
#include <iostream>
#include <memory>

#include "StatsReporter.h"
#include "metrics_client/metric_collector_conf.h"
#include "metrics_client/metrics.h"

namespace facebook::velox {

// This is a ByteDance reporter using Metrics Client
class MetricsStatsReporter : public facebook::velox::BaseStatsReporter {
  cpputil::metrics2::MetricCollectorConf metricsConf =
      cpputil::metrics2::MetricCollectorConf();
  std::string PSM;
  std::string clusterName;

 public:
  mutable std::unordered_map<std::string, StatType> counterTypeMap;
  MetricsStatsReporter(std::string PSM, std::string clusterName)
      : PSM(PSM), clusterName(clusterName) {
    metricsConf.namespace_prefix = PSM;
    cpputil::metrics2::Metrics::init(metricsConf);
  }

  MetricsStatsReporter() {
    PSM = getenv("TCE_PSM");
    clusterName = getenv("TCE_CLUSTER");
    facebook::velox::MetricsStatsReporter(PSM, clusterName);
  }

  void addStatExportType(const char* key, StatType statType) const override {
    counterTypeMap[key] = statType;
  }

  void addStatExportType(folly::StringPiece key, StatType statType)
      const override {
    counterTypeMap[key.str()] = statType;
  }

  void addStatValue(const std::string& key, const size_t value) const override {
    if (counterTypeMap.find(key) == counterTypeMap.end()) {
      LOG(ERROR) << "Metrics: Unable to emit to Metrics. Metrics key \"" << key
                 << "\" is not registered";
      return;
    }
    StatType type = counterTypeMap[key];
    std::string tag = "cluster=" + clusterName;
    switch (type) {
      case facebook::velox::StatType::AVG:
        cpputil::metrics2::Metrics::emit_store(key, (double)value, tag);
        break;
      case facebook::velox::StatType::COUNT:
        cpputil::metrics2::Metrics::emit_store(key, (double)value, tag);
        break;
      case facebook::velox::StatType::RATE:
        cpputil::metrics2::Metrics::emit_rate_counter(key, (double)value, tag);
        break;
      case facebook::velox::StatType::SUM:
        cpputil::metrics2::Metrics::emit_counter(key, (double)value, tag);
        break;
    }
  }

  void addStatValue(const char* key, const size_t value) const override {
    std::string keyString(key);
    addStatValue(keyString, value);
  }

  void addStatValue(folly::StringPiece key, size_t value) const override {
    addStatValue(key.str(), value);
  }
};

#define METRICS_REPORT_ADD_STAT_VALUE(k, ...)                   \
  {                                                             \
    auto reporter = folly::Singleton<                           \
        facebook::velox::MetricsStatsReporter>::try_get_fast(); \
    if (LIKELY(reporter != nullptr)) {                          \
      reporter->addStatValue((k), ##__VA_ARGS__);               \
    }                                                           \
  }

#define METRICS_REPORT_ADD_STAT_EXPORT_TYPE(k, t)               \
  {                                                             \
    auto reporter = folly::Singleton<                           \
        facebook::velox::MetricsStatsReporter>::try_get_fast(); \
    if (LIKELY(reporter != nullptr)) {                          \
      reporter->addStatExportType((k), (t));                    \
    }                                                           \
  }

} // namespace facebook::velox
