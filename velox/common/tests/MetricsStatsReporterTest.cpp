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

#include "velox/common/base/MetricsStatsReporter.h"
#include <folly/Singleton.h>
#include <folly/init/Init.h>
#include <gtest/gtest.h>
#include <unordered_map>
#include "velox/common/base/StatsReporter.h"

namespace facebook::velox {

class MetricsStatsReporterTest : public testing::Test {
 protected:
  void SetUp() override {}
  void TearDown() override {}
};

// Internal Metrics API https://metrics-fe-us.byted.org/web/plot/metrics
TEST_F(MetricsStatsReporterTest, metricsReporter) {
  auto metricsReporter =
      facebook::velox::MetricsStatsReporter("dp.presto.cppworker", "test");

  metricsReporter.addStatExportType("key1", StatType::COUNT);
  metricsReporter.addStatExportType("key2", StatType::SUM);
  metricsReporter.addStatExportType("key3", StatType::RATE);
  EXPECT_EQ(StatType::COUNT, metricsReporter.counterTypeMap["key1"]);
  EXPECT_EQ(StatType::SUM, metricsReporter.counterTypeMap["key2"]);
  EXPECT_EQ(StatType::RATE, metricsReporter.counterTypeMap["key3"]);
  metricsReporter.addStatValue("key1", 10);
  metricsReporter.addStatValue("key2", 20);
  metricsReporter.addStatValue("key3", 30);
};

} // namespace facebook::velox
