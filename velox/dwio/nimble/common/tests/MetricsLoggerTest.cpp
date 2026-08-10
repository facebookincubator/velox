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

#include <gtest/gtest.h>

#include "velox/dwio/nimble/common/MetricsLogger.h"

namespace facebook::nimble::test {

// --- StripeLoadMetrics::serialize ---

TEST(MetricsLoggerTest, StripeLoadMetricsSerialize) {
  StripeLoadMetrics metrics{
      .stripeIndex = 3,
      .rowsInStripe = 1000,
      .streamCount = 5,
      .totalStreamSize = 4096,
      .cpuUsec = 100,
      .wallTimeUsec = 200,
  };

  auto obj = metrics.serialize();
  EXPECT_EQ(obj["stripeIndex"].asInt(), 3);
  EXPECT_EQ(obj["rowsInStripe"].asInt(), 1000);
  EXPECT_EQ(obj["streamCount"].asInt(), 5);
  EXPECT_EQ(obj["totalStreamSize"].asInt(), 4096);
}

TEST(MetricsLoggerTest, StripeLoadMetricsSerializeDefaults) {
  StripeLoadMetrics metrics{
      .stripeIndex = 0,
      .rowsInStripe = 0,
      .cpuUsec = 0,
      .wallTimeUsec = 0,
  };

  auto obj = metrics.serialize();
  EXPECT_EQ(obj["streamCount"].asInt(), 0);
  EXPECT_EQ(obj["totalStreamSize"].asInt(), 0);
}

// --- StripeFlushMetrics::serialize ---

TEST(MetricsLoggerTest, StripeFlushMetricsSerialize) {
  StripeFlushMetrics metrics{
      .inputSize = 50000,
      .rowCount = 10000,
      .stripeSize = 30000,
      .trackedMemory = 65536,
      .flushCpuUsec = 500,
      .flushWallTimeUsec = 600,
  };

  auto obj = metrics.serialize();
  EXPECT_EQ(obj["inputSize"].asInt(), 50000);
  EXPECT_EQ(obj["rowCount"].asInt(), 10000);
  EXPECT_EQ(obj["stripeSize"].asInt(), 30000);
  EXPECT_EQ(obj["trackedMemory"].asInt(), 65536);
}

// --- FileCloseMetrics::serialize ---

TEST(MetricsLoggerTest, FileCloseMetricsSerialize) {
  FileCloseMetrics metrics{
      .rowCount = 100000,
      .inputSize = 500000,
      .stripeCount = 5,
      .fileSize = 300000,
      .encodingCpuNs = 2000000,
      .encodingWallNs = 3000000,
  };

  auto obj = metrics.serialize();
  EXPECT_EQ(obj["rowCount"].asInt(), 100000);
  EXPECT_EQ(obj["inputSize"].asInt(), 500000);
  EXPECT_EQ(obj["stripeCount"].asInt(), 5);
  EXPECT_EQ(obj["fileSize"].asInt(), 300000);
  EXPECT_EQ(obj["encodingCpuNs"].asInt(), 2000000);
  EXPECT_EQ(obj["encodingWallNs"].asInt(), 3000000);
}

// --- LoggingScope ---

TEST(MetricsLoggerTest, LoggingScopeNullInitially) {
  // Without any LoggingScope, getLogger should return nullptr
  EXPECT_EQ(LoggingScope::getLogger(), nullptr);
}

TEST(MetricsLoggerTest, LoggingScopeSetAndClear) {
  MetricsLogger logger;
  {
    LoggingScope scope(logger);
    EXPECT_EQ(LoggingScope::getLogger(), &logger);
  }
  // After scope destruction, logger should be null again
  EXPECT_EQ(LoggingScope::getLogger(), nullptr);
}

TEST(MetricsLoggerTest, LoggingScopeNestedScopes) {
  MetricsLogger logger1;
  MetricsLogger logger2;
  {
    LoggingScope scope1(logger1);
    EXPECT_EQ(LoggingScope::getLogger(), &logger1);
    {
      LoggingScope scope2(logger2);
      EXPECT_EQ(LoggingScope::getLogger(), &logger2);
    }
    // After inner scope destruction, logger is null (not restored to logger1)
    EXPECT_EQ(LoggingScope::getLogger(), nullptr);
  }
}

// --- Default MetricsLogger virtual methods are safe no-ops ---

TEST(MetricsLoggerTest, DefaultMethodsAreNoOps) {
  MetricsLogger logger;
  // These should all execute without error
  logger.logException(LogOperation::Write, "test error");
  logger.logStripeLoad(StripeLoadMetrics{});
  logger.logStripeFlush(StripeFlushMetrics{});
  logger.logFileClose(FileCloseMetrics{});
  logger.logCompressionContext("test context");
}

} // namespace facebook::nimble::test
