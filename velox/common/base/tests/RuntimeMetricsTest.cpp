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

#include "velox/common/base/RuntimeMetrics.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "velox/common/base/ConcurrentRuntimeStatWriter.h"
#include "velox/common/base/VeloxException.h"
#include "velox/common/base/tests/GTestUtils.h"

namespace facebook::velox {

class RuntimeMetricsTest : public testing::Test {
 protected:
  static void testMetric(
      const RuntimeMetric& rm1,
      int64_t expectedSum,
      uint64_t expectedCount,
      int64_t expectedMin = std::numeric_limits<int64_t>::max(),
      int64_t expectedMax = std::numeric_limits<int64_t>::min()) {
    EXPECT_EQ(expectedSum, rm1.sum);
    EXPECT_EQ(expectedCount, rm1.count);
    EXPECT_EQ(expectedMin, rm1.min);
    EXPECT_EQ(expectedMax, rm1.max);
  }
};

TEST_F(RuntimeMetricsTest, basic) {
  RuntimeMetric rm1;
  testMetric(rm1, 0, 0);

  rm1.addValue(5);
  testMetric(rm1, 5, 1, 5, 5);

  rm1.addValue(11);
  testMetric(rm1, 16, 2, 5, 11);

  rm1.addValue(3);
  testMetric(rm1, 19, 3, 3, 11);

  ASSERT_EQ(
      fmt::format(
          "sum:{}, count:{}, min:{}, max:{}, avg: {}",
          rm1.sum,
          rm1.count,
          rm1.min,
          rm1.max,
          rm1.sum / rm1.count),
      rm1.toString());

  RuntimeMetric rm2;

  rm1.merge(rm2);
  testMetric(rm1, 19, 3, 3, 11);

  rm2.addValue(53);
  rm1.merge(rm2);
  testMetric(rm1, 72, 4, 3, 53);

  rm1.aggregate();
  testMetric(rm1, 72, 1, 72, 72);

  RuntimeMetric rm3;
  rm3.aggregate();
  testMetric(rm3, 0, 0, 0, 0);

  RuntimeMetric byteRm(RuntimeCounter::Unit::kBytes);
  byteRm.addValue(5);
  ASSERT_EQ(byteRm.toString(), "sum:5B, count:1, min:5B, max:5B, avg: 5B");

  RuntimeMetric timeRm(RuntimeCounter::Unit::kNanos);
  timeRm.addValue(2'000);
  ASSERT_EQ(
      timeRm.toString(),
      "sum:2.00us, count:1, min:2.00us, max:2.00us, avg: 2.00us");
}

TEST_F(RuntimeMetricsTest, saturateCast) {
  auto maxUint64 = std::numeric_limits<uint64_t>::max();
  RuntimeMetric rm{
      saturateCast(maxUint64),
      maxUint64,
      saturateCast(maxUint64),
      saturateCast(maxUint64)};

  auto maxInt64 = std::numeric_limits<int64_t>::max();
  EXPECT_EQ(rm.sum, maxInt64);
  EXPECT_EQ(rm.count, maxUint64);
  EXPECT_EQ(rm.min, maxInt64);
  EXPECT_EQ(rm.max, maxInt64);
}

TEST_F(RuntimeMetricsTest, mergeCounter) {
  RuntimeMetric rm(RuntimeCounter::Unit::kBytes);

  rm.merge(RuntimeCounter(10, RuntimeCounter::Unit::kBytes));
  testMetric(rm, 10, 1, 10, 10);

  rm.merge(RuntimeCounter(30, RuntimeCounter::Unit::kBytes));
  testMetric(rm, 40, 2, 10, 30);

  VELOX_ASSERT_THROW(
      rm.merge(RuntimeCounter(1, RuntimeCounter::Unit::kNanos)),
      "Unit mismatch for runtime stat");
}

class SetThreadLocalRuntimeStatTest : public testing::Test {};

TEST_F(SetThreadLocalRuntimeStatTest, singleMetric) {
  ConcurrentRuntimeStatWriter collector;
  RuntimeStatWriterScopeGuard guard(&collector);

  RuntimeMetric metric(RuntimeCounter::Unit::kNone);
  metric.addValue(10);
  metric.addValue(20);
  metric.addValue(30);

  setThreadLocalRuntimeStat("test.metric", metric);

  const auto stats = collector.runtimeStats();
  ASSERT_EQ(stats.count("test.metric"), 1);
  const auto& result = stats.at("test.metric");
  EXPECT_EQ(result.count, 3);
  EXPECT_EQ(result.sum, 60);
  EXPECT_EQ(result.min, 10);
  EXPECT_EQ(result.max, 30);
}

TEST_F(SetThreadLocalRuntimeStatTest, existingMetric) {
  ConcurrentRuntimeStatWriter collector;
  RuntimeStatWriterScopeGuard guard(&collector);

  RuntimeMetric first(RuntimeCounter::Unit::kNone);
  first.addValue(100);
  setThreadLocalRuntimeStat("test.metric", first);

  RuntimeMetric second(RuntimeCounter::Unit::kNone);
  second.addValue(5);
  second.addValue(15);
  setThreadLocalRuntimeStat("test.metric", second);

  const auto stats = collector.runtimeStats();
  ASSERT_EQ(stats.count("test.metric"), 1);
  const auto& result = stats.at("test.metric");
  EXPECT_EQ(result.count, 2);
  EXPECT_EQ(result.sum, 20);
  EXPECT_EQ(result.min, 5);
  EXPECT_EQ(result.max, 15);
}

TEST_F(SetThreadLocalRuntimeStatTest, emptyMetric) {
  ConcurrentRuntimeStatWriter collector;
  RuntimeStatWriterScopeGuard guard(&collector);

  RuntimeMetric empty(RuntimeCounter::Unit::kNone);
  setThreadLocalRuntimeStat("test.empty", empty);

  const auto stats = collector.runtimeStats();
  ASSERT_EQ(stats.count("test.empty"), 1);
  EXPECT_EQ(stats.at("test.empty").count, 0);
}

TEST_F(SetThreadLocalRuntimeStatTest, noWriter) {
  RuntimeMetric metric(RuntimeCounter::Unit::kNanos);
  metric.addValue(42);
  setThreadLocalRuntimeStat("test.nowriter", metric);
}

TEST_F(SetThreadLocalRuntimeStatTest, noopWriter) {
  ConcurrentRuntimeStatWriter collector;
  RuntimeStatWriterScopeGuard collecting(&collector);

  {
    NoopRuntimeStatWriter noop;
    RuntimeStatWriterScopeGuard discarding(&noop);
    addThreadLocalRuntimeStat("test.noop", RuntimeCounter(1));
  }
  EXPECT_THAT(collector.runtimeStats(), testing::IsEmpty());

  addThreadLocalRuntimeStat("test.after", RuntimeCounter(1));
  EXPECT_EQ(collector.runtimeStats().count("test.after"), 1);
}

} // namespace facebook::velox
