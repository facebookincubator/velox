// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <gtest/gtest.h>
#include "velox/dwio/common/IoStatistics.h"

using namespace ::testing;

TEST(IOStatistics, IncMetricValue) {
  facebook::velox::dwio::common::IoStatistics ioStats;
  ioStats.incMetric("foo", 10);
  ioStats.incMetric("foo", 20);
  ioStats.incMetric("bar", 30);
  ASSERT_EQ(ioStats.getMetricValue("foo"), 30);
  ASSERT_EQ(ioStats.getMetricValue("bar"), 30);
  ASSERT_EQ(ioStats.getMetricValue("barbar"), 0);
}

TEST(IOStatistics, Merge) {
  facebook::velox::dwio::common::IoStatistics ioStats1;
  ioStats1.incMetric("bar1", 30);
  ioStats1.incMetric("bar2", 30);

  facebook::velox::dwio::common::IoStatistics ioStats2;
  ioStats2.incMetric("bar2", 30);
  ioStats2.incMetric("bar3", 40);

  ioStats1.merge(ioStats2);
  ASSERT_EQ(ioStats1.getMetricValue("bar1"), 30);
  ASSERT_EQ(ioStats1.getMetricValue("bar2"), 60);
  ASSERT_EQ(ioStats1.getMetricValue("bar3"), 40);
}
