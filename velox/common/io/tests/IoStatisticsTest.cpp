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

#include "velox/common/io/IoStatistics.h"
#include <gtest/gtest.h>

namespace facebook::velox::io {

TEST(IoStatisticsTest, latencyBreakdown) {
  IoStatistics stats;

  stats.storageReadLatencyUs().increment(100);
  stats.ssdCacheReadLatencyUs().increment(50);
  stats.cacheWaitLatencyUs().increment(25);
  stats.coalescedSsdLoadLatencyUs().increment(30);
  stats.coalescedStorageLoadLatencyUs().increment(45);

  EXPECT_EQ(stats.storageReadLatencyUs().count(), 1);
  EXPECT_EQ(stats.storageReadLatencyUs().sum(), 100);
  EXPECT_EQ(stats.ssdCacheReadLatencyUs().sum(), 50);
  EXPECT_EQ(stats.cacheWaitLatencyUs().sum(), 25);
  EXPECT_EQ(stats.coalescedSsdLoadLatencyUs().sum(), 30);
  EXPECT_EQ(stats.coalescedStorageLoadLatencyUs().sum(), 45);

  IoStatistics stats2;
  stats2.storageReadLatencyUs().increment(200);
  stats.merge(stats2);

  EXPECT_EQ(stats.storageReadLatencyUs().count(), 2);
  EXPECT_EQ(stats.storageReadLatencyUs().sum(), 300);
}

} // namespace facebook::velox::io
