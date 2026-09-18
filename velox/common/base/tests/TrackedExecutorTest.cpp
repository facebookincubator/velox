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

#include "velox/common/base/TrackedExecutor.h"
#include "velox/common/base/ConcurrentRuntimeStatWriter.h"

#include <folly/BenchmarkUtil.h>
#include <folly/executors/InlineExecutor.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <atomic>
#include <stdexcept>
#include <string>

#include "velox/common/base/RuntimeMetrics.h"

namespace facebook::velox {
namespace {

class TrackedExecutorTest : public testing::Test {};

TEST_F(TrackedExecutorTest, reportsOneSamplePerCallback) {
  // Run callbacks inline so the per-metric counts are deterministic.
  TrackedExecutor tracked{
      folly::getKeepAliveToken(folly::InlineExecutor::instance())};

  constexpr int kNumTasks{4};
  for (int i = 0; i < kNumTasks; ++i) {
    tracked.add([] {
      // Spin to produce non-zero wall and cpu time for the callback.
      double sink{0};
      for (int j = 0; j < 1'000'000; ++j) {
        sink += static_cast<double>(j) * 0.5;
      }
      folly::doNotOptimizeAway(sink);
    });
  }

  ConcurrentRuntimeStatWriter writer;
  tracked.reportTo(writer);
  const auto metrics = writer.runtimeStats();

  ASSERT_THAT(
      metrics,
      testing::UnorderedElementsAre(
          testing::Key("executorWaitNanos"),
          testing::Key("executorExecutionWallNanos"),
          testing::Key("executorExecutionCpuNanos")));

  const auto& wait = metrics.at("executorWaitNanos");
  const auto& wall = metrics.at("executorExecutionWallNanos");
  const auto& cpu = metrics.at("executorExecutionCpuNanos");

  // Every scheduled callback contributes one sample to each metric.
  EXPECT_EQ(wait.count, kNumTasks);
  EXPECT_EQ(wall.count, kNumTasks);
  EXPECT_EQ(cpu.count, kNumTasks);

  EXPECT_EQ(wait.unit, RuntimeCounter::Unit::kNanos);
  EXPECT_EQ(wall.unit, RuntimeCounter::Unit::kNanos);
  EXPECT_EQ(cpu.unit, RuntimeCounter::Unit::kNanos);

  EXPECT_GT(wall.sum, 0);
  EXPECT_GT(cpu.sum, 0);
  EXPECT_GE(wait.sum, 0);
}

TEST_F(TrackedExecutorTest, keepsMetricCountsAlignedWhenCallbackThrows) {
  TrackedExecutor tracked{
      folly::getKeepAliveToken(folly::InlineExecutor::instance())};

  // Each metric records one sample even when the callback throws.
  std::atomic<bool> shouldThrow{true};
  EXPECT_THROW(
      tracked.add([&shouldThrow] {
        if (shouldThrow.load()) {
          throw std::runtime_error("boom");
        }
      }),
      std::runtime_error);

  ConcurrentRuntimeStatWriter writer;
  tracked.reportTo(writer);
  const auto metrics = writer.runtimeStats();

  EXPECT_EQ(metrics.at("executorWaitNanos").count, 1);
  EXPECT_EQ(metrics.at("executorExecutionWallNanos").count, 1);
  EXPECT_EQ(metrics.at("executorExecutionCpuNanos").count, 1);
}

} // namespace
} // namespace facebook::velox
