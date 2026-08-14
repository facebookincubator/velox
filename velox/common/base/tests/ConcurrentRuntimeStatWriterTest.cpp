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

#include "velox/common/base/ConcurrentRuntimeStatWriter.h"

#include <fmt/format.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <thread>
#include <vector>

#include "velox/common/base/Exceptions.h"
#include "velox/common/base/tests/GTestUtils.h"

namespace facebook::velox {

class ConcurrentRuntimeStatWriterTest : public testing::Test {
 protected:
  ConcurrentRuntimeStatWriter writer_;
};

TEST_F(ConcurrentRuntimeStatWriterTest, addAccumulates) {
  writer_.addRuntimeStat(
      "wall", RuntimeCounter(10, RuntimeCounter::Unit::kNanos));
  writer_.addRuntimeStat(
      "wall", RuntimeCounter(30, RuntimeCounter::Unit::kNanos));

  const auto stats = writer_.runtimeStats();
  const auto& wall = stats.at("wall");
  EXPECT_EQ(wall.count, 2);
  EXPECT_EQ(wall.sum, 40);
  EXPECT_EQ(wall.min, 10);
  EXPECT_EQ(wall.max, 30);
  EXPECT_EQ(wall.unit, RuntimeCounter::Unit::kNanos);
}

TEST_F(ConcurrentRuntimeStatWriterTest, setReplaces) {
  RuntimeMetric preset(RuntimeCounter::Unit::kBytes);
  preset.addValue(5);
  preset.addValue(15);
  writer_.setRuntimeStat("bytes", preset);

  RuntimeMetric replacement(RuntimeCounter::Unit::kBytes);
  replacement.addValue(100);
  writer_.setRuntimeStat("bytes", replacement);

  // The second metric replaces the first rather than merging into it.
  const auto stats = writer_.runtimeStats();
  const auto& bytes = stats.at("bytes");
  EXPECT_EQ(bytes.count, 1);
  EXPECT_EQ(bytes.sum, 100);
}

TEST_F(ConcurrentRuntimeStatWriterTest, unitMismatchThrows) {
  writer_.addRuntimeStat("m", RuntimeCounter(10, RuntimeCounter::Unit::kNanos));
  VELOX_ASSERT_THROW(
      writer_.addRuntimeStat(
          "m", RuntimeCounter(20, RuntimeCounter::Unit::kBytes)),
      "Unit mismatch for runtime stat");

  // The rejected sample leaves the metric untouched.
  const auto stats = writer_.runtimeStats();
  const auto& metric = stats.at("m");
  EXPECT_EQ(metric.count, 1);
  EXPECT_EQ(metric.sum, 10);
  EXPECT_EQ(metric.unit, RuntimeCounter::Unit::kNanos);
}

TEST_F(ConcurrentRuntimeStatWriterTest, unitMismatchThrowsAfterSet) {
  // setRuntimeStat replaces without checking the unit, so the add path is what
  // catches a mismatch on a name seeded by a set.
  RuntimeMetric seeded(RuntimeCounter::Unit::kBytes);
  seeded.addValue(10);
  writer_.setRuntimeStat("m", seeded);

  VELOX_ASSERT_THROW(
      writer_.addRuntimeStat(
          "m", RuntimeCounter(20, RuntimeCounter::Unit::kNanos)),
      "Unit mismatch for runtime stat");
}

TEST_F(ConcurrentRuntimeStatWriterTest, unitHelpersApplyTheirUnit) {
  writer_.addCount("splits", 3);
  writer_.addBytes("read", 128);

  const auto stats = writer_.runtimeStats();
  EXPECT_EQ(stats.at("splits").sum, 3);
  EXPECT_EQ(stats.at("splits").unit, RuntimeCounter::Unit::kNone);
  EXPECT_EQ(stats.at("read").sum, 128);
  EXPECT_EQ(stats.at("read").unit, RuntimeCounter::Unit::kBytes);
}

TEST_F(ConcurrentRuntimeStatWriterTest, clearDropsEverything) {
  writer_.addRuntimeStat("a", RuntimeCounter(1));
  writer_.addRuntimeStat("b", RuntimeCounter(2));
  writer_.clear();
  EXPECT_THAT(writer_.runtimeStats(), testing::IsEmpty());

  // A cleared name may take a new unit, since nothing survives the clear.
  writer_.addRuntimeStat("a", RuntimeCounter(5, RuntimeCounter::Unit::kBytes));
  const auto stats = writer_.runtimeStats();
  EXPECT_EQ(stats.at("a").sum, 5);
  EXPECT_EQ(stats.at("a").unit, RuntimeCounter::Unit::kBytes);
}

TEST_F(ConcurrentRuntimeStatWriterTest, concurrentAddIsLossless) {
  constexpr int32_t kNumThreads{8};
  constexpr int32_t kSamplesPerThread{2'000};

  // Every thread adds to one shared name and to a name only it uses, so the
  // test covers both contended and uncontended keys.
  std::vector<std::thread> threads;
  threads.reserve(kNumThreads);
  for (int32_t thread = 0; thread < kNumThreads; ++thread) {
    threads.emplace_back([&, thread] {
      const auto ownName = fmt::format("perThread{}", thread);
      for (int32_t sample = 0; sample < kSamplesPerThread; ++sample) {
        writer_.addCount("shared", 1);
        writer_.addCount(ownName, 1);
      }
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }

  const auto stats = writer_.runtimeStats();
  const auto& shared = stats.at("shared");
  EXPECT_EQ(shared.count, kNumThreads * kSamplesPerThread);
  EXPECT_EQ(shared.sum, kNumThreads * kSamplesPerThread);

  for (int32_t thread = 0; thread < kNumThreads; ++thread) {
    const auto& perThread = stats.at(fmt::format("perThread{}", thread));
    EXPECT_EQ(perThread.count, kSamplesPerThread);
    EXPECT_EQ(perThread.sum, kSamplesPerThread);
  }
}

} // namespace facebook::velox
