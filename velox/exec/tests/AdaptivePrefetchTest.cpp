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

#include "velox/exec/AdaptivePrefetch.h"
#include <gtest/gtest.h>
#include <functional>
#include <limits>
#include <thread>
#include "velox/common/testutil/TestValue.h"

namespace facebook::velox::exec {
namespace {

TEST(AdaptivePrefetchTest, returnsInitialLookAheadDuringMeasurement) {
  AdaptivePrefetch prefetch(1000);
  for (int i = 0; i < 16; ++i) {
    EXPECT_EQ(prefetch.lookAhead(), 4);
  }
}

TEST(AdaptivePrefetchTest, slowIterationsClampToMin) {
  AdaptivePrefetch prefetch(1000);
  for (int i = 0; i < 16; ++i) {
    prefetch.lookAhead();
    std::this_thread::sleep_for(std::chrono::microseconds(10));
  }
  EXPECT_EQ(prefetch.lookAhead(), 4);
}

TEST(AdaptivePrefetchTest, fastIterationsProduceHighLookAhead) {
  AdaptivePrefetch prefetch(1000);
  for (int i = 0; i < 16; ++i) {
    prefetch.lookAhead();
  }
  auto lookAhead = prefetch.lookAhead();
  EXPECT_GE(lookAhead, 4);
  EXPECT_LE(lookAhead, 32);
}

#ifndef NDEBUG
TEST(AdaptivePrefetchTest, measuredLookAhead) {
  common::testutil::TestValue::enable();
  const struct {
    std::chrono::nanoseconds::rep elapsedNs;
    int32_t lookAhead;
  } testCases[] = {
      {0, 32},
      {-1, 32},
      {1, 32},
      {200, 32},
      {201, 31},
      {400, 16},
      {1'600, 4},
      {1'601, 4},
      {std::numeric_limits<std::chrono::nanoseconds::rep>::max(), 4},
  };
  for (const auto& testCase : testCases) {
    SCOPED_TRACE(testCase.elapsedNs);
    SCOPED_TESTVALUE_SET(
        "facebook::velox::exec::AdaptivePrefetch::computeLookAhead",
        std::function<void(std::chrono::nanoseconds::rep*)>(
            [&](auto* elapsedNs) { *elapsedNs = testCase.elapsedNs; }));
    for (int32_t numIterations : {4, 5, 16, 17, 20, 48, 49, 64}) {
      SCOPED_TRACE(numIterations);
      AdaptivePrefetch prefetch(numIterations);
      for (int32_t i = 0; i < numIterations; ++i) {
        const auto lookAhead = i < 16 ? 4 : testCase.lookAhead;
        EXPECT_EQ(
            prefetch.lookAhead(),
            i + lookAhead < numIterations ? lookAhead : 0);
      }
    }
  }
  common::testutil::TestValue::disable();
}
#endif

TEST(AdaptivePrefetchTest, returnsZeroNearEnd) {
  AdaptivePrefetch prefetch(20);
  int zeroCount = 0;
  for (int i = 0; i < 20; ++i) {
    if (prefetch.lookAhead() == 0) {
      ++zeroCount;
    }
  }
  EXPECT_GT(zeroCount, 0);
}

} // namespace
} // namespace facebook::velox::exec
