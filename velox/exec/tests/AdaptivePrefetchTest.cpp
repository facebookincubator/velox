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
#include <thread>

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
  EXPECT_GT(prefetch.lookAhead(), 4);
}

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
