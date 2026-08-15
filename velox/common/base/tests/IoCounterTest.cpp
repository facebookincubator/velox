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

#include "velox/common/base/IoCounter.h"

#include <limits>

#include <gtest/gtest.h>

namespace facebook::velox::io {

TEST(IoCounterTest, increment) {
  IoCounter counter;
  EXPECT_EQ(counter.count(), 0);
  EXPECT_EQ(counter.sum(), 0);
  EXPECT_EQ(counter.min(), std::numeric_limits<uint64_t>::max());
  EXPECT_EQ(counter.max(), 0);

  counter.increment(100);
  counter.increment(200);
  counter.increment(50);

  EXPECT_EQ(counter.count(), 3);
  EXPECT_EQ(counter.sum(), 350);
  EXPECT_EQ(counter.min(), 50);
  EXPECT_EQ(counter.max(), 200);
}

TEST(IoCounterTest, merge) {
  IoCounter counter;
  counter.increment(100);
  counter.increment(200);

  IoCounter other;
  other.increment(50);
  other.increment(500);

  counter.merge(other);
  EXPECT_EQ(counter.count(), 4);
  EXPECT_EQ(counter.sum(), 850);
  EXPECT_EQ(counter.min(), 50);
  EXPECT_EQ(counter.max(), 500);
}

TEST(IoCounterTest, copyConstructor) {
  IoCounter counter;
  counter.increment(100);
  counter.increment(200);
  counter.increment(50);

  IoCounter copied(counter);
  EXPECT_EQ(copied.count(), 3);
  EXPECT_EQ(copied.sum(), 350);
  EXPECT_EQ(copied.min(), 50);
  EXPECT_EQ(copied.max(), 200);

  copied.increment(1);
  EXPECT_EQ(counter.count(), 3);
  EXPECT_EQ(counter.sum(), 350);
}

TEST(IoCounterTest, copyAssignment) {
  IoCounter counter;
  counter.increment(100);
  counter.increment(200);

  IoCounter assigned;
  assigned.increment(999);
  assigned = counter;
  EXPECT_EQ(assigned.count(), 2);
  EXPECT_EQ(assigned.sum(), 300);
  EXPECT_EQ(assigned.min(), 100);
  EXPECT_EQ(assigned.max(), 200);
}

TEST(IoCounterTest, copyDefaultEmpty) {
  IoCounter counter;
  IoCounter copied(counter);
  EXPECT_EQ(copied.count(), 0);
  EXPECT_EQ(copied.sum(), 0);
  EXPECT_EQ(copied.min(), std::numeric_limits<uint64_t>::max());
  EXPECT_EQ(copied.max(), 0);

  copied.increment(1);
  EXPECT_EQ(counter.count(), 0);
  EXPECT_EQ(counter.sum(), 0);
}

} // namespace facebook::velox::io
