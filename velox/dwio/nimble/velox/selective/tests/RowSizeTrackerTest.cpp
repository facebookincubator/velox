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

#include "velox/dwio/nimble/velox/selective/RowSizeTracker.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

using namespace ::testing;
using namespace facebook::velox;

namespace facebook::nimble {

class RowSizeTrackerTest : public velox::test::VectorTestBase, public Test {
 protected:
  static void SetUpTestCase() {
    memory::MemoryManager::initialize(memory::MemoryManager::Options{});
  }

  void testInit(
      velox::RowTypePtr schema,
      std::vector<vector_size_t> variableLengthNodes) {
    // Create the same TypeWithId as reference.
    std::shared_ptr<const dwio::common::TypeWithId> schemaWithId =
        dwio::common::TypeWithId::create(schema);

    RowSizeTracker tracker(schemaWithId);

    EXPECT_EQ(tracker.cellSizes_.size(), schemaWithId->maxId() + 1);

    velox::bits::forEachBit(
        tracker.variableLengthNodes_.allBits(),
        0,
        tracker.variableLengthNodes_.end(),
        true,
        [&](velox::vector_size_t col) {
          EXPECT_EQ(
              1,
              std::count(
                  variableLengthNodes.begin(), variableLengthNodes.end(), col))
              << fmt::format(
                     "variable length col {} not correctly marked after initialization",
                     col);
        });
  }
};

TEST_F(RowSizeTrackerTest, initialization) {
  testInit(ROW({INTEGER(), VARCHAR()}), {2});
  testInit(ROW({INTEGER(), ROW({VARCHAR(), DOUBLE()})}), {3});
  testInit(ROW({ARRAY(INTEGER()), ROW({VARCHAR(), DOUBLE()})}), {1, 2, 4});
  testInit(
      ROW({ARRAY(INTEGER()), ROW({ROW({VARCHAR(), DOUBLE()}), VARCHAR()})}),
      {1, 2, 5, 7});
  testInit(
      ROW(
          {ARRAY(INTEGER()),
           ROW(
               {ROW({VARCHAR(), DOUBLE(), MAP(INTEGER(), ROW({REAL()}))}),
                VARCHAR()})}),
      {1, 2, 5, 7, 8, 9, 10, 11});
}

TEST_F(RowSizeTrackerTest, basicUpdates) {
  auto schema = ROW({ARRAY(INTEGER()), ROW({VARCHAR(), DOUBLE()})});
  std::shared_ptr<const dwio::common::TypeWithId> schemaWithId =
      dwio::common::TypeWithId::create(schema);
  RowSizeTracker tracker(schemaWithId);
  for (size_t i = 0; i < 6; ++i) {
    tracker.applyProjection(i, /*projected=*/true);
  }
  tracker.finalizeProjection();

  tracker.update(1, 2, 2);
  tracker.update(2, 4, 2);
  tracker.update(3, 8, 2);
  tracker.update(4, 16, 2);
  tracker.update(5, 32, 2);
  EXPECT_EQ(tracker.getCurrentMaxRowSize(), 31);

  tracker.update(2, 32, 4);
  EXPECT_EQ(tracker.getCurrentMaxRowSize(), 37);
  tracker.update(2, 4, 4);
  EXPECT_EQ(tracker.getCurrentMaxRowSize(), 37);

  tracker.update(4, 8, 2);
  EXPECT_EQ(tracker.getCurrentMaxRowSize(), 37);
}

TEST_F(RowSizeTrackerTest, partialMarterialization) {
  auto schema = ROW({ARRAY(INTEGER()), ROW({VARCHAR(), DOUBLE()})});
  std::shared_ptr<const dwio::common::TypeWithId> schemaWithId =
      dwio::common::TypeWithId::create(schema);
  constexpr size_t kFallbackValue = 1UL << 20;
  // Updating all variable length columns first
  {
    RowSizeTracker tracker(schemaWithId);
    for (size_t i = 0; i < 6; ++i) {
      tracker.applyProjection(i, /*projected=*/true);
    }
    tracker.finalizeProjection();

    tracker.update(1, 0, 2);
    EXPECT_EQ(tracker.getCurrentMaxRowSize(), kFallbackValue);
    tracker.update(2, 128, 2);
    EXPECT_EQ(tracker.getCurrentMaxRowSize(), kFallbackValue);
    tracker.update(4, 48, 2);
    EXPECT_EQ(tracker.getCurrentMaxRowSize(), 88);
    tracker.update(5, 32, 2);
    EXPECT_EQ(tracker.getCurrentMaxRowSize(), 104);
    tracker.update(3, 8, 2);
    EXPECT_EQ(tracker.getCurrentMaxRowSize(), 108);
  }
  // Updating all variable length columns last
  {
    RowSizeTracker tracker(schemaWithId);
    for (size_t i = 0; i < 6; ++i) {
      tracker.applyProjection(i, /*projected=*/true);
    }
    tracker.finalizeProjection();

    tracker.update(5, 32, 2);
    EXPECT_EQ(tracker.getCurrentMaxRowSize(), kFallbackValue);
    tracker.update(3, 8, 2);
    EXPECT_EQ(tracker.getCurrentMaxRowSize(), kFallbackValue);
    tracker.update(1, 0, 2);
    EXPECT_EQ(tracker.getCurrentMaxRowSize(), kFallbackValue);
    tracker.update(2, 128, 2);
    EXPECT_EQ(tracker.getCurrentMaxRowSize(), kFallbackValue);
    tracker.update(4, 48, 2);
    EXPECT_EQ(tracker.getCurrentMaxRowSize(), 108);
  }
  // Updating in traversal order
  {
    RowSizeTracker tracker(schemaWithId);
    for (size_t i = 0; i < 6; ++i) {
      tracker.applyProjection(i, /*projected=*/true);
    }
    tracker.finalizeProjection();

    tracker.update(1, 0, 2);
    EXPECT_EQ(tracker.getCurrentMaxRowSize(), kFallbackValue);
    tracker.update(2, 128, 2);
    EXPECT_EQ(tracker.getCurrentMaxRowSize(), kFallbackValue);
    tracker.update(3, 8, 2);
    EXPECT_EQ(tracker.getCurrentMaxRowSize(), kFallbackValue);
    tracker.update(4, 48, 2);
    EXPECT_EQ(tracker.getCurrentMaxRowSize(), 92);
    tracker.update(5, 32, 2);
    EXPECT_EQ(tracker.getCurrentMaxRowSize(), 108);
  }
}

TEST_F(RowSizeTrackerTest, oversizeRow) {
  auto schema = ROW({ARRAY(INTEGER()), ROW({VARCHAR(), DOUBLE()})});
  std::shared_ptr<const dwio::common::TypeWithId> schemaWithId =
      dwio::common::TypeWithId::create(schema);
  RowSizeTracker tracker(schemaWithId);
  for (size_t i = 0; i < 6; ++i) {
    tracker.applyProjection(i, /*projected=*/true);
  }
  tracker.finalizeProjection();

  constexpr size_t kFallbackValue = 1UL << 20;
  EXPECT_EQ(tracker.getCurrentMaxRowSize(), kFallbackValue);
  // One materialized node already surpassed the fallback value 1MB.
  tracker.update(2, 4 * kFallbackValue, 2);
  EXPECT_EQ(tracker.getCurrentMaxRowSize(), 2 * kFallbackValue);

  tracker.update(4, 4 * kFallbackValue, 2);
  EXPECT_EQ(tracker.getCurrentMaxRowSize(), 4 * kFallbackValue);
}
} // namespace facebook::nimble
