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

#include "velox/common/memory/MemoryTier.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "velox/common/base/Exceptions.h"
#include "velox/common/base/tests/GTestUtils.h"

namespace facebook::velox::memory {
namespace {

using ::testing::ElementsAre;

TEST(MemoryTierTest, accessors) {
  MemoryTier tier({{"rmm"}, {"pinnedHost"}, {"cxl", "dram"}});
  EXPECT_EQ(tier.numTiers(), 3);
  EXPECT_THAT(tier.tierAt(0), ElementsAre("rmm"));
  EXPECT_THAT(tier.tierAt(1), ElementsAre("pinnedHost"));
  EXPECT_THAT(tier.tierAt(2), ElementsAre("cxl", "dram"));
}

TEST(MemoryTierTest, tierAtOutOfRangeThrows) {
  MemoryTier tier({{"rmm"}, {"dram"}});
  VELOX_ASSERT_THROW(tier.tierAt(2), "MemoryTier level out of range");
}

TEST(MemoryTierTest, emptyHierarchyThrows) {
  VELOX_ASSERT_THROW(MemoryTier({}), "MemoryTier requires at least one tier");
}

TEST(MemoryTierTest, emptyLevelThrows) {
  VELOX_ASSERT_THROW(
      MemoryTier({{"rmm"}, {}}), "MemoryTier level must have at least one tag");
}

TEST(MemoryTierTest, emptyTagThrows) {
  VELOX_ASSERT_THROW(
      MemoryTier({{"rmm"}, {""}}), "MemoryTier tag must not be empty");
}

TEST(MemoryTierTest, duplicateTagWithinLevelThrows) {
  VELOX_ASSERT_THROW(
      MemoryTier({{"cxl", "cxl"}}), "Duplicate tag in MemoryTier: cxl");
}

TEST(MemoryTierTest, duplicateTagAcrossLevelsThrows) {
  VELOX_ASSERT_THROW(
      MemoryTier({{"dram"}, {"dram"}}), "Duplicate tag in MemoryTier: dram");
}

TEST(MemoryTierTest, levelOf) {
  MemoryTier tier({{"rmm"}, {"pinnedHost"}, {"cxl", "dram"}});
  EXPECT_EQ(tier.levelOf("rmm"), 0);
  EXPECT_EQ(tier.levelOf("pinnedHost"), 1);
  EXPECT_EQ(tier.levelOf("cxl"), 2);
  EXPECT_EQ(tier.levelOf("dram"), 2);
  EXPECT_EQ(tier.levelOf("unknown"), std::nullopt);
}

TEST(MemoryTierTest, toString) {
  MemoryTier tier({{"rmm"}, {"pinnedHost"}, {"cxl", "dram"}});
  EXPECT_EQ(tier.toString(), "MemoryTier[[rmm], [pinnedHost], [cxl, dram]]");
}

} // namespace
} // namespace facebook::velox::memory
