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

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <numeric>

#include "velox/experimental/torchwave/LaunchPartition.h"

namespace torch::wave {
namespace {

using ::testing::ElementsAre;

// A small machine, so a wave is 8 blocks and the arithmetic in the
// expectations stays checkable by hand. 64 KB of shared memory per SM means an
// op asking for 32 KB drops to 2 blocks per SM and one asking for 64 KB to 1.
GridDevice smallDevice() {
  return {
      .numSMs = 4,
      .maxBlocksPerSM = 2,
      .sharedPerSM = 64 * 1024,
      .staticSharedPerBlock = 0};
}

GridOp op(float cost, int32_t maxBlocks, int32_t dynamicShared = 0) {
  return {
      .cost = cost,
      .maxBlocks = maxBlocks,
      .dynamicShared = dynamicShared,
      .alwaysSingleBlock = false,
      .hasBarrier = false};
}

// Op index of every block, in emit order.
std::vector<int32_t> blockOps(const LaunchPlan& plan) {
  std::vector<int32_t> ops;
  ops.reserve(plan.blocks.size());
  for (const auto& block : plan.blocks) {
    ops.push_back(block.op);
  }
  return ops;
}

// Blocks in each segment, in launch order.
std::vector<int32_t> segmentSizes(const LaunchPlan& plan) {
  std::vector<int32_t> sizes;
  sizes.reserve(plan.segments.size());
  for (const auto& segment : plan.segments) {
    sizes.push_back(segment.numBlocks);
  }
  return sizes;
}

// Every op must see one contiguous 0..n-1 series of blockInOp values, and n
// must be what its blocks report as numBlocksInOp -- a block computes its
// slice from the pair, so a gap or a mismatch silently skips work. Straddling
// a launch boundary must not disturb that.
void expectCompleteBlockSeries(const LaunchPlan& plan) {
  std::vector<std::vector<int32_t>> seen(plan.blocksPerOp.size());
  for (const auto& block : plan.blocks) {
    ASSERT_LT(block.op, static_cast<int32_t>(seen.size()));
    seen[block.op].push_back(block.blockInOp);
  }
  for (size_t i = 0; i < seen.size(); ++i) {
    std::sort(seen[i].begin(), seen[i].end());
    std::vector<int32_t> expected(plan.blocksPerOp[i]);
    std::iota(expected.begin(), expected.end(), 0);
    EXPECT_EQ(seen[i], expected) << "op " << i;
  }
}

// Segments must tile the block array exactly: the launch site hands each one a
// pointer into the middle of it, so a gap or an overlap would run the wrong
// blocks or none at all.
void expectSegmentsTileBlocks(const LaunchPlan& plan) {
  int32_t next = 0;
  for (const auto& segment : plan.segments) {
    EXPECT_EQ(segment.firstBlock, next);
    EXPECT_GT(segment.numBlocks, 0);
    next += segment.numBlocks;
  }
  EXPECT_EQ(next, static_cast<int32_t>(plan.blocks.size()));
}

// A cooperative launch runs only if every one of its blocks is co-resident, so
// one carrying more than its occupancy allows does not run at all -- it fails
// with "too many blocks in cooperative launch".
void expectCooperativeSegmentsFitAWave(
    const LaunchPlan& plan,
    const GridDevice& device) {
  for (size_t s = 0; s < plan.segments.size(); ++s) {
    const auto& segment = plan.segments[s];
    if (!segment.cooperative) {
      continue;
    }
    EXPECT_LE(
        segment.numBlocks,
        device.numSMs * blocksPerSM(device, segment.dynamicShared))
        << "segment " << s;
  }
}

TEST(LaunchPartitionTest, sharedMemoryCutsBlocksPerSM) {
  const auto device = smallDevice();
  EXPECT_EQ(blocksPerSM(device, 0), 2);
  EXPECT_EQ(blocksPerSM(device, 32 * 1024), 2);
  EXPECT_EQ(blocksPerSM(device, 33 * 1024), 1);
  // Never below one block: a block that does not fit still has to run.
  EXPECT_EQ(blocksPerSM(device, 1024 * 1024), 1);
}

TEST(LaunchPartitionTest, skewIsTheMakespanOverABalancedWave) {
  // Two ops of equal cost, each on half the wave: perfectly balanced.
  const std::vector<GridOp> balanced = {op(100, 8), op(100, 8)};
  EXPECT_NEAR(gridStats(balanced, {4, 4}, smallDevice()).skew, 1.0f, 0.01f);

  // Same total work, but one op holds it all on one block. Its 200/1 against a
  // balanced 200/8 is eight times the makespan.
  const std::vector<GridOp> lopsided = {op(200, 8), op(0.0f, 8)};
  EXPECT_NEAR(gridStats(lopsided, {1, 7}, smallDevice()).skew, 8.0f, 0.01f);
}

TEST(LaunchPartitionTest, countsOpsStuckOnOneBlock) {
  // Only the op that could have used more blocks is starved; an
  // alwaysSingleBlock op (maxBlocks 1) is on one block by construction.
  const std::vector<GridOp> ops = {op(10, 4), op(10, 1), op(10, 4)};
  const auto stats = gridStats(ops, {1, 1, 3}, smallDevice());
  EXPECT_EQ(stats.numStarved, 1);
}

TEST(LaunchPartitionTest, reportsOccupancyLostToOneHungryOp) {
  // The 48 KB op leaves room for one block per SM, so the whole step launches
  // at 1 although the other two would run at 2. It holds a tenth of the cost.
  const std::vector<GridOp> ops = {op(90, 8), op(90, 8), op(20, 8, 48 * 1024)};
  const auto stats = gridStats(ops, {4, 3, 1}, smallDevice());
  EXPECT_EQ(stats.occupancy, 1);
  EXPECT_EQ(stats.bestOccupancy, 2);
  EXPECT_EQ(stats.targetBlocks, 4);
  EXPECT_NEAR(stats.poison, 0.1f, 0.01f);
}

TEST(LaunchPartitionTest, singleLaunchKeepsOpOrderByDefault) {
  const std::vector<GridOp> ops = {op(1, 4), op(100, 4)};
  const auto plan = singleLaunchPlan(ops, {2, 3}, /*orderByLatency=*/false);

  EXPECT_THAT(blockOps(plan), ElementsAre(0, 0, 1, 1, 1));
  EXPECT_THAT(segmentSizes(plan), ElementsAre(5));
  expectCompleteBlockSeries(plan);
  expectSegmentsTileBlocks(plan);
}

TEST(LaunchPartitionTest, singleLaunchPutsTheLongPolesFirst) {
  // op 1 projects at 100/3 per block against op 0's 1/2, so its blocks go out
  // first and op 0 backfills.
  const std::vector<GridOp> ops = {op(1, 4), op(100, 4)};
  const auto plan = singleLaunchPlan(ops, {2, 3}, /*orderByLatency=*/true);

  EXPECT_THAT(blockOps(plan), ElementsAre(1, 1, 1, 0, 0));
  EXPECT_THAT(segmentSizes(plan), ElementsAre(5));
  expectCompleteBlockSeries(plan);
}

TEST(LaunchPartitionTest, gateIgnoresABalancedStep) {
  const std::vector<GridOp> ops = {op(100, 8), op(100, 8)};
  const auto stats = gridStats(ops, {4, 4}, smallDevice());

  EXPECT_FALSE(shouldPartition(stats, PartitionParams{}));
  EXPECT_FALSE(partitionLaunches(ops, smallDevice(), stats, PartitionParams{})
                   .has_value());
}

TEST(LaunchPartitionTest, gateFiresOnLostOccupancyWithoutSkew) {
  // Balanced, so nothing to rebalance, but the cheap 48 KB op is costing the
  // other two half their occupancy. That alone is worth a split.
  const std::vector<GridOp> ops = {op(100, 8), op(100, 8), op(1, 8, 48 * 1024)};
  const auto stats = gridStats(ops, {2, 2, 1}, smallDevice());

  EXPECT_EQ(stats.occupancy, 1);
  EXPECT_EQ(stats.bestOccupancy, 2);
  EXPECT_TRUE(shouldPartition(stats, PartitionParams{}));
}

TEST(LaunchPartitionTest, gateIgnoresOccupancyTheExpensiveOpsNeed) {
  // Same lost occupancy, but now the hungry op holds most of the cost, so the
  // occupancy it forces is the one the step needs anyway.
  const std::vector<GridOp> ops = {op(10, 8), op(10, 8), op(200, 8, 48 * 1024)};
  auto stats = gridStats(ops, {1, 1, 2}, smallDevice());
  // Keep skew out of it; this test is about the poison term alone.
  stats.skew = 1.0f;

  EXPECT_LT(stats.occupancy, stats.bestOccupancy);
  EXPECT_FALSE(shouldPartition(stats, PartitionParams{}));
}

TEST(LaunchPartitionTest, starvedStepBecomesSeveralFullWaves) {
  // Sixteen ops too cheap to earn a block plus one that wants the whole wave:
  // seventeen ops for eight blocks, which is the starvation case. Every op
  // still gets its block, but they come out as three packed waves instead of
  // one launch three waves deep.
  std::vector<GridOp> ops = {op(1000, 8)};
  for (int32_t i = 0; i < 16; ++i) {
    ops.push_back(op(1, 1));
  }
  const std::vector<int32_t> blocks(ops.size(), 1);
  const auto stats = gridStats(ops, blocks, smallDevice());
  const auto plan =
      partitionLaunches(ops, smallDevice(), stats, PartitionParams{});

  ASSERT_TRUE(plan.has_value());
  EXPECT_EQ(plan->blocksPerOp[0], 8);
  EXPECT_THAT(segmentSizes(*plan), ElementsAre(8, 8, 8));
  expectCompleteBlockSeries(*plan);
  expectSegmentsTileBlocks(*plan);
}

TEST(LaunchPartitionTest, hungryOpGetsALaunchOfItsOwnAtItsOwnOccupancy) {
  // The 48 KB op runs 1 block per SM, the others 2, so they cannot share a
  // launch. The split is what lets the cheap ops keep the wave of 8 the hungry
  // one would otherwise have cut to 4.
  const std::vector<GridOp> ops = {
      op(100, 8), op(100, 8), op(20, 4, 48 * 1024)};
  const auto stats = gridStats(ops, {2, 1, 1}, smallDevice());
  const auto plan =
      partitionLaunches(ops, smallDevice(), stats, PartitionParams{});

  ASSERT_TRUE(plan.has_value());
  ASSERT_EQ(plan->segments.size(), 2);
  // Most of the cost is in the shared-memory-free pair, so their launch goes
  // first and asks for no shared memory at all.
  EXPECT_EQ(plan->segments[0].dynamicShared, 0);
  EXPECT_EQ(plan->segments[1].dynamicShared, 48 * 1024);
  expectCompleteBlockSeries(*plan);
  expectSegmentsTileBlocks(*plan);
}

// Segments an op's blocks appear in.
std::vector<int32_t> segmentsWith(const LaunchPlan& plan, int32_t opIndex) {
  std::vector<int32_t> found;
  for (size_t s = 0; s < plan.segments.size(); ++s) {
    const auto& segment = plan.segments[s];
    for (int32_t b = 0; b < segment.numBlocks; ++b) {
      if (plan.blocks[segment.firstBlock + b].op == opIndex) {
        found.push_back(static_cast<int32_t>(s));
        break;
      }
    }
  }
  return found;
}

// The wants a step of this shape produces never exceed one wave, so the
// barrier and budget rules below cannot be reached through gridStats. Both
// guard the packer's contract with its caller rather than a case makeGrid
// produces today, so they are exercised against a wider target than this
// device has -- which is what a real machine would hand them.
GridStats statsWithTarget(
    const std::vector<GridOp>& ops,
    int32_t targetBlocks) {
  const std::vector<int32_t> blocks(ops.size(), 1);
  auto stats = gridStats(ops, blocks, smallDevice());
  stats.targetBlocks = targetBlocks;
  return stats;
}

TEST(LaunchPartitionTest, aBarrierOpIsShrunkToALaunchRatherThanSplit) {
  // opBarrier waits for numBlocksInOp arrivals, so every block of the op has
  // to be co-resident in one launch. Its group holds a wave of 8, so it is cut
  // to 8 however much work it carries -- and blocksPerOp has to say 8 too, or
  // the blocks that did launch would skip the slices of the ones that did not.
  std::vector<GridOp> ops = {op(1000, 64), op(10, 8), op(10, 8)};
  ops[0].hasBarrier = true;
  const auto plan = partitionLaunches(
      ops, smallDevice(), statsWithTarget(ops, 64), PartitionParams{});

  ASSERT_TRUE(plan.has_value());
  EXPECT_EQ(plan->blocksPerOp[0], 8);
  EXPECT_THAT(segmentsWith(*plan, 0), ElementsAre(0));
  // Only the launch holding the barrier op has to be cooperative; the ops with
  // no barrier keep the ordinary launch, which has no co-residency limit.
  ASSERT_EQ(plan->segments.size(), 2);
  EXPECT_TRUE(plan->segments[0].cooperative);
  EXPECT_FALSE(plan->segments[1].cooperative);
  expectCompleteBlockSeries(*plan);
  expectSegmentsTileBlocks(*plan);
}

TEST(LaunchPartitionTest, aSingleBlockBarrierOpNeedsNoCooperativeLaunch) {
  // opBarrier waits for numBlocksInOp arrivals, which a lone block satisfies
  // as soon as it runs. Demanding a cooperative launch for it would cap the
  // grid at what fits co-resident for no reason.
  std::vector<GridOp> ops = {op(100, 8), op(1, 1)};
  ops[1].hasBarrier = true;

  auto plan = singleLaunchPlan(ops, {4, 1}, /*orderByLatency=*/false);
  ASSERT_EQ(plan.segments.size(), 1);
  EXPECT_FALSE(plan.segments[0].cooperative);

  plan = singleLaunchPlan(ops, {4, 2}, /*orderByLatency=*/false);
  EXPECT_TRUE(plan.segments[0].cooperative);
}

TEST(LaunchPartitionTest, aBarrierFreeOpMayStraddleTwoLaunches) {
  // Three ops of 3 blocks each into a wave of 8: the third does not fit whole.
  // It has no barrier, so its blocks find their slice from blockInOp and
  // numBlocksInOp wherever they were launched from, and the remainder simply
  // runs in the next launch.
  std::vector<GridOp> ops = {op(100, 8), op(100, 8), op(100, 8)};
  for (int32_t i = 0; i < 12; ++i) {
    ops.push_back(op(0.1f, 1));
  }
  const std::vector<int32_t> blocks(ops.size(), 1);
  const auto stats = gridStats(ops, blocks, smallDevice());
  const auto plan =
      partitionLaunches(ops, smallDevice(), stats, PartitionParams{});

  ASSERT_TRUE(plan.has_value());
  EXPECT_EQ(plan->blocksPerOp[2], 3);
  EXPECT_THAT(segmentsWith(*plan, 2), ElementsAre(0, 1));
  expectCompleteBlockSeries(*plan);
  expectSegmentsTileBlocks(*plan);
}

TEST(LaunchPartitionTest, aThinTailIsFoldedBackIntoThePreviousLaunch) {
  // Nine ops, one block each, into a wave of eight. Serializing a whole launch
  // for the ninth block would cost more than the imbalance it was split off to
  // fix, so it rides along with the first launch and the step stays whole.
  const std::vector<GridOp> ops(9, op(1, 1));
  const std::vector<int32_t> blocks(ops.size(), 1);
  const auto stats = gridStats(ops, blocks, smallDevice());

  EXPECT_FALSE(partitionLaunches(ops, smallDevice(), stats, PartitionParams{})
                   .has_value());
}

TEST(LaunchPartitionTest, barrierOpsAreSizedToWholeWaves) {
  // Two barrier ops wanting five blocks each, into a wave of eight. Packed
  // greedily they take a launch apiece and neither fills one, and folding the
  // tail back would put ten blocks in a cooperative launch that holds eight.
  // Their block counts are free, so the pair is scaled to four blocks each
  // instead: the same work, one full wave, one launch.
  std::vector<GridOp> ops = {op(600, 6), op(600, 6), op(200, 8), op(200, 8)};
  ops[0].hasBarrier = true;
  ops[1].hasBarrier = true;
  const auto plan = partitionLaunches(
      ops, smallDevice(), statsWithTarget(ops, 64), PartitionParams{});

  ASSERT_TRUE(plan.has_value());
  EXPECT_THAT(plan->blocksPerOp, ElementsAre(4, 4, 6, 6));
  ASSERT_EQ(plan->segments.size(), 2);
  EXPECT_TRUE(plan->segments[0].cooperative);
  EXPECT_EQ(plan->segments[0].numBlocks, 8);
  EXPECT_FALSE(plan->segments[1].cooperative);
  expectCooperativeSegmentsFitAWave(*plan, smallDevice());
  expectCompleteBlockSeries(*plan);
  expectSegmentsTileBlocks(*plan);
}

TEST(LaunchPartitionTest, splittableWorkFillsWhatABarrierOpLeavesOfItsWave) {
  // A barrier op that can only use three blocks leaves five of its wave empty,
  // and a cooperative launch cannot be widened to use them. Ops of the same
  // occupancy class can ride along in that space rather than wait for a launch
  // of their own: the launch's shared memory is the max over its ops, and the
  // class is exactly the set that yields one blocksPerSM, so the passengers
  // cannot cost the barrier op the co-residency it was sized against.
  std::vector<GridOp> ops = {op(600, 3), op(200, 8), op(200, 8)};
  ops[0].hasBarrier = true;
  const auto plan = partitionLaunches(
      ops, smallDevice(), statsWithTarget(ops, 64), PartitionParams{});

  ASSERT_TRUE(plan.has_value());
  EXPECT_THAT(plan->blocksPerOp, ElementsAre(3, 8, 8));
  EXPECT_THAT(segmentSizes(*plan), ElementsAre(8, 8, 3));
  ASSERT_FALSE(plan->segments.empty());
  EXPECT_TRUE(plan->segments[0].cooperative);
  // The last op straddles the cooperative launch and one of its own.
  EXPECT_THAT(segmentsWith(*plan, 2), ElementsAre(0, 2));
  expectCooperativeSegmentsFitAWave(*plan, smallDevice());
  expectCompleteBlockSeries(*plan);
  expectSegmentsTileBlocks(*plan);
}

TEST(LaunchPartitionTest, blocksStayWithinTheReservedBudget) {
  // Four ops that would each want sixteen blocks. maxWaves caps the step at
  // two waves, which is exactly what the block array was reserved for.
  const std::vector<GridOp> ops = {
      op(1000, 64), op(1000, 64), op(1000, 64), op(1000, 64)};
  const auto plan = partitionLaunches(
      ops,
      smallDevice(),
      statsWithTarget(ops, 64),
      PartitionParams{.maxWaves = 2});

  ASSERT_TRUE(plan.has_value());
  EXPECT_THAT(plan->blocksPerOp, ElementsAre(4, 4, 4, 4));
  EXPECT_THAT(segmentSizes(*plan), ElementsAre(8, 8));
  expectCompleteBlockSeries(*plan);
  expectSegmentsTileBlocks(*plan);
}

TEST(LaunchPartitionTest, aLongerMinimumBlockKeepsTheStepInOneLaunch) {
  // The same step that splits into four launches on an even division stays
  // whole once a block is required to be worth 100 units of cost: four ops of
  // 100 want one block each, which is half a wave.
  const std::vector<GridOp> ops = {
      op(100, 8), op(100, 8), op(100, 8), op(100, 8)};
  const auto stats = gridStats(ops, {2, 2, 2, 2}, smallDevice());
  const auto plan = partitionLaunches(
      ops,
      smallDevice(),
      stats,
      PartitionParams{.maxWaves = 4, .minBlockCost = 100.0f});

  EXPECT_FALSE(plan.has_value());
}

} // namespace
} // namespace torch::wave
