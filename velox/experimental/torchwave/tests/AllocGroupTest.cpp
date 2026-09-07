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

#include <ATen/cuda/CUDAContext.h>
#include <folly/ScopeGuard.h>
#include <folly/init/Init.h>

#include "velox/experimental/torchwave/AllocGroup.h"
#include "velox/experimental/torchwave/WaveConfig.h"

namespace torch::wave {
namespace {

using ::testing::ElementsAre;

TEST(AllocGroupTest, alignsToSixteenBytes) {
  EXPECT_EQ(alignAllocSize(0), 0);
  EXPECT_EQ(alignAllocSize(1), 16);
  EXPECT_EQ(alignAllocSize(16), 16);
  EXPECT_EQ(alignAllocSize(17), 32);
}

// The grouping is by (allocStep, freeStep): two values that die at different
// steps must not share a buffer, or the longer-lived one pins the shorter one's
// bytes for as long as it lives.
TEST(AllocGroupTest, groupsBySharedLifetime) {
  std::vector<AllocLifetime> lifetimes = {
      {.actualId = 10, .dtype = at::kFloat, .allocStep = 0, .freeStep = 3},
      {.actualId = 11, .dtype = at::kInt, .allocStep = 0, .freeStep = 3},
      {.actualId = 12, .dtype = at::kFloat, .allocStep = 0, .freeStep = 5},
      {.actualId = 13, .dtype = at::kFloat, .allocStep = 1, .freeStep = 3},
  };
  auto groups = buildAllocGroups(lifetimes);

  ASSERT_EQ(groups.size(), 3);
  EXPECT_THAT(groups[0].actualIds, ElementsAre(10, 11));
  EXPECT_EQ(groups[0].allocStep, 0);
  EXPECT_EQ(groups[0].freeStep, 3);
  EXPECT_THAT(groups[1].actualIds, ElementsAre(12));
  EXPECT_EQ(groups[1].freeStep, 5);
  EXPECT_THAT(groups[2].actualIds, ElementsAre(13));
  EXPECT_EQ(groups[2].allocStep, 1);
}

// A value with no release step outlives the invocation. Grouping it would keep
// every other slot in its group alive with it, so it is handed back to be
// allocated on its own.
TEST(AllocGroupTest, escapingValuesAreNotGrouped) {
  std::vector<AllocLifetime> lifetimes = {
      {.actualId = 20, .dtype = at::kFloat, .allocStep = 0, .freeStep = 2},
      {.actualId = 21, .dtype = at::kFloat, .allocStep = 0, .freeStep = -1},
  };
  std::vector<nativert::ValueId> ungrouped;
  auto groups = buildAllocGroups(lifetimes, &ungrouped);

  ASSERT_EQ(groups.size(), 1);
  EXPECT_THAT(groups[0].actualIds, ElementsAre(20));
  EXPECT_THAT(ungrouped, ElementsAre(21));
}

// Groups whose shapes are known without waiting are emitted before the ones
// that need a transfer to land, so the host does every layout and allocation it
// can before it blocks on the device.
TEST(AllocGroupTest, syncFreeGroupsComeFirst) {
  std::vector<AllocLifetime> lifetimes = {
      {.actualId = 30,
       .dtype = at::kFloat,
       .allocStep = 0,
       .freeStep = 1,
       .needsSync = true},
      {.actualId = 31,
       .dtype = at::kFloat,
       .allocStep = 0,
       .freeStep = 4,
       .needsSync = false},
  };
  auto groups = buildAllocGroups(lifetimes);

  ASSERT_EQ(groups.size(), 2);
  EXPECT_FALSE(groups[0].needsSync);
  EXPECT_THAT(groups[0].actualIds, ElementsAre(31));
  EXPECT_TRUE(groups[1].needsSync);
  EXPECT_THAT(groups[1].actualIds, ElementsAre(30));
}

TEST(AllocGroupTest, rejectsFreeBeforeAlloc) {
  std::vector<AllocLifetime> lifetimes = {
      {.actualId = 40, .dtype = at::kFloat, .allocStep = 3, .freeStep = 1},
  };
  EXPECT_THROW(buildAllocGroups(lifetimes), c10::Error);
}

// Between being sized and being carved a member carries a shape-only
// placeholder, which is all the sizing pass needs to read a shape. It is not
// enough for the places that keep the tensor itself -- a host-side view built
// from it, an in-place output aliased to it -- so a value its own node borrows
// that way stays out. Merely reading it, and borrowing it in a later node, are
// both fine.
TEST(AllocGroupTest, valuesBorrowedByTheirOwnNodeAreNotGrouped) {
  const std::vector<NodeFootprint> nodes = {
      {.node = 0,
       .lastStep = 1,
       .launches =
           {{.step = 0,
             .writes = {50, 51, 52},
             .writeDtypes = {at::kFloat, at::kInt, at::kFloat},
             .writeNeedsSync = {false, false, false}},
            // %51 is only read by its own node, which the placeholder covers.
            {.step = 1, .reads = {51}}},
       .borrowedIds = {50}},
      {.node = 1,
       .launches = {{.step = 0, .reads = {50, 51, 52}}},
       .releasedIds = {50, 51, 52},
       .releaseSteps = {0, 0, 0},
       // A borrow after the group is carved is harmless.
       .borrowedIds = {52}},
  };

  AllocGroupStats stats;
  auto lifetimes = graphAllocLifetimes(nodes, &stats);
  auto groups = buildAllocGroups(lifetimes);

  EXPECT_EQ(stats.numAllocated, 3);
  EXPECT_EQ(stats.numBorrowedInOwnNode, 1);
  ASSERT_EQ(groups.size(), 1);
  EXPECT_THAT(groups[0].actualIds, ElementsAre(51, 52));
}

// The point of the whole-graph scan: a value produced by one node and released
// by a later one is grouped with the others that share both points, which a
// per-node scan cannot see because the release is not in the node that
// allocates.
TEST(AllocGroupTest, lifetimesAcrossNodes) {
  const std::vector<NodeFootprint> nodes = {
      {.node = 0,
       .launches =
           {{.step = 0,
             .writes = {10, 11, 12},
             .writeDtypes = {at::kFloat, at::kFloat, at::kFloat},
             .writeNeedsSync = {false, false, false}}}},
      {.node = 1,
       .lastStep = 1,
       .launches = {{.step = 0, .reads = {10, 11}}},
       .releasedIds = {10, 11},
       .releaseSteps = {1, 1}},
      {.node = 2, .launches = {{.step = 0, .reads = {12}}}},
  };

  AllocGroupStats stats;
  auto lifetimes = graphAllocLifetimes(nodes, &stats);
  auto groups = buildAllocGroups(lifetimes);

  // %12 is read but never released -- a graph output -- so it stays out.
  EXPECT_EQ(stats.numAllocated, 3);
  EXPECT_EQ(stats.numGrouped, 2);
  EXPECT_EQ(stats.numEscaping, 1);
  EXPECT_EQ(stats.numBorrowedInOwnNode, 0);
  ASSERT_EQ(groups.size(), 1);
  EXPECT_THAT(groups[0].actualIds, ElementsAre(10, 11));
  EXPECT_EQ(groups[0].allocNode, 0);
  EXPECT_EQ(groups[0].freeNode, 1);
  EXPECT_EQ(groups[0].freeStep, 1);
}

// Two values released by different nodes have different lifetimes, however
// alike their allocation looks: sharing a buffer would hold the shorter-lived
// one's bytes until the longer-lived one died.
TEST(AllocGroupTest, differentReleaseNodesDoNotShare) {
  const std::vector<NodeFootprint> nodes = {
      {.node = 0,
       .launches =
           {{.step = 0,
             .writes = {10, 11},
             .writeDtypes = {at::kFloat, at::kFloat},
             .writeNeedsSync = {false, false}}}},
      {.node = 1, .releasedIds = {10}, .releaseSteps = {0}},
      {.node = 2, .releasedIds = {11}, .releaseSteps = {0}},
  };

  auto groups = buildAllocGroups(graphAllocLifetimes(nodes));

  ASSERT_EQ(groups.size(), 2);
  EXPECT_THAT(groups[0].actualIds, ElementsAre(10));
  EXPECT_EQ(groups[0].freeNode, 1);
  EXPECT_THAT(groups[1].actualIds, ElementsAre(11));
  EXPECT_EQ(groups[1].freeNode, 2);
}

// A value no node releases lives to the end of the graph, so it gets no release
// point and buildAllocGroups will keep it out of every group.
TEST(AllocGroupTest, unreleasedValueEscapes) {
  const std::vector<NodeFootprint> nodes = {
      {.node = 0,
       .launches =
           {{.step = 0,
             .writes = {60},
             .writeDtypes = {at::kFloat},
             .writeNeedsSync = {false}},
            {.step = 2, .reads = {60}}}},
  };

  AllocGroupStats stats;
  auto lifetimes = graphAllocLifetimes(nodes, &stats);

  ASSERT_EQ(lifetimes.size(), 1);
  EXPECT_EQ(lifetimes[0].freeStep, -1);
  EXPECT_EQ(stats.numEscaping, 1);
}

// A release the grid contradicts -- a launch reads the value after the point it
// is said to go at -- means the two analyses disagree. Carving a buffer on that
// reading would free it under a live reader, so the value is left alone.
TEST(AllocGroupTest, readAfterReleaseIsNotGrouped) {
  const std::vector<NodeFootprint> nodes = {
      {.node = 0,
       .launches =
           {{.step = 0,
             .writes = {70},
             .writeDtypes = {at::kFloat},
             .writeNeedsSync = {false}}}},
      {.node = 1,
       .launches = {{.step = 0, .reads = {70}}},
       .releasedIds = {70},
       .releaseSteps = {0}},
      {.node = 2, .launches = {{.step = 0, .reads = {70}}}},
  };

  AllocGroupStats stats;
  auto lifetimes = graphAllocLifetimes(nodes, &stats);

  ASSERT_EQ(lifetimes.size(), 1);
  EXPECT_EQ(lifetimes[0].freeStep, -1);
  EXPECT_EQ(stats.numReadAfterFree, 1);
}

// Which of two writes allocates the buffer is a question of execution order,
// not of the plan, so a value written at more than one point is left alone
// rather than carved at a point that may already hold a live buffer.
TEST(AllocGroupTest, multiplyWrittenValueIsNotGrouped) {
  const std::vector<NodeFootprint> nodes = {
      {.node = 0,
       .lastStep = 1,
       .launches =
           {{.step = 0,
             .writes = {70},
             .writeDtypes = {at::kFloat},
             .writeNeedsSync = {false}},
            {.step = 1,
             .writes = {70},
             .writeDtypes = {at::kFloat},
             .writeNeedsSync = {false}}}},
      {.node = 1, .releasedIds = {70}, .releaseSteps = {0}},
  };

  AllocGroupStats stats;
  auto lifetimes = graphAllocLifetimes(nodes, &stats);

  ASSERT_EQ(lifetimes.size(), 1);
  EXPECT_EQ(lifetimes[0].allocStep, 0);
  EXPECT_EQ(lifetimes[0].freeStep, -1);
  EXPECT_EQ(stats.numMultiWrite, 1);
}

// needsSync rides through to the lifetime, which is what orders the group
// after the ones the host can lay out without waiting.
TEST(AllocGroupTest, needsSyncPropagates) {
  const std::vector<NodeFootprint> nodes = {
      {.node = 0,
       .launches =
           {{.step = 0,
             .writes = {80, 81},
             .writeDtypes = {at::kFloat, at::kFloat},
             .writeNeedsSync = {true, false}}}},
      {.node = 1,
       .lastStep = 1,
       .launches = {{.step = 1, .reads = {80, 81}}},
       .releasedIds = {80, 81},
       .releaseSteps = {1, 1}},
  };

  auto lifetimes = graphAllocLifetimes(nodes);

  ASSERT_EQ(lifetimes.size(), 2);
  EXPECT_TRUE(lifetimes[0].needsSync);
  EXPECT_FALSE(lifetimes[1].needsSync);

  // End to end: the sync-free one is emitted first even though it is the
  // higher value id.
  auto groups = buildAllocGroups(lifetimes);
  ASSERT_EQ(groups.size(), 2);
  EXPECT_FALSE(groups[0].needsSync);
  EXPECT_THAT(groups[0].actualIds, ElementsAre(81));
  EXPECT_TRUE(groups[1].needsSync);
  EXPECT_THAT(groups[1].actualIds, ElementsAre(80));
}

// An operand that crosses the concat's kernel boundary, as concatSetOutputs
// describes one: an earlier kernel produces it and its extent is read straight
// off the value.
ConcatInputInfo boundaryOperand(nativert::ValueId valueId) {
  ConcatInputInfo operand;
  operand.valueId = valueId;
  operand.isSubgraphInput = true;
  operand.sizeExpr.op = SizeShortcut::kMax;
  operand.sizeExpr.values = {valueId};
  return operand;
}

// An operand the concat's own kernel computes, whose extent follows 'from'.
ConcatInputInfo internalOperand(
    nativert::ValueId valueId,
    nativert::ValueId from) {
  ConcatInputInfo operand;
  operand.valueId = valueId;
  operand.sizeExpr.op = SizeShortcut::kMax;
  operand.sizeExpr.values = {from};
  return operand;
}

ConcatFootprint oneDimCat(
    nativert::ValueId resultId,
    std::vector<ConcatInputInfo> operands) {
  ConcatFootprint concat;
  concat.resultId = resultId;
  concat.dtype = at::kFloat;
  concat.dim = 0;
  concat.outRank = 1;
  concat.operands = std::move(operands);
  return concat;
}

// One launch producing the operands and a later one joining them: the concat's
// result is placed where the operands are made, so each of them writes into the
// region it occupies instead of into a buffer the concat would copy.
TEST(AllocGroupTest, concatPlacesItsResultAheadOfTheOperands) {
  const std::vector<NodeFootprint> nodes = {
      {.node = 0,
       .launches =
           {{.step = 0,
             .writes = {10, 11, 12},
             .writeDtypes = {at::kFloat, at::kFloat, at::kFloat},
             .writeNeedsSync = {false, true, false}}}},
      {.node = 1,
       .launches =
           {{.step = 0,
             .reads = {10, 11, 12},
             .writes = {13},
             .writeDtypes = {at::kFloat},
             .writeNeedsSync = {false},
             .concats = {oneDimCat(
                 13,
                 {boundaryOperand(10),
                  boundaryOperand(11),
                  boundaryOperand(12)})}}}},
  };

  folly::F14FastSet<nativert::ValueId> claimed;
  AllocGroupStats stats;
  auto groups = graphConcatGroups(nodes, claimed, &stats);

  ASSERT_EQ(groups.size(), 1);
  EXPECT_THAT(groups[0].actualIds, ElementsAre(10, 11, 12));
  EXPECT_EQ(groups[0].allocNode, 0);
  EXPECT_EQ(groups[0].allocStep, 0);
  // One member's shape only lands with a transfer, so the whole layout waits.
  EXPECT_TRUE(groups[0].needsSync);
  ASSERT_NE(groups[0].concat, nullptr);
  EXPECT_EQ(groups[0].concat->resultId, 13);
  EXPECT_THAT(groups[0].concat->memberOfOperand, ElementsAre(0, 1, 2));
  EXPECT_EQ(stats.numConcatGroups, 1);
  EXPECT_EQ(stats.numConcatMembers, 3);

  // The result and every carved operand are the concat group's to allocate, so
  // the lifetime pass has to leave them alone: grouping them again would carve
  // a second buffer over the first.
  EXPECT_THAT(claimed, ::testing::UnorderedElementsAre(10, 11, 12, 13));
  AllocGroupStats lifetimeStats;
  auto lifetimes = graphAllocLifetimes(nodes, &lifetimeStats, &claimed);
  EXPECT_EQ(lifetimeStats.numInConcatGroup, 4);
  EXPECT_EQ(lifetimeStats.numGrouped, 0);
  EXPECT_TRUE(buildAllocGroups(lifetimes).empty());
}

// Every operand gets a view of its region, not just the ones the group's own
// step produces. The group still lands at step 1, the earliest point all three
// extents are known, but operand 10 -- materialized back at step 0 -- is carved
// too and is copied into its view rather than keeping a buffer the concat would
// have to copy from.
TEST(AllocGroupTest, concatCarvesEveryOperand) {
  const std::vector<NodeFootprint> nodes = {
      {.node = 0,
       .lastStep = 1,
       .launches =
           {{.step = 0,
             .writes = {10},
             .writeDtypes = {at::kFloat},
             .writeNeedsSync = {false}},
            {.step = 1,
             .writes = {11, 12},
             .writeDtypes = {at::kFloat, at::kFloat},
             .writeNeedsSync = {false, false}}}},
      {.node = 1,
       .launches =
           {{.step = 0,
             .writes = {13},
             .writeDtypes = {at::kFloat},
             .writeNeedsSync = {false},
             .concats = {oneDimCat(
                 13,
                 {boundaryOperand(10),
                  boundaryOperand(11),
                  boundaryOperand(12)})}}}},
  };

  folly::F14FastSet<nativert::ValueId> claimed;
  auto groups = graphConcatGroups(nodes, claimed, nullptr);

  ASSERT_EQ(groups.size(), 1);
  EXPECT_EQ(groups[0].allocStep, 1);
  EXPECT_THAT(groups[0].actualIds, ElementsAre(10, 11, 12));
  EXPECT_THAT(groups[0].concat->memberOfOperand, ElementsAre(0, 1, 2));
  EXPECT_THAT(claimed, ::testing::UnorderedElementsAre(10, 11, 12, 13));
}

// The whole-graph plan reports every value a concat group places. Nothing else
// may hand one of them a buffer -- elementwise reuse in particular, which would
// otherwise take a placed output over to its input's buffer and leave the
// region of the result unwritten, silently, since the concat no longer copies a
// placed operand in. The gate that prevents it reads this set, so an empty one
// puts the wrong values back on the table without failing anything else.
TEST(AllocGroupTest, planReportsWhatAConcatGroupPlaces) {
  auto& config = WaveConfig::get();
  const auto savedAlloc = config.enableAllocGroup;
  const auto savedConcat = config.enableConcatAllocGroup;
  const auto savedFree = config.freeIntermediates;
  const auto savedCg = config.isCg;
  config.enableAllocGroup = true;
  config.enableConcatAllocGroup = true;
  config.freeIntermediates = true;
  config.isCg = true;
  SCOPE_EXIT {
    config.enableAllocGroup = savedAlloc;
    config.enableConcatAllocGroup = savedConcat;
    config.freeIntermediates = savedFree;
    config.isCg = savedCg;
  };

  const std::vector<NodeFootprint> nodes = {
      {.node = 0,
       .lastStep = 1,
       .launches =
           {{.step = 0,
             .writes = {10},
             .writeDtypes = {at::kFloat},
             .writeNeedsSync = {false}},
            {.step = 1,
             .writes = {11, 12},
             .writeDtypes = {at::kFloat, at::kFloat},
             .writeNeedsSync = {false, false}}}},
      {.node = 1,
       .launches =
           {{.step = 0,
             .writes = {13},
             .writeDtypes = {at::kFloat},
             .writeNeedsSync = {false},
             .concats = {oneDimCat(
                 13,
                 {boundaryOperand(10),
                  boundaryOperand(11),
                  boundaryOperand(12)})}}}},
  };

  const auto graphPlan = buildGraphAllocGroupPlan(nodes);
  EXPECT_THAT(
      graphPlan.concatPlaced, ::testing::UnorderedElementsAre(10, 11, 12, 13));
}

// A two-operand concat is left to the ordinary path: it constrains its
// producers' layout exactly as much and there is one allocation to save.
TEST(AllocGroupTest, concatOfTwoOperandsIsLeftAlone) {
  const std::vector<NodeFootprint> nodes = {
      {.node = 0,
       .launches =
           {{.step = 0,
             .writes = {10, 11},
             .writeDtypes = {at::kFloat, at::kFloat},
             .writeNeedsSync = {false, false}}}},
      {.node = 1,
       .launches =
           {{.step = 0,
             .writes = {12},
             .writeDtypes = {at::kFloat},
             .writeNeedsSync = {false},
             .concats = {oneDimCat(
                 12, {boundaryOperand(10), boundaryOperand(11)})}}}},
  };

  folly::F14FastSet<nativert::ValueId> claimed;
  AllocGroupStats stats;

  EXPECT_TRUE(graphConcatGroups(nodes, claimed, &stats).empty());
  EXPECT_EQ(stats.numConcatTooFew, 1);
  EXPECT_TRUE(claimed.empty());
}

// An operand whose extent the device settles is not knowable when the result
// would have to be laid out, and the offsets of every operand after it depend
// on it. The whole concat stays on the ordinary path.
TEST(AllocGroupTest, concatWithADeviceSizedOperandIsLeftAlone) {
  auto concat = oneDimCat(
      13, {boundaryOperand(10), boundaryOperand(11), boundaryOperand(12)});
  concat.operands[1].hasShapeOnDevice = true;
  const std::vector<NodeFootprint> nodes = {
      {.node = 0,
       .launches =
           {{.step = 0,
             .writes = {10, 11, 12},
             .writeDtypes = {at::kFloat, at::kFloat, at::kFloat},
             .writeNeedsSync = {false, false, false}}}},
      {.node = 1,
       .launches =
           {{.step = 0,
             .writes = {13},
             .writeDtypes = {at::kFloat},
             .writeNeedsSync = {false},
             .concats = {concat}}}},
  };

  folly::F14FastSet<nativert::ValueId> claimed;
  AllocGroupStats stats;

  EXPECT_TRUE(graphConcatGroups(nodes, claimed, &stats).empty());
  EXPECT_EQ(stats.numConcatOnDevice, 1);
}

// An operand the concat's own kernel computes is not there to measure, but its
// size expression is over values that are, so the result can still be laid out
// around it. It is not carved -- the concat writes it in place as always.
TEST(AllocGroupTest, concatMeasuresAnInternalOperandFromItsSizeExpression) {
  const std::vector<NodeFootprint> nodes = {
      {.node = 0,
       .launches =
           {{.step = 0,
             .writes = {10, 11},
             .writeDtypes = {at::kFloat, at::kFloat},
             .writeNeedsSync = {false, false}}}},
      {.node = 1,
       .launches =
           {{.step = 0,
             .writes = {13},
             .writeDtypes = {at::kFloat},
             .writeNeedsSync = {false},
             .concats = {oneDimCat(
                 13,
                 {boundaryOperand(10),
                  boundaryOperand(11),
                  internalOperand(20, 10)})}}}},
  };

  folly::F14FastSet<nativert::ValueId> claimed;
  auto groups = graphConcatGroups(nodes, claimed, nullptr);

  ASSERT_EQ(groups.size(), 1);
  EXPECT_THAT(groups[0].actualIds, ElementsAre(10, 11));
  EXPECT_THAT(groups[0].concat->memberOfOperand, ElementsAre(0, 1, -1));
}

// A reserve function reads whatever it likes, and nothing says what it needs is
// there when the result would be laid out. Running it early is refused rather
// than risked.
TEST(AllocGroupTest, concatWithAnInternalReserveIsLeftAlone) {
  auto concat = oneDimCat(
      13, {boundaryOperand(10), boundaryOperand(11), internalOperand(20, 10)});
  concat.operands[2].reserveShape = [](NodeCP,
                                       nativert::ExecutionFrame&,
                                       const FormalToActual&,
                                       NodeCP,
                                       const NodeMap&) {
    return std::vector<std::vector<Dim>>{};
  };
  const std::vector<NodeFootprint> nodes = {
      {.node = 0,
       .launches =
           {{.step = 0,
             .writes = {10, 11},
             .writeDtypes = {at::kFloat, at::kFloat},
             .writeNeedsSync = {false, false}}}},
      {.node = 1,
       .launches =
           {{.step = 0,
             .writes = {13},
             .writeDtypes = {at::kFloat},
             .writeNeedsSync = {false},
             .concats = {concat}}}},
  };

  folly::F14FastSet<nativert::ValueId> claimed;
  AllocGroupStats stats;

  EXPECT_TRUE(graphConcatGroups(nodes, claimed, &stats).empty());
  EXPECT_EQ(stats.numConcatUnplaceableOperand, 1);
}

// A view has no storage of its own to redirect, so it is not carved -- but it
// does not sink the concat either. Its extent is the host's to compute like any
// other operand's, so the group still forms over the operands that can be
// carved and the view is copied into the region it occupies.
TEST(AllocGroupTest, concatCarvesAroundAViewOperand) {
  auto concat = oneDimCat(
      13, {boundaryOperand(10), boundaryOperand(11), boundaryOperand(12)});
  concat.operands[2].isView = true;
  const std::vector<NodeFootprint> nodes = {
      {.node = 0,
       .launches =
           {{.step = 0,
             .writes = {10, 11, 12},
             .writeDtypes = {at::kFloat, at::kFloat, at::kFloat},
             .writeNeedsSync = {false, false, false}}}},
      {.node = 1,
       .launches =
           {{.step = 0,
             .writes = {13},
             .writeDtypes = {at::kFloat},
             .writeNeedsSync = {false},
             .concats = {concat}}}},
  };

  folly::F14FastSet<nativert::ValueId> claimed;
  AllocGroupStats stats;

  auto groups = graphConcatGroups(nodes, claimed, &stats);
  ASSERT_EQ(groups.size(), 1);
  EXPECT_THAT(groups[0].actualIds, ElementsAre(10, 11));
  EXPECT_THAT(groups[0].concat->memberOfOperand, ElementsAre(0, 1, -1));
  EXPECT_EQ(stats.numConcatUnplaceableOperand, 0);
}

// A view cannot value-convert, so an operand the concat would promote to the
// result's dtype has to keep its own buffer and be copied in.
TEST(AllocGroupTest, concatDoesNotCarveAPromotedOperand) {
  const std::vector<NodeFootprint> nodes = {
      {.node = 0,
       .launches =
           {{.step = 0,
             .writes = {10, 11, 12},
             .writeDtypes = {at::kFloat, at::kInt, at::kFloat},
             .writeNeedsSync = {false, false, false}}}},
      {.node = 1,
       .launches =
           {{.step = 0,
             .writes = {13},
             .writeDtypes = {at::kFloat},
             .writeNeedsSync = {false},
             .concats = {oneDimCat(
                 13,
                 {boundaryOperand(10),
                  boundaryOperand(11),
                  boundaryOperand(12)})}}}},
  };

  folly::F14FastSet<nativert::ValueId> claimed;
  auto groups = graphConcatGroups(nodes, claimed, nullptr);

  ASSERT_EQ(groups.size(), 1);
  EXPECT_THAT(groups[0].actualIds, ElementsAre(10, 12));
  EXPECT_THAT(groups[0].concat->memberOfOperand, ElementsAre(0, -1, 1));
}

// cat([x, y, x]) joins one value at two positions, and one buffer cannot be two
// regions of the result. Neither occurrence is carved; the value keeps its own
// buffer and the concat copies it into both.
TEST(AllocGroupTest, concatDoesNotCarveARepeatedOperand) {
  const std::vector<NodeFootprint> nodes = {
      {.node = 0,
       .launches =
           {{.step = 0,
             .writes = {10, 11},
             .writeDtypes = {at::kFloat, at::kFloat},
             .writeNeedsSync = {false, false}}}},
      {.node = 1,
       .launches =
           {{.step = 0,
             .writes = {13},
             .writeDtypes = {at::kFloat},
             .writeNeedsSync = {false},
             .concats = {oneDimCat(
                 13,
                 {boundaryOperand(10),
                  boundaryOperand(11),
                  boundaryOperand(10)})}}}},
  };

  folly::F14FastSet<nativert::ValueId> claimed;
  auto groups = graphConcatGroups(nodes, claimed, nullptr);

  ASSERT_EQ(groups.size(), 1);
  EXPECT_THAT(groups[0].actualIds, ElementsAre(11));
  EXPECT_THAT(groups[0].concat->memberOfOperand, ElementsAre(-1, 0, -1));
}

// Joined off the outermost axis, an operand's region of the result is a pitched
// band. Whether it can be carved is the producer's to say: an op that indexes
// its output linearly would corrupt the band, so only one declaring
// ArgumentMeta::mayWriteStrided is handed the view. The rest are copied in.
TEST(AllocGroupTest, concatOffTheOutermostAxisNeedsAStridedWriter) {
  auto concat = oneDimCat(
      13, {boundaryOperand(10), boundaryOperand(11), boundaryOperand(12)});
  concat.outRank = 2;
  concat.dim = 1;
  concat.operands[1].mayWriteStrided = true;
  const std::vector<NodeFootprint> nodes = {
      {.node = 0,
       .launches =
           {{.step = 0,
             .writes = {10, 11, 12},
             .writeDtypes = {at::kFloat, at::kFloat, at::kFloat},
             .writeNeedsSync = {false, false, false}}}},
      {.node = 1,
       .launches =
           {{.step = 0,
             .writes = {13},
             .writeDtypes = {at::kFloat},
             .writeNeedsSync = {false},
             .concats = {concat}}}},
  };

  folly::F14FastSet<nativert::ValueId> claimed;
  AllocGroupStats stats;

  auto groups = graphConcatGroups(nodes, claimed, &stats);
  // Only operand 11's producer says it can fill a pitched band, so it is the
  // one carved; 10 and 12 would corrupt theirs and are copied in instead.
  ASSERT_EQ(groups.size(), 1);
  EXPECT_EQ(groups[0].concat->dim, 1);
  EXPECT_THAT(groups[0].actualIds, ElementsAre(11));
  EXPECT_THAT(groups[0].concat->memberOfOperand, ElementsAre(-1, 0, -1));
  EXPECT_EQ(stats.numConcatStridedBand, 2);
}

// Nothing is installed unless a collector is alive, so the ordinary allocation
// path is what runs everywhere the mode is not driving.
TEST(AllocGroupCollectorTest, notInstalledByDefault) {
  EXPECT_EQ(currentAllocCollector(), nullptr);
}

TEST(AllocGroupCollectorTest, installsAndUninstalls) {
  AllocGroup group;
  group.actualIds = {5};
  group.dtypes = {at::kFloat};
  {
    AllocGroupCollector collector(group);
    EXPECT_EQ(currentAllocCollector(), &collector);
  }
  EXPECT_EQ(currentAllocCollector(), nullptr);
}

// Two live collectors would both claim the same output, and which one won
// would depend on nesting order. Rejected rather than resolved.
TEST(AllocGroupCollectorTest, rejectsNesting) {
  AllocGroup group;
  group.actualIds = {5};
  group.dtypes = {at::kFloat};
  AllocGroupCollector collector(group);
  EXPECT_THROW({ AllocGroupCollector nested(group); }, c10::Error);
}

// A member reports its shape and is not allocated; a non-member is refused, so
// its caller allocates it the ordinary way.
TEST(AllocGroupCollectorTest, capturesMembersOnly) {
  AllocGroup group;
  group.actualIds = {7, 9};
  group.dtypes = {at::kFloat, at::kInt};
  AllocGroupCollector collector(group);

  EXPECT_FALSE(collector.complete());
  EXPECT_THAT(collector.missing(), ElementsAre(7, 9));

  EXPECT_TRUE(collector.capture(9, std::vector<int64_t>{2, 3}));
  EXPECT_FALSE(collector.capture(8, std::vector<int64_t>{4}));
  EXPECT_FALSE(collector.complete());
  EXPECT_THAT(collector.missing(), ElementsAre(7));

  EXPECT_TRUE(collector.capture(7, std::vector<int64_t>{5}));
  EXPECT_TRUE(collector.complete());
  EXPECT_TRUE(collector.missing().empty());

  // Laid out in the group's order, not the order the ops happened to size
  // them in, so a slot sits at the same offset on every execution.
  const auto& requests = collector.requests();
  ASSERT_EQ(requests.size(), 2);
  EXPECT_EQ(requests[0].actualId, 7);
  EXPECT_EQ(requests[0].dtype, at::kFloat);
  EXPECT_THAT(requests[0].dims, ElementsAre(5));
  EXPECT_EQ(requests[1].actualId, 9);
  EXPECT_EQ(requests[1].dtype, at::kInt);
  EXPECT_THAT(requests[1].dims, ElementsAre(2, 3));
}

// A deferred op is sized again once its transfer lands. The second shape is the
// settled one; keeping the first would size the slot from a stale count.
TEST(AllocGroupCollectorTest, resizeKeepsTheLaterShape) {
  AllocGroup group;
  group.actualIds = {3};
  group.dtypes = {at::kLong};
  AllocGroupCollector collector(group);

  EXPECT_TRUE(collector.capture(3, std::vector<int64_t>{16}));
  EXPECT_TRUE(collector.capture(3, std::vector<int64_t>{40}));

  ASSERT_EQ(collector.requests().size(), 1);
  EXPECT_THAT(collector.requests()[0].dims, ElementsAre(40));
}

// The plan is what execution indexes into: which values share a buffer, and
// which groups a given step of a given node has to materialize.
TEST(AllocGroupTest, planIndexesGroupsByStep) {
  const std::vector<NodeFootprint> nodes = {
      {.node = 0,
       .lastStep = 2,
       .launches =
           {{.step = 0,
             .writes = {20, 21, 22, 24},
             .writeDtypes = {at::kFloat, at::kFloat, at::kFloat, at::kFloat},
             .writeNeedsSync = {false, false, true, false}},
            {.step = 2,
             .writes = {23},
             .writeDtypes = {at::kFloat},
             .writeNeedsSync = {false}}}},
      {.node = 1,
       .lastStep = 1,
       .releasedIds = {20, 21, 22},
       .releaseSteps = {1, 1, 1}},
      {.node = 2, .releasedIds = {23}, .releaseSteps = {0}},
  };

  const auto graphPlan = buildGraphAllocGroupPlan(nodes);
  ASSERT_EQ(graphPlan.perNode.size(), 3);
  const auto& plan = graphPlan.perNode[0];

  // %24 is never released. Everything else is allocated by node 0, so every
  // group is node 0's however far downstream it goes.
  EXPECT_THAT(plan.ungrouped, ElementsAre(24));
  EXPECT_TRUE(graphPlan.perNode[1].groups.empty());
  ASSERT_EQ(plan.groupsByStep.size(), 3);
  // Step 0 holds the sync-free pair and, after it, the one that must wait.
  ASSERT_EQ(plan.groupsByStep[0].size(), 2);
  EXPECT_FALSE(plan.groups[plan.groupsByStep[0][0]].needsSync);
  EXPECT_THAT(
      plan.groups[plan.groupsByStep[0][0]].actualIds, ElementsAre(20, 21));
  EXPECT_TRUE(plan.groups[plan.groupsByStep[0][1]].needsSync);
  EXPECT_TRUE(plan.groupsByStep[1].empty());
  ASSERT_EQ(plan.groupsByStep[2].size(), 1);
  EXPECT_EQ(plan.groups[plan.groupsByStep[2][0]].freeNode, 2);

  EXPECT_EQ(plan.groupOfValue.count(24), 0);
  EXPECT_EQ(plan.groupOfValue.at(20), plan.groupOfValue.at(21));
  EXPECT_NE(plan.groupOfValue.at(20), plan.groupOfValue.at(22));

  EXPECT_EQ(graphPlan.stats.numAllocated, 5);
  EXPECT_EQ(graphPlan.stats.numGroups, 3);
  EXPECT_EQ(graphPlan.stats.numCrossNodeGroups, 3);
  EXPECT_EQ(graphPlan.stats.largestGroup, 2);
}

// Every group belongs to the node that allocates it, however far downstream it
// is released -- that node is the only one whose step sizes its members.
TEST(AllocGroupTest, planSplitsGroupsByAllocatingNode) {
  const std::vector<NodeFootprint> nodes = {
      {.node = 0,
       .launches =
           {{.step = 0,
             .writes = {10, 11},
             .writeDtypes = {at::kFloat, at::kFloat},
             .writeNeedsSync = {false, false}}}},
      {.node = 1,
       .launches =
           {{.step = 0,
             .reads = {10, 11},
             .writes = {12},
             .writeDtypes = {at::kFloat},
             .writeNeedsSync = {false}}}},
      {.node = 2,
       .launches = {{.step = 0, .reads = {12}}},
       .releasedIds = {10, 11, 12},
       .releaseSteps = {0, 0, 0}},
  };

  const auto graphPlan = buildGraphAllocGroupPlan(nodes);

  ASSERT_EQ(graphPlan.perNode.size(), 3);
  ASSERT_EQ(graphPlan.perNode[0].groups.size(), 1);
  EXPECT_THAT(graphPlan.perNode[0].groups[0].actualIds, ElementsAre(10, 11));
  ASSERT_EQ(graphPlan.perNode[1].groups.size(), 1);
  EXPECT_THAT(graphPlan.perNode[1].groups[0].actualIds, ElementsAre(12));
  EXPECT_TRUE(graphPlan.perNode[2].groups.empty());
  EXPECT_EQ(graphPlan.stats.numCrossNodeGroups, 2);
}

class AllocGroupCudaTest : public ::testing::Test {
 protected:
  void SetUp() override {
    if (at::cuda::device_count() == 0) {
      GTEST_SKIP() << "No CUDA device";
    }
  }
};

// The point of the mechanism: many outputs, one allocator call, and each slot a
// correctly shaped view at a 16-byte aligned offset of the one buffer.
TEST_F(AllocGroupCudaTest, carvesOneBufferIntoAlignedSlots) {
  std::vector<AllocRequest> requests = {
      {.actualId = 1, .dtype = at::kFloat, .dims = {3}}, // 12 bytes -> 16
      {.actualId = 2, .dtype = at::kInt, .dims = {5}}, // 20 bytes -> 32
      {.actualId = 3, .dtype = at::kDouble, .dims = {2, 2}}, // 32 bytes
  };
  auto group = allocateAllocGroup(requests);

  EXPECT_EQ(group.totalBytes, 16 + 32 + 32);
  EXPECT_THAT(group.offsets, ElementsAre(0, 16, 48));
  ASSERT_EQ(group.slots.size(), 3);

  for (const auto& slot : group.slots) {
    EXPECT_TRUE(slot.is_cuda());
    // Every slot is backed by the group's single allocation, which is what
    // makes the group free itself when the last view dies.
    EXPECT_TRUE(slot.is_alias_of(group.base));
  }
  EXPECT_THAT(group.slots[0].sizes(), ElementsAre(3));
  EXPECT_EQ(group.slots[0].scalar_type(), at::kFloat);
  EXPECT_THAT(group.slots[1].sizes(), ElementsAre(5));
  EXPECT_EQ(group.slots[1].scalar_type(), at::kInt);
  EXPECT_THAT(group.slots[2].sizes(), ElementsAre(2, 2));
  EXPECT_EQ(group.slots[2].scalar_type(), at::kDouble);
  for (const auto& slot : group.slots) {
    EXPECT_TRUE(slot.is_contiguous());
  }
}

// Slots must not overlap: the alignment padding is what keeps a write to one
// from landing in its neighbour.
TEST_F(AllocGroupCudaTest, slotsDoNotOverlap) {
  std::vector<AllocRequest> requests = {
      {.actualId = 1, .dtype = at::kFloat, .dims = {3}},
      {.actualId = 2, .dtype = at::kFloat, .dims = {4}},
  };
  auto group = allocateAllocGroup(requests);

  group.slots[0].fill_(1.0f);
  group.slots[1].fill_(2.0f);

  EXPECT_TRUE(
      at::equal(
          group.slots[0].cpu(), at::full({3}, 1.0f, at::TensorOptions())));
  EXPECT_TRUE(
      at::equal(
          group.slots[1].cpu(), at::full({4}, 2.0f, at::TensorOptions())));
}

// The allocation outlives the local handle: the slots hold it up, which is what
// lets the caller drop the group struct and keep only the frame values.
TEST_F(AllocGroupCudaTest, baseOutlivesTheGroupHandle) {
  at::Tensor slot;
  {
    std::vector<AllocRequest> requests = {
        {.actualId = 1, .dtype = at::kFloat, .dims = {8}}};
    auto group = allocateAllocGroup(requests);
    slot = group.slots[0];
    slot.fill_(7.0f);
  }
  EXPECT_TRUE(at::equal(slot.cpu(), at::full({8}, 7.0f, at::TensorOptions())));
}

// A zero-element output still gets a slot, so callers can index positionally.
TEST_F(AllocGroupCudaTest, zeroElementSlot) {
  std::vector<AllocRequest> requests = {
      {.actualId = 1, .dtype = at::kFloat, .dims = {0}},
      {.actualId = 2, .dtype = at::kFloat, .dims = {2}},
  };
  auto group = allocateAllocGroup(requests);

  ASSERT_EQ(group.slots.size(), 2);
  EXPECT_EQ(group.slots[0].numel(), 0);
  EXPECT_EQ(group.slots[1].numel(), 2);
  EXPECT_THAT(group.offsets, ElementsAre(0, 0));
}

} // namespace
} // namespace torch::wave

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  folly::Init init{&argc, &argv};
  return RUN_ALL_TESTS();
}
