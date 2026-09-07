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

#pragma once

#include <cstdint>
#include <functional>
#include <optional>
#include <vector>

namespace torch::wave {

/// Occupancy and width of the device a step's grid is laid out for. Passed in
/// rather than read from the driver so the layout can be exercised without a
/// GPU.
struct GridDevice {
  /// Streaming multiprocessors the grid spreads over.
  int32_t numSMs{1};

  /// Blocks of this step's kernel that stay resident on one SM when no op in
  /// it asks for dynamic shared memory.
  int32_t maxBlocksPerSM{1};

  /// Shared memory one SM has for all of its resident blocks.
  int32_t sharedPerSM{0};

  /// Static shared memory every block of this kernel takes, whatever op it
  /// runs.
  int32_t staticSharedPerBlock{0};

  /// Blocks of this step's kernel that stay resident on one SM at the given
  /// dynamic shared memory, as the DRIVER computes it. Read by coopBlocksPerSM,
  /// which bounds a cooperative launch; blocksPerSM keeps the estimate below,
  /// since changing that would move ops between occupancy classes and
  /// repartition the graph.
  ///
  /// The division is only an approximation of what the driver does: the driver
  /// also rounds each block's shared allocation up to the hardware granularity,
  /// subtracts the shared memory reserved per block, and honours the L1/shared
  /// carveout. Each of those can only lower the true figure, so the division
  /// can come out a block per SM too high -- and a cooperative launch is packed
  /// to exactly this number, so an over-estimate is not slack but a grid the
  /// driver refuses outright.
  ///
  /// Left unset by tests, which have no kernel to ask and want the layout to be
  /// reproducible without a GPU.
  std::function<int32_t(int32_t dynamicShared)> occupancyFor;
};

/// What laying out a step's launches needs to know about one of its ops.
/// Reduced from LaunchData and KernelOperation so the layout depends on
/// neither.
struct GridOp {
  /// numElements * unitCost * costAdjustFactor: the op's share of the step's
  /// work, in the units blocks are divided by.
  float cost{0};

  /// Most blocks this op may be given.
  int32_t maxBlocks{1};

  /// extern __shared__ bytes the op needs. Only the ops sharing a launch with
  /// it pay for them.
  int32_t dynamicShared{0};

  /// Set when the op folds its cross-block barriers into __syncthreads and is
  /// only correct as a single block.
  bool alwaysSingleBlock{false};

  /// Set when the op spin-waits until all of its blocks arrive (opBarrier), so
  /// they must be co-resident in one launch and the op may not be split
  /// across two.
  bool hasBarrier{false};
};

/// One kernel launch of a step, over a contiguous range of the step's blocks.
struct LaunchSegment {
  /// Index into the step's block array where this launch starts.
  int32_t firstBlock{0};

  int32_t numBlocks{0};

  /// Dynamic shared memory to launch with: the max over the ops in THIS
  /// launch, not over the whole step. Launching the ops that need none apart
  /// from the one that does is what keeps a single hungry op from cutting
  /// everyone's occupancy.
  int32_t dynamicShared{0};

  /// Set when a barrier op in this launch spans more than one block. opBarrier
  /// waits for all of them to arrive, which only a cooperative launch keeps
  /// co-resident. A barrier op on a single block passes its barrier as soon as
  /// it runs, so it needs nothing.
  bool cooperative{false};
};

/// One thread block of the emitted grid.
struct GridBlock {
  /// Index into the step's ops.
  int32_t op{0};

  /// This block's position among that op's blocks, counted over the whole
  /// step: an op split across two launches still numbers its blocks 0..n-1,
  /// which is what lets it compute its slice the same way either way.
  int32_t blockInOp{0};
};

/// How far a step's grid is from a balanced, fully occupied one. Computed for
/// every step; reported under WaveConfig::kGrid and summarized in the
/// performance report.
struct GridStats {
  int32_t numOps{0};

  /// Blocks one wave holds at the occupancy the step actually launches with.
  int32_t targetBlocks{0};

  /// Blocks the step was given.
  int32_t totalBlocks{0};

  float totalCost{0};

  /// The step's makespan over that of a perfectly balanced, fully occupied
  /// one: max_i(cost_i / blocks_i) divided by totalCost / targetBlocks. 1
  /// means there is nothing to win. With every op on one block it degenerates
  /// to the cost spread of the step, which is what block starvation costs.
  float skew{0};

  /// Ops left on a single block although they could have used more. This is
  /// what block starvation looks like: once numOps reaches targetBlocks the
  /// pro-rata split has nothing left to give.
  int32_t numStarved{0};

  /// Blocks per SM the step launches with, i.e. after the step-wide max
  /// dynamic shared memory has cut it.
  int32_t occupancy{0};

  /// Best blocks per SM any op of the step would reach launched on its own.
  /// Above 'occupancy' means some op is taxing the others.
  int32_t bestOccupancy{0};

  /// Share of the step's cost belonging to the ops that ask for the most
  /// dynamic shared memory. A small share with occupancy below bestOccupancy
  /// is a cheap op holding the whole step down.
  float poison{0};

  /// Launches the step was split into. 1 unless the packer ran.
  int32_t numSegments{1};
};

/// The blocks of one step, grouped into the launches that run them.
struct LaunchPlan {
  /// Blocks each op ends up with, parallel to the ops. The packer may differ
  /// from what was passed in; a single launch never does.
  std::vector<int32_t> blocksPerOp;

  /// The grid, flattened over the segments in launch order.
  std::vector<GridBlock> blocks;

  /// Contiguous ranges of 'blocks', one kernel launch each.
  std::vector<LaunchSegment> segments;
};

/// Knobs of the packer, from WaveConfig.
struct PartitionParams {
  /// Most launches one step may be split into, and so the multiple of one
  /// wave the block array is reserved for.
  int32_t maxWaves{3};

  /// GridStats::skew at and above which a step is worth splitting.
  float skewThreshold{1.3f};

  /// Cost a block must be worth for the packer to open one, so a split does
  /// not produce blocks too short to be worth scheduling. 0 falls back to an
  /// even division of the step over one wave.
  float minBlockCost{0};

  /// Mean block clocks over the slowest, measured on the previous execution of
  /// this step. 0 when nothing has been measured.
  double measuredUtil{0};

  /// Blocks that execution ran.
  int32_t measuredBlocks{0};

  /// Measured utilization at and above which a step is left at the width it
  /// already had. Sizing by quantum asks how many blocks the work is worth, but
  /// a step whose blocks already finish together has no idle capacity for more
  /// blocks to fill: widening it only buys serialized launches and host time
  /// spent filling BlockInfo. On the ROO graph, four steps already at 90-98%
  /// were tripled to the wave budget for +1.2 ms and slightly WORSE
  /// utilization.
  float keepWidthUtil{0.85f};
};

/// Blocks of this kernel that fit on one SM alongside each other when each
/// takes 'dynamicShared' bytes of dynamic shared memory.
int32_t blocksPerSM(const GridDevice& device, int32_t dynamicShared);

/// True when this step, left as a single launch, would go out as a cooperative
/// grid wider than the device can hold co-resident -- which the driver refuses
/// outright. Such a step must be split whatever its skew, since the alternative
/// is not a slower plan but a failed launch.
/// True when 'ops' laid out as one launch would need a cooperative grid wider
/// than the device holds co-resident, so the step has to be split whatever the
/// skew gate says. On the 1k ROO graph this fires on two steps, both of which
/// otherwise fail the launch outright.
bool exceedsCooperativeCapacity(
    const std::vector<GridOp>& ops,
    const std::vector<int32_t>& blocksPerOp,
    const GridDevice& device);

/// The same, but for a launch that has to be cooperative: the driver's own
/// figure via GridDevice::occupancyFor when there is one, else the estimate.
/// Never above blocksPerSM, since the occupancy classes and the BlockInfo
/// reservation are both derived from that and a launch may not outgrow them.
///
/// Separate from blocksPerSM because only a cooperative launch is packed to
/// exactly its capacity. An ordinary launch may exceed it -- the hardware runs
/// the surplus in a later wave -- so it keeps the estimate, and using the
/// driver's figure for it would repartition the graph for no gain.
int32_t coopBlocksPerSM(
    const GridDevice& device,
    int32_t classOccupancy,
    int32_t dynamicShared);

/// Measures the balance of a grid that gives op i blocksPerOp[i] blocks.
GridStats gridStats(
    const std::vector<GridOp>& ops,
    const std::vector<int32_t>& blocksPerOp,
    const GridDevice& device);

/// Lays the step out as the single launch it is today. With 'orderByLatency'
/// the ops are emitted in descending cost per block instead of in index
/// order, so the long poles start at t=0 and the cheap ops backfill SMs as
/// those retire. Blocks are dispatched to SMs roughly in index order, which is
/// what makes the order matter at all.
LaunchPlan singleLaunchPlan(
    const std::vector<GridOp>& ops,
    const std::vector<int32_t>& blocksPerOp,
    bool orderByLatency);

/// Blocks per op sized against a common per-block work quantum, with the total
/// rounded up to a whole number of waves.
///
/// The pro-rata split divides one wave's worth of blocks among the ops, so what
/// an op gets depends on how many other ops the step has. This asks the other
/// question: how many blocks of about 'params.minBlockCost' is each op's work
/// worth? The answer is a property of the op alone, and the step emits as many
/// waves as the sum needs -- up to params.maxWaves, after which it scales back.
///
/// Having sized them, the total is rounded UP to a whole wave and the surplus
/// given to the ops with the most work per block, so the last wave is not left
/// half empty. An op already at its maxBlocks, or marked alwaysSingleBlock,
/// takes no part in that.
///
/// Ops whose maxBlocks is 1 -- a launch too small to divide -- still take a
/// block each. That floor is why a step of many tiny ops can exceed a wave
/// however it is sized; the difference is that it now costs an extra launch
/// rather than the large ops' blocks.
std::vector<int32_t> sizeByQuantum(
    const std::vector<GridOp>& ops,
    const GridDevice& device,
    const PartitionParams& params);

/// Whether 'stats' says the step has something to gain from being split:
/// either it is badly skewed, or one op's shared memory is cutting the
/// occupancy of ops that hold most of the work.
bool shouldPartition(const GridStats& stats, const PartitionParams& params);

/// Splits a step into balanced launches: ops are sized against a common block
/// quantum, grouped by the occupancy their shared memory allows, and packed
/// into full waves of that group's capacity. A barrier op is shrunk to fit a
/// launch but never split across two; any other op may straddle, since its
/// blocks only need blockInOp and numBlocksInOp to find their slice.
///
/// Returns nullopt when the step is no better off split -- one launch came
/// out of the packing, or there is nothing to pack.
/// 'presized' overrides the block counts the packer would compute for itself,
/// for a caller that has sized the step by another rule -- sizeByQuantum. It
/// also forces a plan out: with counts supplied, a packing that came out as a
/// single launch is still the answer, not a reason to fall back to the
/// pro-rata split.
std::optional<LaunchPlan> partitionLaunches(
    const std::vector<GridOp>& ops,
    const GridDevice& device,
    const GridStats& stats,
    const PartitionParams& params,
    const std::vector<int32_t>* presized = nullptr);

} // namespace torch::wave
