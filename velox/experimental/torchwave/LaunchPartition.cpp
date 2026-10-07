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

#include "velox/experimental/torchwave/LaunchPartition.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <numeric>

namespace torch::wave {

int32_t blocksPerSM(const GridDevice& device, int32_t dynamicShared) {
  int32_t perSM = std::max(1, device.maxBlocksPerSM);
  if (dynamicShared > 0 && device.sharedPerSM > 0) {
    const int32_t perBlock = dynamicShared + device.staticSharedPerBlock;
    perSM = std::max(1, std::min(perSM, device.sharedPerSM / perBlock));
  }
  return perSM;
}

int32_t coopBlocksPerSM(
    const GridDevice& device,
    int32_t classOccupancy,
    int32_t dynamicShared) {
  if (!device.occupancyFor) {
    return classOccupancy;
  }
  const int32_t fromDriver = device.occupancyFor(dynamicShared);
  if (fromDriver <= 0) {
    return classOccupancy;
  }
  // Only ever downwards, and only away from 'classOccupancy'. The occupancy
  // classes, the block budget and the BlockInfo reservation are all derived
  // from blocksPerSM; raising the figure here would let a launch outgrow a
  // reservation sized from that one, and returning anything but
  // 'classOccupancy' when there is no driver to ask would repartition a graph
  // that has no GPU behind it.
  return std::max(1, std::min(classOccupancy, fromDriver));
}

namespace {

// Cost per block of op 'index', i.e. how long it is projected to keep the
// blocks it was given busy. Balancing a launch means levelling this.
float latencyOf(
    const std::vector<GridOp>& ops,
    const std::vector<int32_t>& blocksPerOp,
    int32_t index) {
  const int32_t blocks = blocksPerOp[index];
  return blocks > 0 ? ops[index].cost / static_cast<float>(blocks) : 0.0f;
}

// Op indices in the order their blocks should be emitted. Blocks are
// dispatched to SMs roughly in index order, so the ops projected to run
// longest go first and the cheap ones backfill SMs as the long ones retire.
std::vector<int32_t> latencyOrder(
    const std::vector<GridOp>& ops,
    const std::vector<int32_t>& blocksPerOp) {
  std::vector<int32_t> order(ops.size());
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&](int32_t lhs, int32_t rhs) {
    return latencyOf(ops, blocksPerOp, lhs) > latencyOf(ops, blocksPerOp, rhs);
  });
  return order;
}

// Largest dynamic shared memory any op needs.
int32_t maxDynamicShared(const std::vector<GridOp>& ops) {
  int32_t bytes = 0;
  for (const auto& op : ops) {
    bytes = std::max(bytes, op.dynamicShared);
  }
  return bytes;
}

// The same over a subset, which is what a launch holding only those ops will
// ask the driver for.
int32_t maxDynamicShared(
    const std::vector<GridOp>& ops,
    const std::vector<int32_t>& members) {
  int32_t bytes = 0;
  for (auto op : members) {
    bytes = std::max(bytes, ops.at(op).dynamicShared);
  }
  return bytes;
}

// Whether a launch holding these ops has to be cooperative.
bool needsCooperative(
    const std::vector<GridOp>& ops,
    const std::vector<int32_t>& blocksPerOp,
    int32_t index) {
  return ops[index].hasBarrier && blocksPerOp[index] > 1;
}

} // namespace

bool exceedsCooperativeCapacity(
    const std::vector<GridOp>& ops,
    const std::vector<int32_t>& blocksPerOp,
    const GridDevice& device) {
  int32_t total = 0;
  bool anyCooperative = false;
  for (size_t i = 0; i < ops.size(); ++i) {
    const int32_t blocks = i < blocksPerOp.size() ? blocksPerOp[i] : 0;
    total += std::max(0, blocks);
    anyCooperative = anyCooperative ||
        needsCooperative(ops, blocksPerOp, static_cast<int32_t>(i));
  }
  if (!anyCooperative) {
    return false;
  }
  // Unsplit, the whole step goes out as one cooperative launch, so its total
  // is what the device has to hold co-resident.
  const int32_t capacity = std::max(1, device.numSMs) *
      coopBlocksPerSM(device,
                      blocksPerSM(device, maxDynamicShared(ops)),
                      maxDynamicShared(ops));
  return total > capacity;
}

GridStats gridStats(
    const std::vector<GridOp>& ops,
    const std::vector<int32_t>& blocksPerOp,
    const GridDevice& device) {
  GridStats stats;
  stats.numOps = static_cast<int32_t>(ops.size());
  const int32_t stepShared = maxDynamicShared(ops);
  stats.occupancy = blocksPerSM(device, stepShared);
  stats.bestOccupancy = stats.occupancy;
  for (const auto& op : ops) {
    stats.bestOccupancy =
        std::max(stats.bestOccupancy, blocksPerSM(device, op.dynamicShared));
  }
  stats.targetBlocks = std::max(1, device.numSMs) * stats.occupancy;

  float worstLatency = 0;
  float poisonCost = 0;
  for (size_t i = 0; i < ops.size(); ++i) {
    stats.totalCost += ops[i].cost;
    const int32_t blocks = i < blocksPerOp.size() ? blocksPerOp[i] : 0;
    stats.totalBlocks += blocks;
    if (blocks > 0) {
      worstLatency =
          std::max(worstLatency, ops[i].cost / static_cast<float>(blocks));
    }
    if (blocks == 1 && ops[i].maxBlocks > 1) {
      ++stats.numStarved;
    }
    if (stepShared > 0 && ops[i].dynamicShared == stepShared) {
      poisonCost += ops[i].cost;
    }
  }
  if (stats.totalCost > 0) {
    stats.skew = worstLatency /
        (stats.totalCost / static_cast<float>(stats.targetBlocks));
    stats.poison = poisonCost / stats.totalCost;
  }
  return stats;
}

LaunchPlan singleLaunchPlan(
    const std::vector<GridOp>& ops,
    const std::vector<int32_t>& blocksPerOp,
    bool orderByLatency) {
  LaunchPlan plan;
  plan.blocksPerOp = blocksPerOp;
  plan.blocksPerOp.resize(ops.size(), 0);

  int32_t total = 0;
  for (auto blocks : plan.blocksPerOp) {
    total += std::max(0, blocks);
  }
  plan.blocks.reserve(total);

  if (orderByLatency) {
    for (auto index : latencyOrder(ops, plan.blocksPerOp)) {
      for (int32_t block = 0; block < plan.blocksPerOp[index]; ++block) {
        plan.blocks.push_back({index, block});
      }
    }
  } else {
    for (size_t i = 0; i < ops.size(); ++i) {
      for (int32_t block = 0; block < plan.blocksPerOp[i]; ++block) {
        plan.blocks.push_back({static_cast<int32_t>(i), block});
      }
    }
  }
  bool cooperative = false;
  for (size_t i = 0; i < ops.size(); ++i) {
    cooperative = cooperative ||
        needsCooperative(ops, plan.blocksPerOp, static_cast<int32_t>(i));
  }
  plan.segments.push_back(
      {.firstBlock = 0,
       .numBlocks = static_cast<int32_t>(plan.blocks.size()),
       .dynamicShared = maxDynamicShared(ops),
       .cooperative = cooperative});
  return plan;
}

namespace {

// Share of the step's cost that may sit in the ops setting the step-wide max
// dynamic shared memory before splitting them off stops being worth it. Above
// this the expensive ops are the hungry ones, so the occupancy they force is
// the occupancy the step needs anyway.
constexpr float kPoisonFraction = 0.25f;

// How full an emitted launch must be, as a fraction of its group's capacity.
// Same-stream launches serialize, so a launch far short of a full wave leaves
// most of the machine idle for its whole duration.
constexpr float kMinLaunchFill = 0.8f;

// A run of consecutive blocks of one op inside one launch. An op split across
// two launches contributes one chunk to each.
struct Chunk {
  int32_t op{0};
  int32_t firstBlockInOp{0};
  int32_t count{0};
};

// Ops only share a launch when they need the same occupancy, since a launch's
// shared memory is the max over what is in it. Ops needing a cooperative
// launch are kept apart as well: that launch is capped at what fits
// co-resident, so mixing splittable work into it would strand the rest.
struct GroupKey {
  bool cooperative{false};
  int32_t occupancy{1};

  bool operator<(const GroupKey& other) const {
    if (cooperative != other.cooperative) {
      return cooperative < other.cooperative;
    }
    return occupancy < other.occupancy;
  }
};

// One occupancy class, packed.
struct PackedGroup {
  GroupKey key;
  float cost{0};
  std::vector<std::vector<Chunk>> launches;
  /// Ops in this class, needed to re-derive the launch's shared memory when
  /// topping a cooperative launch back up.
  std::vector<int32_t> members;
};

// Most blocks a step may emit over all of its launches. Matches the multiple
// of one wave StepVectors::blockCapacity reserves the block array for, so the
// packer cannot outrun the reservation.
int32_t blockBudget(const GridDevice& device, const PartitionParams& params) {
  const int32_t oneWave =
      std::max(1, device.numSMs) * std::max(1, device.maxBlocksPerSM);
  return std::max(1, params.maxWaves) * oneWave;
}

// Scales 'want' down until it fits 'budget', keeping every op on at least one
// block. Only the blocks above that floor are scaled, so the ops with the most
// to give up give up the most.
void shrinkToBudget(std::vector<int32_t>& want, int32_t budget) {
  int64_t total = 0;
  for (auto blocks : want) {
    total += blocks;
  }
  if (total <= budget) {
    return;
  }
  const auto floorTotal = static_cast<int64_t>(want.size());
  if (budget <= floorTotal) {
    std::fill(want.begin(), want.end(), 1);
    return;
  }
  const double scale = static_cast<double>(budget - floorTotal) /
      static_cast<double>(total - floorTotal);
  total = 0;
  for (auto& blocks : want) {
    blocks = 1 + static_cast<int32_t>((blocks - 1) * scale);
    total += blocks;
  }
  while (total > budget) {
    auto* widest = &*std::max_element(want.begin(), want.end());
    if (*widest <= 1) {
      break;
    }
    --*widest;
    --total;
  }
}

// shrinkToBudget over the subset of 'want' named by 'members'.
void shrinkMembersToBudget(
    std::vector<int32_t>& want,
    const std::vector<int32_t>& members,
    int32_t budget) {
  std::vector<int32_t> subset;
  subset.reserve(members.size());
  for (auto op : members) {
    subset.push_back(want.at(op));
  }
  shrinkToBudget(subset, budget);
  for (size_t i = 0; i < members.size(); ++i) {
    want.at(members[i]) = subset.at(i);
  }
}

// Waves to size one cooperative occupancy class to. Rounding down and then
// shrinking to fit is what makes the waves full: a class wanting 1.2 waves
// packs better as one wave of fatter blocks than as a full wave plus a tail
// launch holding a fifth of the machine, since the work is the same either way
// and the tail costs a whole extra serialization.
int32_t cooperativeWaves(
    const std::vector<int32_t>& want,
    const std::vector<int32_t>& members,
    int32_t capacity,
    int32_t maxWaves) {
  int64_t total = 0;
  for (auto op : members) {
    total += std::min(want.at(op), capacity);
  }
  return std::clamp(
      static_cast<int32_t>(total / std::max(1, capacity)),
      1,
      std::max(1, maxWaves));
}

// Packs one cooperative occupancy class. A barrier op is never split --
// opBarrier waits for every block of its op and only a single launch can hold
// them co-resident -- but its block count is free, so once the class is at its
// wave target an op that does not fit the current launch is trimmed into what
// is left rather than opening another one.
//
// Trimming only ever removes blocks, so every launch stays within 'capacity'
// whatever the wave estimate was. That is the invariant packGroup and
// mergeThinTail could not offer between them: packGroup fills a launch to
// exactly capacity and mergeThinTail then appends a tail to it, which on a
// cooperative launch overruns the device's co-residency limit outright.
std::vector<std::vector<Chunk>> packCooperative(
    std::vector<int32_t>& want,
    const std::vector<int32_t>& members,
    int32_t capacity,
    int32_t targetLaunches) {
  std::vector<std::vector<Chunk>> launches(1);
  int32_t remaining = capacity;
  for (auto op : members) {
    int32_t blocks = std::min(want.at(op), capacity);
    if (blocks > remaining) {
      // A launch with nothing left always yields: an op trimmed to no blocks
      // would never run at all.
      if (remaining <= 0 ||
          static_cast<int32_t>(launches.size()) < targetLaunches) {
        launches.emplace_back();
        remaining = capacity;
      } else {
        blocks = remaining;
      }
    }
    want.at(op) = blocks;
    launches.back().push_back({.op = op, .firstBlockInOp = 0, .count = blocks});
    remaining -= blocks;
  }
  if (launches.back().empty()) {
    launches.pop_back();
  }
  return launches;
}

// LPT packing of one occupancy class: ops in descending cost, each opening a
// new launch when the current one is full. A barrier op is shrunk to the
// launch capacity rather than split, since opBarrier waits for all of its
// blocks and only a single launch can hold them co-resident. Shrinking it
// rewrites its entry in 'want', which is what the emitted blocks report as
// numBlocksInOp -- an op told it has more blocks than were launched would skip
// the slices of the ones that never ran.
std::vector<std::vector<Chunk>> packGroup(
    const std::vector<GridOp>& ops,
    std::vector<int32_t>& want,
    const std::vector<int32_t>& members,
    int32_t capacity) {
  std::vector<std::vector<Chunk>> launches(1);
  int32_t remaining = capacity;
  for (auto op : members) {
    if (ops.at(op).hasBarrier) {
      const int32_t blocks = std::min(want.at(op), capacity);
      want.at(op) = blocks;
      if (blocks > remaining && !launches.back().empty()) {
        launches.emplace_back();
        remaining = capacity;
      }
      launches.back().push_back(
          {.op = op, .firstBlockInOp = 0, .count = blocks});
      remaining -= blocks;
      if (remaining <= 0) {
        launches.emplace_back();
        remaining = capacity;
      }
      continue;
    }
    int32_t placed = 0;
    while (placed < want.at(op)) {
      const int32_t take = std::min(want.at(op) - placed, remaining);
      launches.back().push_back(
          {.op = op, .firstBlockInOp = placed, .count = take});
      placed += take;
      remaining -= take;
      if (remaining <= 0) {
        launches.emplace_back();
        remaining = capacity;
      }
    }
  }
  if (launches.back().empty()) {
    launches.pop_back();
  }
  return launches;
}

int32_t blocksIn(const std::vector<Chunk>& launch) {
  int32_t blocks = 0;
  for (const auto& chunk : launch) {
    blocks += chunk.count;
  }
  return blocks;
}

// Folds a launch that is far short of a full wave back into the one before it.
// Splitting trades one skewed wave for several packed ones plus their tails,
// which only pays while each is close to full; a thin tail costs a whole
// serialized launch for a sliver of the machine.
//
// The merged launch can exceed 'capacity', which an ordinary launch may do --
// the hardware runs the surplus in a second wave. Only a cooperative launch may
// not, and those are packed by packCooperative, which leaves no thin tail to
// fold.
void mergeThinTail(
    std::vector<std::vector<Chunk>>& launches,
    int32_t capacity) {
  const auto minBlocks =
      static_cast<int32_t>(kMinLaunchFill * static_cast<float>(capacity));
  while (launches.size() > 1 && blocksIn(launches.back()) < minBlocks) {
    auto tail = std::move(launches.back());
    launches.pop_back();
    launches.back().insert(launches.back().end(), tail.begin(), tail.end());
  }
}

// Fills what a cooperative launch leaves of its wave with splittable work from
// the same occupancy class. A cooperative launch is capped at what fits
// co-resident, so when the barrier ops do not reach that cap the rest of the
// machine would otherwise idle through the launch while the class's non-barrier
// work waits its turn in a launch of its own.
//
// Donating within one occupancy class is what makes this safe. A launch's
// shared memory is the max over its ops, and every member of a class yields the
// same blocksPerSM, so the donated blocks cannot lower the co-residency the
// barrier ops were sized against. Blocks are taken off the end of the donor's
// last launch, which is the one a thin tail would have landed in.
void backfillCooperative(
    std::vector<PackedGroup>& packed,
    const std::vector<GridOp>& ops,
    const GridDevice& device) {
  const int32_t numSMs = std::max(1, device.numSMs);
  for (auto& coop : packed) {
    if (!coop.key.cooperative || coop.launches.size() != 1) {
      continue;
    }
    // The same bound packCooperative filled to, for the same reason.
    int32_t slack = std::max(
                        1,
                        numSMs *
                            coopBlocksPerSM(
                                device,
                                coop.key.occupancy,
                                maxDynamicShared(ops, coop.members))) -
        blocksIn(coop.launches.front());
    if (slack <= 0) {
      continue;
    }
    for (auto& donor : packed) {
      if (donor.key.cooperative || donor.key.occupancy != coop.key.occupancy) {
        continue;
      }
      while (slack > 0 && !donor.launches.empty()) {
        auto& last = donor.launches.back();
        if (last.empty()) {
          donor.launches.pop_back();
          continue;
        }
        auto& chunk = last.back();
        const int32_t take = std::min(slack, chunk.count);
        coop.launches.front().push_back(
            {.op = chunk.op,
             .firstBlockInOp = chunk.firstBlockInOp + chunk.count - take,
             .count = take});
        chunk.count -= take;
        slack -= take;
        if (chunk.count == 0) {
          last.pop_back();
        }
      }
      break;
    }
  }
}

// Orders one launch's chunks by descending projected latency, for the same
// reason latencyOrder does it across a whole step.
void orderChunks(
    std::vector<Chunk>& launch,
    const std::vector<GridOp>& ops,
    const std::vector<int32_t>& want) {
  std::stable_sort(
      launch.begin(), launch.end(), [&](const Chunk& lhs, const Chunk& rhs) {
        return latencyOf(ops, want, lhs.op) > latencyOf(ops, want, rhs.op);
      });
}

} // namespace

std::vector<int32_t> sizeByQuantum(
    const std::vector<GridOp>& ops,
    const GridDevice& device,
    const PartitionParams& params) {
  std::vector<int32_t> want(ops.size(), 1);
  if (ops.empty()) {
    return want;
  }
  const int32_t oneWave =
      std::max(1, device.numSMs) * std::max(1, device.maxBlocksPerSM);

  float totalCost = 0;
  for (const auto& op : ops) {
    totalCost += std::max(0.0f, op.cost);
  }
  if (totalCost <= 0) {
    return want;
  }

  // What one block is worth. The measured figure when there is one -- it is
  // the only one that knows what the ops actually do -- and an even division
  // of the step over a wave otherwise, which is where the pro-rata split
  // starts from too.
  float quantum = params.minBlockCost > 0
      ? params.minBlockCost
      : totalCost / static_cast<float>(oneWave);
  if (quantum <= 0) {
    return want;
  }

  auto sizeAt = [&](float q) {
    int64_t total = 0;
    for (size_t i = 0; i < ops.size(); ++i) {
      const int32_t cap =
          ops[i].alwaysSingleBlock ? 1 : std::max(1, ops[i].maxBlocks);
      const auto scaled = static_cast<int32_t>(std::lround(ops[i].cost / q));
      want[i] = std::clamp(scaled, 1, cap);
      total += want[i];
    }
    return total;
  };

  int64_t total = sizeAt(quantum);

  // Too much for the waves we may emit: lengthen the block until it fits,
  // rather than trimming the tallest ops afterwards. Bisection because an op
  // at its cap or on its floor does not shrink with the quantum, so the total
  // is not proportional to it.
  const int64_t budget = blockBudget(device, params);
  if (total > budget) {
    float lo = quantum;
    float hi = quantum * 2;
    for (int32_t i = 0; i < 40 && sizeAt(hi) > budget; ++i) {
      lo = hi;
      hi *= 2;
    }
    for (int32_t i = 0; i < 40; ++i) {
      const float mid = (lo + hi) / 2;
      if (mid <= lo || mid >= hi) {
        break;
      }
      if (sizeAt(mid) > budget) {
        lo = mid;
      } else {
        hi = mid;
      }
    }
    total = sizeAt(hi);
    if (total > budget) {
      shrinkToBudget(want, static_cast<int32_t>(budget));
      return want;
    }
  }

  // A step whose blocks already finished together does not get widened. The
  // quantum says how much work there is, not whether the machine has room to
  // absorb it in parallel: expanding an already-full step turns one wave into
  // several serialized launches running the same work. Only the measurement can
  // distinguish the two cases -- the cost-model skew is derived from the same
  // costs this sizing uses, so it calls a mis-costed step balanced.
  if (params.measuredBlocks > 0 &&
      params.measuredUtil >= params.keepWidthUtil &&
      total > params.measuredBlocks) {
    shrinkToBudget(want, params.measuredBlocks);
    return want;
  }

  // Round up to a whole wave and hand the surplus to whoever has the most work
  // left per block. Half a wave of idle SMs is free capacity, and the op that
  // would use it is by definition the one holding the step up.
  const int64_t waves = (total + oneWave - 1) / oneWave;
  const int64_t targetTotal = std::min<int64_t>(waves * oneWave, budget);
  for (int64_t spare = targetTotal - total; spare > 0; --spare) {
    int32_t best = -1;
    float bestLatency = 0;
    for (size_t i = 0; i < ops.size(); ++i) {
      const int32_t cap =
          ops[i].alwaysSingleBlock ? 1 : std::max(1, ops[i].maxBlocks);
      if (want.at(i) >= cap) {
        continue;
      }
      const float latency = ops[i].cost / static_cast<float>(want.at(i));
      if (latency > bestLatency) {
        bestLatency = latency;
        best = static_cast<int32_t>(i);
      }
    }
    if (best < 0) {
      break;
    }
    ++want.at(best);
  }
  return want;
}

bool shouldPartition(const GridStats& stats, const PartitionParams& params) {
  if (stats.numOps < 2 || stats.totalCost <= 0) {
    return false;
  }
  if (stats.skew >= params.skewThreshold) {
    return true;
  }
  // No skew to fix, but a cheap op is still cutting everyone's occupancy.
  return stats.occupancy < stats.bestOccupancy &&
      stats.poison < kPoisonFraction;
}

std::optional<LaunchPlan> partitionLaunches(
    const std::vector<GridOp>& ops,
    const GridDevice& device,
    const GridStats& stats,
    const PartitionParams& params,
    const std::vector<int32_t>* presized) {
  if (ops.empty() || stats.targetBlocks <= 0 || stats.totalCost <= 0) {
    return std::nullopt;
  }

  // Cost one block is worth. Under an even division of the step over one wave
  // the assignment reproduces today's pro-rata split; a larger minBlockCost
  // makes blocks fewer and longer, so a step is only split when the work
  // genuinely does not fit in a wave of blocks that size.
  std::vector<int32_t> want;
  if (presized != nullptr) {
    want = *presized;
    want.resize(ops.size(), 1);
  } else {
    const float quantum = std::max(
        params.minBlockCost,
        stats.totalCost / static_cast<float>(stats.targetBlocks));
    if (quantum <= 0) {
      return std::nullopt;
    }
    want.assign(ops.size(), 1);
    for (size_t i = 0; i < ops.size(); ++i) {
      const auto scaled =
          static_cast<int32_t>(std::lround(ops[i].cost / quantum));
      want.at(i) = std::clamp(scaled, 1, std::max(1, ops[i].maxBlocks));
    }
    shrinkToBudget(want, blockBudget(device, params));
  }

  // A barrier op that came out on one block passes its barrier as soon as it
  // runs, so it needs no cooperative launch and can ride along with everything
  // else rather than costing a serialized launch of its own.
  std::map<GroupKey, std::vector<int32_t>> groups;
  for (size_t i = 0; i < ops.size(); ++i) {
    const GroupKey key{
        .cooperative = needsCooperative(ops, want, static_cast<int32_t>(i)),
        .occupancy = blocksPerSM(device, ops[i].dynamicShared)};
    groups[key].push_back(static_cast<int32_t>(i));
  }

  std::vector<PackedGroup> packed;
  for (auto& [key, members] : groups) {
    std::stable_sort(
        members.begin(), members.end(), [&](int32_t lhs, int32_t rhs) {
          return ops[lhs].cost > ops[rhs].cost;
        });
    // A cooperative launch is packed to exactly its capacity, so that figure
    // has to be one the driver will accept rather than an estimate of it. An
    // ordinary launch may exceed its capacity -- the hardware runs the surplus
    // in a later wave -- so it keeps the estimate and the occupancy classes
    // stay as they were.
    const int32_t capacity = std::max(
        1,
        std::max(1, device.numSMs) *
            (key.cooperative
                 ? coopBlocksPerSM(
                       device, key.occupancy, maxDynamicShared(ops, members))
                 : key.occupancy));
    PackedGroup group;
    group.key = key;
    group.members = members;
    if (key.cooperative) {
      const int32_t waves =
          cooperativeWaves(want, members, capacity, params.maxWaves);
      shrinkMembersToBudget(want, members, waves * capacity);
      group.launches = packCooperative(want, members, capacity, waves);
    } else {
      group.launches = packGroup(ops, want, members, capacity);
      mergeThinTail(group.launches, capacity);
    }
    for (auto member : members) {
      group.cost += ops[member].cost;
    }
    packed.push_back(std::move(group));
  }
  backfillCooperative(packed, ops, device);

  size_t numLaunches = 0;
  for (const auto& group : packed) {
    for (const auto& launch : group.launches) {
      numLaunches += launch.empty() ? 0 : 1;
    }
  }
  if (numLaunches <= 1 && presized == nullptr) {
    return std::nullopt;
  }
  std::stable_sort(
      packed.begin(), packed.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.cost > rhs.cost;
      });

  LaunchPlan plan;
  plan.blocksPerOp = want;
  for (auto& group : packed) {
    for (auto& launch : group.launches) {
      orderChunks(launch, ops, want);
      LaunchSegment segment;
      segment.firstBlock = static_cast<int32_t>(plan.blocks.size());
      for (const auto& chunk : launch) {
        segment.dynamicShared =
            std::max(segment.dynamicShared, ops[chunk.op].dynamicShared);
        segment.cooperative =
            segment.cooperative || needsCooperative(ops, want, chunk.op);
        for (int32_t block = 0; block < chunk.count; ++block) {
          plan.blocks.push_back({chunk.op, chunk.firstBlockInOp + block});
        }
      }
      segment.numBlocks =
          static_cast<int32_t>(plan.blocks.size()) - segment.firstBlock;
      if (segment.numBlocks > 0) {
        plan.segments.push_back(segment);
      }
    }
  }
  return plan;
}

} // namespace torch::wave
