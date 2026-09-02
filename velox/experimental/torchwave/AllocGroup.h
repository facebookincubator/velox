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

#include <ATen/ATen.h>
#include <torch/nativert/executor/ExecutionFrame.h>

#include <torch/nativert/graph/Graph.h>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include "velox/experimental/torchwave/Cat.h"
#include "velox/experimental/torchwave/CompiledOp.h"

namespace torch::wave {

/// Byte alignment every slot in a group buffer starts at. Wide enough for the
/// widest vector load the kernels use, so a slot's base is as aligned as a
/// separate allocation's would have been.
constexpr int64_t kAllocGroupAlign = 16;

/// Rounds a byte count up to kAllocGroupAlign.
int64_t alignAllocSize(int64_t bytes);

/// One output to be carved out of a group's buffer. The shape is settled per
/// execution, by the reserve function of the op that produces the value, so a
/// request is built fresh each run rather than cached with the plan.
struct AllocRequest {
  /// Frame value the view is written to.
  nativert::ValueId actualId{-1};
  c10::ScalarType dtype{c10::ScalarType::Float};
  std::vector<int64_t> dims;
};

/// The buffer backing one group for one execution, plus the views carved from
/// it. Every view shares the base's storage, so the allocation is freed when
/// the last of them goes out of scope -- no bookkeeping and no explicit free.
struct AllocGroupBuffer {
  at::Tensor base;
  std::vector<at::Tensor> slots;
  /// Offset of each slot in the base, parallel to 'slots'.
  std::vector<int64_t> offsets;
  int64_t totalBytes{0};
};

/// Lays out 'requests' end to end at kAllocGroupAlign, makes one CUDA byte
/// allocation of the total, and returns a view of it per request with that
/// request's shape and dtype.
///
/// The views are ordinary tensors sharing one storage: writing through one
/// leaves the others alone because their extents do not overlap, and the base
/// dies with the last surviving view. A request of zero elements still gets a
/// (zero-element) view, so callers can index slots positionally.
AllocGroupBuffer allocateAllocGroup(const std::vector<AllocRequest>& requests);

/// Where one allocation lives, as a point at which it is produced and a point
/// at which it is released. A point is a (node, step) pair: the node is the
/// position of a CompiledNode in the graph's execution order and the step is an
/// index into that node's compiled grid, which the cooperative-grid mode
/// settles before the first execution -- so both bounds are known without
/// running anything.
///
/// Node -1 in both points is the degenerate single-node case, where the two
/// steps are comparable on their own.
struct AllocLifetime {
  nativert::ValueId actualId{-1};
  c10::ScalarType dtype{c10::ScalarType::Float};

  /// Node whose launch produces the value.
  int32_t allocNode{-1};

  /// Step of 'allocNode' whose launch produces the value.
  int32_t allocStep{-1};

  /// Node that releases the value -- the one its last use in the whole graph
  /// falls in. -1 for a value that is never released (a graph output, or one
  /// nothing reads), which is never grouped: its buffer would pin the whole
  /// group for as long as it lives.
  int32_t freeNode{-1};

  /// Step of 'freeNode' at whose end the value is released. -1 alongside
  /// freeNode -1, and never negative otherwise.
  int32_t freeStep{-1};

  /// True when the shape is only known once a device-to-host transfer has
  /// landed: a reserve that reads a returned count, or an output the device
  /// sizes. Groups holding one of these cannot be laid out until the host has
  /// waited, so they are ordered after the ones that can.
  bool needsSync{false};
};

/// Lays a group out as the result of a fused cat / stack instead of packing its
/// slots end to end. The buffer is the concat result itself and each member is
/// the region of it one operand occupies, so the kernel that produces the
/// operand writes straight into the result and the concat copies nothing.
///
/// A concat leaves no room for the padding between slots an ordinary group
/// inserts, and needs none: the result is one dense tensor of one dtype, so
/// every operand already starts at a whole number of its elements.
struct ConcatGroupLayout {
  /// Frame value of the concat result, which the group allocates and writes.
  nativert::ValueId resultId{-1};
  c10::ScalarType dtype{c10::ScalarType::Float};
  bool isStack{false};
  int32_t dim{0};
  int8_t outRank{-1};

  /// Every operand of the concat, in the order the result joins them, with the
  /// value ids already translated to the frame's.
  std::vector<ConcatInputInfo> operands;

  /// Index into the group's members for each operand, or -1 for one the group
  /// does not carve: an operand an earlier step already materialized, or one
  /// the concat's own kernel produces. The layout only reads such an operand's
  /// extent, to place the ones after it; its data the concat still copies in.
  std::vector<int32_t> memberOfOperand;
};

/// Allocations that share a lifetime, laid out in one buffer.
struct AllocGroup {
  int32_t allocNode{-1};
  int32_t allocStep{-1};
  int32_t freeNode{-1};
  int32_t freeStep{-1};
  bool needsSync{false};
  /// Values in the group, in the order their slots are laid out.
  std::vector<nativert::ValueId> actualIds;
  std::vector<c10::ScalarType> dtypes;

  /// Set when the group's buffer is a concat result rather than a plain byte
  /// buffer. Null for an ordinary lifetime group.
  std::shared_ptr<const ConcatGroupLayout> concat;
};

/// Groups the lifetimes produced at the same point and released at the same
/// point.
///
/// Only allocations with a known release point are grouped: one whose buffer is
/// never released would keep every other slot in its group alive with it,
/// turning a lifetime-shaped allocation into a leak of the whole group. Those
/// are returned in 'ungrouped' for the caller to allocate singly.
///
/// Within one (allocation point, release point) pair the sync-free allocations
/// are kept in a separate group from the ones that must wait, and every
/// sync-free group is ordered before every waiting one, so the host lays out
/// and allocates everything it can before it blocks on the device.
std::vector<AllocGroup> buildAllocGroups(
    const std::vector<AllocLifetime>& lifetimes,
    std::vector<nativert::ValueId>* ungrouped = nullptr);

/// True when the allocation-group path should run: the config selects it, the
/// cooperative grid is in force -- which is what fixes the step boundaries an
/// allocation's lifetime is expressed in before the first execution -- and the
/// intermediates are freed, without which no group's buffer is ever released
/// and there are no lifetimes to fold.
bool allocGroupEnabled();

/// A fused cat / stack as the whole-graph scan sees it: the result and the
/// operands it joins, in frame values, with every id already translated out of
/// the KernelOperation's formal subgraph.
struct ConcatFootprint {
  nativert::ValueId resultId{-1};
  c10::ScalarType dtype{c10::ScalarType::Float};
  bool isStack{false};
  int32_t dim{0};
  int8_t outRank{-1};
  /// The operands in join order, sizeExpr translated to frame values.
  std::vector<ConcatInputInfo> operands;
};

/// What one launch contributes to the lifetime scan: the step it runs at, the
/// values it reads, and the outputs it really allocates. Separating this from
/// the grid walk is what lets the lifetime rules be tested without a compiled
/// graph.
struct LaunchFootprint {
  int32_t step{-1};
  std::vector<nativert::ValueId> reads;
  /// Allocated outputs, parallel arrays: value, element type, and whether the
  /// shape needs a transfer to land before it is known.
  std::vector<nativert::ValueId> writes;
  std::vector<c10::ScalarType> writeDtypes;
  std::vector<bool> writeNeedsSync;

  /// Fused concats this launch runs. Empty for all but the few launches that
  /// hold one.
  std::vector<ConcatFootprint> concats;
};

/// What one node contributes to the whole-graph lifetime scan: the launches of
/// its compiled grid and the release point of every value whose last use in the
/// graph falls in it. Separating this from the grid walk is what lets the
/// cross-node rules be tested without a compiled graph.
struct NodeFootprint {
  /// Position of the node in the graph's execution order.
  int32_t node{-1};

  /// Index of the last step the node runs, which is where a released value
  /// whose readers are not known goes.
  int32_t lastStep{0};

  /// One entry per launch of the node's grid, with 'step' an index into that
  /// grid.
  std::vector<LaunchFootprint> launches;

  /// Values whose last use in the whole graph falls in this node -- the node's
  /// last-use set -- and, parallel to it, the step of this node at which each
  /// is released.
  std::vector<nativert::ValueId> releasedIds;
  std::vector<int32_t> releaseSteps;

  /// Values whose tensor OBJECT, not merely whose shape, this node's sizing
  /// pass takes and keeps: the operand a host-side view node is built from, the
  /// self an in-place output is aliased to, the argument a mutating op writes
  /// through. See AllocGroupStats::numBorrowedInOwnNode.
  std::vector<nativert::ValueId> borrowedIds;
};

/// What the whole-graph scan made of the allocations it saw. Every allocated
/// output ends up in exactly one of the grouped or the not-grouped counts, so
/// the report can account for all of them and name what the mode cannot fold.
struct AllocGroupStats {
  /// Allocated tensor outputs the scan described.
  int32_t numAllocated{0};
  int32_t numGrouped{0};

  // Why the rest were left out, summing to numAllocated - numGrouped.

  /// Never released: a graph output, or a value nothing reads.
  int32_t numEscaping{0};
  /// Its tensor object is taken and kept by the sizing pass of the node that
  /// allocates it.
  ///
  /// A member carries a shape-only placeholder between being sized and being
  /// carved, which is enough for everything that reads a shape -- a size
  /// expression, a reserve function, the grid choice. It is not enough for the
  /// three places that keep the tensor itself: a host-side view node builds its
  /// view from it, an in-place output is aliased to it, a mutating op writes
  /// through it. Those would keep the placeholder, and nothing would ever
  /// replace it.
  int32_t numBorrowedInOwnNode{0};
  /// Written by launches at more than one point. Which write allocates the
  /// buffer is then a question of execution order rather than of the plan, and
  /// a group carved at the wrong one would replace a live buffer.
  int32_t numMultiWrite{0};
  /// Released before the point that produces it, which is not a lifetime this
  /// can reason about.
  int32_t numBackwardFree{0};
  /// Read after the point it is released at, meaning the last-use analysis and
  /// the grid disagree. Grouping it would carve a buffer that is freed while a
  /// launch still reads it.
  int32_t numReadAfterFree{0};
  /// Taken by a concat group, which allocates it as part of a concat result
  /// rather than on its own: an operand the concat joins, or the result.
  int32_t numInConcatGroup{0};

  int32_t numGroups{0};
  /// Groups released in a later node than the one that allocates them, which
  /// is the consolidation a per-node scan cannot see.
  int32_t numCrossNodeGroups{0};
  int32_t largestGroup{0};

  /// Groups the same values would form if the release point were the releasing
  /// node alone, ignoring which of its steps. The difference against numGroups
  /// is what insisting on the exact step costs.
  int32_t numGroupsByNode{0};

  /// Concats whose result the pass places ahead of its operands, and the
  /// operands those groups carve. Each carved operand is one allocation and one
  /// concat copy that do not happen.
  int32_t numConcatGroups{0};
  int32_t numConcatMembers{0};
  /// Fused concats the pass looked at and could not place, by reason. A concat
  /// that is placed contributes to none of these.
  int32_t numConcatTooFew{0};
  int32_t numConcatOnDevice{0};
  int32_t numConcatUnplaceableOperand{0};
  int32_t numConcatNoMembers{0};
  /// Joined on an axis other than the outermost, so an operand's region of the
  /// result is a strided band rather than a run.
  int32_t numConcatStridedBand{0};
};

/// Turns per-node footprints into one lifetime per allocated value.
///
/// The allocation point is the point of the launch that writes the value. The
/// release point is where the node that last-uses it says it goes, which is the
/// only place that knows: a value produced in one node and read in another is
/// released by the reader, several nodes downstream of its allocation. A value
/// no node releases gets no release point and is left for the caller to
/// allocate singly, as is one its own node uses again -- see
/// AllocGroupStats::numUsedInOwnNode for why.
///
/// Output is sorted by (allocation point, value) so a slot lands at the same
/// offset on every execution.
/// 'claimed' names the values a concat group already allocates, which are left
/// out of the lifetime grouping and counted under numInConcatGroup: allocating
/// them twice would replace what the concat placed.
std::vector<AllocLifetime> graphAllocLifetimes(
    const std::vector<NodeFootprint>& nodes,
    AllocGroupStats* stats = nullptr,
    const folly::F14FastSet<nativert::ValueId>* claimed = nullptr);

/// Turns the fused concats the scan saw into the groups that place a concat
/// result ahead of the operands that fill it, and adds every value those groups
/// allocate -- the result and the operands they carve -- to 'claimed'.
///
/// A concat qualifies when the host can lay its result out at a point strictly
/// before the concat's own launch. That needs every operand's extent to be
/// knowable there: an operand an earlier launch produces is measured from the
/// frame, and one the concat's own kernel produces from its size expression, so
/// an operand whose extent the device settles, one that is a view of somebody
/// else's buffer, and one that reaches the host only through a reserve function
/// all disqualify the concat. The members -- the operands the group carves --
/// are those produced by the latest launch among them, which is the last point
/// at which redirecting a write is still possible.
///
/// Concats of two operands are left alone: the layout constraints they impose
/// on their producers are the same, and there is one allocation to save.
std::vector<AllocGroup> graphConcatGroups(
    const std::vector<NodeFootprint>& nodes,
    folly::F14FastSet<nativert::ValueId>& claimed,
    AllocGroupStats* stats = nullptr);

/// Describes every launch of the cooperative grid of every op in 'ops'.
///
/// Only outputs that are really allocated are described: a view, a delegated
/// output, an in-place alias of another value and a shape-only tensor all
/// either borrow someone else's storage or have none, so none of them belongs
/// in a group. An output whose shape comes from a reserve function or is set by
/// the device is marked needsSync, since its size is not known until a transfer
/// has landed.
///
/// Step indices are comparable across ops because the composite runs step 0 of
/// every op, then step 1, and so on. Reading the grid is only meaningful when
/// it cannot change mid-run, which is why this is cooperative-grid only.
/// 'borrowedIds', when given, collects the values whose tensor object the
/// sizing pass keeps, for NodeFootprint::borrowedIds.
std::vector<LaunchFootprint> collectLaunchFootprints(
    std::vector<OpInvocation>& ops,
    const IdToValueMap& idToValue,
    const ValueTypes& types,
    std::vector<nativert::ValueId>* borrowedIds = nullptr);

/// Walks every compiled node of the graph and returns what the whole-graph
/// lifetime scan runs on: each node's launches, and the step of that node at
/// which each of its last-use values is released.
std::vector<NodeFootprint> collectNodeFootprints(
    const std::vector<std::unique_ptr<CompiledNode>>& nodes,
    const IdToValueMap& idToValue,
    const ValueTypes& types);

/// The grouping decided for one invocation -- the groups it allocates, which
/// it may not be the node that releases. Settled before the graph's first
/// execution and reused by every later one: only the shapes change per run;
/// which values share a buffer, and in what order they sit in it, do not.
struct AllocGroupPlan {
  std::vector<AllocGroup> groups;

  /// Allocated values of this node that belong to no group. Kept for the
  /// report: they are what the mode cannot fold together.
  std::vector<nativert::ValueId> ungrouped;

  /// Group index per grouped value. A value absent from this is allocated the
  /// ordinary way even while the plan is in force, which is what lets the mode
  /// cover the values it can reason about and leave the rest alone.
  folly::F14FastMap<nativert::ValueId, size_t> groupOfValue;

  /// Index into 'groups' of the groups allocated at 'step', in the order they
  /// must be materialized: everything layoutable without waiting on the device
  /// first. Empty for a step that allocates nothing.
  std::vector<std::vector<size_t>> groupsByStep;
};

/// The grouping decided for a whole graph. A group's members are always
/// allocated by one node, so the plan splits cleanly per node even though the
/// lifetimes that shaped it do not.
struct GraphAllocGroupPlan {
  /// One plan per node, in the graph's execution order. A node that groups
  /// nothing gets an empty plan rather than being left out, so this can be
  /// indexed by node position.
  std::vector<AllocGroupPlan> perNode;
  AllocGroupStats stats;

  /// Every value a concat group places: the results and the operands carved
  /// out of them. These do not get a buffer of their own, so nothing else may
  /// hand them one -- see WaveGraph::isConcatPlaced.
  folly::F14FastSet<nativert::ValueId> concatPlaced;
};

/// Builds the whole-graph plan from the footprints. Cheap enough to run once
/// per graph and never again; it touches no frame and allocates nothing on the
/// device.
GraphAllocGroupPlan buildGraphAllocGroupPlan(
    const std::vector<NodeFootprint>& nodes);

/// Renders what the plan folded together and what it could not, as the lines
/// the mode prints once per graph under the kTiming trace bit.
std::string allocGroupPlanReport(const GraphAllocGroupPlan& plan);

/// Builds the whole-graph plan for 'graph' and hands each compiled node the
/// slice of it that node allocates. Called once, on the first execution that
/// runs in allocation-group mode; the plan depends only on the compiled grids,
/// so every later execution reuses it.
void installGraphAllocGroupPlans(WaveGraph& graph, const ValueTypes& types);

/// Diverts the allocation of one group's members while it is installed.
///
/// The members of a group are sized by the ops that produce them, spread over
/// however many launches the step holds, and the buffer cannot be carved until
/// the last of them is known. So sizing and allocation, which the ordinary path
/// does together per output, are split: with a collector installed the sizing
/// path records the shape it would have allocated and leaves the frame slot
/// alone, and the group is materialized once every member has reported.
///
/// Installed on the calling thread for the collector's lifetime. Nesting is
/// rejected: two live collectors would race for the same output.
class AllocGroupCollector {
 public:
  /// Collects for every group in 'groups' at once. A step's groups are all
  /// sized by the same pass over its launches, so one collector covers them
  /// and each is materialized as soon as its own members are complete --
  /// which, for the groups that need no transfer, is before the host waits.
  ///
  /// An empty list is legal and collects nothing: it is what a step with no
  /// groups installs, so the parameter fill is suppressed uniformly for every
  /// step of the mode rather than only for the steps that group something.
  explicit AllocGroupCollector(const std::vector<const AllocGroup*>& groups);
  explicit AllocGroupCollector(const AllocGroup& group);
  ~AllocGroupCollector();

  // The installed pointer is this object's address, so a copy or a move would
  // leave the thread-local naming an object that is not the one collecting.
  AllocGroupCollector(const AllocGroupCollector&) = delete;
  AllocGroupCollector& operator=(const AllocGroupCollector&) = delete;
  AllocGroupCollector(AllocGroupCollector&&) = delete;
  AllocGroupCollector& operator=(AllocGroupCollector&&) = delete;

  /// Records 'dims' as the shape of 'actualId' and returns true when the value
  /// belongs to one of the collected groups, meaning the caller must not
  /// allocate it. Returns false for anything else, which the caller allocates
  /// as usual.
  ///
  /// A member sized twice in one step keeps the later shape: the second sizing
  /// is the settled one (a deferred op re-runs after its transfer lands).
  bool capture(nativert::ValueId actualId, c10::IntArrayRef dims);

  size_t numGroups() const {
    return groups_.size();
  }

  /// True once every member of group 'g' has been sized. A group that is not
  /// complete cannot be laid out; materializing it would leave the unsized
  /// members None.
  bool complete(size_t g) const;

  /// True for a group whose slots have already been handed to the frame, so a
  /// second materialization pass over the same step skips it. The sync-free
  /// groups are materialized before the host waits and the rest after, and
  /// both passes walk the same list.
  bool materialized(size_t g) const {
    return groups_[g].materialized;
  }

  void markMaterialized(size_t g) {
    groups_[g].materialized = true;
  }

  /// True when 'actualId' belongs to one of the collected groups, sized or
  /// not. What it answers, for a caller that finds the value's frame slot
  /// empty, is "this one is about to be carved" rather than "this one was
  /// never produced".
  bool owns(nativert::ValueId actualId) const {
    return slotOf_.count(actualId) != 0;
  }

  /// Members of group 'g' that were never sized, for the report.
  std::vector<nativert::ValueId> missing(size_t g) const;

  /// The requests of group 'g' the sizing pass actually reached, in the group's
  /// own order.
  ///
  /// A member it never reached is left out rather than carved: the value was
  /// allocated some other way -- an elementwise output written in place over a
  /// reusable input, an output the op does not size at all -- and its frame
  /// slot must not be replaced by a slot of this group's. Only meaningful once
  /// the step's sizing is over; before that, an unreached member is one that
  /// has not been sized yet.
  std::vector<AllocRequest> sizedRequests(size_t g) const;

  /// Whether group 'g' needs a device-to-host transfer to have landed before
  /// its shapes are known.
  bool needsSync(size_t g) const;

  /// The concat layout of group 'g', or null when it is an ordinary lifetime
  /// group whose slots are packed end to end.
  const ConcatGroupLayout* concatLayout(size_t g) const;

  /// Which members of group 'g' the sizing pass reached, in the group's own
  /// order.
  const std::vector<bool>& sizedMask(size_t g) const {
    return groups_[g].sized;
  }

  /// The requests of group 'g', in the group's own order, so a slot lands at
  /// the same offset on every execution regardless of the order the ops sized
  /// them in. Only meaningful once complete(g).
  const std::vector<AllocRequest>& requests(size_t g) const {
    return groups_[g].requests;
  }

  /// Convenience for the single-group case.
  bool complete() const {
    return complete(0);
  }
  const std::vector<AllocRequest>& requests() const {
    return requests(0);
  }
  std::vector<nativert::ValueId> missing() const {
    return missing(0);
  }

 private:
  struct Collected {
    const AllocGroup* group{nullptr};
    std::vector<AllocRequest> requests;
    std::vector<bool> sized;
    bool materialized{false};
  };

  /// Which group a member belongs to, and where in that group's requests it
  /// sits. One lookup per allocation, which is why this is a map rather than a
  /// scan over the groups.
  struct Slot {
    size_t group{0};
    size_t index{0};
  };

  std::vector<Collected> groups_;
  folly::F14FastMap<nativert::ValueId, Slot> slotOf_;
};

/// The collector installed on this thread, or nullptr when the ordinary
/// allocation path is in force.
AllocGroupCollector* currentAllocCollector();

/// Allocates one buffer for 'requests' and writes each slot into the frame
/// under its request's value id.
///
/// The frame holds the only surviving references once this returns, so the
/// group is freed when the last of those slots is released -- which is the step
/// the group was formed around. Returns the buffer so a caller that wants the
/// byte total (the allocation trace, the timing report) can read it without
/// going back through the frame.
AllocGroupBuffer materializeAllocGroup(
    const std::vector<AllocRequest>& requests,
    nativert::ExecutionFrame& frame);

/// Allocates the concat result 'layout' describes and writes it to the frame,
/// then hands every member the region of it its operand occupies.
///
/// 'requests' are the group's members in the group's own order and 'sized' says
/// which of them the sizing pass reached; an operand the group meant to carve
/// but the pass never sized keeps whatever the ordinary path gave it and is
/// measured from the frame like any other uncarved operand, so the result is
/// still laid out correctly -- that operand's data is simply copied in by the
/// concat rather than written in place.
AllocGroupBuffer materializeConcatGroup(
    const ConcatGroupLayout& layout,
    const std::vector<AllocRequest>& requests,
    const std::vector<bool>& sized,
    nativert::ExecutionFrame& frame);

/// True when the concat groups of graphConcatGroups should be built: the
/// allocation-group mode is running and the config selects them.
bool concatAllocGroupEnabled();

} // namespace torch::wave
