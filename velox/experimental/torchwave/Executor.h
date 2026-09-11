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

#include <torch/nativert/executor/GraphExecutorBase.h>
#include <torch/nativert/executor/Weights.h>
#include <deque>
#include "velox/experimental/torchwave/Compile.h"
#include "velox/experimental/torchwave/CompiledOp.h"
#include "velox/experimental/torchwave/LaunchPartition.h"
#include "velox/experimental/torchwave/Registry.h"
#include "velox/experimental/torchwave/Utils.h"
#include "velox/experimental/wave/common/Buffer.h"

namespace facebook::velox::wave {
class GpuArena;
} // namespace facebook::velox::wave

namespace torch::wave {

struct ModelContext;

/// Initializes process-wide GPU resources (arenas, stream/event pools).
/// Does nothing if no GPU is available. Safe to call multiple times.
void initialize();

/// Copies tensors from host to device. Allocates a contiguous pinned buffer,
/// copies tensor storage into it, allocates device tensors via PyTorch, and
/// enqueues async H2D transfers on 'stream'. The caller must wait on 'stream'
/// before using the output tensors.
void tensorsToDevice(
    const std::vector<at::Tensor>& in,
    std::vector<at::Tensor>& out,
    facebook::velox::wave::Stream& stream);

/// Copies tensors from device to host. Allocates a contiguous pinned buffer,
/// enqueues an async D2H transfer on 'stream', and builds output tensors
/// backed by the pinned buffer. The caller must wait on 'stream' before
/// accessing the output data.
void tensorsToHost(
    const std::vector<at::Tensor>& in,
    std::vector<at::Tensor>& out,
    facebook::velox::wave::Stream& stream);

/// Debug info pointers for a single kernel launch, linking the pinned (host)
/// and device copies so that getDebugInfo() can queue D2H transfers.
struct LaunchDebugInfo {
  DebugInfo* pinnedInfo;
  DebugInfo* deviceInfo;
  int32_t numBlocks;
  int32_t sequenceNumber;
  int32_t stepIdx;
};

/// Device-timeline events bracketing one step's GPU work. Wave events are
/// absent for a step with no fused kernel, standalone events for a step with no
/// device-side eager op. The *Begin events exist only under the kTiming trace
/// bit; the *Done events are always present because they carry the cross-stream
/// ordering that replaces the host stream syncs.
struct StepEvents {
  int32_t sequenceNumber{0};
  int32_t stepIdx{0};
  EventP waveBegin;
  EventP waveDone;
  EventP standaloneBegin;
  EventP standaloneDone;
};

/// Per-launch metadata stored in thread-local alongside the DebugInfo.
struct LaunchMeta {
  int32_t sequenceNumber{0};
  int32_t stepIdx{0};
  int32_t numBlocks{0};
  int64_t gatherUs{0};
  int64_t gridUs{0};
  int64_t allocUs{0};
  // The part of allocUs spent inside the allocator itself. allocUs minus this
  // is shape arithmetic and building the views over what was allocated.
  int64_t allocCallUs{0};
  int64_t fillUs{0};
  int64_t kernelUs{0};
  int64_t standaloneUs{0};
  // Wall time of the metadata-only shortcut-standalone batch (one
  // hardware_timestamp pair around the tight loop).
  int64_t shortcutUs{0};
  bool standaloneBound{false};
  bool noDtoH{false};
  int64_t inputBytes{0};
  int64_t outputBytes{0};
  int64_t currentBytes{0};
  // Time spent on reference-frame device-to-host copy and comparison for this
  // step (debug-only; excluded from the reported e2e time).
  int64_t refCheckUs{0};
  // Bytes of copying the clone-elision pass saved, charged to this step.
  int64_t elidedCloneBytes{0};
  // Allocation groups carved at this step, and the outputs they covered. Zero
  // outside the allocation-group mode, and also for a step of it whose outputs
  // all escape the invocation.
  int32_t allocGroups{0};
  int32_t allocGroupTensors{0};
  // Device-measured spans from this step's events (kTiming only). kernelUs and
  // standaloneUs above are host wall times around issuing the work; these are
  // what the GPU actually spent on it.
  int64_t kernelGpuUs{0};
  int64_t standaloneGpuUs{0};
  // Device idle before this step: the gap between the end of the previous
  // step's GPU work and the start of this one's.
  int64_t gpuIdleUs{0};
  // Ops in this step that read a value an earlier step returned over a D2H.
  // See StepVectors for the details.
  int32_t d2hDepFused{0};
  int32_t d2hDepStandalone{0};
  int32_t d2hDepShortcut{0};
  int32_t d2hDepOnPrevStep{0};
  int32_t d2hNearestProducer{-1};
  int32_t viewNodeDescs{0};
  // Total ops in the step, so the dependent counts read as a fraction.
  int32_t numFused{0};
  int32_t numStandalone{0};
  int32_t numShortcut{0};
  // How balanced this step's grid came out, and how many launches it ran as.
  GridStats gridStats;

  /// The launches the step ran as, each owning a contiguous range of the block
  /// array the DebugInfo of this step is indexed by. Empty for a step that was
  /// not split. Kept so the balance report can score each launch on its own:
  /// they run back to back, so a block in one was never able to run beside a
  /// block in another.
  std::vector<LaunchSegment> segments;
};

/// Per-thread debug info from the most recent wave execution. Populated by
/// executeWave when WaveConfig::keepStatsOnThread is true.
struct WaveThreadInfo {
  std::vector<std::vector<DebugInfo>> debugInfo;
  std::vector<LaunchMeta> launchMeta;
  std::string errors;
  /// Standalone execution times, sorted descending. Indices line up with
  /// standaloneLabels and standaloneTargets.
  std::vector<int64_t> standaloneTimes;
  /// Human-readable label for each standalone, parallel to standaloneTimes.
  std::vector<std::string> standaloneLabels;
  /// Operator target string for each standalone, parallel to standaloneTimes.
  std::vector<std::string> standaloneTargets;
  /// Peak torch CUDA caching-allocator allocated bytes reached over the covered
  /// run, read from the allocator's high-water mark (peak stats are reset at
  /// the start of the run), so it captures transient intra-step peaks. Filled
  /// when trace kTiming is on.
  int64_t peakBytes{0};
  /// Total device idle across the run, summed from the per-step gpuIdleUs: wall
  /// time in which neither the wave stream nor the eager standalones had work
  /// on the device, because the host was still preparing the next launch.
  int64_t gpuIdleUs{0};
  /// Last-use values released before their node's last step, and at it.
  int32_t numLastUseEarly{0};
  int32_t numLastUseAtNodeEnd{0};
  /// Deferred device-to-host transfers and the executed steps they spanned.
  int32_t numDeferredReturns{0};
  int64_t deferredStepSpan{0};
  /// Host run-ahead depth over the run: total, samples, max, and how many steps
  /// began with the device queue already drained.
  int64_t runAheadSum{0};
  int32_t runAheadSamples{0};
  int32_t runAheadMax{0};
  int32_t numDrainedStarts{0};
  /// Drains forced by the delayed-free ceiling, and the peak bytes of freeing
  /// the run-ahead was holding.
  int32_t numMemoryStalls{0};
  int64_t maxDelayedFreeSeen{0};
  /// Ops deferred to a step's second pass, the steps that needed one, and the
  /// host time spent freeing frame values.
  int32_t numDeferredOps{0};
  int32_t numDeferredSteps{0};
  int32_t numGridRedos{0};
  int64_t freeUs{0};
  /// Donation pool: allocations served from a released buffer of the same size,
  /// those that fell through to the allocator, buffers dropped at the ceiling,
  /// and the host time the pool itself cost.
  int64_t donationHits{0};
  int64_t donationMisses{0};
  int64_t donationEvictions{0};
  int64_t donationUs{0};

  /// Performance report, filled when trace kTiming is on.
  std::string perfReport;
};

/// Returns the thread-local WaveThreadInfo for the current thread.
const WaveThreadInfo& waveThreadInfo();

/// Lifecycle of a step's kernel outputs, used to bundle intermediate freeing
/// with wave-stream syncs (see WaveConfig::freeIntermediates).
enum class ExecutionStage {
  /// Reset state at the start of executeWave; nothing allocated yet.
  kNotStarted,
  /// The step's kernel op outputs have been allocated (kernels launched).
  kAllocated,
  /// A wave-stream wait has completed since kAllocated, so this step's kernels
  /// are known to be done; its freeable buffers may be released.
  kSynced,
};

/// Preallocated vectors reused across executions for a given
/// (compositeInvocation, stepIdx) pair.  Avoids per-step heap allocation
/// in CompositeInvocation::execute and makeGrid.
struct StepVectors {
  /// Byte offset of the BlockInfo array within the step's pinned and device
  /// buffers. The kernel params come first, so their offsets depend only on
  /// which ops are in the step, not on the block count -- which is what lets
  /// them be filled before makeGrid has settled that count. The BlockInfo
  /// array sits above them and is written last.
  int64_t blockInfoOffset{0};

  /// Byte offset of every kernel param slot this step can use, flattened over
  /// the ops; opSlotBegin[i] is where op i's slots start and opSlotBegin.back()
  /// is the end. Each slot is reserved for the largest of the op's grid
  /// variants, and an op gets as many slots as its widest variant needs, so
  /// switching a variant never moves another op's params. That is what makes it
  /// safe to fill params before makeGrid has run and before a variant that
  /// depends on a pending transfer has been chosen. Built once per step by
  /// layoutParamSlots; paramRegionBytes is the total.
  std::vector<int64_t> slotOffsets;
  std::vector<int32_t> opSlotBegin;
  int64_t paramRegionBytes{0};

  /// Upper bound on the blocks makeGrid can produce for this step. The
  /// BlockInfo and DebugInfo regions are sized for it rather than for the
  /// actual count, so the buffer size -- and therefore its base address -- is
  /// settled before makeGrid runs. Without that, a later size change would
  /// reallocate and move the base out from under params already filled.
  int32_t blockCapacity{0};

  /// Every frame value this step can read, over ALL of its grid variants:
  /// each kernel launch's KernelOperation::orderingInputs(), each standalone
  /// launch's node inputs, and the ProjectOperation subgraph leaves, translated
  /// formal -> actual. Built once by layoutParamSlots, from the compiled grids
  /// alone, so it is available before gatherLaunches has picked a variant --
  /// which is what lets a step decide whether it must wait for a pending
  /// device-to-host transfer before it sizes and allocates anything. Being a
  /// union over variants it is a superset of what the step actually reads,
  /// which is the safe direction: a read missed here is a step running against
  /// a frame value that has not landed yet.
  folly::F14FastSet<nativert::ValueId> readIds;

  /// The same read set split per op, parallel to the invocation's ops. Lets a
  /// step defer only the ops that actually read a pending transfer instead of
  /// stalling whole. readIds is the union of these.
  std::vector<folly::F14FastSet<nativert::ValueId>> opReadIds;

  /// One bit per (id % 64) of opReadIds[i], so testing an op against the
  /// pending transfers is a single AND against the same signature built from
  /// the pending ids -- no hash lookup in the per-op path. Zero overlap proves
  /// independence; a hit falls through to the exact set. Actual ids throughout;
  /// the formal-to-actual translation happens once, when this is built.
  std::vector<uint64_t> opReadSignature;

  /// How many kernel / standalone / shortcut launches each op contributed at
  /// this step, recorded by the first pass. The second pass adds these to its
  /// running indices for the ops it is skipping, so it lands on a deferred op's
  /// slots without re-walking the launches of everything before it.
  std::vector<int32_t> opKernelCount;
  std::vector<int32_t> opStandaloneCount;
  std::vector<int32_t> opShortcutCount;

  /// Used by CompositeInvocation::execute / gatherLaunches.
  std::vector<LaunchData> kernels;
  std::vector<LaunchData> standalones;
  /// Metadata-only standalones with a StandaloneShortcut, split out from
  /// 'standalones' so they run in a tight switch loop (no per-op timing, no
  /// stream sync) and are timed as one batch.
  std::vector<LaunchData> shortcutStandalones;
  std::vector<int64_t> paramOffsets;

  /// Used by makeGrid (output).
  std::vector<BlockInfo> blocks;
  std::vector<int32_t> launchIndices;

  /// The kernel launches 'blocks' is run as, in order. Always at least one
  /// entry; more only when WaveConfig::partitionLaunches split the step. Each
  /// launch takes its own slice of 'blocks' and its own dynamic shared memory,
  /// which is where the occupancy a split buys actually lands.
  std::vector<LaunchSegment> segments;

  /// How balanced this step's grid came out, from the last makeGrid that ran
  /// for it. Reported per step under WaveConfig::kGrid and summarized in the
  /// performance report; also what the partitioning gate reads.
  GridStats gridStats;

  /// Thread-block clocks this step spent per unit of cost, measured on its
  /// previous execution. Turns WaveConfig::minBlockUs into the cost units the
  /// block quantum is expressed in. 0 until the first execution has been
  /// measured, which falls the quantum back to an even division of the step.
  double clocksPerCost{0};

  /// How well the blocks of this step's previous execution filled the time its
  /// slowest one took: mean block clocks over max, the same figure the per-step
  /// balance line reports. 0 until measured.
  ///
  /// This is the only honest signal that a step needs MORE blocks. The
  /// cost-model skew in gridStats cannot tell -- it is derived from the same
  /// costs the sizing uses, so a step whose costs are wrong looks balanced to
  /// it. A step already near 1.0 here has nothing to gain from a wider grid and
  /// everything to lose, since the extra blocks arrive as serialized launches.
  double measuredUtil{0};

  /// Blocks the previous execution actually ran, so a step that measured well
  /// can be held to what already worked.
  int32_t measuredBlocks{0};

  /// Used by makeGrid (internal temporaries).
  std::vector<float> costs;
  std::vector<int32_t> maxBlocks;
  std::vector<int32_t> numBlocksPerLaunch;

  /// Output of selectGrid, recycled across executions.
  std::vector<GridChoice> gridChoices;

  /// Cached size bounds for makeGrid reuse. If every launch's numElements falls
  /// within [sizesLower[i], sizesUpper[i]], the previous makeGrid result
  /// (blocks, launchIndices, numBlocksPerLaunch) can be reused.
  std::vector<int64_t> sizesLower;
  std::vector<int64_t> sizesUpper;
  int32_t cachedBlockSize{0};

  /// Bitmap of grid choices from selectGrid (1 = singleBlock, 0 = default).
  /// Used to detect when grid choice changed, invalidating all step caches.
  uint64_t gridChoiceBitmap{0};

  /// True if sizes matched so that makeGrid results can be reused.
  bool hasGridCache{false};

  /// True if grid choice matched so that Values and their frame offset in
  /// LaunchData can be reused. If false, these are reconstructed when
  /// constructing the frame.
  bool hasLaunchCache{false};

  /// Set by gatherLaunches when an existing LaunchData's grid is switched
  /// (e.g. from default to single-block) due to isGridChoice.
  bool gridChanged{false};

  /// Set by gatherLaunches when any kernel op in this step has op barriers
  /// (multi-block synchronization). Causes cooperative launch.
  bool isCgGrid{false};

  /// Set by gatherLaunches when this step has at least one standalone that does
  /// device-side work (a non-shortcut op). Shortcut standalones only manipulate
  /// host-side tensor metadata and never read wave-stream buffers, so when this
  /// is false the wave stream need not be synced before running them.
  bool hasGpuStandalones{false};

  // Timing fields, populated when kTiming trace bit or printTiming is on.
  int64_t gatherUs{0};
  int64_t gridUs{0};
  int64_t allocUs{0};
  // The part of allocUs spent inside the allocator itself. allocUs minus this
  // is shape arithmetic and building the views over what was allocated.
  int64_t allocCallUs{0};
  int64_t fillUs{0};
  int64_t kernelUs{0};
  int64_t standaloneUs{0};
  // Wall time of the metadata-only shortcut-standalone batch (one
  // hardware_timestamp pair around the tight loop).
  int64_t shortcutUs{0};
  bool standaloneBound{false};
  bool noDtoH{false};
  int64_t inputBytes{0};
  int64_t outputBytes{0};
  // Torch CUDA caching allocator's currently-allocated bytes, sampled right
  // after this step's kernel and standalones ran (kTiming trace only).
  int64_t currentBytes{0};
  // Time spent on reference-frame device-to-host copy and comparison for this
  // step (debug-only; excluded from the reported e2e time).
  int64_t refCheckUs{0};
  // Bytes of copying the clone-elision pass saved, charged to this step: for
  // each of the node's elided clone inputs that first has a tensor here,
  // numel * element size * the number of clones elided for it. Filled only
  // when the kTiming trace bit is on.
  int64_t elidedCloneBytes{0};

  // Allocation groups carved at this step and the number of outputs they
  // covered, i.e. the allocator calls the grouping replaced with one call each.
  // Always maintained, not just under kTiming: they are counts already to hand
  // where the groups are materialized, not a measurement.
  int32_t allocGroups{0};
  int32_t allocGroupTensors{0};

  // Device-measured spans from this step's events, and the device idle that
  // preceded it (kTiming only). See LaunchMeta for what each one means.
  int64_t kernelGpuUs{0};
  int64_t standaloneGpuUs{0};
  int64_t gpuIdleUs{0};

  // How many of this step's ops read a value an earlier step sent back over a
  // device-to-host transfer, split by op kind. These are the ops that cannot
  // start before that transfer has landed; every other op in the step could in
  // principle be prepared while it is still in flight. A shortcut counts as
  // dependent if any of its inputs is such a value, and it then taints its own
  // outputs, so a fused op reaching a returned scalar only through a chain of
  // metadata-only ops is counted too. kTiming only.
  int32_t d2hDepFused{0};
  int32_t d2hDepStandalone{0};
  int32_t d2hDepShortcut{0};
  // Of the dependent ops, how many depend on the immediately preceding step --
  // the case where nothing can be overlapped.
  int32_t d2hDepOnPrevStep{0};
  // Distance in executed steps back to the nearest producing step, -1 if none.
  int32_t d2hNearestProducer{-1};
  // Fused output descriptors built by a host-side view. Their shape/offset
  // operands come from the view node, not from the op's inputs, so they are a
  // path by which a fused op can depend on a returned scalar without naming it.
  int32_t viewNodeDescs{0};

  // Lifecycle stage for bundling frees with wave-stream syncs. Reset to
  // kNotStarted at the start of executeWave, set to kAllocated once this step's
  // kernel outputs are allocated, and advanced to kSynced by the first
  // wave-stream wait thereafter (see advanceSyncedStages).
  ExecutionStage executionStage{ExecutionStage::kNotStarted};

  // ExecutionState::executedSteps at the point this step ran, i.e. the index
  // LaunchData::d2hProducer and PendingReturn::executedStep are expressed in.
  // -1 before the step has run in this execution.
  int32_t executedStep{-1};

  // Frame value ids to release when this step reaches kSynced (and
  // WaveConfig::freeIntermediates is on). Set on a node's last executed step to
  // that node's ProjectNode::lastUse values, so the node's last-use tensors are
  // freed by the sync that advances this step to kSynced -- no extra sync.
  std::vector<nativert::ValueId> lastUseIds;

  // Bytes behind lastUseIds, summed as each id is stamped and subtracted from
  // ExecutionState::delayedFreeBytes in one go when this step is swept.
  int64_t lastUseBytes{0};
};

/// A step's device-to-host transfer that has been enqueued but whose pinned
/// buffer has not been parsed into the frame yet (WaveConfig::deferD2h). All
/// transfers go on the wave stream, so issue order is also completion order and
/// resolving one entry implies every earlier one has landed.
struct PendingReturn {
  int32_t sequenceNumber{0};
  int32_t stepIdx{0};

  /// ExecutionState::executedSteps at the producing step -- the same index
  /// LaunchData::d2hProducer carries on a launch that reads one of the values
  /// this transfer brings back.
  int32_t executedStep{0};

  /// Completes once the transfer has landed; recorded after the
  /// deviceToHostAsync. Owned by ExecutionState::stepEvents.
  facebook::velox::wave::Event* waveDone{nullptr};
};

/// Holds runtime state for executing a WaveGraph.  Pooled by WaveGraph
/// so that reusable buffers survive across calls.
struct ExecutionState {
  FrameP frame{nullptr};
  const ValueTypes* valueTypes{nullptr};
  WaveGraph* waveGraph{nullptr};
  facebook::velox::wave::GpuArena* deviceArena{nullptr};
  facebook::velox::wave::GpuArena* pinnedArena{nullptr};
  StreamPool* streamPool{nullptr};

  /// Reusable CUDA stream, obtained from streamPool at the start of execution
  /// and returned on scope exit.
  std::unique_ptr<facebook::velox::wave::Stream> stream;

  const folly::F14FastMap<NodeCP, nativert::OpKernel*>* kernelMap{nullptr};
  const folly::F14FastMap<NodeCP, int32_t>* standaloneIndices{nullptr};
  std::vector<StandaloneStats>* standaloneStats{nullptr};

  // Standalone nodes skipped during step execution due to None inputs.
  std::vector<NodeCP>* deferredStandalones{nullptr};

  // Fusion-coverage counters for the --trace summary: how many ops actually ran
  // as eager single-op GPU standalones vs. metadata-only host shortcuts (each
  // counted once, at its execution site, after the nodeOutputsComputed/None
  // skip guards).  Reset at the start of each executeWave.
  int64_t numStandalonesRun{0};
  int64_t numShortcutsRun{0};

  /// Per-launch debug info collected during execution.
  std::vector<LaunchDebugInfo> launchDebugInfos;

  /// Counters for reference frame verification (owned by executor).
  int64_t* numRefTensorsChecked{nullptr};
  int64_t* numRefNodesChecked{nullptr};

  /// Reusable device buffers indexed by [sequenceNumber][stepIdx].
  std::vector<std::vector<facebook::velox::wave::WaveBufferPtr>> deviceBuffers;
  /// Reusable pinned buffers indexed by
  /// [sequenceNumber][stepIdx]. These are on;ly to be used for their
  /// particular sequence and step and must not be
  /// overwritten. Constant parts of the frame are kept in these
  /// buffers between runs of the pipeline.
  std::vector<std::vector<facebook::velox::wave::WaveBufferPtr>> pinnedBuffers;

  /// Preallocated per-step vectors indexed by [sequenceNumber][stepIdx].
  std::vector<std::vector<StepVectors>> stepVectors;

  /// Value tracing state for the current execution.
  TraceState traceState;

  /// Value ids that passed reference verification. Used by reverify to detect
  /// corruption of previously correct values.
  std::vector<nativert::ValueId> verifiedIds;

  /// Generation counter from the last execution. Incremented by returnFrame
  /// to signal that launch caches are stale.
  uint64_t lastFrameGeneration{0};

  /// Per-step GPU timeline events in global execution order, parallel to
  /// launchDebugInfos. Returned to the event pools at the end of the run. A
  /// deque because a step holds a reference to its entry while later steps are
  /// appended, and lastWaveDone / lastStandaloneDone point into it.
  std::deque<StepEvents> stepEvents;

  /// Reference event for the run's device timeline, recorded on the wave stream
  /// before the first step. Every other timing event is read as an offset from
  /// this one, which is what puts the wave stream and the torch stream on a
  /// single comparable timeline. kTiming only.
  EventP timelineBase;

  /// Most recently recorded completion events, used to build the cross-stream
  /// ordering edges. Owned by stepEvents; a step can be missing one side, so
  /// these are carried forward rather than indexed by step.
  facebook::velox::wave::Event* lastWaveDone{nullptr};
  facebook::velox::wave::Event* lastStandaloneDone{nullptr};

  /// Index into stepEvents of the oldest step not yet advanced to kSynced.
  /// advanceCompletedStages resumes from here, so the sweep costs one query per
  /// step over the run rather than re-walking every step on every launch.
  size_t syncedCursor{0};

  /// Value ids an earlier step sent back over a device-to-host transfer, mapped
  /// to the executed-step index that produced them. Diagnostic only: it is what
  /// lets a step report which of its ops must wait for a transfer to land.
  folly::F14FastMap<nativert::ValueId, int32_t> returnedAtStep;

  /// Count of executed steps so far, spanning nodes. Indexes returnedAtStep.
  int32_t executedSteps{0};

  /// Transfers whose pinned buffers have not been parsed into the frame yet,
  /// oldest first. Always empty unless WaveConfig::deferD2h is on.
  std::deque<PendingReturn> pendingReturns;

  /// How many transfers ran deferred, and the total number of executed steps
  /// they stayed in flight for (resolution step minus producing step). A span
  /// of one is a transfer resolved by the very next step, i.e. no better than
  /// waiting at the producer; what the deferral buys is the excess over that.
  int32_t numDeferredReturns{0};
  int64_t deferredStepSpan{0};

  /// Sampled once per step, just before the host starts interpreting it: how
  /// many already-issued steps the device has not finished yet. Zero means the
  /// queue had drained and this step's whole interpretation is exposed as idle;
  /// that is the quantity deeper run-ahead has to raise. kTiming only, because
  /// each sample costs a cudaEventQuery per step still in flight.
  int64_t runAheadSum{0};
  int32_t runAheadSamples{0};
  int32_t runAheadMax{0};
  int32_t numDrainedStarts{0};

  /// Bytes stamped for release on steps the device has not finished, i.e. the
  /// memory run-ahead is holding onto. Checked against
  /// WaveConfig::maxDelayedFree before a step allocates.
  int64_t delayedFreeBytes{0};

  /// How many times that ceiling forced a drain, and the high-water mark of
  /// delayedFreeBytes -- which is how much peak the run-ahead is costing.
  int32_t numMemoryStalls{0};
  int64_t maxDelayedFreeSeen{0};

  /// Ops whose setup a step's first pass had to leave for the second, and how
  /// many steps needed a second pass at all.
  int32_t numDeferredOps{0};
  int32_t numDeferredSteps{0};

  /// Steps whose grid variant changed mid-run, forcing the setup pass to be
  /// redone. Under a forced cooperative grid the choice is fixed, so this must
  /// stay at zero; anything else means a step is paying for two passes.
  int32_t numGridRedos{0};

  /// Host time spent releasing frame values. Reported next to the allocation
  /// time as the part of the interpretation that compiling the host path
  /// cannot remove.
  int64_t freeUs{0};

  /// Buffers whose last use has passed but which are held here instead of being
  /// handed back to the caching allocator, keyed by storage bytes. A later wave
  /// kernel takes one instead of allocating: allocation costs the same per call
  /// whatever the size, so skipping the call is the whole win. Nothing needs to
  /// be synced -- the wave stream is in order, so the donor's readers have
  /// finished before the taker's kernel starts, and the storage never stops
  /// being owned. Only wave-path kernel outputs take part; a standalone may run
  /// on the torch stream, where that ordering does not hold.
  ///
  /// Cleared after the closing stream syncs (see clearDonationPool), so an
  /// exception on the way out cannot strand the buffers.
  folly::F14FastMap<int64_t, std::vector<at::Tensor>> donatable;

  /// Bytes currently parked in 'donatable'.
  int64_t donatableBytes{0};

  /// Allocations served from the pool, and those that had to fall through to
  /// the allocator.
  int64_t donationHits{0};
  int64_t donationMisses{0};

  /// Buffers dropped to stay under the delayed-free ceiling.
  int64_t donationEvictions{0};

  /// Host time spent in the pool itself (take + donate + evict), so its
  /// overhead can be read against the allocation time it removes.
  int64_t donationUs{0};

  /// Values released during this run. The fusion-coverage report needs them
  /// because it infers "this node never ran" from an empty frame slot, and a
  /// released intermediate leaves exactly that -- so without this every value
  /// freeIntermediates reclaims would be counted as an uncovered op. Only
  /// populated when tracing, which is the only time the report is produced.
  folly::F14FastSet<nativert::ValueId> freedValueIds;

  /// Last-use values stamped onto a step before their node's last one, and
  /// onto the last one. Says how much WaveConfig::stepLastUse actually moved.
  int32_t numLastUseEarly{0};
  int32_t numLastUseAtNodeEnd{0};

  /// Under trace kFrame only: every value already stamped for release, mapped
  /// to the step that stamped it. A later step reading one of them means the
  /// reader analysis missed a reader and the buffer was freed too early, which
  /// otherwise only shows up as a wrong result.
  folly::F14FastMap<nativert::ValueId, int32_t> releasedAtStep;
};

/// Offers a released buffer to the donation pool. Takes ownership only when the
/// tensor is a solely-owned CUDA storage -- a view or a second frame slot over
/// the same memory means it is not actually free. Returns true when the pool
/// took it, in which case the caller must not also release it.
bool donateFreedTensor(ExecutionState& state, const at::Tensor& tensor);

/// Returns a tensor of 'dims' and 'dtype' backed by a pooled buffer of exactly
/// 'bytes', or an undefined tensor when the pool has none. Exact sizes only:
/// best-fit hands multi-megabyte buffers to small requests and wastes far more
/// than it saves.
at::Tensor takeDonatedTensor(
    ExecutionState& state,
    int64_t bytes,
    c10::ScalarType dtype,
    c10::IntArrayRef dims);

/// Drops pooled buffers, largest first, while the pool holds more than
/// 'limitBytes'. Called when an allocation misses and the parked bytes would
/// otherwise push past the delayed-free ceiling.
void evictDonatable(ExecutionState& state, int64_t limitBytes);

/// Releases every pooled buffer. Call after the closing stream syncs.
void clearDonationPool(ExecutionState& state);

/// Advances every step vector currently in kAllocated to kSynced (call right
/// after a wave-stream wait, when those steps' kernels are known complete).
/// When WaveConfig::freeIntermediates is on, releases each newly-synced step's
/// lastUseIds from the frame, bundling that freeing into the sync that just
/// happened rather than adding a dedicated one.
void advanceSyncedStages(ExecutionState& state);

/// Waits on the wave stream and then advances synced stages (see
/// advanceSyncedStages). Used everywhere the wave stream is drained so freeing
/// is bundled with the wait.
void syncWaveStream(ExecutionState& state);

/// Releases 'ids' from the frame right away. For last-use values that have to
/// be stamped onto a step a blocking sync already swept to kSynced: the sweeps
/// only ever visit a kAllocated step, so nothing would free them otherwise.
/// The caller must have waited for that step, which the sweep implies.
void freeLastUseNow(
    ExecutionState& state,
    const std::vector<nativert::ValueId>& ids);

/// Advances to kSynced every kAllocated step whose waveDone event has already
/// completed, and frees its lastUseIds. The non-blocking counterpart of
/// advanceSyncedStages: a device-side event wait orders the GPU but tells the
/// host nothing, so a step must not be declared synced on one. Event::query()
/// is a cheap non-blocking host check that does establish it. Freeing happens
/// slightly later than under the old host-sync scheme, which raises the
/// transient memory peak a little.
void advanceCompletedStages(ExecutionState& state);

/// Copies the return values a step's kernels sent back from its pinned buffer
/// into the execution frame: tensor shapes via resize_, scalars via setIValue.
/// The caller must have waited for that step's transfer.
void processReturnData(
    StepVectors& sv,
    nativert::ExecutionFrame& frame,
    uint8_t* pinnedBase);

/// Waits for and parses every pending return produced at executed step
/// 'throughStep' or earlier, oldest first (see ExecutionState::pendingReturns).
/// A negative 'throughStep' resolves nothing; one past the newest entry
/// resolves all of them. Cheap when the transfer has already landed, which is
/// the case at every call site that follows a stream wait.
void resolvePendingReturns(ExecutionState& state, int32_t throughStep);

/// Resolves everything still pending. For the points where the frame has to be
/// complete regardless of what is read next: the end of a run, and the debug
/// paths that walk arbitrary values.
void resolveAllPendingReturns(ExecutionState& state);

/// Samples how many already-issued steps the device has not finished, and folds
/// it into the run-ahead counters. Call once per step, immediately before its
/// interpretation begins, so a sample of zero means that interpretation runs
/// against an empty queue and is fully exposed as idle. Does nothing unless the
/// kTiming trace bit is on.
void sampleRunAhead(ExecutionState& state);

/// TSC ticks per microsecond, calibrated once. Use with
/// folly::hardware_timestamp() for anything timed per op: std::chrono here is
/// backed by kvm-clock at tens of microseconds a call, so a per-op chrono pair
/// costs far more than the work it measures.
double tscTicksPerMicro();

/// True when TW_ALLOC_TRACE is set in the environment. Enables an event line
/// per kernel-output allocation and per released frame value, so the fraction
/// of allocations a same-size freed buffer could have served can be measured
/// offline. Read once; the check is a load.
bool allocTraceEnabled();

/// Emits one allocation/free event: "ALLOCEV <kind> <id> <bytes>", where kind
/// is keep / resize / alloc for a kernel output and free for a released value.
/// Emission order is host program order, which is what a donation-pool
/// simulation needs.
void logAllocEvent(const char* kind, int32_t valueId, int64_t bytes);

/// Emits an allocation event carrying the STATIC size key the compiler already
/// knows for the output ("dyn" when the shape comes from a reserveShape lambda,
/// which is opaque before execution), and remembers it for 'valueId' so the
/// matching free event can report the same key. Two values with equal keys are
/// provably the same byte size without running anything, which is what a
/// compile-time donation edge needs.
void logKeyedAllocEvent(
    const char* kind,
    int32_t valueId,
    int64_t bytes,
    const std::string& sizeKey);

/// The static size key recorded for 'valueId', or "?" if none was seen.
const std::string& recordedSizeKey(int32_t valueId);

/// Stamps 'id' onto 'sv' for release when that step is swept, and accounts its
/// bytes against the delayed-free ceiling.
void addLastUseId(ExecutionState& state, StepVectors& sv, nativert::ValueId id);

/// Drains both streams when the bytes waiting to be freed on already-issued
/// steps exceed WaveConfig::maxDelayedFree, giving the memory back before the
/// next step allocates. Call before a step sizes or allocates anything. Returns
/// true if it drained. Both streams, not just the wave one: advanceSyncedStages
/// releases on a wave-stream wait alone, and under run-ahead an eager
/// standalone on the torch stream can still be reading one of those buffers.
bool enforceDelayedFreeLimit(ExecutionState& state);

/// Returns the process-wide Stream wrapping the CUDA stream that eager
/// standalones are dispatched to, for recording events on and waiting for it.
facebook::velox::wave::Stream& torchStream();

/// Makes 'stream' wait, device-side, for everything already enqueued on the
/// torch stream. A wave stream is non-blocking, so it no longer implicitly
/// waits for the legacy default stream: anything a wave stream touches that
/// eager work produced -- or that the caching allocator could hand it while an
/// eager op still uses it -- needs this edge first.
void waitForTorchStream(facebook::velox::wave::Stream& stream);

/// Acquires a step's events, appends them to state.stepEvents and returns the
/// entry. Under kTiming all four events come from the timing-enabled pool;
/// otherwise only the ordering (*Done) events are taken, from the untimed pool.
StepEvents& newStepEvents(ExecutionState& state, int32_t seq, int32_t stepIdx);

/// Returns every event held by 'state' to its pool. Must run only after the
/// streams that recorded them have been waited on, so no event is recycled
/// while still pending.
void releaseStepEvents(ExecutionState& state);

/// Executes a single node via its OpKernel with tracing and error logging.
/// When 'traceState' is non-null, traces the node's input values before the op
/// and its produced output values after, for any ids in --trace_values.
void executeNode(
    NodeCP node,
    nativert::OpKernel* kernel,
    nativert::ExecutionFrame& frame,
    TraceState* traceState = nullptr);

/// Runs standalone launches by mapping each formal node to the actual node
/// via OpInvocation::nodeMap(), executing it via the corresponding OpKernel.
/// When 'timing' is true, syncs the PyTorch default stream after each op and
/// records its elapsed time in standaloneStats; when false, no clock is read
/// and standaloneStats is left untouched.
void runStandalones(
    const std::vector<LaunchData>& standalones,
    ExecutionState& state,
    const folly::F14FastMap<NodeCP, nativert::OpKernel*>& kernelMap,
    const folly::F14FastMap<NodeCP, int32_t>& standaloneIndices,
    std::vector<StandaloneStats>& standaloneStats,
    bool timing);

/// Runs the metadata-only shortcut standalones in a tight loop (just the
/// per-op switch in runStandaloneShortcut), with no per-op timing or stream
/// sync. When 'timing' is true, records the whole batch's wall time (via
/// folly::hardware_timestamp) into 'outUs'.
void runShortcutStandalones(
    const std::vector<LaunchData>& shortcuts,
    ExecutionState& state,
    bool timing,
    int64_t& outUs);

/// Builds BlockInfo grid for a set of LaunchData entries. Uses preallocated
/// vectors in 'sv' (blocks, launchIndices, costs, maxBlocks,
/// numBlocksPerLaunch). Returns the block size (threads per block).
/// 'maxBlocksPerSM' is the kernel's occupancy at zero dynamic shared memory and
/// 'staticSharedPerBlock' its static shared memory; both are needed to bound a
/// cooperative grid when an op in the step asks for dynamic shared memory.
/// 'occupancyFor' answers the same question from the driver at a given dynamic
/// shared memory, and supersedes the other two when given: a cooperative launch
/// is packed to exactly that figure, so deriving it any other way risks a grid
/// the driver refuses. See GridDevice::occupancyFor.
int32_t makeGrid(
    std::vector<LaunchData>& launches,
    StepVectors& sv,
    int32_t maxBlocksPerSM = 0,
    int32_t staticSharedPerBlock = 0,
    const std::function<int32_t(int32_t dynamicShared)>& occupancyFor =
        nullptr);

/// Looks up 'value' in 'map' and returns the corresponding tensor from 'frame'.
at::Tensor paramTensor(
    ValueCP value,
    nativert::ExecutionFrame& frame,
    const FormalToActual& map);

/// Returns the shape of the largest tensor reachable from
/// node->inputs()[ordinal] by tracing through elementwise producers that have
/// no frame entry. Stops at values that exist in the frame.
std::vector<std::vector<Dim>> elementwiseInputShape(
    NodeCP node,
    nativert::ExecutionFrame& frame,
    const FormalToActual& map,
    int32_t ordinal);

/// Looks up 'value' in 'map' and returns the corresponding SymInt from 'frame'.
int64_t paramSymInt(
    ValueCP value,
    nativert::ExecutionFrame& frame,
    const FormalToActual& map);

/// Returns the int64 value for a named argument that may be either an input
/// value (translated via map) or an attribute (on the actual node found via
/// nodeMap).
int64_t paramIntByName(
    NodeCP node,
    std::string_view name,
    nativert::ExecutionFrame& frame,
    const FormalToActual& map,
    const NodeMap& nodeMap);

/// Returns the int64 list for a named argument that may be either an input
/// value / prim.ListPack (translated via map) or a vector<int64_t> attribute
/// (on the actual node found via nodeMap).
std::vector<int64_t> paramIntListByName(
    NodeCP node,
    std::string_view name,
    nativert::ExecutionFrame& frame,
    const FormalToActual& map,
    const NodeMap& nodeMap);

/// Formats a NodeMap as human-readable text with one formal -> actual pair
/// per entry, each node printed with NodePrinter stopping at inputs.
std::string printNodeMap(const NodeMap& nodeMap);

/// Executes a WaveGraph as a GraphExecutorBase subclass, allowing it to
/// be used wherever the standard nativert executors are used.
class WaveGraphExecutor : public nativert::GraphExecutorBase {
 public:
  /// Takes exclusive ownership of the nativert::Graph in modelContext and
  /// mutates it internally during WaveGraph compilation. The graph must not
  /// be used externally after construction.
  explicit WaveGraphExecutor(std::unique_ptr<ModelContext> modelContext);

  /// Creates an ExecutionFrame on CPU with all constants and weights
  /// pre-filled.
  std::unique_ptr<nativert::ExecutionFrame> makeFrame();

  /// Creates an ExecutionFrame whose persistent tensors have been copied to
  /// device.
  std::unique_ptr<nativert::ExecutionFrame> makeDeviceFrame();

  /// Executes with a pooled device frame. The frame is obtained from the pool,
  /// inputs are filled, the wave graph runs, outputs are extracted and
  /// decoupled from the frame, and the frame is returned to the pool.
  std::vector<c10::IValue> execute(
      nativert::ExecutionFrame& frame,
      std::vector<c10::IValue> inputs) override;

  std::vector<c10::IValue> executeWithPrefilledFrame(
      nativert::ExecutionFrame& frame) override;

  /// Runs the graph on positional 'inputs' (in graph user-input order) using a
  /// pooled device frame and returns the user outputs. Convenience wrapper over
  /// getFrame()/fillUserInputs()/executeWithPrefilledFrame()/returnFrame() for
  /// callers that have inputs but no frame (e.g. TorchWaveModel::run).
  std::vector<c10::IValue> runInputs(std::vector<c10::IValue> inputs);

  /// Returns a frame from the pool, creating one if needed.
  std::unique_ptr<nativert::ExecutionFrame> getFrame();

  /// Returns 'frame' to the pool after clearing non-persistent values.
  void returnFrame(std::unique_ptr<nativert::ExecutionFrame> frame);

  /// Returns a human-readable error string from the most recent execution.
  /// Checks thread-local debug info for non-zero error lines and maps each
  /// error back to the originating kernel op via the WaveGraph structure.
  std::string errorString() const;

  /// Produces a performance report with per-node timing, throughput,
  /// thread block balance, and top consumers. Called inside executeWave
  /// while execution state is live.
  std::string makePerfReport(ExecutionState& state, int64_t wallUs) const;

  /// Returns standalone execution stats: pairs of (node string, micros)
  /// from the most recent execution. The node string is formatted as in
  /// Launch::toString for standalone nodes.
  std::vector<std::pair<std::string, int64_t>> getStandaloneStats() const;

  WaveGraph* waveGraph() const {
    return waveGraph_.get();
  }

  const nativert::Graph& graph() const {
    return *modelContext_->graph;
  }

  int64_t numRefTensorsChecked() const {
    return numRefTensorsChecked_;
  }

  int64_t numRefNodesChecked() const {
    return numRefNodesChecked_;
  }

  void addRefTensorsChecked(int64_t count) {
    numRefTensorsChecked_ += count;
  }

  void addRefNodesChecked(int64_t count) {
    numRefNodesChecked_ += count;
  }

 private:
  /// Runs the WaveGraph on the given frame.
  void executeWave(nativert::ExecutionFrame& frame, WaveGraph& waveGraph);

  /// Transfers device-side debug info to host and stores in thread-local
  /// WaveThreadInfo.
  void collectDebugInfo(ExecutionState& state);

  /// Adjusts per-launch costAdjustFactor based on actual vs expected thread
  /// block clock distribution. Invalidates the grid cache when the adjustment
  /// exceeds 1.1x.
  void adjustCosts(ExecutionState& state);

  std::unique_ptr<ModelContext> modelContext_;

  std::unique_ptr<WaveGraph> waveGraph_;

  /// Maps each nativert Node to its OpKernel, built once at construction.
  folly::F14FastMap<NodeCP, nativert::OpKernel*> kernelMap_;

  /// Pool of device-side ExecutionFrames with persistent tensors on GPU.
  std::unique_ptr<Pool<nativert::ExecutionFrame>> framePool_;

  uint64_t frameGeneration_{0};

  int64_t numRefTensorsChecked_{0};
  int64_t numRefNodesChecked_{0};
};

} // namespace torch::wave
