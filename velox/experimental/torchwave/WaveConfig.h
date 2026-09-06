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
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <c10/core/ScalarType.h>
#include <folly/CPortability.h>

namespace c10 {
struct IValue;
}

namespace torch::wave {

struct WaveConfig;

/// Returns a mutable reference to the thread-local WaveConfig override pointer.
/// While it is non-null, WaveConfig::get() returns the pointee instead of the
/// global singleton, so wave graphs compiled and executed with different
/// configs can run concurrently on different threads. Null on threads with no
/// active override.
WaveConfig*& waveConfigOverride();

/// Process-wide configuration for wave graph execution (block size, tracing,
/// grid hints).
struct WaveConfig {
  static constexpr int32_t kNodes = 1;
  static constexpr int32_t kLaunches = 2;
  static constexpr int32_t kTensors = 4;
  static constexpr int32_t kFrame = 8;
  static constexpr int32_t kTiming = 16;

  int32_t blockSize{256};
  bool allStandalone{false};

  /// If non-zero, use this as the number of SMs instead of reading from the
  /// device.
  int32_t numSms{0};

  /// Trace bit mask. kNodes prints node headers, kLaunches prints per-launch
  /// details.
  int32_t trace{0};

  /// If set, forces the grid choice between single-block and multi-block
  /// variants. If nullopt, the choice is made based on input size.
  std::optional<bool> useSingleBlock;

  /// If set and true, use the cooperative grid variant when available.
  std::optional<bool> isCg;

  /// If true, ops with both a barrier-based and a single-pass cooperative-grid
  /// form (masked_select_jagged) use the single-pass one. Only has an effect in
  /// cooperative-grid mode.
  bool singlePassSelect{false};

  /// If true, cumsum, exclusive sum and masked_select are registered with a
  /// single decoupled look-back implementation instead of the single-block,
  /// multi-kernel and cooperative-grid variants. Read once, by
  /// registerBuiltins(), so it must be set before initialize().
  bool singlePass{false};

  /// Reference values keyed by ValueId for verifying intermediates.
  std::unordered_map<int32_t, c10::IValue>* referenceFrame{nullptr};

  /// If non-empty, save the wave execution frame to this path.
  std::string saveReferenceFramePath;

  // If non-empty, cache compiled CUDA kernels (cubin) in this directory.
  std::string kernelCacheDir;

  // Max pointer variables in elementwise codegen before inlining storage
  // expressions.
  int32_t maxElementwiseVars{7};

  // Character threshold for extracting elementwise subtrees into
  // __device__ __noinline__ helpers. 0 disables extraction.
  int32_t outOfLineExprSize{10'000};

  // Print timing for wave graph execution.
  bool printTiming{false};

  // Attribute GPU time to individual eager standalone ops by syncing the torch
  // stream after each one. That sync is what makes per-op numbers possible and
  // is also the largest perturbation in the measurement: it serializes the
  // standalones against each other and against the wave stream, so the per-step
  // device times and the GPU-idle figure stop reflecting what an untraced run
  // does. Off by default under kTiming, where the per-step standalone event
  // pair gives an unperturbed device measurement instead.
  bool perOpStandaloneTiming{false};

  // Comma-separated list of value ids to trace during execution.
  std::string traceValues;

  // Max elements printed per tensor when tracing values. 0 means no limit.
  int32_t tensorPrintElementLimit{100};

  // Re-verify all previously passed reference values on each step to detect
  // corruption.
  bool reverify{false};

  // If true, copy per-block debug info from device to thread-local storage
  // before returning the execution state to the pool.
  bool keepStatsOnThread{true};

  // If true, throw after execution if any block reported an error.
  bool throwOnError{true};

  // If true, skip the elementwise fast path and always generate the slow
  // path with complexIdx.
  bool noElementwiseFastPath{false};

  // If true, log reference mismatches but continue execution instead of
  // throwing.
  bool continueAfterMismatch{false};

  // Enable device-side debug printfs. Emergency use only.
  bool kernelDebugOutput{false};

  // Compile kernels with -lineinfo so compute-sanitizer can attribute a fault
  // to a source line. Read once, from initialize(), because wave freezes its
  // NVRTC flags on the first compile -- setting it later has no effect.
  // Optimization stays on (unlike -G, which ptxas rejects at -O>0). It changes
  // the kernel cache key, so the first run after enabling it recompiles.
  bool kernelLineInfo{false};

  // Launch kernel once per block for debugging, waiting between launches.
  // Each kernel op runs as a standalone invocation so device-side errors
  // can be attributed to a single op.
  bool debugSingleOps{false};

  // If true, adjust per-op cost multipliers after each execution based on
  // actual thread block clock distribution.
  bool autoAdjustCost{true};

  // If true, reuse a value's buffer in place when an op is its unique last use
  // (turning copying ops into in-place ops), and drop clones that no consumer
  // needs. On by default.
  bool enableReuse{true};

  // If true, run the pre-partition read-only clone elision pass. Only consulted
  // when enableReuse is set; separated from it so the pass can be A/B'd against
  // the post-partition in-place rewrite alone.
  bool elideClones{true};

  // Force a launch boundary after a multi-block (non-cooperative) scan so every
  // cross-block consumer of its output reads a fully materialized buffer from a
  // later stream-ordered launch, and fence a multi-block cat's shifted copies
  // with a grid-wide opBarrier before an in-kernel consumer reads them.
  // Without this, a fused cat consumer that reads a scan output (or a
  // shift-by-offset cat element) cross-block within one kernel is ordered only
  // by intra-block __syncthreads(), which is insufficient across
  // non-co-resident blocks and produces stale reads.  On by default (a
  // correctness fix); the race harness flips it off for the racy A/B arm.
  bool scanOutputReturnBarrier{true};

  // If true, release the frame tensors of each ProjectNode's last-use values
  // right after that node's composite invocation executes, instead of keeping
  // them until the whole graph finishes. On by default. It cannot be turned on
  // before this commit: a concat alloc group's buffer is only ever released
  // when this is set, and until the decomposition here the group carved a cat
  // element that was a view of a wave-produced value, so the released buffer
  // was reused under a live reader (cumsumOffsetsReproTest).
  bool freeIntermediates{true};

  // If true, release each last-use value after the last STEP that reads it
  // rather than after the node's last step. A node's ops finish at different
  // steps, so a value read only by a short op stays live for the whole node
  // under the coarser scheme. Only consulted when freeIntermediates is set.
  bool stepLastUse{true};

  // If true, drain both streams at the end of every step, so a step's freeable
  // buffers are back in the caching allocator before the next step allocates.
  // Serializes the pipeline and costs wall time; it exists to make the peak
  // memory reflect the release schedule rather than how far the host ran ahead.
  bool syncEachStep{false};

  // If true, do not block the host on a step's device-to-host transfer at the
  // step that issues it. The transfer is recorded as pending and its pinned
  // buffer is parsed into the frame at the first later step that can read one
  // of the values it brings back, so a step that reads none of them does its
  // sizing, allocation, parameter fill and launch while the transfer is still
  // in flight. On by default.
  bool deferD2h{true};

  // If true, drop the host-side stream waits that are not a real data or memory
  // dependency, so the host can queue as many steps ahead of the device as it
  // can interpret. Today that is the default-stream drain at the end of every
  // composite invocation that ran an eager standalone: the cross-stream
  // ordering it used to provide is now carried device-side by the
  // lastStandaloneDone / lastWaveDone event edges, and a step's buffers are
  // only freed once its own completion events have been observed, so the drain
  // costs a host stall per node and buys nothing. Needs deferD2h to be of any
  // use -- otherwise every transfer still stops the host at its producing step.
  // Off by default.
  bool runAhead{false};

  // Ceiling, in bytes, on how much freeable memory may sit in already-issued
  // but not yet completed steps. Running ahead delays every free until the
  // device catches up, so the peak grows by roughly the bytes in flight; when
  // this is exceeded the host drains both streams before allocating any more,
  // trading the run-ahead back for the memory. 0 disables the check. Only
  // meaningful with freeIntermediates, which is what makes the frees delayable.
  int64_t maxDelayedFree{1LL << 30};

  // If true, hold a released wave-kernel buffer instead of returning it to the
  // caching allocator, and hand it to a later wave kernel that needs exactly
  // the same number of bytes. Allocation costs roughly the same per call at any
  // size, so skipping the call is the win, not the memory. Safe without a sync
  // because the wave stream is in order and the buffer is never unowned. Off by
  // default.
  bool donateBuffers{false};

  // Bytes of donatable buffers carried between executions. The pool is trimmed
  // to this at the end of each run. Distinct from maxDelayedFree, which bounds
  // freeing held up by run-ahead within a run: carrying buffers across runs
  // keeps them out of the caching allocator entirely, so hoarding the big ones
  // makes every allocation the pool does NOT serve more expensive.
  int64_t donationCarryBytes{64LL << 20};

  // If true, run the pre-partition metadata-duplication pass: rematerialize a
  // multiply-used metadata getter (sym_size / sym_numel) once per use site so
  // it stops being a shared value. A value with more than one user is a CSE
  // border, which the partitioner turns into a top-level output of its
  // ProjectNode -- a frame slot, an output and a parameter block per step. A
  // metadata getter reads only shape fields, so recomputing it is free, but it
  // only pays when its tensor operand is available at the use site anyway;
  // otherwise the border just moves from the getter's output to its input. Off
  // by default.
  bool duplicateMetadata{false};

  // EXPERIMENT, off by default and not correct yet. If true, a metadata getter
  // whose only reachable user is the output node stops being counted as a use
  // of its operand when the partitioner builds its levels, so the operand does
  // not become a CSE border on the getter's account. The getter is put back
  // before the last layer is built, so it still runs and still occupies its
  // output slot.
  //
  // What it is for: on the ROO preproc graph 255 of the 297 top-level exprs are
  // returned sym_size getters, and 248 of them have an operand whose only other
  // reachable user is one consumer. Each of those is a border that exists
  // purely because the size is returned, and each pushes its operand's producer
  // into an earlier layer. Removing them is what would give the last layer more
  // to carve.
  //
  // Why it is not correct yet: dropping the reference takes those 248 operands
  // to a single user, so their producers stop being borders and fuse into that
  // consumer -- and 238 of the 255 are produced by aten.slice (a view) or
  // aten.zeros, neither of which leaves a tensor in the frame once inlined. The
  // getter would then read the size of something that does not exist. The
  // shapes are all host-computable, so the fix is shape propagation rather than
  // a frame read, but that is not written. Use this only to measure what the
  // partitioning change is worth.
  bool deferSizeOutputs{false};

  // If true, the graph optimizer assumes every producer-less value (model
  // input, weight, or constant) is contiguous, so downstream passes may treat
  // them as densely laid out. When on, executeWave verifies each such tensor is
  // actually contiguous and throws otherwise. Off by default.
  bool inputContiguous{false};

  // Merge nodes that compute the same thing from the same operands, before
  // partitioning. Split in two because the two halves pay off differently: the
  // compute half removes real work, while the view half only removes graph
  // nodes -- and duplicating views per consumer is something duplicateMetadata
  // does on purpose, so merging them can work against the partitioner.
  bool cseCompute{false};
  bool cseViews{false};

  // Split a list-producing op into one node per tensor before partitioning, so
  // each column reaches the block allocator with its own cost instead of one
  // op's grid covering all of them. Also folds a chain of such ops where the
  // op supports it. On by default; the switch is for A/B against the list form.
  bool decomposeLists{true};

  // If true, a column folds its producer's gather into its own even when the
  // producer has several readers, provided every reader is a consumer that can
  // absorb a chain itself. The sole-reader rule two foldable consumers can
  // never satisfy: whichever rule runs first sees the other as an outside
  // reader and declines, and the second then sees a gather already reading the
  // buffer, so neither folds and the intermediate is always written. Folding
  // into both removes it, at the cost of running the producer's steps once per
  // consumer.
  //
  // On by default, which only measurement could decide, because a reader that
  // folds beside one that then declines for its own reasons leaves the buffer
  // AND duplicates the work. On the ROO preproc at 1k rows it takes every one
  // of the 23 columns where a select reads a flip: kernel time 39.6 -> 37.1 ms
  // and peak GPU RAM 27.97 -> 26.93 GB, with no column left materialized for a
  // consumer that declined.
  bool foldSharedChains{true};

  // If true, cooperative-grid mode expands tw.masked_select_jagged into its
  // multi-kernel stages instead of the single-node cg form. The stages reserve
  // the output list to the exact selected count, which the cg form cannot do:
  // with no host round trip it must over-allocate to the mask length and set
  // the real shape on device. The stages stay in separate launches inside the
  // cg grid because each names its predecessor through inputFromPreviousKernel,
  // which breaks that producer into its own kernel whatever the grid mode.
  bool mkSelect{false};

  // If true, allocate the outputs that share a lifetime out of one buffer
  // instead of one allocator call each. Allocation costs roughly the same per
  // call at any size, so the win is the call count: a step's outputs that all
  // die at the same later step become one allocation carved into views. Only
  // consulted in the cooperative-grid mode, where the grid -- and with it the
  // step boundaries an allocation's lifetime is expressed in -- is settled
  // before the first execution. Selects a separate execute path
  // (executeAllocGroups) rather than branching inside the per-op one, and turns
  // off buffer donation, whose size-matched reuse assumes per-output
  // allocations.
  bool enableAllocGroup{true};

  // If true, a fused cat / stack of more than two operands gets an allocation
  // group of its own: the whole result is allocated at the step that produces
  // its operands, and each operand's frame slot is the region of the result it
  // occupies, so the kernel that produces it writes in place and the concat
  // copies nothing. Separate from enableAllocGroup, which it rides on, so the
  // two can be measured apart.
  bool enableConcatAllocGroup{true};

  // If false, the lifetime grouping is skipped and only the concat groups are
  // formed. Both still ride on enableAllocGroup, which they need for the plan
  // to be installed at all; this splits the two apart so the concat grouping's
  // own cost and benefit can be read off without the lifetime grouping's much
  // larger numbers on top of it.
  bool enableLifetimeAllocGroup{true};

  // If true, a fused cat / stack of more than two operands stops emitting one
  // copy per operand into its own kernel and instead pushes every operand into
  // a kernel op of its own in the previous step. Each of those is sized by the
  // operand it writes and gets its own share of the grid, so the operands fill
  // the result side by side instead of walking a chain of __concatCopy calls in
  // one block. The concat then becomes a kernel break that copies nothing.
  bool parallelConcatFill{false};

  // If true, alongside each composite kernel also compile one single-op kernel
  // per op it contains, named <composite>_op_<opCode>. Diagnostic only: the
  // per-op kernels are never launched for results, they exist so the register /
  // shared / local memory and occupancy numbers logged after graph construction
  // are available at one-op resolution instead of only for the fused whole.
  // Their compiles are queued with the composite's, so the extra cost is
  // compile parallelism rather than serial latency. Off by default.
  bool configPerOp{false};

  /// Returns the active config: the thread-local override set by
  /// waveConfigOverride() when non-null, otherwise the process-wide singleton.
  /// The singleton is not thread-safe; all of its mutations must happen before
  /// concurrent reads.
  FOLLY_EXPORT static WaveConfig& get() {
    if (auto* configOverride = waveConfigOverride()) {
      return *configOverride;
    }
    static WaveConfig instance;
    return instance;
  }

  /// Returns a compact, comma-separated list of the settings whose value
  /// differs from its default (e.g. "trace=16, autoAdjustCost=false,
  /// freeIntermediates=true"), or "defaults" when every field is at its
  /// default. Used in the performance report so a run's active configuration is
  /// self-documenting.
  std::string toString() const;
};

} // namespace torch::wave
