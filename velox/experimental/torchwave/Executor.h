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
#include "velox/experimental/torchwave/Compile.h"
#include "velox/experimental/torchwave/CompiledOp.h"
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

/// Per-launch metadata stored in thread-local alongside the DebugInfo.
struct LaunchMeta {
  int32_t sequenceNumber{0};
  int32_t stepIdx{0};
  int32_t numBlocks{0};
  int64_t gatherUs{0};
  int64_t gridUs{0};
  int64_t allocUs{0};
  int64_t fillUs{0};
  int64_t kernelUs{0};
  int64_t standaloneUs{0};
  bool standaloneBound{false};
  bool noDtoH{false};
  int64_t inputBytes{0};
  int64_t outputBytes{0};
};

/// Per-thread debug info from the most recent wave execution. Populated by
/// executeWave when WaveConfig::keepStatsOnThread is true.
struct WaveThreadInfo {
  std::vector<std::vector<DebugInfo>> debugInfo;
  std::vector<LaunchMeta> launchMeta;
  std::string errors;
  /// Standalone execution times, sorted descending. Paired with labels.
  std::vector<int64_t> standaloneTimes;
  std::vector<std::string> standaloneLabels;
  /// Performance report, filled when trace kTiming is on.
  std::string perfReport;
};

/// Returns the thread-local WaveThreadInfo for the current thread.
const WaveThreadInfo& waveThreadInfo();

/// Preallocated vectors reused across executions for a given
/// (compositeInvocation, stepIdx) pair.  Avoids per-step heap allocation
/// in CompositeInvocation::execute and makeGrid.
struct StepVectors {
  /// Used by CompositeInvocation::execute / gatherLaunches.
  std::vector<LaunchData> kernels;
  std::vector<LaunchData> standalones;
  std::vector<int64_t> paramOffsets;

  /// Used by makeGrid (output).
  std::vector<BlockInfo> blocks;
  std::vector<int32_t> launchIndices;

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

  // Timing fields, populated when kTiming trace bit or printTiming is on.
  int64_t gatherUs{0};
  int64_t gridUs{0};
  int64_t allocUs{0};
  int64_t fillUs{0};
  int64_t kernelUs{0};
  int64_t standaloneUs{0};
  bool standaloneBound{false};
  bool noDtoH{false};
  int64_t inputBytes{0};
  int64_t outputBytes{0};
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
};

/// Executes a single node via its OpKernel with tracing and error logging.
void executeNode(
    NodeCP node,
    nativert::OpKernel* kernel,
    nativert::ExecutionFrame& frame);

/// Runs standalone launches by mapping each formal node to the actual node
/// via OpInvocation::nodeMap(), executing it via the corresponding OpKernel,
/// and recording timing in standaloneStats.
void runStandalones(
    const std::vector<LaunchData>& standalones,
    ExecutionState& state,
    const folly::F14FastMap<NodeCP, nativert::OpKernel*>& kernelMap,
    const folly::F14FastMap<NodeCP, int32_t>& standaloneIndices,
    std::vector<StandaloneStats>& standaloneStats);

/// Builds BlockInfo grid for a set of LaunchData entries. Uses preallocated
/// vectors in 'sv' (blocks, launchIndices, costs, maxBlocks,
/// numBlocksPerLaunch). Returns the block size (threads per block).
int32_t makeGrid(
    std::vector<LaunchData>& launches,
    StepVectors& sv,
    int32_t maxBlocksPerSM = 0);

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
