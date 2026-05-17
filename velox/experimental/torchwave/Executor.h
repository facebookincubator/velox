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
};

/// Preallocated vectors reused across executions for a given
/// (compositeInvocation, stepIdx) pair.  Avoids per-step heap allocation
/// in CompositeInvocation::execute and makeGrid.
struct StepVectors {
  // Used by CompositeInvocation::execute / gatherLaunches.
  std::vector<LaunchData> kernels;
  std::vector<LaunchData> standalones;
  std::vector<int64_t> paramOffsets;

  // Precomputed largestInput per kernel launch for step 0, populated by
  // selectGrid so that gatherLaunches avoids recomputing it.
  std::vector<int64_t> inputSizes;

  // Used by makeGrid (output).
  std::vector<BlockInfo> blocks;
  std::vector<int32_t> launchIndices;

  // Used by makeGrid (internal temporaries).
  std::vector<float> costs;
  std::vector<int32_t> maxBlocks;
  std::vector<int32_t> numBlocksPerLaunch;

  // Output of selectGrid, recycled across executions.
  std::vector<GridChoice> gridChoices;

  // Cached size bounds for makeGrid reuse. If every launch's numElements falls
  // within [sizesLower[i], sizesUpper[i]], the previous makeGrid result
  // (blocks, launchIndices, numBlocksPerLaunch) can be reused.
  std::vector<int64_t> sizesLower;
  std::vector<int64_t> sizesUpper;
  int32_t cachedBlockSize{0};

  // Bitmap of grid choices from selectGrid (1 = singleBlock, 0 = default).
  // Used to detect when grid choice changed, invalidating all step caches.
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
};

/// Holds runtime state for executing a WaveGraph.  Pooled by WaveGraph
/// so that reusable buffers survive across calls.
struct ExecutionState {
  FrameP frame{nullptr};
  const ValueTypes* valueTypes{nullptr};
  WaveGraph* waveGraph{nullptr};
  facebook::velox::wave::GpuArena* deviceArena{nullptr};
  facebook::velox::wave::GpuArena* pinnedArena{nullptr};
  facebook::velox::wave::GpuArena* managedArena{nullptr};
  StreamPool* streamPool{nullptr};
  EventPool* eventPool{nullptr};

  /// Reusable CUDA stream, obtained from streamPool at the start of execution
  /// and returned on scope exit.
  std::unique_ptr<facebook::velox::wave::Stream> stream;

  const std::unordered_map<NodeCP, nativert::OpKernel*>* kernelMap{nullptr};
  const std::unordered_map<NodeCP, int32_t>* standaloneIndices{nullptr};
  std::vector<StandaloneStats>* standaloneStats{nullptr};

  /// Per-launch debug info collected during execution (owned by executor).
  std::vector<LaunchDebugInfo>* launchDebugInfos{nullptr};

  /// Reusable device buffers indexed by [sequenceNumber][stepIdx].
  std::vector<std::vector<facebook::velox::wave::WaveBufferPtr>> deviceBuffers;
  /// Reusable pinned buffers indexed by [sequenceNumber][stepIdx].
  std::vector<std::vector<facebook::velox::wave::WaveBufferPtr>> pinnedBuffers;
  /// Preallocated per-step vectors indexed by [sequenceNumber][stepIdx].
  std::vector<std::vector<StepVectors>> stepVectors;
};

/// Runs standalone launches by mapping each formal node to the actual node
/// via OpInvocation::nodeMap(), executing it via the corresponding OpKernel,
/// and recording timing in standaloneStats.
void runStandalones(
    const std::vector<LaunchData>& standalones,
    ExecutionState& state,
    const std::unordered_map<NodeCP, nativert::OpKernel*>& kernelMap,
    const std::unordered_map<NodeCP, int32_t>& standaloneIndices,
    std::vector<StandaloneStats>& standaloneStats);

/// Builds BlockInfo grid for a set of LaunchData entries. Uses preallocated
/// vectors in 'sv' (blocks, launchIndices, costs, maxBlocks,
/// numBlocksPerLaunch). Returns the block size (threads per block).
int32_t makeGrid(
    const std::vector<LaunchData>& launches,
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
    FormalToActual map,
    int32_t ordinal);

/// Looks up 'value' in 'map' and returns the corresponding SymInt from 'frame'.
int64_t paramSymInt(
    ValueCP value,
    nativert::ExecutionFrame& frame,
    const FormalToActual& map);

/// Executes a WaveGraph as a GraphExecutorBase subclass, allowing it to
/// be used wherever the standard nativert executors are used.
class WaveGraphExecutor : public nativert::GraphExecutorBase {
 public:
  WaveGraphExecutor(
      nativert::Graph& graph,
      std::vector<std::unique_ptr<nativert::OpKernel>> nodeKernels,
      const nativert::ExecutorConfig& executorConfig,
      std::shared_ptr<nativert::Weights> weights);

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

  /// Returns per-block DebugInfo from the most recent execution. Transfers
  /// device-side debug info to host. The outer vector has one element per
  /// kernel launch; the inner vector has one DebugInfo per block in that
  /// launch.
  std::vector<std::vector<DebugInfo>> getDebugInfo();

  /// Returns standalone execution stats: pairs of (node string, micros)
  /// from the most recent execution. The node string is formatted as in
  /// Launch::toString for standalone nodes.
  std::vector<std::pair<std::string, int64_t>> getStandaloneStats() const;

  WaveGraph* waveGraph() const {
    return waveGraph_.get();
  }

 private:
  /// Runs the WaveGraph on the given frame.
  void executeWave(nativert::ExecutionFrame& frame, WaveGraph& waveGraph);

  std::shared_ptr<nativert::Weights> weights_;

  /// Stable storage for TensorMeta objects referenced by the initial
  /// ValueTypes passed to WaveGraph.
  std::vector<std::unique_ptr<nativert::TensorMeta>> metaStore_;

  std::unique_ptr<WaveGraph> waveGraph_;

  /// Maps each nativert Node to its OpKernel, built once at construction.
  std::unordered_map<NodeCP, nativert::OpKernel*> kernelMap_;

  /// Pool of device-side ExecutionFrames with persistent tensors on GPU.
  std::unique_ptr<Pool<nativert::ExecutionFrame>> framePool_;

  /// Per-launch debug info from the most recent execution.
  std::vector<LaunchDebugInfo> launchDebugInfos_;
};

} // namespace torch::wave
