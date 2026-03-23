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
/// copies tensor storage into it, allocates a matching device buffer, and
/// enqueues an async H2D transfer on 'stream'. The output tensors in 'out'
/// share the device buffer and point into the appropriate offsets. The returned
/// WaveBufferPtr keeps the device memory alive as long as any output tensor
/// references it.
facebook::velox::wave::WaveBufferPtr tensorsToDevice(
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

/// Holds runtime state for executing a WaveGraph.
struct ExecutionState {
  nativert::ExecutionFrame& frame;
  const ValueTypes* valueTypes{nullptr};
  facebook::velox::wave::GpuArena* deviceArena{nullptr};
  facebook::velox::wave::GpuArena* pinnedArena{nullptr};
  facebook::velox::wave::GpuArena* managedArena{nullptr};
  StreamPool* streamPool{nullptr};
  EventPool* eventPool{nullptr};
};

/// Builds BlockInfo grid for a set of OpInvocations given their numElements.
/// Populates 'blocks' with the BlockInfo array, 'opIndices' with the op index
/// for each block, and clears 'nextOps'.
void makeGrid(
    const std::vector<OpInvocation>& ops,
    const std::vector<int64_t>& numElements,
    std::vector<BlockInfo>& blocks,
    std::vector<int32_t>& opIndices,
    std::vector<OpInvocation*>& nextOps);

/// Computes output shapes for elementwise operations given the dedupped input
/// values' shapes.
std::vector<std::vector<Dim>> elementwiseOutputShape(
    const std::vector<const nativert::Value*>& inputs,
    const nativert::Node* node,
    nativert::ExecutionFrame& frame,
    FormalToActual map);

/// Executes a WaveGraph as a GraphExecutorBase subclass, allowing it to
/// be used wherever the standard nativert executors are used.
class WaveGraphExecutor : public nativert::GraphExecutorBase {
 public:
  WaveGraphExecutor(
      const nativert::Graph& graph,
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

 private:
  /// Runs the WaveGraph on the given frame.
  void executeWave(nativert::ExecutionFrame& frame, const WaveGraph& waveGraph);

  /// Builds ValueTypes from the graph's tensor metadata.
  void initValueTypes();

  std::shared_ptr<nativert::Weights> weights_;

  /// Stable storage for TensorMeta objects referenced by valueTypes_.
  std::vector<std::unique_ptr<nativert::TensorMeta>> metaStore_;

  ValueTypes valueTypes_;

  std::unique_ptr<WaveGraph> waveGraph_;

  /// Pool of device-side ExecutionFrames with persistent tensors on GPU.
  std::unique_ptr<Pool<nativert::ExecutionFrame>> framePool_;
};

} // namespace torch::wave
