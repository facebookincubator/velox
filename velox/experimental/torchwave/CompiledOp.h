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

#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include <ATen/core/ivalue.h>
#include <folly/container/F14Set.h>

#include "velox/experimental/torchwave/WaveGraph.h"
#include "velox/experimental/wave/common/Cuda.h"

namespace torch::wave {

class OpInvocation;
struct StepVectors;

/// Represents launch of a single KernelOperation or standalone Node.
struct Launch {
  Launch() = default;

  /// Constructs a standalone launch, setting up cpuOnly arg copies if needed.
  Launch(NodeCP standaloneNode, const ValueTypes& types, WaveGraph& waveGraph);

  NodeCP standalone{nullptr};
  KernelOperation* op{nullptr};

  /// Corresponds to orderedInputs in 'op'.
  std::vector<ValueCP> values;

  /// Indices into constants in enclosing OpInvocation.
  std::vector<int32_t> constantIndices;

  /// For standalone ops with cpuOnly args: the CPU-side copy values and their
  /// corresponding device-side originals, at matching indices.
  std::vector<ValueCP> argOnCpu;
  std::vector<ValueCP> argOnDevice;

  /// Returns "kernel: <op toString>" for fused ops or
  /// "standalone <subgraph toString>" for standalone nodes where the subgraph
  /// root is the standalone node and inputs are its direct inputs.
  std::string toString(Listing mode = kExprs) const;
};

/// Represents a grid of parallel / consecutive operations for completing a
/// single top level computation. The launches in the inner vector are parallel
/// and data independent. Where Node is nullptr, these hit different opcodes of
/// one composite kernel. Where Node is set, the Nodes are invoked in parallel.
/// The outer vector represents consecutive steps. A multikernel op, e.g. a
/// multiblock reduction has different pieces of it in launches wit consecutive
/// outer vector indices.
using LaunchGrid = std::vector<std::vector<Launch>>;

class CompileCtx;

/// Represents a top level result producing expression in a ProjectNode. There
/// is one per distinct shape of expression. A single ProjectNode can have many
/// outputs with the same compute on different inputs.
class ProjectOperation {
 public:
  explicit ProjectOperation(const Subgraph& sg);

  const Subgraph& subgraph() const {
    return subgraph_;
  }

  LaunchGrid& grid() {
    return grid_;
  }

  LaunchGrid& singleBlockGrid() {
    return singleBlockGrid_;
  }

  LaunchGrid& cgGrid() {
    return cgGrid_;
  }

  int32_t singleBlockMaxSize() const {
    return singleBlockMaxSize_;
  }

  std::string toString(Listing mode = kExprs) const;

  /// Values created during grid generation (e.g. multikernel variant
  /// intermediates) that are not part of the original subgraph.
  const std::vector<ValueCP>& extraValues() const {
    return extraValues_;
  }

 private:
  friend class CompileCtx;
  Subgraph subgraph_;

  // Sequence of kernel launches to process a multi-block wide input.
  LaunchGrid grid_;

  // If set, there is a single block variant that can process the input more
  // efficiently with fewer launches if the input is small enough.
  LaunchGrid singleBlockGrid_;

  // If set, a cooperative grid variant using cgVariant metadata.
  LaunchGrid cgGrid_;

  // Single-block reduction is more efficient at small sizes because it avoids
  // the overhead of multi-kernel synchronization. Use it when the largest
  // input has at most this many elements. WaveConfig::useSingleBlock overrides.
  int32_t singleBlockMaxSize_{0};

  // Values from newTensorValue/newScalarValue that appear in this
  // ProjectOperation's grids. These need special binding handling when the
  // ProjectOperation is reused for a different actual subgraph.
  std::vector<ValueCP> extraValues_;
};

/// Binds a ProjectOperation to an actual subgraph with concrete value mappings.
class OpInvocation {
 public:
  OpInvocation(
      ProjectOperation* projectOp,
      const Subgraph& sg,
      std::deque<c10::IValue>& storage);

  ProjectOperation* projectOp() const {
    return projectOp_;
  }

  const FormalToActual& bindings() const {
    return bindings_;
  }

  const std::vector<const c10::IValue*>& constants() const {
    return constants_;
  }

  /// Maps each node in the projectOp's formal subgraph to the corresponding
  /// node in the actual subgraph passed at construction.
  const std::unordered_map<NodeCP, NodeCP>& nodeMap() const {
    return nodeMap_;
  }

  /// Adds a binding from a formal value id to an actual value id.
  void addBinding(int32_t formalId, int32_t actualId) {
    bindings_[formalId] = actualId;
  }

 private:
  ProjectOperation* projectOp_;
  FormalToActual bindings_;
  std::vector<const c10::IValue*> constants_;
  std::unordered_map<NodeCP, NodeCP> nodeMap_;
};

/// Compiled CUDA kernel containing one or more ProjectOperations.
class CompositeKernel {
 public:
  CompositeKernel(
      std::vector<std::unique_ptr<ProjectOperation>>&& ops,
      std::vector<std::unique_ptr<KernelOperation>>&& kernelOps,
      const std::unordered_set<std::string>& includes);

  /// Launches the kernel on the given stream.
  void launch(
      int32_t numBlocks,
      int32_t numThreads,
      int32_t sharedMemory,
      facebook::velox::wave::Stream* stream,
      void** args);

  /// Launches the kernel as a cooperative grid.
  void launchCooperative(
      int32_t numBlocks,
      int32_t numThreads,
      int32_t sharedMemory,
      facebook::velox::wave::Stream* stream,
      void** args);

  /// Returns occupancy information for the compiled kernel. Returns a
  /// default KernelInfo if no GPU is available.
  facebook::velox::wave::KernelInfo kernelInfo() const;

  std::string toString(Listing mode = kExprs) const;

  const std::string& entryPoint() const {
    return entryPoint_;
  }

  const std::string& text() const {
    return text_;
  }

  void warmup();

 private:
  std::unique_ptr<facebook::velox::wave::CompiledKernel> kernel_;
  std::string entryPoint_;
  std::string text_;
  std::vector<std::unique_ptr<ProjectOperation>> ops_;
  std::vector<std::unique_ptr<KernelOperation>> kernelOpStorage_;
};

/// Records the grid variant (single-block vs multi-block) chosen for a
/// ProjectOperation.
struct GridChoice {
  int32_t numElements;
  bool singleBlock;
  LaunchGrid* grid;
};

/// Tracks a TensorList argument and its element Tensors within kernel
/// parameters. A TensorList is passed as a TensorList header (size + pointer
/// array) at listOffset, and each element Tensor is also placed individually
/// in the same parameter block at its own offset. The TensorList's pointer
/// array references these element Tensors by device address.
///
/// During H2D setup, fillTensorListParam fills each element Tensor descriptor
/// at its offset and records the offsets here. patchTensorListPointers then
/// rewrites the pointer array entries to device-side addresses so the kernel
/// can index into the list. During D2H, processReturnData uses the recorded
/// elementOffsets and elementIds to read back per-element shapes from the
/// pinned buffer into the execution frame.
struct TensorListParam {
  int32_t listOffset{0};
  std::vector<int32_t> elementOffsets;
  std::vector<nativert::ValueId> elementIds;
};

/// Runtime state for a single kernel launch: actual value IDs, parameter
/// offsets, and return info.
struct LaunchData {
  LaunchData() = default;
  LaunchData(
      const Launch& launch,
      OpInvocation& op,
      const IdToValueMap& idToValue);

  const Launch* launch{nullptr};
  OpInvocation* invocation{nullptr};
  NodeCP standalone{nullptr};
  SizeExpr sizeExpr;
  int64_t numElements{0};
  std::vector<nativert::ValueId> actualInputs;
  std::vector<nativert::ValueId> actualOutputs;
  std::vector<OutputDesc> actualOutputDescs;

  /// Type kind from the Value for each output, parallel to actualOutputs.
  std::vector<nativert::Type::Kind> actualOutputTypes;

  /// After first use, the tensors and scalars and the offset to set in the
  /// existing invocation frame.
  std::vector<nativert::ValueId> tensorsInFrame;
  std::vector<int32_t> tensorOffsets;
  folly::F14FastSet<size_t> shapeOnlyTensorIndices;
  std::vector<nativert::ValueId> scalarsInFrame;
  std::vector<int32_t> scalarOffsets;
  std::vector<nativert::ValueId> returnValues;
  std::vector<int32_t> returnOffsets;
  /// Type kind for each return value, parallel to returnValues.
  std::vector<nativert::Type::Kind> returnTypes;

  std::vector<TensorListParam> tensorLists;
};

class CompositeInvocation {
 public:
  CompositeInvocation(
      std::unique_ptr<CompositeKernel> kernel,
      std::vector<OpInvocation> ops,
      std::deque<c10::IValue> ivalueStorage,
      int32_t sequenceNumber);

  /// Executes this composite invocation: allocates outputs, builds the grid,
  /// copies params to pinned+device memory, and enqueues the H2D transfer.
  void execute(ExecutionState& state);

  std::string toString(Listing mode = kExprs, int32_t ordinal = 0) const;

  const std::vector<OpInvocation>& ops() const {
    return ops_;
  }

  CompositeKernel* kernel() const {
    return kernel_.get();
  }

 private:
  /// Launches the kernel. In debug_single_ops mode, launches once per block
  /// with all other blocks' opcodes set to -1 and waits after each launch.
  /// 'betweenLaunchAndSync' is called after the kernel launch (and D2H
  /// scheduling if any) but before the stream sync, to overlap host work
  /// with the GPU. In debug mode it is called after all block-by-block
  /// launches and transfers.
  void launch(
      int32_t numBlocks,
      int32_t blockSize,
      uint8_t* pinnedBase,
      uint8_t* deviceBase,
      int64_t totalPinnedBytes,
      int32_t returnBegin,
      int32_t returnEnd,
      DebugInfo* deviceDebugBase,
      facebook::velox::wave::Stream* stream,
      const StepVectors& sv,
      int32_t stepIdx,
      std::function<void()> betweenLaunchAndSync = nullptr);

  /// Collects LaunchData from all OpInvocations at the given step index.
  void gatherLaunches(
      const ExecutionState& state,
      std::vector<GridChoice>& grids,
      int32_t stepIdx,
      StepVectors& sv);

  /// Copies return values from the pinned buffer back into the execution frame.
  void processReturnData(
      StepVectors& sv,
      nativert::ExecutionFrame& frame,
      uint8_t* pinnedBase);

  /// Prints per-step trace: step header and per-launch details.
  void traceStep(
      int32_t stepIdx,
      const StepVectors& sv,
      const std::vector<GridChoice>& gridChoices);

  std::unique_ptr<CompositeKernel> kernel_;
  std::vector<OpInvocation> ops_;
  std::deque<c10::IValue> ivalueStorage_;
  int32_t sequenceNumber_;
};

/// Represents a single ProjectNode in a stack of ProjectNodes. Contains a graph
/// of CompositeKernels and a binding of their parameters to slots in the
/// execution state.
class CompiledNode {
 public:
  explicit CompiledNode(std::unique_ptr<CompositeInvocation> kernels)
      : kernels_(std::move(kernels)) {}

  /// Executes this node using the given execution state.
  void execute(ExecutionState& state);

  const CompositeInvocation* kernels() const {
    return kernels_.get();
  }

  std::string toString(Listing mode = kExprs, int32_t ordinal = 0) const;

 private:
  // The outer array represents parallel launchable sequences kernels. The inner
  // array is a sequence of consecutive kernels.
  std::unique_ptr<CompositeInvocation> kernels_;
};

} // namespace torch::wave
