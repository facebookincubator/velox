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

#include "velox/experimental/torchwave/WaveGraph.h"
#include "velox/experimental/wave/common/Cuda.h"

namespace torch::wave {

class OpInvocation;
struct StepVectors;

/// Represents launch of a single KernelOperation or standalone Node.
struct Launch {
  NodeCP standalone{nullptr};
  KernelOperation* op{nullptr};

  /// Corresponds to irderedInputs in 'op'.
  std::vector<ValueCP> values;

  // Indices into constants in enclosing OpInvocation.
  std::vector<int32_t> constantIndices;

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
  ProjectOperation(const Subgraph& sg, CompileCtx& ctx);

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

  // Use single block variant if the largest input is <= than this
  int32_t singleBlockMaxSize_{0};

  // Values from newTensorValue/newScalarValue that appear in this
  // ProjectOperation's grids. These need special binding handling when the
  // ProjectOperation is reused for a different actual subgraph.
  std::vector<ValueCP> extraValues_;
};

struct ActualParameter {
  ValueCP value;
  std::optional<c10::IValue> constant;
};

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

 private:
  std::unique_ptr<facebook::velox::wave::CompiledKernel> kernel_;
  std::vector<std::unique_ptr<ProjectOperation>> ops_;
  std::vector<std::unique_ptr<KernelOperation>> kernelOpStorage_;
};

struct GridChoice {
  int32_t numElements;
  bool singleBlock;
  LaunchGrid* grid;
};

/// Data for a single launch within a step of the grid.
struct TensorListParam {
  int32_t listOffset;
  std::vector<int32_t> elementOffsets;
  std::vector<nativert::ValueId> elementIds;
};

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
  int64_t numElements;
  std::vector<nativert::ValueId> actualInputs;
  std::vector<nativert::ValueId> actualOutputs;
  std::vector<OutputDesc> actualOutputDescs;

  // Type kind from the Value for each output, parallel to actualOutputs.
  std::vector<nativert::Type::Kind> actualOutputTypes;

  /// After first use, the tensors and scalars and the offset to set in the
  /// existing invocation frame.
  std::vector<nativert::ValueId> tensorsInFrame;
  std::vector<int32_t> tensorOffsets;
  std::unordered_set<size_t> shapeOnlyTensorIndices;
  std::vector<nativert::ValueId> scalarsInFrame;
  std::vector<int32_t> scalarOffsets;
  std::vector<nativert::ValueId> returnValues;
  std::vector<int32_t> returnOffsets;
  // Type kind for each return value, parallel to returnValues.
  std::vector<nativert::Type::Kind> returnTypes;

  std::vector<TensorListParam> tensorLists;
};

struct CompositeInvocation {
  /// Executes this composite invocation: allocates outputs, builds the grid,
  /// copies params to pinned+device memory, and enqueues the H2D transfer.
  void execute(ExecutionState& state);

  /// Launches the kernel. In debug_single_ops mode, launches once per block
  /// with all other blocks' opcodes set to -1 and waits after each launch.
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
      int32_t stepIdx);

  /// Collects LaunchData from all OpInvocations at the given step index.
  /// Launches with a non-null kernel op go into 'kernels', others into
  /// 'standalones'.
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

  std::string toString(Listing mode = kExprs, int32_t ordinal = 0) const;

  std::unique_ptr<CompositeKernel> kernel;
  std::vector<OpInvocation> ops;
  /// Stable storage for IValues referenced by OpInvocation::constants_.
  std::deque<c10::IValue> ivalueStorage;
  /// Sequence number of this CompositeInvocation within its WaveGraph, starting
  /// at 0.
  int32_t sequenceNumber{0};
  /// Kernel occupancy info, cached at construction.
  facebook::velox::wave::KernelInfo kernelInfo;
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
