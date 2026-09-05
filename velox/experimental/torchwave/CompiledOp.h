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
#include <unordered_set>
#include <utility>
#include <vector>

#include <ATen/core/ivalue.h>
#include <folly/container/F14Set.h>

#include "velox/experimental/torchwave/WaveGraph.h"
#include "velox/experimental/wave/common/Cuda.h"

namespace torch::wave {

class OpInvocation;
struct StepVectors;
struct StepEvents;
// Defined in AllocGroup.h, which includes this file.
struct AllocGroupPlan;

/// Represents launch of a single KernelOperation or standalone Node.
struct Launch {
  Launch() = default;

  /// Constructs a standalone launch, setting up cpuOnly arg copies if needed.
  Launch(NodeCP standaloneNode, const ValueTypes& types, WaveGraph& waveGraph);

  NodeCP standalone{nullptr};
  KernelOperation* op{nullptr};

  /// For a standalone metadata-only op, the specific host-side shortcut, or
  /// kNone. Set from the node target in the standalone constructor.
  StandaloneShortcut standaloneShortcut{StandaloneShortcut::kNone};

  /// True if 'standalone' only manipulates tensor metadata (no real compute).
  /// Initialized from the op's Metadata, or true for prim.ListPack (which has
  /// no registry entry).
  bool metadataOnly{false};

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
  const NodeMap& nodeMap() const {
    return nodeMap_;
  }

  /// Adds a binding from a formal value id to an actual value id.
  void addBinding(int32_t formalId, int32_t actualId) {
    bindings_[formalId] = actualId;
  }

  std::string toString() const;

 private:
  ProjectOperation* projectOp_;
  FormalToActual bindings_;
  std::vector<const c10::IValue*> constants_;
  NodeMap nodeMap_;
};

/// Compiled CUDA kernel containing one or more ProjectOperations.
class CompositeKernel {
 public:
  CompositeKernel(
      std::vector<std::unique_ptr<ProjectOperation>>&& ops,
      std::vector<std::unique_ptr<KernelOperation>>&& kernelOps,
      const std::unordered_set<std::string>& includes,
      int32_t kernelId);

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

  /// Waits for the per-op diagnostic kernels queued under
  /// WaveConfig::configPerOp and returns each one's entry point and occupancy,
  /// in the order the ops appear in this kernel. Empty when configPerOp is off
  /// or no GPU is present.
  std::vector<std::pair<std::string, facebook::velox::wave::KernelInfo>>
  perOpKernelInfo();

  const std::vector<std::unique_ptr<KernelOperation>>& kernelOps() const {
    return kernelOpStorage_;
  }

 private:
  // One single-op kernel built beside the composite when configPerOp is set.
  // Diagnostic only: never launched for results, only warmed up so its
  // occupancy can be read.
  struct PerOpKernel {
    int32_t opCode{0};
    std::string entryPoint;
    std::unique_ptr<facebook::velox::wave::CompiledKernel> kernel;
  };

  std::unique_ptr<facebook::velox::wave::CompiledKernel> kernel_;
  std::string entryPoint_;
  std::string text_;
  std::vector<std::unique_ptr<ProjectOperation>> ops_;
  std::vector<std::unique_ptr<KernelOperation>> kernelOpStorage_;
  std::vector<PerOpKernel> perOpKernels_;
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

  /// For a metadata-only standalone shortcut (launch->standaloneShortcut !=
  /// kNone): the op's operands in c10 schema order (first-to-last for
  /// prim.ListPack, which has no schema). args[i] is the value operand, or
  /// nullptr when that operand is an integer constant -- in which case
  /// intArgs[i] holds the constant. intList holds an all-integer list operand
  /// (e.g. aten.view's size) for direct pass-through to the ATen primitive.
  std::vector<ValueCP> args;
  std::vector<int64_t> intArgs;
  std::vector<int64_t> intList;

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
  /// Offsets of non-tensor (scalar) kernel outputs. These get a zero
  /// placeholder before launch and are overwritten by the kernel; they must
  /// never be filled from the frame (unlike scalarsInFrame, which are inputs),
  /// since the frame slot is None until this kernel produces the value.
  std::vector<int32_t> scalarOutputOffsets;
  std::vector<nativert::ValueId> returnValues;
  std::vector<int32_t> returnOffsets;
  /// Type kind for each return value, parallel to returnValues.
  std::vector<nativert::Type::Kind> returnTypes;

  std::vector<TensorListParam> tensorLists;

  /// Executed-step index of the latest step whose device-to-host transfer
  /// produced a value this launch reads -- directly, through a chain of
  /// metadata-only shortcuts, or through a host-side view operand of an output
  /// descriptor. -1 when the launch reads nothing that came back over a
  /// transfer, which is what makes it safe to size, allocate and fill while an
  /// earlier step's transfer is still in flight. Set by markD2hDependencies
  /// before the step runs.
  int32_t d2hProducer{-1};

  float costAdjustFactor{1};
  float expectedFraction{0};
};

class CompositeInvocation {
 public:
  CompositeInvocation(
      std::unique_ptr<CompositeKernel> kernel,
      std::vector<OpInvocation> ops,
      std::deque<c10::IValue> ivalueStorage,
      int32_t sequenceNumber,
      std::vector<nativert::ValueId> lastUseIds,
      std::vector<std::vector<int32_t>> lastUseReaderOps = {},
      std::vector<nativert::ValueId> reusableIds = {},
      std::vector<Launch> prePassStandalones = {},
      std::vector<std::pair<nativert::ValueId, int32_t>> elidedCloneInputs =
          {});

  /// Out of line because allocGroupPlan_ points to a type this header only
  /// forward-declares.
  ~CompositeInvocation();

  /// Executes this composite invocation: allocates outputs, builds the grid,
  /// copies params to pinned+device memory, and enqueues the H2D transfer.
  void execute(ExecutionState& state);

  std::string toString(Listing mode = kExprs, int32_t ordinal = 0) const;

  const std::vector<OpInvocation>& ops() const {
    return ops_;
  }

  /// Non-const because describing a launch of an op's grid binds it, which the
  /// whole-graph allocation-group scan does for every op of every node.
  std::vector<OpInvocation>& ops() {
    return ops_;
  }

  CompositeKernel* kernel() const {
    return kernel_.get();
  }

  int32_t sequenceNumber() const {
    return sequenceNumber_;
  }

  const std::vector<nativert::ValueId>& lastUseIds() const {
    return lastUseIds_;
  }

  const std::vector<std::vector<int32_t>>& lastUseReaderOps() const {
    return lastUseReaderOps_;
  }

  /// True when this invocation releases its last-use values as their readers
  /// run out of grid steps rather than all at its last step. The
  /// allocation-group plan has to agree with it: a group's buffer is freed when
  /// the last of its slots is, so the plan and the release path must place that
  /// point the same way.
  bool stepLevelRelease() const;

  /// Installs the groups this invocation allocates, built once for the whole
  /// graph before its first execution.
  void setAllocGroupPlan(std::unique_ptr<AllocGroupPlan> plan);

 private:
  /// Launches the kernel. In debug_single_ops mode, launches once per block
  /// with all other blocks' opcodes set to -1 and waits after each launch.
  /// 'betweenLaunchAndSync' is called after the kernel launch (and D2H
  /// scheduling if any) but before the stream sync, to overlap host work
  /// with the GPU. In debug mode it is called after all block-by-block
  /// launches and transfers. With 'deferReturn', the host wait that makes the
  /// returned data readable is left to the caller, which records the transfer
  /// in ExecutionState::pendingReturns instead.
  void launch(
      int32_t numBlocks,
      int32_t blockSize,
      uint8_t* pinnedBase,
      uint8_t* deviceBase,
      int64_t h2dBytes,
      int32_t returnBegin,
      int32_t returnEnd,
      DebugInfo* deviceDebugBase,
      facebook::velox::wave::Stream* stream,
      const StepVectors& sv,
      int32_t stepIdx,
      bool deferReturn,
      const std::function<void()>& betweenLaunchAndSync = nullptr,
      StepEvents* events = nullptr);

  /// Reserves a param slot for every kernel launch any grid variant of any op
  /// can put at 'stepIdx', each sized for the largest variant, and fills
  /// sv.slotOffsets / sv.opSlotBegin / sv.paramRegionBytes. Also fills
  /// sv.readIds. Idempotent: both depend only on the compiled grids, so they
  /// are built once per step.
  void layoutParamSlots(int32_t stepIdx, StepVectors& sv);

  /// Why an op's setup was left for the second pass.
  enum class DeferReason {
    /// Reads a value whose device-to-host transfer has not landed.
    kTransfer,
    /// Allocating it would push past WaveConfig::maxDelayedFree.
    kMemory,
  };

  /// Running position over the three parallel per-launch arrays a step fills:
  /// sv.kernels, sv.standalones and sv.shortcutStandalones. It advances over a
  /// deferred op as well, whose launches keep their indices so that deferring
  /// one op cannot move a later op's parameter offsets.
  struct LaunchCursor {
    int32_t kernel{0};
    int32_t standalone{0};
    int32_t shortcut{0};
  };

  /// Picks the grid variant 'data' should launch under -- single-block,
  /// multi-block or cooperative -- from its element count, and rebuilds 'data'
  /// and 'largestId' from the new variant if the choice moved. Returns true if
  /// it moved, in which case the caller re-reads the grid and step it cached.
  /// Only meaningful for a launch whose op isGridChoice().
  bool chooseGridVariant(
      ExecutionState& state,
      GridChoice& gridChoice,
      OpInvocation& op,
      int32_t stepIdx,
      size_t launchIndex,
      bool hasByLargestInput,
      LaunchData& data,
      nativert::ValueId& largestId);

  /// Zeroes 'data.numElements' when a tensor the launch reads or writes but
  /// does not produce itself is still None, which makes makeGrid give it no
  /// blocks. Under a cooperative grid, where the whole step launches as one
  /// kernel and an op therefore cannot be skipped, recovers a grid size from
  /// the static input shapes instead.
  void sizeForUnreadyOperands(
      LaunchData& data,
      const Launch& launch,
      nativert::ExecutionFrame& frame);

  /// Fills 'data's parameter block at 'paramOffset', patches its TensorList
  /// pointers to device addresses, and folds the offsets of the return values
  /// it writes into 'returnBegin' / 'returnEnd'.
  void fillLaunchParamBlock(
      LaunchData& data,
      nativert::ExecutionFrame& frame,
      uint8_t* pinnedBase,
      uint8_t* deviceBase,
      int64_t paramOffset,
      int32_t& returnBegin,
      int32_t& returnEnd);

  /// Sets up the ops at 'stepIdx' that can be set up now: sizes them, allocates
  /// their outputs and fills their parameter blocks, all in one pass per op so
  /// each is touched once and tested for dependencies once.
  ///
  /// An op that reads a value a pending transfer has not delivered, or that
  /// would allocate past the delayed-free ceiling, is skipped and appended to
  /// 'deferred'; run the pass again with 'deferredOnly' after resolving what
  /// they need. The LaunchData, parameter offsets and launch indices of every
  /// op are established on the first pass regardless, so the second pass
  /// cannot move another op's parameters.
  ///
  /// 'returnBegin' / 'returnEnd' accumulate as min/max over absolute offsets
  /// rather than assuming the ops are filled in ascending order, which the
  /// two-pass split breaks.
  void gatherLaunches(
      ExecutionState& state,
      std::vector<GridChoice>& grids,
      int32_t stepIdx,
      StepVectors& sv,
      uint8_t* pinnedBase,
      uint8_t* deviceBase,
      bool deferredOnly,
      std::vector<std::pair<size_t, DeferReason>>& deferred,
      int32_t& returnBegin,
      int32_t& returnEnd);

  /// Fills the parameter block of every kernel launch in 'sv' in one sweep,
  /// accumulating the return-data span the same way the inline fill does.
  ///
  /// Only the allocation-group path uses this. The ordinary path fills each
  /// launch in the pass that sizes it, which it can because that pass also
  /// allocates the launch's outputs; a grouped launch has no output tensor
  /// until its whole group has been sized and carved, so its fill has to wait
  /// for every group of the step.
  void fillStepParams(
      ExecutionState& state,
      StepVectors& sv,
      uint8_t* pinnedBase,
      uint8_t* deviceBase,
      int32_t& returnBegin,
      int32_t& returnEnd);

  /// The allocation-group execute path, selected by WaveConfig at the top of
  /// execute(). Defined in AllocGroup.cpp: it is a parallel implementation of
  /// the step loop, not a variant of it, and keeping it out of this file is
  /// what stops the two from growing shared branches.
  ///
  /// Differs from execute() only in how outputs are allocated. Instead of one
  /// allocator call per output as each op is sized, the step's outputs that
  /// share a lifetime are sized first and carved out of one buffer, sync-free
  /// groups before the host waits on any transfer and the rest after.
  void executeAllocGroups(ExecutionState& state);

  /// Stamps onto step 'stepIdx' every last-use value not yet released whose
  /// reading ops have all run out of grid steps, recording the step in
  /// 'releaseStep' (parallel to lastUseIds_, -1 until released). Called once
  /// per executed step, after gatherLaunches has settled its grid choices.
  void releaseLastUseAtStep(
      ExecutionState& state,
      const std::vector<GridChoice>& grids,
      int32_t stepIdx,
      StepVectors& sv,
      std::vector<int32_t>& releaseStep);

  /// Prints per-step trace: step header and per-launch details.
  void traceStep(
      int32_t stepIdx,
      const StepVectors& sv,
      const std::vector<GridChoice>& gridChoices);

  std::unique_ptr<CompositeKernel> kernel_;
  std::vector<OpInvocation> ops_;
  std::deque<c10::IValue> ivalueStorage_;
  int32_t sequenceNumber_;

  // Frame value ids whose last use across the graph is in this node (graph
  // outputs excluded). When WaveConfig::freeIntermediates is set, their frame
  // tensors are released at the end of execute().
  std::vector<nativert::ValueId> lastUseIds_;

  // Parallel to lastUseIds_: the indices into ops_ of the ops that read that
  // value. Under WaveConfig::stepLastUse the value is released at the step
  // where the last of them runs out of grid steps. Empty means the readers are
  // unknown, which falls back to releasing at the node's last step.
  std::vector<std::vector<int32_t>> lastUseReaderOps_;

  // Frame value ids whose buffer an elementwise output may reuse in place --
  // reusable last-use boundary inputs and expr-local overwritable temps, from
  // ProjectNode::reusableValues_/overwritableTemps_. Gated by
  // WaveConfig::enableReuse.
  folly::F14FastSet<nativert::ValueId> reusableIds_;

  // Standalone ops from the maxFusedNodes pre-pass.  Executed at the
  // start of execute() before any kernel step, so their outputs are
  // available for SizeExpr evaluation.
  std::vector<Launch> prePassStandalones_;

  // The allocation grouping for this invocation, built on the first execution
  // in that mode and reused by every later one. Held by pointer so CompiledOp.h
  // does not have to see AllocGroup.h, which includes it.
  std::unique_ptr<AllocGroupPlan> allocGroupPlan_;

  // Frame value ids that were the input of a clone the in-place pass elided in
  // this node, paired with the number of clones elided for that value (from
  // ProjectNode::elidedCloneCounts). Used to report the copying saved; read
  // only when the kTiming trace bit is on.
  std::vector<std::pair<nativert::ValueId, int32_t>> elidedCloneInputs_;
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

  CompositeInvocation* kernels() {
    return kernels_.get();
  }

  std::string toString(Listing mode = kExprs, int32_t ordinal = 0) const;

 private:
  // The outer array represents parallel launchable sequences kernels. The inner
  // array is a sequence of consecutive kernels.
  std::unique_ptr<CompositeInvocation> kernels_;
};

} // namespace torch::wave
