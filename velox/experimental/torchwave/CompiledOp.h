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
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <ATen/core/ivalue.h>
#include <folly/container/F14Set.h>

#include "velox/experimental/torchwave/Utils.h"
#include "velox/experimental/torchwave/WaveGraph.h"
#include "velox/experimental/wave/common/Cuda.h"

namespace torch::wave {

class OpInvocation;
struct StepVectors;
struct StepEvents;
// Defined in AllocGroup.h, which includes this file.
class AllocGroupCollector;
struct AllocGroupPlan;
struct ConcatGroupLayout;
struct LaunchOp;
struct OutputOp;
struct StepCode;
struct StepProgram;

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

  /// Blocks per SM this op would rather run at, or 0 for unspecified. Taken
  /// from WaveConfig::preferBlocksPerSm by opcode when the launch is built, so
  /// the partitioner does not have to consult the config per step. A non-zero
  /// value both keeps the op out of launches with a different preference and
  /// narrows its launch's capacity to numSMs times this. See the config field
  /// for what it is for.
  int32_t blocksPerSm{0};

  /// Corresponds to orderedInputs in 'op'.
  std::vector<ValueCP> values;

  /// Earliest step this launch may be placed in, or -1 for no floor. Placement
  /// otherwise puts a launch one step after the last of its inputs, which is as
  /// early as its data allows -- right for compute, wrong for a copy that fills
  /// a band of a concat result. That band exists only once the concat's
  /// allocation group has been carved, so such a copy has to sit no earlier
  /// than the step whose head lays the concat out, however early its own source
  /// happens to be ready.
  int32_t minLevel{-1};

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

  /// The entry point of the module of step 'stepIdx' at 'blockSize' threads.
  int32_t entryFor(int32_t blockSize, int32_t stepIdx) const;

  /// Blocks of entry point 'entry' resident on one SM at its block size and no
  /// dynamic shared memory. KernelInfo::maxOccupancy0 is the same figure at
  /// 256 threads.
  int32_t maxOccupancy(int32_t entry) const;

  /// The largest maxOccupancy of any entry point, which bounds the blocks any
  /// step of this kernel can launch.
  int32_t maxOccupancyOfEntries() const;

  /// Launches entry point 'entry' at its block size on the given stream.
  void launch(
      int32_t entry,
      int32_t numBlocks,
      int32_t sharedMemory,
      facebook::velox::wave::Stream* stream,
      void** args);

  /// Launches entry point 'entry' as a cooperative grid.
  void launchCooperative(
      int32_t entry,
      int32_t numBlocks,
      int32_t sharedMemory,
      facebook::velox::wave::Stream* stream,
      void** args);

  /// Returns occupancy information for entry point 'entry'. Returns a default
  /// KernelInfo if no GPU is available.
  facebook::velox::wave::KernelInfo kernelInfo(int32_t entry) const;

  int32_t numEntries() const {
    return static_cast<int32_t>(entryBlockSize_.size());
  }

  /// Names entry point 'entry' by its module and block size, followed by its
  /// KernelInfo, for logs.
  std::string entryInfoString(int32_t entry) const;

  /// Blocks of entry point 'entry' that stay resident on one SM at
  /// 'dynamicSharedBytes' of dynamic shared memory, as the driver computes it.
  /// 0 when there is no compiled kernel to ask. Sizing a cooperative launch by
  /// anything else risks a grid the driver will refuse, so the launch
  /// partitioner reads this rather than dividing KernelInfo's zero-shared
  /// figure down.
  int32_t occupancy(int32_t entry, int32_t dynamicSharedBytes) const;

  std::string toString(Listing mode = kExprs) const;

  const std::string& entryPoint() const {
    return entryPoint_;
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

  // The compiled modules, one per distinct set of ops that share a step.
  std::vector<std::unique_ptr<facebook::velox::wave::CompiledKernel>> modules_;
  std::string entryPoint_;
  std::vector<std::unique_ptr<ProjectOperation>> ops_;
  std::vector<std::unique_ptr<KernelOperation>> kernelOpStorage_;

  // Per entry point: its block size.
  std::vector<int32_t> entryBlockSize_;
  // Per entry point: its module, and its index among that module's entries.
  std::vector<int32_t> entryModule_;
  std::vector<int32_t> entryLocal_;
  // The module of each step of the node, -1 for a step that launches no
  // kernel op.
  std::vector<int32_t> stepModule_;

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

  /// The operands of the host-side views of the outputs (OutputDesc::viewNode,
  /// e.g. a leaf-input slice) other than the viewed tensor: a view takes its
  /// shape and offset from them, not from this op's inputs -- a slice end is
  /// typically an item() of a device result. Set by setViewReadIds at bind.
  std::vector<nativert::ValueId> viewReadIds;

  float costAdjustFactor{1};
  float expectedFraction{0};
};

/// A host-side view of a fused output (OutputDesc::viewNode) as a metadata
/// shortcut: the same direct tensor arithmetic a standalone shortcut op runs,
/// in place of a boxed dispatch to the view op's kernel.
struct ViewRecipe {
  Launch launch;
  LaunchData data;
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

  /// Bound path, first pass of binding: fixes each op's grid, builds the
  /// launches of every step of this node for 'state' and adds the values they
  /// bring back to the host to 'returned'.
  void bindLaunches(
      ExecutionState& state,
      folly::F14FastSet<int32_t>& returned);

  /// Bound path, second pass: builds each step's program -- sizes, outputs,
  /// fills, the wait -- and writes the parameters that never change. 'returned'
  /// is every value any launch of the graph brings back.
  void bindPrograms(
      ExecutionState& state,
      const folly::F14FastSet<int32_t>& returned);

  /// The groups this invocation allocates, or null while none are installed.
  const std::vector<Launch>& prePassStandalones() const {
    return prePassStandalones_;
  }

  const AllocGroupPlan* allocGroupPlan() const {
    return allocGroupPlan_.get();
  }

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

  /// Per-execution state of one node's step loop: what the step setup of either
  /// path and the shared step epilogue both read and update.
  struct NodeRun {
    ExecutionState& state;
    bool doTiming{false};
    bool doEventTiming{false};
    bool ranStandalones{false};
    bool countElidedClones{false};
    std::vector<bool> elidedCounted;
    bool stepLastUse{false};
    std::vector<int32_t> lastUseReleaseStep;
    int32_t lastExecStep{-1};
    int32_t blockSize{0};
    int64_t reuseCount0{0};
    int64_t reuseBytes0{0};
  };

  NodeRun startNodeRun(ExecutionState& state);

  /// Carves the step's allocation groups that are ready: before the host waits
  /// on a transfer, the complete groups that need none; after, all the rest.
  void materializeStepGroups(
      ExecutionState& state,
      AllocGroupCollector& collector,
      StepVectors& sv,
      bool afterWait,
      bool doTiming);

  /// Writes the shape of each value 'concat's operand sizes read that has no
  /// table entry yet but is an output of a later launch of this node -- a
  /// chain gather in a later step whose extent a fused elementwise operand
  /// takes. Runs that output's own Size early, which its inputs allow once
  /// they are in the table; the launch that writes it sizes it again later.
  void sizeConcatReadsAhead(
      const ConcatGroupLayout& concat,
      ExecutionState& state);

  /// Adds the bytes of this node's elided clones to the first step where their
  /// inputs exist.
  void addElidedClones(NodeRun& run, StepVectors& sv);

  /// Everything a step does once its parameter blocks are filled: D2H
  /// bookkeeping, step-level release, the grid and BlockInfo array, the
  /// launch with the step's standalones, return data and reference checks.
  void finishStep(
      NodeRun& run,
      int32_t stepIdx,
      StepVectors& sv,
      uint8_t* pinnedBase,
      uint8_t* deviceBase,
      int64_t totalPinnedBytes,
      int32_t returnBegin,
      int32_t returnEnd,
      const SetupCounters& setupBefore,
      std::vector<GridChoice>& currentGridChoices);

  /// The node's end: the default-stream sync and the release of the values
  /// whose last use is this node.
  void finishNode(NodeRun& run);

  /// Runs the node's steps from their bound programs.
  void executeBound(ExecutionState& state);

  /// Sizes a bound launch and places its outputs.
  void runBoundLaunch(
      const LaunchOp& launch,
      StepProgram& program,
      StepVectors& sv,
      ExecutionState& state);

  void runBoundOutput(
      const OutputOp& output,
      StepProgram& program,
      ExecutionState& state);

  /// A launch whose tensor input it does not produce itself, or whose output,
  /// is absent this execution runs on no blocks.
  void checkUnreadyOperands(
      const LaunchOp& launch,
      const StepCode& code,
      LaunchData& data,
      ExecutionState& state);

  /// Fills lastUseByStep_ from the bound grids.
  void bindLastUseSteps(const std::vector<GridChoice>& grids, int32_t numSteps);

  /// Stamps onto step 'stepIdx' every last-use value not yet released whose
  /// reading ops have all run out of grid steps, recording the step in
  /// 'releaseStep' (parallel to lastUseIds_, -1 until released). Called once
  /// per executed step.
  void releaseLastUseAtStep(
      ExecutionState& state,
      int32_t stepIdx,
      StepVectors& sv,
      std::vector<int32_t>& releaseStep);

  /// Prints per-step trace: step header and per-launch details.
  void traceStep(
      int32_t stepIdx,
      const StepVectors& sv,
      const std::vector<GridChoice>& gridChoices);

  /// By step, the indices into lastUseIds_ of the values whose readers are all
  /// done after that step.
  std::vector<std::vector<int32_t>> lastUseByStep_;
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
