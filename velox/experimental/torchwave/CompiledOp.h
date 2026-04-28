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

#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include <ATen/core/ivalue.h>

#include <torch/nativert/graph/Graph.h>
#include <torch/nativert/graph/TensorMeta.h>
#include "velox/experimental/torchwave/KernelParams.h"
#include "velox/experimental/torchwave/Registry.h"
#include "velox/experimental/wave/common/Cuda.h"

namespace torch::wave {

class CompileCtx;
struct ExecutionState;
class OpInvocation;
struct StepVectors;

/// Map from Value::id() to Value*, pre-built from the graph for fast lookups.
using IdToValueMap = std::unordered_map<int32_t, ValueCP>;

/// Describes how to determine the number of blocks to run for a
/// KernelOperation based on input sizes.
struct SizeExpr {
  SizeShortcut op{SizeShortcut::kNone};
  std::vector<nativert::ValueId> values;
  std::vector<SizeExpr> args;

  /// Accesses all Values and calls recursively on args and combines the
  /// results (numel() of Values) by 'op'. (max or sum). If largestOut is
  /// not null and op is kMax, assigns the ValueId with the largest numel.
  int64_t numElements(FrameP frame, nativert::ValueId* largestOut = nullptr)
      const;

  /// Substitutes Values in 'this' according to 'bindings' and returns a deep
  /// copy with the Values replaced.
  SizeExpr toActual(
      const FormalToActual& bindings,
      const IdToValueMap& idToValue) const;
};

/// Describes a single output of a KernelOperation.
struct OutputDesc {
  /// Determines the shape to allocate for this output.
  OutputReserveFunc reserveShape;

  /// True if device code sets the dims. If reserveFunc is nullptr,
  /// and shapeSetOnDevice is true, then the device code does a view and
  /// the host reads back the modified parameter block to update the
  /// c10::Tensor on host. For cases like single pass stream
  /// compaction, the reserveFunc will reserve the max and the actual
  /// compacted size will be read for the same c10::Tensor in a read
  /// queued after the kernel.
  bool shapeSetOnDevice{false};

  bool neededOnHost{false};

  /// If this is a view that is not allocated, it still gets a storage from some
  /// Value.
  ValueCP storageFrom{nullptr};

  /// Shortcut for all elementwise, where we already know the largest input and
  /// can set the output shape by that.
  bool byLargestInput{false};

  SizeExpr sizeExpr;
};

/// Describes a subgraph rooted at a single node with its leaf inputs and
/// their tensor metadata.
struct Subgraph {
  NodeCP root;
  std::vector<ValueCP> inputs;
  std::vector<const nativert::TensorMeta*> inputTypes;

  /// Returns a map from node ordinal to the number of attributes seen before
  /// that node, for each node that has attributes. Traverses inner nodes in the
  /// same order as listConstants. Ordinals are assigned in pre-order (root = 0,
  /// leftmost input = 1, etc.).
  std::unordered_map<int32_t, int32_t> makeConstantIndices() const;

  /// Returns the pre-order ordinal of 'node' within this subgraph.
  int32_t nodeOrdinal(NodeCP node) const;

  /// Returns a human-readable expression string for this subgraph.
  /// Nodes are printed as target(arg1, arg2, ...) recursively, stopping at
  /// inputs_ which are printed as %id.
  std::string toString() const;
};

/// Collects scalar constants from a subgraph's nodes into 'storage' and
/// returns pointers into the stable storage.
std::vector<const c10::IValue*> listConstants(
    const Subgraph& sg,
    std::deque<c10::IValue>& storage);

/// Represents an operation offered by a composite kernel. For example a fused
/// expression like a + b * c + scalar. At run time the operation may span a
/// variable number of blocks in an invocation of a composite kernel.
class KernelOperation {
 public:
  KernelOperation(
      const Subgraph& sg,
      int32_t opCode,
      const CompileCtx& compileCtx);

  int32_t paramOffset(ValueCP value) const;

  int32_t attrOffset(NodeCP node, std::string_view attr) const;

  void addSharedDeclaration(const std::string& decl);

  /// Sets the CUDA code text from the given stream and clears the stream.
  void setCode(std::stringstream& code);

  const std::string& code() const {
    return text_;
  }

  int32_t opCode() const {
    return opCode_;
  }

  /// Returns the largest numel among tensor inputs (orderedInputs where index
  /// < numInputs). Non-tensor inputs count as 1.
  int64_t largestInput(
      nativert::ExecutionFrame& frame,
      const FormalToActual& map) const;

  /// Recurses through the subgraph and collects outputs that need memory
  /// allocation. For each input, checks the node's argumentMeta to determine
  /// whether the producer's output must be in memory. An output is added when
  /// its returnMeta has isRegister false or when inMemory is true. Returns a
  /// pair of output Values and their OutputDescs, with sizeShortcut,
  /// sizeValues, shapeSetOnDevice, and neededOnHost populated from the
  /// registry's ArgumentMeta.
  void setOutputs(
      NodeCP node,
      const std::unordered_set<ValueCP>& subgraphInputs,
      std::vector<ValueCP>& outputValues,
      std::vector<OutputDesc>& outputDescs,
      bool inMemory);

  NodeCP expr() const {
    return expr_;
  }

  int32_t numInputs() const {
    return numInputs_;
  }

  const std::vector<ValueCP>& orderedInputs() const {
    return orderedInputs_;
  }

  const std::vector<OutputDesc>& outputDescs() const {
    return outputDescs_;
  }

  float unitCost() const {
    return unitCost_;
  }

  int32_t constantAreaOffset() const {
    return constantAreaOffset_;
  }

  const std::vector<std::string>& sharedDeclarations() const {
    return sharedDeclarations_;
  }

  bool alwaysSingleBlock() const {
    return alwaysSingleBlock_;
  }

  bool isGridChoice() const {
    return isGridChoice_;
  }

  void setIsGridChoice(bool value) {
    isGridChoice_ = value;
  }

  const SizeExpr& sizeExpr() const {
    return sizeExpr_;
  }

  std::string toString() const;

  // Hash for (Node*, attrName) pairs used as keys in attrOffsets_.
  struct NodeAttrHash {
    size_t operator()(const std::pair<NodeCP, std::string>& key) const {
      auto h1 = std::hash<const void*>{}(key.first);
      auto h2 = std::hash<std::string>{}(key.second);
      return h1 ^ (h2 + 0x9e3779b9 + (h1 << 6) + (h1 >> 2));
    }
  };

 private:
  /// Translates an ArgumentMeta from the registry into an OutputDesc for a
  /// given node and return index. Copies sizeShortcut, shapeSetOnDevice, and
  /// neededOnHost. Expands sizeArgs ordinals to Value pointers from the node's
  /// inputs: for non-list ordinals takes the single Value, for list ordinals
  /// expands via getListElements(). Wraps reserveShape in a lambda if present.
  OutputDesc makeOutputDesc(
      const ArgumentMeta& returnMeta,
      NodeCP node,
      const std::unordered_set<ValueCP>& subgraphInputs);

  /// Builds a SizeExpr tree for the subgraph rooted at 'node' bounded by
  /// subgraph inputs. For all-elementwise subtrees, produces kMax with distinct
  /// leaf Values. For non-elementwise nodes, recurses on sizeArgs inputs and
  /// uses the node's sizeShortcut as the combining op, flattening children
  /// with the same op.
  SizeExpr makeSizeExpr(
      NodeCP node,
      const std::unordered_set<ValueCP>& subgraphInputs);

  /// Op code, unique within a CompositeKernel.
  int32_t opCode_;

  // The represented computation. 'expr' is the top level result. This graph is
  // a tre, except that it may have multiple occurrences of the same Value in
  // 'inputs'. Not filled in in follow ups or single block variant.
  NodeCP expr_;

  // The compilation context, used for accessing node outputs via
  // CompileCtx::outputs() to handle multiblock variant indirection.
  const CompileCtx& compileCtx_;

  // The Values this takes as input. All these are reachable from
  // 'expr'. This operation is usable for any expr that has the same
  // 'expr' with the structure bounded by the save Values. The
  // graphs, including repeated use of the same values in many
  // places must be strictly isomorphic. a+a does not match a+b even
  // if a and b are of the same type as the a in a+a.
  std::unordered_set<ValueCP> inputs_;

  // Cuda program text, to be inserted into the opCode case of the composite
  // kernel.
  std::string text_;

  std::vector<ValueCP> orderedInputs_;

  // Offset of each input Value in the kernel's BlockInfo::params, starting at
  // 0.
  std::unordered_map<ValueCP, int32_t> paramOffsets_;

  // Offset of each node attribute in BlockInfo::params, following the input
  // param offsets at intervals of 8.
  std::unordered_map<std::pair<NodeCP, std::string>, int32_t, NodeAttrHash>
      attrOffsets_;

  // Shared memory declarations for the kernel.
  std::vector<std::string> sharedDeclarations_;

  std::vector<OutputDesc> outputDescs_;

  SizeExpr sizeExpr_;

  float unitCost_{0};

  int32_t numInputs_{0};

  int32_t constantAreaOffset_{0};

  // Unfusable standalone Nodes that can execute in parallel with each other and
  // the kernel.
  std::vector<NodeCP> standalones_;

  // True if any node in the subgraph has alwaysSingleBlock set in its metadata.
  bool alwaysSingleBlock_{false};

  // True if this kernel is present in both grid_ and singleBlockGrid_ (both
  // grids have a kernel at the corresponding position).
  bool isGridChoice_{false};
};

/// Builds a FormalToActual mapping from formal to actual value ids by walking
/// the formal and actual subgraphs in parallel. Maps inputs positionally and
/// outputs by parallel tree walk, stopping at inputs of formalSg.
FormalToActual makeBindings(
    const Subgraph& formalSg,
    const Subgraph& actualSg,
    const KernelOperation& op);

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
  std::string toString() const;
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

  int32_t singleBlockMaxSize() const {
    return singleBlockMaxSize_;
  }

  std::string toString() const;

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

  // Use single block variant if the largest input is <= than this
  int32_t singleBlockMaxSize_{0};

  // Values from newTensorValue/newScalarValue that appear in this
  // ProjectOperation's grids. These need special binding handling when the
  // ProjectOperation is reused for a different actual subgraph.
  std::vector<ValueCP> extraValues_;
};

struct ValueTypes {
  /// Tensor metadata for tensor type values, indexed by Value id().
  std::vector<const nativert::TensorMeta*> types;
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

  /// Returns occupancy information for the compiled kernel. Returns a
  /// default KernelInfo if no GPU is available.
  facebook::velox::wave::KernelInfo kernelInfo() const;

  std::string toString() const;

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
struct LaunchData {
  LaunchData() = default;
  LaunchData(
      const Launch& launch,
      OpInvocation& op,
      const IdToValueMap& idToValue);

  const Launch* launch{nullptr};
  OpInvocation* op{nullptr};
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
  std::vector<nativert::ValueId> scalarsInFrame;
  std::vector<int32_t> scalarOffsets;
  std::vector<nativert::ValueId> returnValues;
  std::vector<int32_t> returnOffsets;
  // Type kind for each return value, parallel to returnValues.
  std::vector<nativert::Type::Kind> returnTypes;
};

struct CompositeInvocation {
  /// Executes this composite invocation: allocates outputs, builds the grid,
  /// copies params to pinned+device memory, and enqueues the H2D transfer.
  void execute(ExecutionState& state);

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

  std::string toString() const;

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

  std::string toString() const;

 private:
  // The outer array represents parallel launchable sequences kernels. The inner
  // array is a sequence of consecutive kernels.
  std::unique_ptr<CompositeInvocation> kernels_;
};

struct StandaloneStats {
  int64_t micros{0};
};

/// Top level container for result of compiling a FX Graph to torch::wave.
/// Multiple Executors can share the same WaveGraph.
class WaveGraph {
 public:
  /// Analyzes 'graph' and creates an execution plan and fused kernels. The
  /// actual tensor content types and ranks come from 'weights'. Normalizes
  /// the graph first to fill in default attribute values from FunctionSchema.
  WaveGraph(nativert::Graph& graph, ValueTypes types);

  ~WaveGraph();

  /// Returns an ExecutionState from the pool, creating one if needed.
  std::unique_ptr<ExecutionState> getState();

  /// Returns 'state' to the pool for reuse.
  void returnState(std::unique_ptr<ExecutionState> state);

  const std::vector<std::unique_ptr<CompiledNode>>& nodes() const {
    return nodes_;
  }

  ValueTypes& types() {
    return types_;
  }

  /// Adds an output to 'node' with the given name and dtype, registers a
  /// TensorMeta entry in types_ for it. The value is recorded in
  /// createdValueDtypes_ for later duplication.
  nativert::Value* newTensorValue(
      nativert::Node* node,
      std::string_view name,
      c10::ScalarType dtype);

  /// Adds a scalar output to 'node' with the given name and dtype.
  /// No TensorMeta is created for this value. The value is recorded in
  /// createdValueDtypes_ for later duplication.
  nativert::Value* newScalarValue(
      nativert::Node* node,
      std::string_view name,
      c10::ScalarType dtype);

  /// Registers a TensorMeta entry for 'value' in types_ with the given dtype.
  void registerTensorMeta(ValueCP value, c10::ScalarType dtype);

  /// Returns true if 'value' was created by newTensorValue or newScalarValue.
  bool isCreatedValue(ValueCP value) const;

  /// Creates a new Value with the same type and dtype as 'original', attached
  /// to an internal placeholder node. Registers it in idToValue_.
  nativert::Value* duplicateValue(ValueCP original);

  /// Returns the underlying nativert graph.
  GraphP graph() {
    return graph_;
  }

  /// Returns the pre-built map from Value::id() to Value*.
  const IdToValueMap& idToValue() const {
    return idToValue_;
  }

  /// Fills in missing attribute defaults from FunctionSchema and creates
  /// multiKernelVariants_ for nodes that have one.
  void normalizeAndAnnotateGraph();

  /// Returns the multikernel variant subgraph for 'node', or nullptr if none.
  const Subgraph* multiKernelVariant(NodeCP node) const {
    auto it = multiKernelVariants_.find(node);
    return it != multiKernelVariants_.end() ? &it->second : nullptr;
  }

  /// Returns the next composite invocation sequence number, starting at 0.
  int32_t nextCompositeInvocationId() {
    return nextCompositeInvocationId_++;
  }

  const std::unordered_set<nativert::ValueId>& syncableValueIds() const {
    return syncableValueIds_;
  }

  void addSyncableValueId(nativert::ValueId id) {
    syncableValueIds_.insert(id);
  }

  const std::unordered_map<NodeCP, int32_t>& standaloneIndices() const {
    return standaloneIndices_;
  }

  std::vector<StandaloneStats>& standaloneStats() {
    return standaloneStats_;
  }

  const std::vector<StandaloneStats>& standaloneStats() const {
    return standaloneStats_;
  }

  std::string toString() const;

 private:
  // The executable graph. the nodes are executed sequentially. Each node has
  // internal prallelism.
  std::vector<std::unique_ptr<CompiledNode>> nodes_;

  ValueTypes types_;

  GraphP graph_{nullptr};

  // Pre-built map from Value::id() to Value* for fast lookups.
  IdToValueMap idToValue_;

  // Owns TensorMeta objects created by newTensorValue so pointers in
  // types_.types remain valid.
  std::vector<std::unique_ptr<nativert::TensorMeta>> metaStorage_;

  // Tracks the c10::ScalarType for each value created by
  // newTensorValue/newScalarValue, enabling duplication of these values.
  std::unordered_map<ValueCP, c10::ScalarType> createdValueDtypes_;

  // Placeholder node used by duplicateValue to attach new Values.
  nativert::Node* placeholderNode_{nullptr};

  // For nodes that have a multikernel implementation, like multiblock
  // reduction, this gives the subgraph to substitute for the Node when
  // generating the multiblock case of a ProjectOperation.
  std::unordered_map<NodeCP, Subgraph> multiKernelVariants_;

  // Counter for assigning sequence numbers to CompositeInvocations.
  int32_t nextCompositeInvocationId_{0};

  // Maps each actual standalone Node to a serial index (0-based).
  std::unordered_map<NodeCP, int32_t> standaloneIndices_;

  // Accumulated timing for each standalone, indexed by standaloneIndices_.
  std::vector<StandaloneStats> standaloneStats_;

  // ValueIds of outputs whose Metadata has shapeSetOnDevice or neededOnHost.
  std::unordered_set<nativert::ValueId> syncableValueIds_;

  // Pool of reusable ExecutionState objects.
  std::mutex statePoolMutex_;
  std::vector<std::unique_ptr<ExecutionState>> statePool_;
};

} // namespace torch::wave
