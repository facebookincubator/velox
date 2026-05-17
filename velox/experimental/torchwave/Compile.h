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

#include <atomic>
#include <deque>

#include <ATen/core/ivalue.h>
#include <torch/nativert/graph/TensorMeta.h>
#include "velox/experimental/torchwave/CompiledOp.h"
#include "velox/experimental/torchwave/ParallelExpr.h"
#include "velox/experimental/torchwave/Registry.h"

namespace torch::wave {

struct ResultSpec {
  ValueCP value{nullptr};
  std::string variable;
};

/// If FLAGS_elt_trace is on, appends an if (threadIdx.x == 0) {printf(...)}
/// statement to 'ss'.
void eltTrace(std::stringstream& ss, std::string_view printf);

bool subgraphsMatch(const Subgraph& left, const Subgraph& right);

/// Adds self-bindings for extra values to an OpInvocation (new ProjectOp case).
void addSelfExtraBindings(
    OpInvocation& op,
    const std::vector<ValueCP>& extraValues);

/// Duplicates extra values and adds formal-to-actual bindings (reused
/// ProjectOp case).
void addDuplicateExtraBindings(
    OpInvocation& op,
    const std::vector<ValueCP>& formalExtras,
    WaveGraph& waveGraph);

struct SubgraphHash {
  size_t operator()(const Subgraph& sg) const;
};

struct SubgraphEqual {
  bool operator()(const Subgraph& left, const Subgraph& right) const {
    return subgraphsMatch(left, right);
  }
};

using SubgraphMap = std::
    unordered_map<Subgraph, ProjectOperation*, SubgraphHash, SubgraphEqual>;

using SubgraphKernelMap =
    std::unordered_map<Subgraph, KernelOperation*, SubgraphHash, SubgraphEqual>;

enum class Context { kTop, kFused, kFusedBreak, kStandalone };

/// Mode for variantSubgraph: kSingle copies as-is, kMulti and kCG expand
/// nodes that have a multikernel or cg variant, respectively.
enum class VariantMode { kSingle, kMulti, kCG };

class CompileCtx {
 public:
  using NodeSet = std::unordered_set<NodeCP>;

  explicit CompileCtx(WaveGraph& waveGraph)
      : waveGraph_(waveGraph), types_{waveGraph.types()} {}

  WaveGraph& waveGraph() {
    return waveGraph_;
  }

  std::unique_ptr<CompiledNode> compileNode(ProjectNode& node);

  ProjectOperation* makeProjectionOperation(const Subgraph& sg);

  /// Clears per-grid state (placed_, grid_) so that makeGrid() starts fresh.
  /// Keeps project ops and kernel ops.
  void newGrid();

  LaunchGrid makeGrid(NodeCP node);

  bool isSingleBlock() const {
    return isSingleBlock_;
  }

  void setIsSingleBlock(bool value) {
    isSingleBlock_ = value;
  }

  bool isCgGrid() const {
    return isCgGrid_;
  }

  void setIsCgGrid(bool value) {
    isCgGrid_ = value;
  }

  /// Returns the next unique opcode for a KernelOperation.
  int32_t nextOpCode() {
    return nextOpCode_++;
  }

  /// Returns the unique leaf input Values for a set of subgraphs. Walks from
  /// each subgraph root following inputs, adding a Value if it is in the
  /// subgraph's own inputs list or its producer is in placed_.
  std::vector<ValueCP> subgraphInputs(
      const std::vector<Subgraph>& subgraphs) const;

  void generateElementwise(
      const std::vector<Subgraph>& subgraphs,
      const std::vector<ResultSpec>& resultSpecs,
      std::string resultStmt = "",
      bool fullBlockResult = false);

  /// Recurses through inputs of 'node', stopping at placed_ and inputs of
  /// generatingOp_'s subgraph. Calls fusedCode on non-elementwise ops with
  /// result specs set to the output Values of the node.
  void generateElementwiseBorder(NodeCP node);

  void generateIndexToOffset(
      const ElementExpr& ee,
      const std::vector<ValueCP>& allInputs);

  void fusedCode(NodeCP node, std::vector<ResultSpec>& resultSpecs);

  /// Returns true if any node reachable from 'value' within the current
  /// generatingOp_ has an output with shapeSetOnDevice.
  bool isSizeSetInThisOp(ValueCP value, std::unordered_set<ValueCP>& visited);

  void functionLoop(NodeCP node);

  std::string elementwiseExpr(
      NodeCP node,
      const KernelOperation& op,
      const std::vector<ValueCP>& inputs,
      bool slowPath = false);

  /// Generates a barrier followed by a __view call for 'dest' as a view of
  /// 'src' at the element offset given by 'offsetExpr'. Uses __syncthreads
  /// for single block mode, OpBarrier otherwise.
  void callView(
      ValueCP src,
      ValueCP dest,
      const std::string& offsetExpr,
      int32_t elementSize);

  /// Emits a __copy<T> call that copies from 'source' (possibly strided) into
  /// contiguous 'dest' at the byte offset given by 'destOffsetExpr' elements.
  void emitCopy(
      ValueCP source,
      ValueCP dest,
      const std::string& destOffsetExpr,
      const std::string& cudaTypeName);

  void emitCode(std::string_view text);

  void emitBarrier();

  void addInclude(std::string_view header);

  std::string declareAttributes(
      NodeCP node,
      const KernelOperation& op,
      const std::vector<ValueCP>& inputs);

  /// Emits setup code for a ScalarList parameter from either a prim.ListPack
  /// value or a constant vector<int64_t> attribute. Allocates space in the
  /// alt params area and returns the setup code string.
  std::string emitScalarListSetup(
      size_t argOrdinal,
      ValueCP value,
      const nativert::Attribute* attr,
      NodeCP node);

  /// Generates a device function call string from the node's Metadata. Emits
  /// the deviceFunc name, optional type template parameters, input/output
  /// params, attributes in schema argument order, shared declarations, and
  /// blockInfo.
  std::string makeCall(
      NodeCP node,
      std::vector<ResultSpec> inputs,
      std::vector<ResultSpec> outputs);

  std::string cudaType(ValueCP value) const;

  /// Returns a CUDA expression for accessing a value's parameter, e.g.
  /// "param<Tensor>(blockInfo, 16)".
  std::string param(ValueCP value, const KernelOperation& op) const;

  /// Returns a CUDA expression for a register-passed element reference. For
  /// tensors generates "elementRef<T>(param<Tensor>(blockInfo, off), idx)", for
  /// scalars generates "*param<T>(blockInfo, off)".
  std::string makeElementRef(ValueCP value, const KernelOperation& op) const;

  /// Declares a temporary variable of the CUDA type for 'scalarType'. Appends a
  /// declaration line to declarations_ and returns the variable name.
  std::string declare(c10::ScalarType scalarType);

  /// Declares a temporary variable matching the Value's type (tensor dtype or
  /// scalar type like int32_t, float, bool).
  std::string declareTemp(ValueCP value);

  Subgraph extractSubgraph(NodeCP node, const NodeSet& inputs, NodeSet& placed);

  bool isElementWise(const nativert::Node& node, const NodeSet& placed = {})
      const;

  bool isSingleBlock(const nativert::Node& node, const NodeSet& placed = {})
      const;

  bool isMultikernel(const nativert::Node& node, const NodeSet& placed = {})
      const;

  Context placeKernels(NodeCP node, Context context);

  void pushdownStandalone(NodeCP node);

  void pushdownFused(NodeCP node);

  std::unique_ptr<KernelOperation> generateFused(const Subgraph& sg);

  void generateFusedInner(const Subgraph& sg);

  /// Fills launch.constantIndices by mapping each attribute in the actual
  /// subgraph to the corresponding index in the project-level constants.
  void fillConstantIndices(const Subgraph& sg, Launch& launch);

  void placeKernelLaunch(Launch launch);

  static int32_t nextKernelId();

  KernelOperation* generatingOp() const {
    return generatingOp_;
  }

  void markPlaced(NodeCP node) {
    placed_.insert(node);
    generatingOp_->allNodes().insert(node);
  }

  bool isPlaced(NodeCP node) const {
    return placed_.count(node);
  }

  /// Returns the original graph node for a variant subgraph copy, or
  /// 'node' itself if not a copy.
  NodeCP originalFromVariant(NodeCP node) const {
    auto it = variantToOriginal_.find(node);
    return it != variantToOriginal_.end() ? it->second : node;
  }

  /// Creates a new nativert::Graph with a deep copy of the contents of 'sg'.
  /// In kSingle mode, copies as-is. In kMulti/kCG, expands nodes that have
  /// a multikernel or cg variant. The graph is owned by the WaveGraph being
  /// constructed. Returns a Subgraph whose root and inputs point into the
  /// new graph.
  Subgraph variantSubgraph(const Subgraph& sg, VariantMode mode);

 private:
  static std::atomic<int32_t> kernelCounter_;

  template <typename Func>
  bool allReachable(
      const nativert::Node& node,
      const NodeSet& placed,
      Func&& predicate,
      NodeSet& visited) const;

  template <typename Func>
  bool anyReachable(
      const nativert::Node& node,
      const NodeSet& placed,
      Func&& predicate,
      NodeSet& visited) const;

  void collectSubgraphInputs(
      NodeCP node,
      const std::unordered_set<ValueCP>& sgInputs,
      std::unordered_set<ValueCP>& seen,
      std::vector<ValueCP>& result) const;

  void generateElementwiseBorderImpl(
      NodeCP node,
      const std::unordered_set<ValueCP>& opInputs,
      NodeSet& visited);

  void elementwiseExprImpl(
      NodeCP node,
      const std::unordered_set<ValueCP>& inputSet,
      const std::vector<ValueCP>& inputs,
      const KernelOperation& op,
      std::stringstream& ss,
      bool slowPath);

  /// Marks matching kernel ops in grid_ and singleBlockGrid_ as grid choices.
  void setGridChoice(ProjectOperation* projectOp);

  /// Scans grids for values created by newTensorValue/newScalarValue and
  /// stores them in projectOp->extraValues_.
  void collectExtraValues(ProjectOperation* projectOp);

  WaveGraph& waveGraph_;
  const ValueTypes& types_;
  bool isSingleBlock_{false};
  bool isCgGrid_{false};
  int32_t nextOpCode_{0};
  std::unordered_set<std::string> includes_;
  std::stringstream code_;
  std::stringstream declarations_;
  int32_t declareCounter_{0};
  const std::unordered_set<NodeCP>* inputs_;
  NodeSet placed_;
  NodeSet placedBeforeNode_;
  NodeSet standaloneNodes_;

  // Offset of param corresponding to Value in the kernel's BlockInfo::params.
  std::unordered_map<ValueCP, int32_t> valueParamOffset_;

  std::vector<OpInvocation> ops_;
  SubgraphMap projectOps_;
  // Stable storage for ProjectOperations so pointers remain valid.
  std::vector<std::unique_ptr<ProjectOperation>> opStorage_;
  std::vector<std::unique_ptr<KernelOperation>> kernelOpStorage_;
  /// Stable storage for IValues so OpInvocation::constants_ pointers remain
  /// valid.
  std::deque<c10::IValue> ivalueStorage_;
  LaunchGrid grid_;

  // Distinct kernel ops for the ProjectOperation being made.
  SubgraphKernelMap projectKernelOps_;

  /// The Subgraph for the ProjectOperation being made.
  const Subgraph* projectOpSubgraph_{nullptr};

  /// Map from node ordinal to constant index for the current ProjectOperation.
  std::unordered_map<int32_t, int32_t> constantMap_;

  // The KernelOperation for which code is being generated.
  KernelOperation* generatingOp_{nullptr};

  // Intermediates within 'generatingOp_' that are backed by device memory.
  std::unordered_set<ValueCP> memoryValues_;

  const ElementExpr* currentElementExpr_{nullptr};

  // Maps each index in leafInputs/allInputs to its tensor-only bit position
  // in the isFastPath bitmask, or -1 for non-tensor inputs.
  std::vector<int32_t> fastPathBitIndex_;

  // Maps variant subgraph copy nodes back to original graph nodes.
  std::unordered_map<NodeCP, NodeCP> variantToOriginal_;
};

} // namespace torch::wave
