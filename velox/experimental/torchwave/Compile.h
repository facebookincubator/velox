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

#include <folly/container/F14Map.h>
#include <atomic>
#include <deque>
#include <map>
#include <tuple>

#include <ATen/core/ivalue.h>
#include <torch/nativert/graph/TensorMeta.h>
#include "velox/experimental/torchwave/CompiledOp.h"
#include "velox/experimental/torchwave/ParallelExpr.h"
#include "velox/experimental/torchwave/Registry.h"
#include "velox/experimental/torchwave/WaveConfig.h"
#include "velox/experimental/torchwave/WaveGraph.h"

namespace torch::wave {

/// Specifies how the result of a compiled expression is delivered. If value
/// is set, the result is written to memory as an element of the tensor
/// identified by value. If variable is set instead, the result is kept as a
/// device-side local (register) under that name, available for consumption
/// by the next fused operation without passing through memory.
struct ResultSpec {
  ValueCP value{nullptr};
  std::string variable;
};

inline std::vector<ResultSpec> outputSpecs(NodeCP node) {
  std::vector<ResultSpec> specs;
  specs.reserve(node->outputs().size());
  for (auto* output : node->outputs()) {
    specs.push_back({output, {}});
  }
  return specs;
}

inline std::vector<ResultSpec> inputSpecs(NodeCP node) {
  std::vector<ResultSpec> specs;
  specs.reserve(node->inputs().size());
  for (const auto& input : node->inputs()) {
    specs.push_back({input.value, {}});
  }
  return specs;
}

// Bits per word for the isFastPath bitmask in elementwise codegen.
constexpr int32_t kBitsPerWord = 32;

/// If WaveConfig::kernelDebugOutput is on, appends an if (threadIdx.x == 0)
/// {printf(...)} statement to 'ss'.
void eltTrace(std::stringstream& ss, std::string_view printf);

std::string cudaAttrType(const nativert::Constant& c);

std::string presentTemplateParams(const Metadata& meta, NodeCP node);

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

/// Hash functor for Subgraph, used to deduplicate identical subgraphs.
struct SubgraphHash {
  size_t operator()(const Subgraph& sg) const;
};

/// Equality functor for Subgraph, used with SubgraphHash.
struct SubgraphEqual {
  bool operator()(const Subgraph& left, const Subgraph& right) const {
    return subgraphsMatch(left, right);
  }
};

using SubgraphMap = std::
    unordered_map<Subgraph, ProjectOperation*, SubgraphHash, SubgraphEqual>;

enum class Context { kTop, kFused, kFusedBreak, kStandalone };

/// Mode for variantSubgraph: kSingle copies as-is, kMulti and kCG expand
/// nodes that have a multikernel or cg variant, respectively.
enum class VariantMode { kSingle, kMulti, kCG };

/// Compilation context that translates a WaveGraph into CUDA kernel code.
class CompileCtx {
 public:
  using NodeSet = std::unordered_set<NodeCP>;

  explicit CompileCtx(WaveGraph& waveGraph)
      : waveGraph_(waveGraph),
        types_{waveGraph.types()},
        allStandalone_{WaveConfig::get().allStandalone} {}

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
      const std::string& resultStmt = "",
      bool fullBlockResult = false,
      // Output Value of a data-dependent scan op (masked_select / nonzero)
      // whose size is written on device; its dims[0] is forced to 0 for an
      // empty input (the element loop that would set it runs zero iterations).
      ValueCP FOLLY_NULLABLE shapeSetOnDeviceResult = nullptr);

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

  void elementwiseExpr(
      ValueCP value,
      const std::string& resultName,
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

  /// Appends text at translation-unit scope, ahead of the kernel function and
  /// inside namespace torch::wave. For an op that codegens something its own
  /// call names -- a type, a constant table -- rather than passes as data.
  void emitHelperCode(std::string_view text);

  /// A number unique within this translation unit, for naming what
  /// emitHelperCode declares. Two congruent nodes share one KernelOperation
  /// and so ask once, but two different ones may want the same declaration
  /// under different names.
  int32_t nextHelperId() {
    return helperId_++;
  }

  void emitBarrier();

  /// Returns true if 'node' reads any input whose producer ran earlier in the
  /// same kernel (generatingOp_) without an intervening barrier.
  bool callNeedsBarrier(NodeCP node);

  /// Returns true if reading 'operand' from memory needs a barrier first:
  /// something earlier in this same kernel writes the storage it names and no
  /// barrier has separated the two. 'consumer' is the node about to read it,
  /// which is not counted as a writer of its own input.
  bool valueNeedsBarrier(ValueCP operand, NodeCP consumer);

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

  /// Returns a reusable temp variable name for the Value's CUDA type. If the
  /// pool for that type is empty, allocates a new name like temp_<type>_<n>.
  /// Call tempDone() when the variable is no longer needed to return it to the
  /// pool.
  std::string useTemp(ValueCP value);

  /// Returns a temp variable name to the reusable pool for the Value's type.
  void tempDone(ValueCP value, const std::string& name);

  Subgraph extractSubgraph(NodeCP node, const NodeSet& inputs, NodeSet& placed);

  bool isElementWise(const nativert::Node& node, const NodeSet& placed = {})
      const;

  bool isSingleBlock(const nativert::Node& node, const NodeSet& placed = {})
      const;

  bool isMultikernel(const nativert::Node& node, const NodeSet& placed = {})
      const;

  Context placeKernels(NodeCP node, Context context);

  void pushdownStandalone(NodeCP node);

  /// Builds the kernel op for 'node's subgraph and places its launch, no
  /// earlier than 'minLevel'. Returns the step it landed in.
  int32_t pushdownFused(NodeCP node, int32_t minLevel = -1);

  /// Ends the kernel of every op in 'value's producer chain whose output extent
  /// the host cannot work out ahead of the launch -- settled on device, or
  /// known only to the producer's own reserve function -- so the value is
  /// materialized in an earlier step and its extent is an ordinary frame
  /// tensor's by the time the consuming launch sizes its outputs. Used for a
  /// cat / stack that must lay its result out on the host: it needs every
  /// operand's extent at one point, so one unmeasurable operand would refuse
  /// the whole concat.
  void breakUnmeasurableProducers(ValueCP value);

  /// Places 'producer' and its own inputs, then emits 'producer' as its own
  /// kernel launch so a consumer reads its output as a materialized border
  /// across a kernel boundary. Used where the whole of 'producer's output must
  /// be visible before the consumer runs, but an in-kernel barrier (which
  /// forces a cooperative, whole-grid-resident launch) is undesirable.
  void breakProducerIntoOwnKernel(NodeCP producer);

  /// Emits the expression that computes one concat operand as its own kernel
  /// launch, so it is sized by that operand and gets its own share of the grid
  /// instead of running as one link in a chain of copies inside the concat's
  /// kernel. Every op the pushdown creates is declared to write
  /// 'concatOutput', which is what orders a reader of the concat result after
  /// all of the operands that fill it. False when the operand has no expression
  /// here to push down, which is what leaves it to be copied instead.
  bool breakConcatOperandIntoOwnKernel(ValueCP operand, ValueCP concatOutput);

  /// The value a copy of 'concat's operand at 'occurrence' writes: a band of
  /// the result, bound at run time by reserveConcatOutput. Creates the copy
  /// node that holds the source, and the value on the first grid variant to
  /// ask. Null when the result has no type to take a dtype from.
  nativert::Value*
  makeConcatCopyDestination(NodeCP concat, int32_t occurrence, ValueCP operand);

  /// Emits the copy that fills one concat operand's band, for an operand that
  /// exists before the concat's result is sized -- a graph input, or a value an
  /// earlier step materialized -- so there is no producing expression to push
  /// down and no write of its own to redirect. Returns false when the operand
  /// has no destination, leaving it to the concat's kernel.
  ///
  /// The code is generated once per element type per composite kernel: the
  /// first copy builds the kernel op, and every later one is another Launch of
  /// that op with its own source and destination. Two hundred copies of one
  /// __copy would otherwise be two hundred bodies in the kernel.
  bool emitConcatOperandCopy(
      ValueCP operand,
      ValueCP destination,
      ValueCP concatOutput,
      int32_t minLevel);

  std::unique_ptr<KernelOperation> generateFused(const Subgraph& sg);

  void generateFusedInner(const Subgraph& sg);

  /// Fills launch.constantIndices by mapping each attribute in the actual
  /// subgraph to the corresponding index in the project-level constants.
  void fillConstantIndices(const Subgraph& sg, Launch& launch);

  /// Places 'launch' one step after the last of its inputs, but never before
  /// launch.minLevel. Returns the step it landed in.
  int32_t placeKernelLaunch(Launch launch);

  /// Returns the next kernel id for this compilation. The counter is per
  /// CompileCtx (one per WaveGraph construction), so kernel names are
  /// deterministic per graph regardless of how many graphs compile
  /// concurrently. This keeps NVRTC cache keys stable (warm-cache hits) and
  /// makes parallel compilation of different configs well-defined.
  int32_t nextKernelId();

  KernelOperation* generatingOp() const {
    return generatingOp_;
  }

  /// The single concat operand this op fills, or -1 when the op is the whole
  /// concat. Set while generating one of the per-operand ops that
  /// parallelConcatFill splits a wide cat / stack into: the special form then
  /// emits the write for that operand alone instead of walking every operand in
  /// one body, which is what keeps each op's parameters to just its own source.
  int32_t concatOperandIndex() const {
    return concatOperandIndex_;
  }

  void setConcatOperandIndex(int32_t index) {
    concatOperandIndex_ = index;
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

  /// The value the operand at 'occurrence' of 'concat' is copied into, or -1 if
  /// nothing has asked for it yet. Keyed by the ORIGINAL concat node: the grid
  /// variants are placed one after another over their own copies of the graph,
  /// and the id names a frame slot, so a variant that minted its own would fill
  /// a slot the other variants' layouts never read.
  nativert::ValueId concatCopyDest(NodeCP concat, int32_t occurrence) const {
    auto it = concatCopyDest_.find({concat, occurrence});
    return it != concatCopyDest_.end() ? it->second : -1;
  }

  void setConcatCopyDest(
      NodeCP concat,
      int32_t occurrence,
      nativert::ValueId destination) {
    concatCopyDest_[{concat, occurrence}] = destination;
  }

  /// True when an op of its own fills the band of 'concat's operand at
  /// 'occurrence', so the concat's kernel emits nothing for it.
  bool concatOperandIsCopied(NodeCP concat, int32_t occurrence) const {
    return concatCopyDest(originalFromVariant(concat), occurrence) >= 0;
  }

  /// Whether one operand of a fused concat writes its own band of the result,
  /// and what decided it. Taken once, while the concat is placed, and read
  /// again by the concat's code generation and by the allocation-group pass, so
  /// the two cannot disagree: an operand one carves and the other copies either
  /// writes through a frame slot nothing bound or overwrites what the copy
  /// moved there.
  struct ConcatCarve {
    /// The allocation group gives this operand a view of its band instead of a
    /// buffer, because a launch other than the concat's own fills it.
    bool groupCarves{false};

    /// A copy op of this operand's own moves it into its band. False both for
    /// an operand the group carves and for one the concat's own kernel writes
    /// in place -- that one is bound to its band by reserveConcatOutput and is
    /// neither a group member nor a copy.
    bool needsCopy{false};

    /// The value the producing launch writes, when the operand is reached
    /// through list plumbing and so is not that value itself. See
    /// ConcatInputInfo::writerId, which this feeds. -1 otherwise.
    nativert::ValueId writerId{-1};

    /// Why, in the words the per-concat report prints.
    std::string reason;
  };

  /// True when 'operand' should be left for 'concat's own kernel to compute
  /// rather than pushed into a kernel of its own a step earlier, because its
  /// producer can write the operand's band directly and the concat is its only
  /// reader. Gated on WaveConfig::concatOperandsInPlace.
  bool concatOperandFusesInPlace(
      ValueCP operand,
      NodeCP concat,
      int64_t dim,
      c10::ScalarType resultDtype) const;

  /// Decides, for every operand of 'concat', whether its producer fills the
  /// operand's band of the result or a copy moves it there.
  ///
  /// The result's layout can be computed at the latest point any operand's
  /// dimensions become known. An operand whose buffer is filled before that
  /// point was already materialized when the result came into being, so there
  /// is no write left to redirect and it is copied; one filled at that point
  /// writes its band directly. So is everything with no write of its own to
  /// give: a view, a value no wave kernel launch writes, one another concat
  /// already carved.
  ///
  /// Must run after every operand's producer has been placed, since that is
  /// what fixes the points it reads.
  void decideConcatCarve(
      NodeCP concat,
      const std::vector<ValueCP>& operands,
      int64_t dim,
      c10::ScalarType resultDtype);

  /// Where the result of 'concat' is laid out, as decideConcatCarve settled it,
  /// or null before the concat has been placed. The allocation group is created
  /// there, and the carve verdicts are expressed against it.
  const WaveGraph::SchedulePoint* concatLayoutPoint(NodeCP concat) const {
    const auto it = concatLayoutPoint_.find(originalFromVariant(concat));
    return it == concatLayoutPoint_.end() ? nullptr : &it->second;
  }

  /// The earliest point the host could know every operand's extent, as opposed
  /// to concatLayoutPoint, which is where the last operand's DATA lands. Null
  /// when no operand puts a floor under it, absent before the concat is placed.
  ///
  /// Not what the layout uses today: the allocation-group collector is built
  /// per step and can only intercept a member sized in its own step, so a group
  /// moved to this earlier point loses the members written later. Recorded so
  /// the gap is visible and so a collector able to span steps has it ready.
  const std::optional<WaveGraph::SchedulePoint>* concatShapePoint(
      NodeCP concat) const {
    const auto it = concatShapePoint_.find(originalFromVariant(concat));
    return it == concatShapePoint_.end() ? nullptr : &it->second;
  }

  /// The decision for the operand at 'occurrence' of 'concat', or null before
  /// the concat has been placed. Keyed by the ORIGINAL concat node, as
  /// concatCopyDest is and for the same reason.
  const ConcatCarve* concatCarve(NodeCP concat, int32_t occurrence) const {
    const auto it =
        concatCarve_.find({originalFromVariant(concat), occurrence});
    return it == concatCarve_.end() ? nullptr : &it->second;
  }

  /// True when a copy op of its own has to move the operand at 'occurrence'
  /// into its band.
  bool concatOperandNeedsCopyOp(NodeCP concat, int32_t occurrence) const {
    const auto* decision = concatCarve(concat, occurrence);
    return decision != nullptr && decision->needsCopy;
  }

 private:
  // Per-CompileCtx (one per WaveGraph construction), not process-wide, so no
  // atomicity is needed: concurrent compilations use distinct CompileCtx
  // instances, keeping kernel ids deterministic per graph for NVRTC cache-key
  // stability.
  int32_t kernelCounter_{0};

  // Records where 'launch' writes each of its outputs and where each becomes
  // measurable, for writtenAt() and realizedAt(). Called as the launch lands,
  // which is the only point that knows the step.
  void recordSchedulePoints(
      const Launch& launch,
      int32_t step,
      const FormalToActual& bindings,
      bool intoGrid);

  // Where 'id' is written and measured, taking the grid being built first: a
  // value this grid's earlier launches write is not in the graph-wide maps
  // until the op is invoked. Null when no launch anywhere writes it.
  const WaveGraph::SchedulePoint* writtenPoint(nativert::ValueId id) const;
  const WaveGraph::SchedulePoint* realizedPoint(nativert::ValueId id) const;

  // Records the schedule points of every launch of 'op's grid, in 'op's own
  // values. Called as each invocation is created: the grid is built over the
  // formal subgraph, and only an invocation knows which frame values its
  // launches actually write.
  void recordInvocationSchedulePoints(OpInvocation& op);

  // True when the expression that computes 'operand' is still unplaced and is
  // not a boundary input, so the kernel being built writes it. Such a value has
  // no schedule point yet: the launch that writes it is placed once the whole
  // expression has been walked.
  bool computedByThisKernel(ValueCP operand) const;

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
      ValueCP value,
      const std::string& resultName,
      const std::unordered_set<ValueCP>& inputSet,
      const std::vector<ValueCP>& inputs,
      const KernelOperation& op,
      bool slowPath);

  std::string formatLeafAccess(
      ValueCP value,
      const std::vector<ValueCP>& inputs,
      const KernelOperation& op,
      bool slowPath);

  std::string buildElementwiseCall(
      const Metadata& meta,
      NodeCP node,
      const KernelOperation& op,
      const std::vector<std::string>& argTexts);

  void maybeExtractOutOfLine(
      ValueCP value,
      const std::string& resultName,
      const std::vector<ValueCP>& inputs,
      size_t codeStart,
      size_t tempLogStart,
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
  // If true, every node is compiled as standalone (no fusion).
  bool allStandalone_{false};
  int32_t nextOpCode_{0};
  // CUDA headers to #include in the generated translation unit.
  std::unordered_set<std::string> includes_;
  // Accumulates the body of the generated CUDA kernel function.
  std::stringstream code_;
  // Accumulates variable declarations emitted before the kernel body.
  std::stringstream declarations_;
  int32_t declareCounter_{0};
  // Boundary input nodes of the subgraph currently being generated.
  const std::unordered_set<NodeCP>* inputs_{nullptr};
  // Nodes whose code has already been emitted in the current kernel.
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

  // Write and realization points of the launches placed in the grid being
  // built, keyed by the values of the subgraph being placed. A launch of this
  // grid has no invocation yet -- one is created only once the whole op is
  // built -- so its outputs are absent from the graph-wide maps until then,
  // and a concat placed in this same grid has to read them from here. Cleared
  // per grid by newGrid().
  folly::F14FastMap<nativert::ValueId, WaveGraph::SchedulePoint> gridWrittenAt_;
  folly::F14FastMap<nativert::ValueId, WaveGraph::SchedulePoint>
      gridRealizedAt_;

  // Set while a project op is built if any concat in it took a carve decision.
  // Such an op is never registered for deduplication -- see the comment at the
  // registration site.
  bool opCarvesAConcat_{false};

  // Position of the node being compiled among the CompiledNodes produced so
  // far, which is the index the allocation-group plan plans by. Advanced only
  // by a compileNode that returns a node, so a node that compiles to nothing
  // does not consume an index.
  int32_t compileNodeIndex_{0};

  /// The Subgraph for the ProjectOperation being made.
  const Subgraph* projectOpSubgraph_{nullptr};

  /// Map from node ordinal to constant index for the current ProjectOperation.
  std::unordered_map<int32_t, int32_t> constantMap_;

  // The KernelOperation for which code is being generated.
  KernelOperation* generatingOp_{nullptr};

  // See concatOperandIndex().
  int32_t concatOperandIndex_{-1};

  // See concatCopyDest(). std::map because the key is a pair; there is one
  // entry per copied concat operand, not one per value.
  std::map<std::pair<NodeCP, int32_t>, nativert::ValueId> concatCopyDest_;

  // See concatLayoutPoint(), keyed by the original concat node.
  std::map<NodeCP, WaveGraph::SchedulePoint> concatLayoutPoint_;
  std::map<NodeCP, std::optional<WaveGraph::SchedulePoint>> concatShapePoint_;

  // See concatCarve(), keyed the same way.
  std::map<std::pair<NodeCP, int32_t>, ConcatCarve> concatCarve_;

  // Operands some concat already carves. One buffer cannot be two bands, so
  // the first concat in graph order takes it and any later one copies.
  std::unordered_set<nativert::ValueId> carvedOperands_;

  // The kernel op generated for the first concat-operand copy of each element
  // type, so the rest are launches of that same op with their own parameters
  // rather than another copy of the code. See emitConcatOperandCopy.
  std::map<c10::ScalarType, KernelOperation*> concatCopyOp_;

  // Why each wide-cat operand did or did not get a kernel op of its own,
  // tallied per concat result id under the timing trace. An operand left fused
  // into the concat is an interior value no launch writes, which the
  // allocation group cannot carve into a band.
  std::map<int32_t, std::map<std::string, int32_t>> concatPushdownSkips_;

  // Intermediates within 'generatingOp_' that are backed by device memory.
  std::unordered_set<ValueCP> memoryValues_;

  const ElementExpr* currentElementExpr_{nullptr};

  // The subgraph root output of the elementwise expression currently being
  // generated (the tensor the loop writes at 'idx'). Passed as the output
  // argument to device functions whose ElementwiseOp has hasOutputArg set.
  ValueCP currentRootOutput_{nullptr};

  // Maps each index in leafInputs/allInputs to its tensor-only bit position
  // in the isFastPath bitmask, or -1 for non-tensor inputs. A value of -1
  // must never be used to compute bit positions — all code paths that read
  // fastPathBitIndex_[i] must first verify the value is a tensor.
  std::vector<int32_t> fastPathBitIndex_;

  // Maps allInputs index to variable name (e.g. "b0") when elementwise
  // variables are generated. When empty, elementwiseExprImpl inlines the
  // storage expression instead.
  std::unordered_map<size_t, std::string> elementwiseVarNames_;

  // Output values of nodes placed before the last emitBarrier. A
  // randomAccess input whose value is in this set does not need a new
  // barrier because its producer is already separated by one.
  std::unordered_set<ValueCP> preBarrierValues_;

  // Maps variant subgraph copy nodes back to original graph nodes.
  std::unordered_map<NodeCP, NodeCP> variantToOriginal_;

  // Pool of reusable temp variable names per CUDA type string.
  std::unordered_map<std::string, std::vector<std::string>> tempNames_;
  // Counter of allocated temp names per CUDA type string.
  std::unordered_map<std::string, int32_t> typeTemps_;

  // Log of (type, name) for each temp used during elementwise expr generation.
  // Used to identify which temps belong to extracted out-of-line helpers.
  std::vector<std::pair<std::string, std::string>> tempUseLog_;

  // Sequential counter for out-of-line elementwise helper functions.
  int32_t outOfLineCounter_{0};

  // Sequential counter handed out by nextHelperId.
  int32_t helperId_{0};

  // Maps each helper function name to the set of bN variable indices it
  // requires (directly or transitively via called helpers).
  std::unordered_map<std::string, std::set<size_t>> helperVarDeps_;

  // Accumulates __device__ __noinline__ helper functions extracted from
  // elementwise expressions that exceed FLAGS_out_of_line_expr_size.
  std::stringstream outOfLineFunctions_;

  // Current project node id, expression ordinal, and distinct op count for
  // labeling kernel operations.
  int32_t currentNodeId_{-1};
  int32_t currentExprOrdinal_{-1};
  int32_t numDistinctOps_{0};
};

} // namespace torch::wave
