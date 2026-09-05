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

#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>

#include <folly/container/F14Map.h>

#include <aten/src/ATen/core/function_schema.h>
#include <torch/nativert/executor/OpKernel.h>
#include <torch/nativert/graph/Graph.h>

namespace torch::wave {

using GraphP = nativert::Graph*;
using NodeCP = const nativert::Node*;
using ValueCP = const nativert::Value*;
using FrameP = nativert::ExecutionFrame*;

class CompileCtx;
class KernelOperation;
class WaveGraph;
struct OutputDesc;
struct ResultSpec;
struct ValueConstraint;
struct ValueTypes;

/// Describes element-wise operations like binary and unary arithmetic.
struct ElementwiseOp {
  std::string functionName;

  /// Number of arguments to pass to the elementwise function. -1 means
  /// auto-detect from FunctionSchema during build().
  int32_t numArgs{-1};

  /// If true, idx is passed as the first argument before inputs and attributes.
  bool hasIdxArg{false};

  /// If true, size (the element count of the first input) is passed after idx.
  /// Requires hasIdxArg.
  bool hasSizeArg{false};

  /// If true, blockInfo is passed as the last argument.
  bool hasBlockInfo{false};

  /// If true, the enclosing elementwise expression's output tensor (the
  /// subgraph root output the loop writes) is passed as a whole-tensor argument
  /// after the schema arguments and before blockInfo. Lets a fused op such as
  /// index_select know the shape it is iterating over, distinct from its own
  /// (possibly broadcast) output shape.
  bool hasOutputArg{false};
};

/// Common cases of determining output size: kNone is a custom function, kMax is
/// largest input, as in all elementwise, kSum is concatenation.
enum class SizeShortcut { kNone, kMax, kSum };

/// Identifies a metadata-only standalone op (no real compute, only tensor
/// metadata manipulation) so the executor can apply a host-side shortcut
/// instead of a generic eager dispatch. kNone means no shortcut applies.
enum class StandaloneShortcut {
  kNone,
  kListPack,
  kListUnpack,
  kView,
  kSlice,
  kSelectInt,
  kUnsqueeze,
  kTranspose,
  kNarrow,
  kUnbind,
  kSplitWithSizes,
  kSqueezeDim,
  kExpand,
  kSymSize,
  kSymNumel,
};

/// Specifies which arguments determine the number of elements a kernel
/// processes.
struct SizeArguments {
  std::vector<int32_t> ordinal;
  std::vector<bool> isList;
};

using Dim = uint32_t;

/// A map from value ids in a lambda matching OutputReserveFunc and
/// the value ids in the frame passed to OutputReserveFunc. The same
/// kernel operation can be invoked with many different sets of
/// inputs to produce sizes for each.
using FormalToActual = folly::F14FastMap<int32_t, int32_t>;

using NodeMap = folly::F14FastMap<NodeCP, NodeCP>;

// Returns shapes to reserve for outputs given inputs. Can return multiple
// shapes for output params that are tuples of tensors.
using OutputReserveFunc = std::function<std::vector<std::vector<Dim>>(
    NodeCP node,
    nativert::ExecutionFrame&,
    const FormalToActual& map,
    NodeCP originalFormalNode,
    const NodeMap& nodeMap)>;

/// Per-argument metadata controlling how an argument is passed to device code.
struct ArgumentMeta {
  /// If can be passed / returned in a register. Caller must read from / assign
  /// to tensor if needed.
  bool isRegister{false};

  /// For outputs that are materialized tensors, function that determines the
  /// size to allocate based on inputs and execution state.
  OutputReserveFunc reserveShape{nullptr};

  /// True if actual size is determined on device, e.g. stream compaction.
  bool shapeSetOnDevice{false};

  /// If true, the host needs to read the value from the invocation frame after
  /// the kernel completes. Introduces a queued D to H transfer and a host side
  /// sync.
  bool neededOnHost{false};

  /// If true, this argument must remain on CPU even when the graph runs on GPU.
  bool cpuOnly{false};

  /// This input/output does not correspond to an actual input or output but
  /// exists only to create an ordering dependency between kernels that depend
  /// on device side results from another.
  bool linkOnly{false};

  /// Set for an output produced by a non-last part of a root op that was split
  /// into several kernel ops (e.g. tw.group_length_guard_head, whose length
  /// outputs are consumed by tw.group_length_guard_final). Such an output is a
  /// real output of the original op, so it must never be released as a per-op
  /// freeable intermediate, even though its producing node is not the kernel
  /// op's root expr. We flag this statically at registration rather than
  /// deciding from downstream uses on purpose: a ProjectOperation is
  /// deduplicated and reused across actual subgraphs, some of which reference
  /// this value externally and some of which do not, so a use-based decision
  /// would be wrong for the shared op.
  bool nonRootOutput{false};

  /// Marks that for an elementwise operation, we want the whole tensor as
  /// opposed to its element for this lane.
  bool wholeTensor{false};

  /// If true, and the value is produced in this kernel, there must be a
  /// kernel-wide barrier between producer and user. E.g. if an elementwise
  /// writes a tensor and another indexes into it at random, there must be
  /// a barrier.
  bool randomAccess{false};

  /// If true, emits a bool template parameter indicating whether this argument
  /// is present with a non-None value. Absent arguments and None-valued
  /// attributes both produce false.
  bool hasPresentTemplateParam{false};

  /// For an output: the kernel maps its writes through the output tensor's
  /// strides rather than indexing its storage linearly, so it stays correct
  /// when the output is a pitched view. This is what lets a concat hand the
  /// producer a strided band of the result to write in place; a producer
  /// without it gets a dense buffer of its own and the concat copies it in.
  ///
  /// It is the output-side dual of Metadata::layoutAgnostic, which is about the
  /// strides an op READS, and it is not implied by an output's contiguity
  /// ValueConstraint -- that states what the value IS, not what its writer
  /// could cope with.
  ///
  /// On a TensorList (or a list of lists) it applies to every element.
  bool mayWriteStrided{false};

  SizeShortcut sizeShortcut{SizeShortcut::kNone};

  SizeArguments sizeArgs;
};

/// Complete metadata for a registered operation, including schema, arguments,
/// and code generation hints.
struct Metadata {
  const c10::FunctionSchema* functionSchema{nullptr};

  std::vector<ArgumentMeta> argumentMeta;

  /// Positional argument names for schema-less ops (no functionSchema, e.g. the
  /// Python _operator.* scalar ops). nativert splits a node's operands across
  /// inputs() (symbolic operands, by name) and attributes() (constant literals,
  /// by name); neither container preserves positional order, only the names do.
  /// forArguments binds each position to the operand carrying argumentNames[i].
  /// These are the Python signature parameter names ("a", "b", ... for the
  /// _operator builtins). A registered name absent from the node, or an operand
  /// the node carries that is not a registered name, is a fatal error, so an op
  /// whose serialized argument names differ from those registered fails loudly
  /// rather than silently miscomputing. Empty for schema-backed ops.
  std::vector<std::string> argumentNames;

  std::vector<ArgumentMeta> returnMeta;

  /// True if all values to be computed by a block must be ready before calling.
  /// True for example for single block stream compaction or first stage of
  /// multi-kernel stream compaction.
  bool hasBarrier{false};

  /// True if this has a fusable single form that requires the invocation to be
  /// single block. For example reduction without kernel boundary. A multikernel
  /// form may exist and if so, makeMultiKernelVariant produces this.
  bool singleBlockIfFused{false};

  /// When a Node has a multikernel (multiple consecutive Nodes) variant, the
  /// non-first Nodes may have many inputs, also ones shared with other stages
  /// of the multikernel op. To get the Nodes in the right order, each non-first
  /// node must have one input that is always an output of the previous Node.
  /// When set, this is the ordinal of this input.
  std::optional<int32_t> inputFromPreviousKernel;

  /// If true, a barrier is needed after this op before consumers can read its
  /// output. In default multi-block mode this forces a kernel break; in CG mode
  /// an opBarrier is emitted instead so the next op stays in the same kernel.
  /// In single-block mode the flag is ignored.
  bool multiBlockReturnBarrier{false};

  /// Like multiBlockReturnBarrier, but only takes effect when the runtime
  /// WaveConfig::scanOutputReturnBarrier toggle is enabled (passed in as the
  /// scanOutputReturnBarrierEnabled argument to isKernelBreak). Set on scan ops
  /// whose multi-block output is read cross-block by fused cat consumers so the
  /// scan ends its launch and the consumer reads a materialized buffer from a
  /// later stream-ordered launch.
  bool scanOutputReturnBarrier{false};

  /// If true, the operation always uses the single block grid variant
  /// regardless of input size.
  bool alwaysSingleBlock{false};

  /// If true, the grid is sized by the sum of the op's input element counts
  /// rather than the largest one. Set this when an op's work is the total over
  /// a tensor list, not the largest member: sizing by the largest gives a grid
  /// that ignores the list length, so the op runs far fewer blocks than it has
  /// independent work.
  bool gridSizeSumsInputs{false};

  /// If true, the op reads tensor metadata (shape, size) rather than
  /// computing on tensor data. When used as a size arg producer, it runs as
  /// a standalone rather than a fused op.
  bool isMetadataGetter{false};

  /// If true, this is a scalar-producing elementwise op (e.g. the Python
  /// _operator.* scalar arithmetic, or sym_size / sym_numel). Such ops have no
  /// FunctionSchema; their operands and result are naked scalars (SymInt /
  /// SymFloat), not tensors. When the top node of an elementwise kernel has
  /// this flag, the elementwise loop runs a single iteration and assigns the
  /// result to a scalar parameter instead of a tensor element.
  bool isScalarElementwise{false};

  /// Translates a single node to a sequence of nodes that must be separated by
  /// kernel boundaries.
  std::function<nativert::Node*(NodeCP single, WaveGraph* waveGraph)>
      makeMultiKernelVariant;

  /// Like makeMultiKernelVariant but for code generation variants.
  std::function<nativert::Node*(NodeCP single, WaveGraph* waveGraph)> cgVariant;

  /// Rewrites one node into a form that produces its outputs tensor by tensor
  /// instead of as a TensorList, so each column becomes a node the rest of the
  /// compiler can see: consumer counting, aliasing, cost-based block shares and
  /// CSE all work per value rather than per bundle. Returns true if it rewrote
  /// the node. Called by decomposeListOps in one traversal, so an op supplies
  /// only its own rule and no pass walks the graph on its behalf.
  std::function<bool(NodeCP node, WaveGraph& waveGraph)> decompose;

  int32_t numBarriers{0};

  /// If true, apply PyTorch arithmetic type promotion rules instead of C++
  /// rules. For example, long * float promotes to double in PyTorch but to
  /// float in C++.
  bool arithmeticPromotion{false};

  /// The input can be overwritten and used as output if there are no concurrent
  /// or subsequent uses of input. True for example of elementwise arithmetic.
  bool inPlaceIfLastUse{false};

  /// If true, this elementwise op's output shape is not derivable from its
  /// operands' shapes (e.g. index_select, whose output resizes one dim to the
  /// index length), so the enclosing expression cannot size its own output by
  /// broadcasting this op's inputs. When set, the op's output is materialized
  /// as a shape-only tensor even when fused into another elementwise, and the
  /// size machinery uses that output's shape (not the op's inputs) as a
  /// broadcast leaf.
  bool sizeFromOutput{false};

  /// True if must be launched as its own kernel sequence  with no fusion.
  bool isStandalone_{false};

  /// If true, the op only supports 1-d (flat) inputs. Falls back to standalone
  /// when any input has rank > 1 or unknown rank.
  bool only1d{false};

  /// If true, the op only manipulates tensor metadata (e.g. a view, slice, or
  /// select) and does no real compute, so a standalone instance can be served
  /// by a host-side shortcut. Set via the builder for known metadata-only ops.
  /// prim.ListPack is also metadata-only but has no Metadata entry; that case
  /// is handled directly where standalone Launches are built.
  bool metadataOnly{false};

  /// If set, called to determine standalone status when isStandalone_ is false.
  std::function<bool(NodeCP, const ValueTypes&)> isStandaloneFunc;

  bool isStandalone(NodeCP node, const ValueTypes& types) const;

  /// Default per-node cost for scheduling. Used by unitCost() when
  /// costFunction is not set.
  float cost{1.0f};

  /// If set, returns the per-element cost for this node given its metadata.
  /// Takes precedence over 'cost' when computing kernel op costs.
  std::function<float(NodeCP, const Metadata&)> costFunction;

  /// Returns the per-element cost for 'node'. If costFunction is set, calls
  /// it; otherwise returns 'cost'.
  float unitCost(NodeCP node) const {
    if (costFunction) {
      return costFunction(node, *this);
    }
    return cost;
  }

  /// If set, the output is a view over the argument at this ordinal.
  std::optional<int32_t> viewOfArg;

  /// Name of the attribute that specifies the output shape (e.g. "size" for
  /// view, "shape" for reshape). Skipped by forEachSortedAttribute.
  std::string shapeAttr;

  /// Attributes to skip in forEachSortedAttribute.
  std::vector<std::string> ignoreAttrs;

  bool isView() const {
    return viewOfArg.has_value();
  }

  // --- In-place-rewrite / scatter-writer metadata (for the functional ->
  // in-place rewrite + clone-elision pass). Tensor operands are named by
  // TENSOR-INPUT ordinal (as inputAt indexes -- constant scalars are dropped);
  // constant scalars that nativert stores as attributes are named by attribute.

  /// Tensor-input ordinal of the input a writing op overwrites in place (the
  /// "self"/target). Set on scatter / index / masked / slice-scatter writers
  /// (= 0). Marks the op as an in-place-rewrite candidate.
  std::optional<int32_t> mutatesArg;

  /// Tensor-input ordinal of the index / mask operand of a scatter/index op.
  std::optional<int32_t> indicesArg;

  /// Tensor-input ordinal of the source / values operand written into self
  /// (unset when the written value is a scalar attribute, e.g. masked_fill).
  std::optional<int32_t> valuesArg;

  /// If true, the op reads all its tensor inputs at arbitrary strides
  /// (elementwise, cat), so a producing clone of a strided input is elidable.
  bool layoutAgnostic{false};

  /// Attribute name of the `dim` argument for dim-wise scatter/index ops
  /// (empty if none); the dim is a constant stored as an attribute.
  std::string dimAttr;

  /// If true, graph normalization rewrites a constant negative "dim" attribute
  /// to its non-negative form and errors if it is out of range for the first
  /// input's rank. Set on the metadata-only view ops whose host-side shortcut
  /// indexes sizes()/strides() directly, so the shortcut needs neither the wrap
  /// nor the check at run time.
  bool normalizeDimAttr{false};

  /// Attribute name of the accumulate / scatter-reduce flag (empty if none).
  /// When true on a node, an in-place FUSED write needs atomics.
  std::string accumulateAttr;

  /// Attribute name of a memory_format argument (empty if none). A clone that
  /// sets it is a layout conversion and must not be elided.
  std::string memoryFormatAttr;

  /// If set, the output rank is taken from the input at this ordinal. Takes
  /// precedence over outputConstraints and the elementwise default.
  std::optional<int32_t> rankArgument;

  /// Returns output constraints given a node and its input constraints in
  /// ValueTypes. If set, called during graph optimization to propagate rank and
  /// other constraints from inputs to outputs.
  std::function<
      std::vector<ValueConstraint>(NodeCP node, const ValueTypes& types)>
      outputConstraints;

  /// Called after setting output constraints during optimization. If it returns
  /// non-empty, each pair's first Value is replaced by the second in all uses.
  std::function<std::vector<std::pair<ValueCP, ValueCP>>(
      NodeCP node,
      ValueTypes& types,
      WaveGraph& waveGraph)>
      maybeReplace;

  /// Called during graph normalization before filling in schema defaults.
  std::function<void(nativert::Node*, const ValueTypes&)> normalize;

  std::unique_ptr<ElementwiseOp> elementwise;

  /// Custom code generation for elementwise ops. If set, elementwiseExprImpl
  /// generates each input as a string and calls this instead of the default
  /// function call pattern.
  std::function<void(std::stringstream&, NodeCP, std::vector<std::string> args)>
      generateCall;

  /// If set, overrides fusedCode for this node. Called instead of the default
  /// code generation path.
  std::function<void(
      NodeCP node,
      const std::vector<ResultSpec>& resultSpecs,
      CompileCtx* ctx)>
      specialForm;

  /// Custom code generation for a non-elementwise op, alongside the default
  /// call rather than instead of it (which is what specialForm is for).
  /// Returns the text of one more template argument, emitted after
  /// templateAttrs, and may declare what that argument names at
  /// translation-unit scope through CompileCtx::emitHelperCode. Lets an op
  /// whose shape arrives as data pass that shape as a type, so the device
  /// function can hold per-shape state in registers instead of in an array a
  /// runtime index subscripts.
  ///
  /// Whatever it reads must be part of the node's dedup identity, or two nodes
  /// sharing one KernelOperation would run the first one's generated type. The
  /// int-list attributes and templateAttrs are; the operands are not.
  std::function<std::string(NodeCP, CompileCtx*)> generateTemplateArg;

  /// device side header to include in the NVRTC translation unit.
  std::string headerFile;

  /// Name of device side function. Arguments are passed as given by
  /// FunctionSchema, inputs are T*, scalars are T, results are T*. T is Tensor
  /// or a scalar type.
  std::string deviceFunc;

  /// List of type, name pairs for __shared__ variables to be declared at head
  /// of containing kernel and then passed as last args in the call to the
  /// function.
  std::vector<std::pair<std::string, std::string>> sharedDecls;

  /// Like sharedDecls but the type is determined at compile time from the dtype
  /// of the input at the given ordinal. Each entry is (argument ordinal,
  /// base name). The variable name is base name + type suffix (e.g.
  /// "counter" + "Float" -> "counterFloat") to avoid collisions when multiple
  /// types appear in one translation unit.
  std::vector<std::pair<int32_t, std::string>> dynamicSharedDecls;

  /// If non-zero, the kernel containing this node is compiled with
  /// __launch_bounds__ for at least this many blocks per SM. Set it on an op
  /// whose device function the compiler would otherwise give so many registers
  /// that it lowers the occupancy of every other op sharing the kernel.
  int32_t minBlocksPerSm{0};

  /// If set, returns the bytes of dynamic (extern __shared__) shared memory
  /// this node's device function needs. The kernel op takes the max over its
  /// nodes and the launch passes that as the kernel's dynamic shared memory
  /// size, so an op that needs a large scratch buffer only costs occupancy in
  /// the launches that contain it.
  std::function<int64_t(NodeCP)> dynamicSharedMemory;

  /// Ordinal value meaning the type comes from the node's dtype attribute.
  static constexpr int32_t kTypeFromDtype = -1;

  /// Ordinals of arguments whose dtype appears as a template parameter of the
  /// device func, set according to the dtype of the arg at the ordinal.
  std::vector<int32_t> typeTemplateParams;

  /// If true, the device function takes WaveConfig::blockSize as its first
  /// template parameter.
  bool hasBlockSizeTemplateParam{false};

  /// If true, the resolved dtype attribute is emitted as an additional type
  /// template parameter after typeTemplateParams. Used for sum/cumsum where the
  /// kernel reads in TIn and accumulates/writes in TOut.
  bool hasDtypeTemplateParam{false};

  /// Attribute names whose values are emitted as template parameters after
  /// typeTemplateParams and hasDtypeTemplateParam, in list order. These
  /// attributes are skipped by forEachSortedAttribute.
  std::vector<std::string> templateAttrs;

  /// Returns true if this elementwise op's result is materialized in memory
  /// rather than kept in a register, i.e. the op writes a whole tensor as a
  /// side effect (the fused in-place scatters, index_put_elt_*, masked_put_).
  /// Such a producer cannot be inlined into a consuming elementwise
  /// expression: codegen emits it as its own expression and the consumer reads
  /// its output back from memory (see
  /// CompileCtx::generateElementwiseBorderImpl). The size machinery must stop
  /// at the same boundary -- the consumer is sized by the materialized output,
  /// not by this op's operands.
  bool isElementwiseBorder() const {
    return elementwise != nullptr && !returnMeta.empty() &&
        !returnMeta[0].isRegister;
  }

  /// Returns true if any argument has isRegister set.
  bool hasRegisterInputs() const {
    for (const auto& am : argumentMeta) {
      if (am.isRegister) {
        return true;
      }
    }
    return false;
  }

  /// Returns true if any argument has hasPresentTemplateParam set.
  bool hasPresentTemplateParams() const {
    for (const auto& am : argumentMeta) {
      if (am.hasPresentTemplateParam) {
        return true;
      }
    }
    return false;
  }

  /// Returns true if the schema argument with the given name has
  /// hasPresentTemplateParam set.
  bool isPresenceTemplateParam(std::string_view name) const {
    if (!functionSchema) {
      return false;
    }
    const auto& args = functionSchema->arguments();
    for (size_t i = 0; i < args.size() && i < argumentMeta.size(); ++i) {
      if (args[i].name() == name && argumentMeta[i].hasPresentTemplateParam) {
        return true;
      }
    }
    return false;
  }

  /// Fills argumentMeta with default ArgumentMeta{} for each schema argument
  /// if argumentMeta is empty. Requires functionSchema to be set.
  void defaultInputMeta();

  /// Fills returnMeta with default ArgumentMeta{} for each schema return
  /// if returnMeta is empty. Requires functionSchema to be set.
  void defaultOutputMeta();

  /// If set, called instead of the default setOutputs logic. The function
  /// receives the same arguments as KernelOperation::setOutputs.
  std::function<void(
      KernelOperation* op,
      NodeCP node,
      const std::unordered_set<ValueCP>& subgraphInputs,
      std::vector<ValueCP>& outputValues,
      std::vector<OutputDesc>& outputDescs,
      bool inMemory,
      bool callerIsElementwise)>
      setOutputs;

  bool isKernelBreak(
      bool isSingleBlock,
      bool isCgGrid = false,
      bool scanOutputReturnBarrierEnabled = false) const {
    for (auto& rm : returnMeta) {
      if (rm.neededOnHost) {
        return true;
      }
    }
    if (isSingleBlock || isCgGrid) {
      return false;
    }
    return multiBlockReturnBarrier ||
        (scanOutputReturnBarrier && scanOutputReturnBarrierEnabled);
  }
};

/// Global registry mapping qualified op names to their Metadata.
class Registry {
 public:
  static void registerMetadata(std::string_view op, Metadata metadata);
  static const Metadata* metadata(std::string_view op);

  /// Removes the entry for 'name' and returns the Metadata. Throws if not
  /// found.
  static Metadata unregister(std::string_view name);

  /// Restores a previously unregistered entry.
  static void restoreRegistry(std::string_view name, Metadata metadata);

  /// Registers an elementwise op by its qualified aten name (e.g.
  /// "torch.ops.aten.add.Tensor"). Looks up the FunctionSchema from the
  /// dispatcher, then creates a Metadata entry with sizeArgs={0},
  /// inPlaceIfLastUse=true, and an ElementwiseOp whose functionName is "__"
  /// followed by the op name part (e.g. "__add").
  static void registerElementwise(std::string_view qualifiedName);

  /// Registers an elementwise op with an explicit CUDA function name and
  /// standalone flag. Use this to create aliases that share the same CUDA
  /// implementation as another op but have different Metadata.
  static void registerElementwiseOp(
      std::string_view qualifiedName,
      std::string_view elementwiseFuncName,
      bool isStandalone);

  /// Stores a FunctionSchema for intrinsics not in the PyTorch dispatcher.
  /// Returns a stable pointer to the stored schema.
  static const c10::FunctionSchema* ownSchema(
      std::unique_ptr<c10::FunctionSchema> schema);

 private:
  static std::unordered_map<std::string, Metadata>& registry();
  static std::vector<std::unique_ptr<c10::FunctionSchema>>& schemaStorage();
};

/// Fluent builder for constructing and registering Metadata entries.
class MetadataBuilder {
 public:
  /// Tag for registering an op that has no FunctionSchema (e.g. the Python
  /// _operator.* scalar ops). The caller must set numArgs / argumentMeta /
  /// returnMeta explicitly since they cannot be derived from a schema.
  struct NoSchema {};

  explicit MetadataBuilder(std::string_view qualifiedName);
  explicit MetadataBuilder(std::unique_ptr<c10::FunctionSchema> schema);
  MetadataBuilder(std::string_view qualifiedName, NoSchema);

  MetadataBuilder& sizeShortcut(SizeShortcut shortcut);
  MetadataBuilder& sizeOrdinal(std::vector<int32_t> ordinal);
  MetadataBuilder& sizeArgsList(std::vector<bool> isList);
  MetadataBuilder& argumentMeta(std::vector<ArgumentMeta> meta);
  MetadataBuilder& argumentNames(std::vector<std::string> names);
  MetadataBuilder& defaultInputMeta();
  MetadataBuilder& returnMeta(std::vector<ArgumentMeta> meta);
  MetadataBuilder& defaultOutputMeta();

  /// Marks every output as one the kernel writes through the output's strides,
  /// so a concat may hand it a pitched band of the result instead of a dense
  /// buffer to be copied in. See ArgumentMeta::mayWriteStrided.
  MetadataBuilder& mayWriteStrided(bool val = true);
  MetadataBuilder& hasBarrier(bool val = true);
  MetadataBuilder& singleBlockIfFused(bool val = true);
  MetadataBuilder& inputFromPreviousKernel(int32_t ordinal);
  MetadataBuilder& multiBlockReturnBarrier(bool val = true);
  MetadataBuilder& scanOutputReturnBarrier(bool val = true);
  MetadataBuilder& alwaysSingleBlock(bool val = true);
  MetadataBuilder& gridSizeSumsInputs(bool val = true);
  MetadataBuilder& metadataGetter(bool val = true);
  MetadataBuilder& makeMultiKernelVariant(
      std::function<nativert::Node*(NodeCP, WaveGraph*)> func);
  MetadataBuilder& cgVariant(
      std::function<nativert::Node*(NodeCP, WaveGraph*)> func);
  MetadataBuilder& decompose(std::function<bool(NodeCP, WaveGraph&)> func);
  MetadataBuilder& numBarriers(int32_t val);
  MetadataBuilder& arithmeticPromotion(bool val = true);
  MetadataBuilder& inPlaceIfLastUse(bool val = true);
  MetadataBuilder& sizeFromOutput(bool val = true);
  MetadataBuilder& isStandalone(bool val = true);
  MetadataBuilder& only1d(bool val = true);
  MetadataBuilder& metadataOnly(bool val = true);
  MetadataBuilder& isStandaloneFunc(
      std::function<bool(NodeCP, const ValueTypes&)> func);
  MetadataBuilder& cost(float val);
  MetadataBuilder& costFunction(
      std::function<float(NodeCP, const Metadata&)> func);
  MetadataBuilder& viewOfArg(int32_t ordinal);
  MetadataBuilder& mutatesArg(int32_t ordinal);
  MetadataBuilder& indicesArg(int32_t ordinal);
  MetadataBuilder& valuesArg(int32_t ordinal);
  MetadataBuilder& layoutAgnostic(bool val = true);
  MetadataBuilder& dimAttr(std::string name);
  MetadataBuilder& normalizeDimAttr(bool val = true);
  MetadataBuilder& accumulateAttr(std::string name);
  MetadataBuilder& memoryFormatAttr(std::string name);
  MetadataBuilder& shapeAttr(std::string name);
  MetadataBuilder& ignoreAttrs(std::vector<std::string> attrs);
  MetadataBuilder& rankArgument(int32_t ordinal);
  MetadataBuilder& outputConstraints(
      std::function<std::vector<ValueConstraint>(NodeCP, const ValueTypes&)>
          func);
  MetadataBuilder& maybeReplace(
      std::function<std::vector<
          std::pair<ValueCP, ValueCP>>(NodeCP, ValueTypes&, WaveGraph&)> func);
  MetadataBuilder& normalize(
      std::function<void(nativert::Node*, const ValueTypes&)> func);
  MetadataBuilder& generateCall(
      std::function<void(std::stringstream&, NodeCP, std::vector<std::string>)>
          func);
  MetadataBuilder& specialForm(
      std::function<void(NodeCP, const std::vector<ResultSpec>&, CompileCtx*)>
          func);
  MetadataBuilder& generateTemplateArg(
      std::function<std::string(NodeCP, CompileCtx*)> func);
  MetadataBuilder& headerFile(std::string file);
  MetadataBuilder& deviceFunc(std::string func);
  MetadataBuilder& sharedDecls(
      std::vector<std::pair<std::string, std::string>> decls);
  MetadataBuilder& dynamicSharedDecls(
      std::vector<std::pair<int32_t, std::string>> decls);
  MetadataBuilder& dynamicSharedMemory(std::function<int64_t(NodeCP)> func);
  MetadataBuilder& minBlocksPerSm(int32_t blocks);
  MetadataBuilder& typeTemplateParams(std::vector<int32_t> params);
  MetadataBuilder& hasBlockSizeTemplateParam(bool val = true);
  MetadataBuilder& hasDtypeTemplateParam(bool val = true);
  MetadataBuilder& templateAttrs(std::vector<std::string> attrs);
  MetadataBuilder& setOutputs(
      std::function<void(
          KernelOperation* op,
          NodeCP node,
          const std::unordered_set<ValueCP>& subgraphInputs,
          std::vector<ValueCP>& outputValues,
          std::vector<OutputDesc>& outputDescs,
          bool inMemory,
          bool callerIsElementwise)> func);

  MetadataBuilder& elementwise();
  MetadataBuilder& elementwiseFunc(std::string funcName);
  MetadataBuilder& numArgs(int32_t n);
  MetadataBuilder& hasIdxArg(bool val = true);
  MetadataBuilder& hasSizeArg(bool val = true);
  MetadataBuilder& hasBlockInfo(bool val = true);
  MetadataBuilder& hasOutputArg(bool val = true);
  MetadataBuilder& isScalarElementwise(bool val = true);

  Metadata build();
  void registerOp();

 private:
  ElementwiseOp& ensureElementwise();

  std::string name_;
  Metadata md_;

  SizeShortcut builderSizeShortcut_{SizeShortcut::kNone};
  SizeArguments builderSizeArgs_;
  bool sizeShortcutSet_{false};
  bool sizeArgsSet_{false};
};

void registerBuiltins();

/// True if the kernel that produces 'value' maps its writes through the output
/// tensor's strides, so it stays correct when handed a pitched view. A concat
/// uses this to decide whether an operand can be given a strided band of the
/// result to fill directly, or whether it needs a dense buffer of its own that
/// the concat then copies in.
///
/// False -- the conservative answer -- for a value with no producer, a producer
/// with no registered metadata, and any op that has not declared
/// ArgumentMeta::mayWriteStrided. A value that is an element of a TensorList
/// output takes the flag from the list, since the flag covers every element.
bool producerMayWriteStrided(ValueCP value);

} // namespace torch::wave
