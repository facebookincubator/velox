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

  /// addition can have alpha
  int32_t numArgs{2};

  /// If true, idx is passed as the first argument before inputs and attributes.
  bool hasIdxArg{false};

  /// If true, size (the element count of the first input) is passed after idx.
  /// Requires hasIdxArg.
  bool hasSizeArg{false};
};

/// Common cases of determining output size: kNone is a custom function, kMax is
/// largest input, as in all elementwise, kSum is concatenation.
enum class SizeShortcut { kNone, kMax, kSum };

struct SizeArguments {
  std::vector<int32_t> ordinal{0};
  std::vector<bool> isList;
};

using Dim = uint32_t;

/// A map from value ids in a lambda matching OutputReserveFunc and
/// the value ids in the frame passed to OutputReserveFunc. The same
/// kernel operation can be invoked with many different sets of
/// inputs to produce sizes for each.
using FormalToActual = folly::F14FastMap<int32_t, int32_t>;

// Returns shapes to reserve for outputs given inputs. Can return multiple
// shapes for output params that are tuples of tensors.
using OutputReserveFunc = std::function<std::vector<std::vector<
    Dim>>(NodeCP node, nativert::ExecutionFrame&, FormalToActual map)>;

struct ArgumentMeta {
  // If can be passed / returned in a register. Caller must read from / assign
  // to tensor if needed.
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

  /// This input/output does not correspond to an actual input or output but
  /// exists only to create an ordering dependency between kernels that depend
  /// on device side results from another.
  bool linkOnly{false};

  /// Marks that for an elementwise operation, we want the whole tensor as
  /// opposed to its element for this lane.
  bool wholeTensor{false};

  /// If true, emits a bool template parameter indicating whether this argument
  /// is present with a non-None value. Absent arguments and None-valued
  /// attributes both produce false.
  bool hasPresentTemplateParam{false};

  SizeShortcut sizeShortcut{SizeShortcut::kNone};

  SizeArguments sizeArgs;
};

struct Metadata {
  const c10::FunctionSchema* functionSchema;

  SizeShortcut sizeShortcut;

  /// The ordinal in arguments for the argument that determines the number of
  /// elements the kernel runs on. If many, the number of lanes to process y the
  /// kernel is the sum of the sizes of the args. If arg is a tensor, we get the
  /// number of elements. If it is a number or zero dim tensor, we use the
  /// number.
  SizeArguments sizeArgs;

  std::vector<ArgumentMeta> argumentMeta;

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

  /// If true, the user of outputs must be in a different kernel launch on the
  /// same stream. Many ops with this property can be in the same kernel as long
  /// as their consumers are in a subsequent one.
  bool kernelBreakForMultiblock{false};

  /// If true, the operation always uses the single block grid variant
  /// regardless of input size.
  bool alwaysSingleBlock{false};

  /// Translates a single node to a sequence of nodes that must be separated by
  /// kernel boundaries.
  std::function<nativert::Node*(NodeCP single, WaveGraph* waveGraph)>
      makeMultiKernelVariant;

  /// Like makeMultiKernelVariant but for code generation variants.
  std::function<nativert::Node*(NodeCP single, WaveGraph* waveGraph)> cgVariant;

  int32_t numBarriers{0};

  /// The input can be overwritten and used as output if there are no concurrent
  /// or subsequent uses of input. True for example of elementwise arithmetic.
  bool inPlaceIfLastUse{false};

  /// True if must be launched as its own kernel sequence  with no fusion.
  bool isStandalone_{false};

  /// If true, the op only supports 1-d (flat) inputs. Falls back to standalone
  /// when any input has rank > 1 or unknown rank.
  bool only1d{false};

  /// If set, called to determine standalone status when isStandalone_ is false.
  std::function<bool(NodeCP, const ValueTypes&)> isStandaloneFunc;

  bool isStandalone(NodeCP node, const ValueTypes& types) const;

  /// Unit cost for scheduling, e.g. proportional block assignment.
  float cost{1.0f};

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
  /// "counter" + "Float" → "counterFloat") to avoid collisions when multiple
  /// types appear in one translation unit.
  std::vector<std::pair<int32_t, std::string>> dynamicSharedDecls;

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

  bool isKernelBreak(bool isSingleBlock) const {
    for (auto& rm : returnMeta) {
      if (rm.neededOnHost) {
        return true;
      }
    }
    return !isSingleBlock && kernelBreakForMultiblock;
  }
};

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
  /// inPlaceIfLastUse=true, and an ElementwiseOp whose functionName is "--"
  /// followed by the op name part (e.g. "--add").
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

class MetadataBuilder {
 public:
  explicit MetadataBuilder(std::string_view qualifiedName);
  explicit MetadataBuilder(std::unique_ptr<c10::FunctionSchema> schema);

  MetadataBuilder& sizeShortcut(SizeShortcut shortcut);
  MetadataBuilder& sizeOrdinal(std::vector<int32_t> ordinal);
  MetadataBuilder& sizeArgsList(std::vector<bool> isList);
  MetadataBuilder& argumentMeta(std::vector<ArgumentMeta> meta);
  MetadataBuilder& defaultInputMeta();
  MetadataBuilder& returnMeta(std::vector<ArgumentMeta> meta);
  MetadataBuilder& defaultOutputMeta();
  MetadataBuilder& hasBarrier(bool val = true);
  MetadataBuilder& singleBlockIfFused(bool val = true);
  MetadataBuilder& inputFromPreviousKernel(int32_t ordinal);
  MetadataBuilder& kernelBreakForMultiblock(bool val = true);
  MetadataBuilder& alwaysSingleBlock(bool val = true);
  MetadataBuilder& makeMultiKernelVariant(
      std::function<nativert::Node*(NodeCP, WaveGraph*)> func);
  MetadataBuilder& cgVariant(
      std::function<nativert::Node*(NodeCP, WaveGraph*)> func);
  MetadataBuilder& numBarriers(int32_t val);
  MetadataBuilder& inPlaceIfLastUse(bool val = true);
  MetadataBuilder& isStandalone(bool val = true);
  MetadataBuilder& only1d(bool val = true);
  MetadataBuilder& isStandaloneFunc(
      std::function<bool(NodeCP, const ValueTypes&)> func);
  MetadataBuilder& cost(float val);
  MetadataBuilder& viewOfArg(int32_t ordinal);
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
  MetadataBuilder& headerFile(std::string file);
  MetadataBuilder& deviceFunc(std::string func);
  MetadataBuilder& sharedDecls(
      std::vector<std::pair<std::string, std::string>> decls);
  MetadataBuilder& dynamicSharedDecls(
      std::vector<std::pair<int32_t, std::string>> decls);
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

  Metadata build();
  void registerOp();

 private:
  ElementwiseOp& ensureElementwise();

  std::string name_;
  Metadata md_;
};

void registerBuiltins();

} // namespace torch::wave
