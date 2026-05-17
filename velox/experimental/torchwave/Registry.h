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
#include <string>
#include <string_view>
#include <unordered_map>

#include <folly/container/F14Map.h>

#include <aten/src/ATen/core/function_schema.h>
#include <torch/nativert/executor/OpKernel.h>
#include <torch/nativert/graph/Graph.h>

namespace torch::wave {

using GraphP = nativert::Graph*;
using NodeCP = const nativert::Node*;
using ValueCP = const nativert::Value*;
using FrameP = nativert::ExecutionFrame*;

class WaveGraph;

/// Describes element-wise operations like binary and unary arithmetic.
struct ElementwiseOp {
  std::string functionName;

  /// addition can have alpha
  int32_t numArgs{2};

  /// Attribute names that map to extra args in the CUDA function, e.g.
  /// {"alpha"} for add/sub, {"min", "max"} for clamp.
  std::vector<std::string> attributeArgs;
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

  /// True if requires a D to H transfer for return status, for example a length
  /// after stream compaction.
  bool hasReturn{false};

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

  /// The input can be overwritten and used as output if there are no concurrent
  /// or subsequent uses of input. True for example of elementwise arithmetic.
  bool inPlaceIfLastUse{false};

  /// True if must be launched as its own kernel sequence  with no fusion.
  bool isStandalone{false};

  /// Unit cost for scheduling, e.g. proportional block assignment.
  float cost{1.0f};

  std::unique_ptr<ElementwiseOp> elementwise;

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

  /// Ordinals of arguments whose dtype appears as a template parameter of the
  /// device func, set according to the dtype of the arg at the ordinal.
  std::vector<int32_t> typeTemplateParams;

  /// If true, the device function takes WaveConfig::blockSize as its first
  /// template parameter.
  bool hasBlockSizeTemplateParam{false};

  /// Returns true if any argument has isRegister set.
  bool hasRegisterInputs() const {
    for (const auto& am : argumentMeta) {
      if (am.isRegister) {
        return true;
      }
    }
    return false;
  }

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
  static void registerElementwise(
      std::string_view qualifiedName,
      std::vector<std::string> attributeArgs = {});

  /// Registers an elementwise op with an explicit CUDA function name and
  /// standalone flag. Use this to create aliases that share the same CUDA
  /// implementation as another op but have different Metadata.
  static void registerElementwiseOp(
      std::string_view qualifiedName,
      std::string_view elementwiseFuncName,
      bool isStandalone,
      std::vector<std::string> attributeArgs = {});

  /// Stores a FunctionSchema for intrinsics not in the PyTorch dispatcher.
  /// Returns a stable pointer to the stored schema.
  static const c10::FunctionSchema* ownSchema(
      std::unique_ptr<c10::FunctionSchema> schema);

 private:
  static std::unordered_map<std::string, Metadata>& registry();
  static std::vector<std::unique_ptr<c10::FunctionSchema>>& schemaStorage();
};

void registerBuiltins();

} // namespace torch::wave
