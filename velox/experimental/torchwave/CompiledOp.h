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
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include <ATen/core/ivalue.h>

#include <torch/nativert/graph/Graph.h>
#include "velox/experimental/torchwave/KernelParams.h"
#include "velox/experimental/torchwave/Registry.h"
#include "velox/experimental/wave/common/Cuda.h"

namespace torch::wave {

struct ExecutionState;
class OpInvocation;

/// Represents an operation offered by   a composite kernel. For example a fused
/// expression like a + b * c + scalar. At run time the operation may span a
/// variable number of blocks in an invocation of a composite kernel.
struct Subgraph;

class KernelOperation {
 public:
  KernelOperation(const Subgraph& sg, int32_t opCode);

  int32_t paramOffset(const nativert::Value* value) const;

  int32_t attrOffset(const nativert::Node* node, std::string_view attr) const;

  void addSharedDeclaration(const std::string& decl);

  /// Sets the CUDA code text from the given stream and clears the stream.
  void setCode(std::stringstream& code);

  const std::string& code() const {
    return text_;
  }

  const std::vector<int32_t>& opCodes() const {
    return opCode_;
  }

  /// Returns the number of elements for this operation given the frame and
  /// bindings. Errors if numElements_ is not set.
  int64_t numElements(
      nativert::ExecutionFrame& frame,
      const FormalToActual& map) const;

  /// Recurses through the subgraph. For elementwise subtrees, adds the output
  /// value and an OutputReserveFunc that calls elementwiseOutputShape. For
  /// non-elementwise nodes, adds outputs where returnMeta has isRegister false.
  void setOutputs(
      const nativert::Node* node,
      const std::unordered_set<const nativert::Value*>& subgraphInputs,
      std::vector<const nativert::Value*>& outputValues,
      std::vector<OutputReserveFunc>& outputReserves);

  const nativert::Node* expr() const {
    return expr_;
  }

  int32_t numInputs() const {
    return numInputs_;
  }

  const std::vector<const nativert::Value*>& orderedInputs() const {
    return orderedInputs_;
  }

  const std::vector<OutputReserveFunc>& outputReserves() const {
    return outputReserve_;
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

  std::string toString() const;

  // Hash for (Node*, attrName) pairs used as keys in attrOffsets_.
  struct NodeAttrHash {
    size_t operator()(
        const std::pair<const nativert::Node*, std::string>& key) const {
      auto h1 = std::hash<const void*>{}(key.first);
      auto h2 = std::hash<std::string>{}(key.second);
      return h1 ^ (h2 + 0x9e3779b9 + (h1 << 6) + (h1 >> 2));
    }
  };

 private:
  /// Op code, unique within a CompositeKernel. A follow up in a multi-kernel
  /// computation can have more than one opcode.
  std::vector<int32_t> opCode_;

  // The represented computation. 'expr' is the top level result. This graph is
  // a tre, except that it may have multiple occurrences of the same Value in
  // 'inputs'. Not filled in in follow ups or single block variant.
  const nativert::Node* expr_;

  // The Values this takes as input. All these are reachable from
  // 'expr'. This operation is usable for any expr that has the same
  // 'expr' with the structure bounded by the save Values. The
  // graphs, including repeated use of the same values in many
  // places must be strictly isomorphic. a+a does not match a+b even
  // if a and b are of the same type as the a in a+a.
  std::unordered_set<const nativert::Value*> inputs_;

  // Cuda program text, to be inserted into the opCode case of the composite
  // kernel.
  std::string text_;

  std::vector<const nativert::Value*> orderedInputs_;

  // Offset of each input Value in the kernel's BlockInfo::params, starting at
  // 0.
  std::unordered_map<const nativert::Value*, int32_t> paramOffsets_;

  // Offset of each node attribute in BlockInfo::params, following the input
  // param offsets at intervals of 8.
  std::unordered_map<
      std::pair<const nativert::Node*, std::string>,
      int32_t,
      NodeAttrHash>
      attrOffsets_;

  // Shared memory declarations for the kernel.
  std::vector<std::string> sharedDeclarations_;

  std::vector<OutputReserveFunc> outputReserve_;

  using NumElementsFunc =
      std::function<int64_t(nativert::ExecutionFrame&, const FormalToActual&)>;
  NumElementsFunc numElements_;

  float unitCost_{0};

  int32_t numInputs_{0};

  int32_t constantAreaOffset_{0};
};

struct ValueTypes;

struct ActualParameter {
  const nativert::Value* value;
  std::optional<c10::IValue> constant;
};

class OpInvocation {
 public:
  /// Allocates output tensors on device using the kernel op's reserve functions
  /// and places them in the frame at the bound actual value IDs. Returns the
  /// numElements for the kernel op.
  int64_t allocateOutput(
      nativert::ExecutionFrame& frame,
      const ValueTypes& types);

  /// Fills the pinned param buffer for this op from the frame. Writes input
  /// and output values using paramOffset from the KernelOperation and actual
  /// values from bindings, then writes constants starting at
  /// constantAreaOffset.
  void fillParams(nativert::ExecutionFrame& frame, uint8_t* paramBase);

  OpInvocation(
      KernelOperation* op,
      const Subgraph& sg,
      std::vector<const c10::IValue*> constants,
      int32_t startOffset = 0);

  KernelOperation* op() const {
    return op_;
  }

  const std::vector<const nativert::Value*>& values() const {
    return values_;
  }

  const std::vector<const c10::IValue*>& constants() const {
    return constants_;
  }

  int32_t paramOffset() const {
    return paramOffset_;
  }

  int32_t paramSize() const {
    return paramSize_;
  }

  const FormalToActual& bindings() const {
    return bindings_;
  }

  std::string toString() const;

 private:
  /// Recurses through formalSg and actualSg in parallel. Stops at inputs of
  /// formalSg. When a node in formalSg has an output that is in the
  /// KernelOperation's outputs, adds a mapping from the formal value id to the
  /// actual value id.
  void makeBindings(const Subgraph& formalSg, const Subgraph& actualSg);
  KernelOperation* op_;
  std::vector<const nativert::Value*> values_;
  std::vector<const c10::IValue*> constants_;
  int32_t paramOffset_{0};
  int32_t paramSize_{0};
  FormalToActual bindings_;
};

class CompositeKernel {
 public:
  CompositeKernel(
      std::vector<std::unique_ptr<KernelOperation>>&& ops,
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
  std::vector<std::unique_ptr<KernelOperation>> ops_;
};

struct CompositeInvocation {
  /// Executes this composite invocation: allocates outputs, builds the grid,
  /// copies params to pinned+device memory, and enqueues the H2D transfer.
  void execute(ExecutionState& state);

  std::string toString() const;

  std::unique_ptr<CompositeKernel> kernel;
  std::vector<OpInvocation> ops;
  /// Stable storage for IValues referenced by OpInvocation::constants_.
  std::deque<c10::IValue> ivalueStorage;
};

/// Represents a single ProjectNode in a stack of ProjectNodes. Contains a graph
/// of CompositeKernels and a binding of their parameters to slots in the
/// execution state.
class CompiledNode {
 public:
  explicit CompiledNode(
      std::vector<std::vector<std::unique_ptr<CompositeInvocation>>>&& kernels)
      : kernels_(std::move(kernels)) {}

  /// Executes this node using the given execution state.
  void execute(ExecutionState& state);

  std::string toString() const;

 private:
  // The outer array represents parallel launchable sequences kernels. The inner
  // array is a sequence of consecutive kernels.
  std::vector<std::vector<std::unique_ptr<CompositeInvocation>>> kernels_;
};

struct ValueTypes;

/// Top level container for result of compiling a FX Graph to torch::wave.
/// Multiple Executors can share the same WaveGraph.
class WaveGraph {
 public:
  /// Analyzes 'graph' and creates an execution plan and fused kernels. The
  /// actual tensor content types and ranks come from 'weights'. Normalizes
  /// the graph first to fill in default attribute values from FunctionSchema.
  WaveGraph(nativert::Graph& graph, const ValueTypes& types);

  const std::vector<std::unique_ptr<CompiledNode>>& nodes() const {
    return nodes_;
  }

  std::string toString() const;

 private:
  // The executable graph. the nodes are executed sequentially. Each node has
  // internal prallelism.
  std::vector<std::unique_ptr<CompiledNode>> nodes_;
};

} // namespace torch::wave
