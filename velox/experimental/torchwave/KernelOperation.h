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
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <ATen/core/ivalue.h>
#include <folly/container/F14Map.h>

#include <torch/nativert/graph/Graph.h>
#include <torch/nativert/graph/TensorMeta.h>
#include "velox/experimental/torchwave/KernelParams.h"
#include "velox/experimental/torchwave/Registry.h"

namespace torch::wave {

class CompileCtx;

enum Listing { kExprs = 0, kGrids, kCode };

/// Map from Value::id() to Value*, pre-built from the graph for fast lookups.
using IdToValueMap = std::unordered_map<int32_t, ValueCP>;

/// Describes inputs and outputs of elementwise expr. Inputs may have to be
/// adjusted to accord to output for broadcast.
struct ElementExpr {
  ValueCP output;
  std::vector<ValueCP> inputs;
  std::unordered_map<ValueCP, int32_t> altParamOffset;
  bool shapeFromThisOp{false};
};

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

  /// Needs a fake tensor to indicate output shape but has no backing memory.
  bool shapeOnly{false};

  /// True if allocation is delegated to another output, e.g. when creating
  /// coordinated views.
  bool delegated{false};

  /// If this is a view that is not allocated, it still gets a storage from some
  /// Value.
  ValueCP storageFrom{nullptr};

  /// If set, this output is produced by executing a standalone view node
  /// (e.g. view, reshape) rather than allocating a new tensor.
  NodeCP viewNode{nullptr};

  /// Shortcut for all elementwise, where we already know the largest input and
  /// can set the output shape by that.
  bool byLargestInput{false};

  SizeExpr sizeExpr;

  bool isList{false};
};

void mergeOutputDesc(OutputDesc& dst, OutputDesc&& src);

bool addOrUpdateOutput(
    std::vector<ValueCP>& outputValues,
    std::vector<OutputDesc>& outputDescs,
    ValueCP value,
    OutputDesc desc);

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
  std::string toString(Listing mode = kExprs) const;
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
  KernelOperation(const Subgraph& sg, int32_t opCode, CompileCtx& compileCtx);

  NodeCP executableNode(NodeCP node);

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
      bool inMemory,
      bool callerIsElementwise = false);

  NodeCP expr() const {
    return expr_;
  }

  int32_t numInputs() const {
    return numInputs_;
  }

  const std::vector<ValueCP>& orderedInputs() const {
    return orderedInputs_;
  }

  bool isInput(ValueCP value) const {
    return inputs_.count(value);
  }

  const std::vector<OutputDesc>& outputDescs() const {
    return outputDescs_;
  }

  /// Returns the set of values backed by memory. Includes all inputs and
  /// outputs whose OutputDesc is not shapeOnly.
  std::unordered_set<ValueCP> memOutputs() const {
    std::unordered_set<ValueCP> result(
        orderedInputs_.begin(), orderedInputs_.begin() + numInputs_);
    for (size_t i = 0; i < outputDescs_.size(); ++i) {
      if (!outputDescs_[i].shapeOnly) {
        result.insert(orderedInputs_[numInputs_ + i]);
      }
    }
    return result;
  }

  float unitCost() const {
    return unitCost_;
  }

  int32_t constantAreaOffset() const {
    return constantAreaOffset_;
  }

  int32_t numConstants() const {
    return numConstants_;
  }

  int32_t altParamOffset() const {
    return altParamOffset_;
  }

  int32_t allocAltParam(int32_t size = sizeof(Tensor)) {
    auto offset = altParamOffset_;
    altParamOffset_ += size;
    return offset;
  }

  int32_t allocateBarrier() {
    auto offset = altParamOffset_;
    barrierCounters_.push_back(offset);
    altParamOffset_ += 8;
    return offset;
  }

  const std::vector<int32_t>& barrierCounters() const {
    return barrierCounters_;
  }

  /// Returns the offset of the first alternate param and the number of
  /// alternate params.
  std::pair<int32_t, int32_t> altParams() const {
    return {
        altParamStart_, (altParamOffset_ - altParamStart_) / sizeof(Tensor)};
  }

  const std::vector<ElementExpr>& elementExprs() const {
    return elementExprs_;
  }

  std::vector<ElementExpr>& elementExprs() {
    return elementExprs_;
  }

  const std::unordered_set<NodeCP>& allNodes() const {
    return allNodes_;
  }

  std::unordered_set<NodeCP>& allNodes() {
    return allNodes_;
  }

  const std::vector<std::string>& sharedDeclarations() const {
    return sharedDeclarations_;
  }

  /// Returns the param offsets of all tensor-type values (inputs and outputs).
  std::vector<int32_t> tensorParamOffsets() const;

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

  std::string toString(Listing mode = kExprs) const;

  const std::string& exprString() const {
    return exprString_;
  }

  const std::unordered_set<nativert::ValueId>& orderingInputs() const {
    return orderingInputs_;
  }

  const std::unordered_set<nativert::ValueId>& orderingOutputs() const {
    return orderingOutputs_;
  }

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

  // The compilation context.
  CompileCtx& compileCtx_;

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

  int32_t numConstants_{0};

  int32_t altParamStart_{0};
  int32_t altParamOffset_{0};

  std::vector<int32_t> barrierCounters_;

  std::vector<ElementExpr> elementExprs_;

  std::unordered_set<NodeCP> allNodes_;

  // Unfusable standalone Nodes that can execute in parallel with each other and
  // the kernel.
  std::vector<NodeCP> standalones_;

  // True if any node in the subgraph has alwaysSingleBlock set in its metadata.
  bool alwaysSingleBlock_{false};

  // True if this kernel is present in both grid_ and singleBlockGrid_ (both
  // grids have a kernel at the corresponding position).
  bool isGridChoice_{false};

  std::string exprString_;

  std::unordered_set<nativert::ValueId> orderingInputs_;
  std::unordered_set<nativert::ValueId> orderingOutputs_;

  std::string makeExprString();
};

/// Builds a FormalToActual mapping from formal to actual value ids by walking
/// the formal and actual subgraphs in parallel. Maps inputs positionally and
/// outputs by parallel tree walk, stopping at inputs of formalSg.
FormalToActual makeBindings(
    const Subgraph& formalSg,
    const Subgraph& actualSg,
    const KernelOperation& op);

} // namespace torch::wave
