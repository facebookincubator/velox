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

struct ValueTypes {
  /// Tensor metadata for tensor type values, indexed by Value id().
  std::vector<const nativert::TensorMeta*> types;
};

struct ResultSpec {
  const nativert::Value* value{nullptr};
  std::string variable;
};

struct Subgraph {
  const nativert::Node* root;
  std::vector<const nativert::Value*> inputs;
  std::vector<const nativert::TensorMeta*> inputTypes;
};

bool subgraphsMatch(const Subgraph& left, const Subgraph& right);

struct SubgraphHash {
  size_t operator()(const Subgraph& sg) const;
};

struct SubgraphEqual {
  bool operator()(const Subgraph& left, const Subgraph& right) const {
    return subgraphsMatch(left, right);
  }
};

using SubgraphMap =
    std::unordered_map<Subgraph, KernelOperation*, SubgraphHash, SubgraphEqual>;

class CompileCtx {
 public:
  using NodeSet = std::unordered_set<const nativert::Node*>;

  explicit CompileCtx(const ValueTypes& types) : types_{types} {}

  std::unique_ptr<CompiledNode> compileNode(ProjectNode& node);

  KernelOperation* makeKernelOperation(const Subgraph& sg);

  void generateElementwise(const Subgraph& sg, const ResultSpec& resultSpec);

  std::vector<const nativert::Value*> elementwiseHead(
      const nativert::Node* node,
      KernelOperation* op);

  void elementWiseBody(
      const nativert::Node* node,
      const KernelOperation& op,
      const std::vector<const nativert::Value*>& inputs,
      std::string resultName,
      std::string resultStmt,
      bool fullBlockResult);

  std::string elementWiseExpr(
      const nativert::Node* node,
      const KernelOperation& op,
      const std::vector<const nativert::Value*>& inputs);

  void addInclude(std::string_view header);

  std::string declareAttributes(
      const nativert::Node* node,
      const KernelOperation& op,
      const std::vector<const nativert::Value*>& inputs);

  std::string cudaType(const nativert::Value* value) const;

  /// Declares a temporary variable of the CUDA type for 'scalarType'. Appends a
  /// declaration line to declarations_ and returns the variable name.
  std::string declare(c10::ScalarType scalarType);

  Subgraph extractSubgraph(
      const nativert::Node* node,
      const NodeSet& inputs,
      const NodeSet& placed);

  bool isElementWise(const nativert::Node& node, const NodeSet& placed = {})
      const;

  bool hasBarrier(const nativert::Node& node, const NodeSet& placed = {}) const;

  bool isSingleBlock(const nativert::Node& node, const NodeSet& placed = {})
      const;

  bool hasStandalone(const nativert::Node& node, const NodeSet& placed = {})
      const;

  bool isMultikernel(const nativert::Node& node, const NodeSet& placed = {})
      const;

  static int32_t nextKernelId();

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

  const ValueTypes& types_;
  int32_t nextOpCode_{0};
  std::unordered_set<std::string> includes_;
  std::stringstream code_;
  std::stringstream declarations_;
  int32_t declareCounter_{0};
  const std::unordered_set<const nativert::Node*>* inputs_;
  NodeSet placed_;

  // Offset of param corresponding to Value in the kernel's BlockInfo::params.
  std::unordered_map<const nativert::Value*, int32_t> valueParamOffset_;

  bool singleBlockTarget_{false};

  std::vector<OpInvocation> ops_;
  // Start of params for the op being produced.
  int32_t paramOffset_{0};
  SubgraphMap kernelOps_;
  // Stable storage for KernelOperations so pointers remain valid.
  std::vector<std::unique_ptr<KernelOperation>> opStorage_;
  // Stable storage for IValues so OpInvocation can hold const pointers.
  std::deque<c10::IValue> ivalueStorage_;
};

} // namespace torch::wave
