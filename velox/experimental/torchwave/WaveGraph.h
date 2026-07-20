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

#include <fmt/format.h>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <folly/container/F14Map.h>

#include <torch/nativert/executor/OpKernel.h>
#include <torch/nativert/graph/Graph.h>
#include <torch/nativert/graph/TensorMeta.h>
#include <torch/nativert/kernels/KernelFactory.h>
#include "velox/experimental/torchwave/KernelOperation.h"
#include "velox/experimental/torchwave/Registry.h"

namespace facebook::velox::wave {
class CompiledKernel;
} // namespace facebook::velox::wave

namespace torch::wave {

class CompiledNode;
class CompileCtx;
class Optimizer;
struct ExecutionState;

/// Rank and layout constraints for a graph value, used during optimization.
struct ValueConstraint {
  int8_t rank{-1};
  /// True when the value is known at compile time to be row-major contiguous
  /// (dense, standard strides). Defaults to false: a wrong `true` could let a
  /// kernel read a strided tensor as dense and corrupt results, whereas a
  /// conservative `false` only costs an unnecessary copy. Set true for ops that
  /// always materialize a fresh dense output (elementwise, cumsum, masked
  /// select, cat, clone, contiguous, factory ops, ...) and propagated by view
  /// and reshape; computed per PyTorch semantics for slice/select.
  bool contiguous{false};
};

/// Per-value tensor metadata and constraints for a WaveGraph.
struct ValueTypes {
  /// Tensor metadata for tensor type values, indexed by Value id().
  std::vector<const nativert::TensorMeta*> types;
  std::vector<ValueConstraint> constraints;

  int8_t rank(ValueCP value) const {
    auto id = value->id();
    TORCH_CHECK(
        id >= 0 && static_cast<size_t>(id) < constraints.size(),
        "Value id out of range: ",
        id);
    return constraints[id].rank;
  }

  /// Whether 'value' is known to be contiguous. Returns false (conservative)
  /// for values with no tracked constraint.
  bool contiguous(ValueCP value) const {
    auto id = value->id();
    if (id < 0 || static_cast<size_t>(id) >= constraints.size()) {
      return false;
    }
    return constraints[id].contiguous;
  }
};

/// Populates 'types' from the graph's tensor metadata. Sizes types and
/// constraints to graph.values().size(), then fills entries from
/// tensorValuesMeta. Allocated TensorMeta objects are appended to 'metaStore'.
void initValueTypes(
    const nativert::Graph& graph,
    ValueTypes& types,
    std::vector<std::unique_ptr<nativert::TensorMeta>>& metaStore);

/// Accumulated execution timing for a standalone (non-fused) operation.
struct StandaloneStats {
  int64_t micros{0};
};

/// Per-node metadata cache for a WaveGraph node.
struct NodeInfo {
  const Metadata* metadata;
};

/// Returns a thread-local reference to the current WaveGraph being compiled.
WaveGraph*& waveGraph();

/// Returns the Metadata for 'node', caching the result in the current
/// WaveGraph's nodeInfos map.
const Metadata* nodeMeta(NodeCP node);

/// Context needed to create and recreate OpKernels. Holds the graph,
/// weights, and config that KernelFactory and WaveGraph require.
struct ModelContext {
  /// Exclusively owns the nativert::Graph. WaveGraph and WaveGraphExecutor
  /// mutate the graph internally during compilation (normalization, op
  /// substitution, value creation). The graph must not be shared.
  std::unique_ptr<nativert::Graph> graph;
  std::shared_ptr<nativert::Weights> weights;
  nativert::ExecutorConfig config;

  /// Creates OpKernels for all nodes in the graph.
  std::vector<std::unique_ptr<nativert::OpKernel>> makeKernels() const {
    nativert::KernelFactory factory;
    auto execKernels =
        factory.initializeNodeKernels(*graph, weights, config, nullptr);
    return std::move(execKernels.nodeKernels);
  }
};

/// Top level container for result of compiling a FX Graph to torch::wave.
/// WaveGraphExecutor owns the WaveGraph . One WaveGraphExecutor can run
/// multiple executions on different threads. The WaveGraph is immutable during
/// execution.
///
/// Compilation stages:
///
/// 1. **Normalize** — Fills in missing attribute defaults from
///    FunctionSchema and substitutes/expands ops that have a multipart
///    implementation.
///
/// 2. **Optimize** — Propagates ValueConstraints (rank, dtype) backwards
///    from outputs through producers. Identifies redundant views and
///    reshapes that can be elided when rank is 1.
///
/// 3. **Partition into ProjectNodes** — Splits the graph into consecutive
///    layers (ProjectNode) where each layer's root expressions are
///    independent. A ProjectNode groups nodes that can execute in a
///    single fused GPU kernel launch.
///
/// 4. **Compile each ProjectNode** — For each ProjectNode, the CompileCtx:
///    a. Extracts subgraphs rooted at each expression. Deduplicates subgraphs
///    that only differ in inputs or constants to limit code size. b. Places
///    nodes as fused (GPU kernel cases) or standalone (host-side
///       OpKernel execution) via placeKernels.
///    c. Generates variant subgraphs (single-block, multi-block,
///       cooperative-grid) for ops with multiple implementations.
///    d. Generates CUDA kernel code (elementwise loops, device function
///       calls, shared memory declarations) for fused operations.
///    e. Compiles the combined kernel source via NVRTC into a
///       CompositeKernel.
///
/// 5. **Build execution plan** — Collects standalone indices, syncable
///    value ids, and per-step launch grids into CompiledNode objects
///    that the WaveGraphExecutor iterates at runtime.
class WaveGraph {
 public:
  /// Borrows the ModelContext (owned by WaveGraphExecutor) and mutates the
  /// graph internally (normalization, op substitution, value creation).
  explicit WaveGraph(ModelContext* modelContext);

  /// Normalizes and optimizes 'graph' without compiling kernels.
  static std::unique_ptr<WaveGraph> optimizeOnly(
      nativert::Graph& graph,
      const ValueTypes& types);

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

  /// Adds a TensorList output to 'node' with the given name and registers it in
  /// idToValue_. No TensorMeta is created (a list has no element-level meta);
  /// element values are obtained via Value::getListElements().
  nativert::Value* newListValue(nativert::Node* node, std::string_view name);

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

  /// Returns a placeholder node in the main graph, creating it on first call.
  nativert::Node* placeholderNode() {
    if (!placeholderNode_) {
      placeholderNode_ = graph_->createNode("tw.placeholder", {});
    }
    return placeholderNode_;
  }

  /// Returns the pre-built map from Value::id() to Value*.
  const IdToValueMap& idToValue() const {
    return idToValue_;
  }

  /// Fills in missing attribute defaults from FunctionSchema and creates
  /// multiKernelVariants_ for nodes that have one.
  void normalizeAndAnnotateGraph();

  /// Propagates constraints for the outputs of 'node' using the shared
  /// Optimizer instance. The optimizer's visited set ensures main-graph
  /// nodes are not re-traversed.
  void optimizeNode(const nativert::Node* node);

  /// Returns the multikernel variant subgraph for 'node', or nullptr if none.
  const Subgraph* multiKernelVariant(NodeCP node) const {
    auto it = multiKernelVariants_.find(node);
    return it != multiKernelVariants_.end() ? &it->second : nullptr;
  }

  /// Returns a unique name by appending _NN to the given name.
  std::string uniqueName(std::string_view name) {
    std::string candidate;
    do {
      candidate = fmt::format("{}_{}", name, nextValueId_++);
    } while (graph_->tryGetValue(candidate) != nullptr);
    return candidate;
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

  const folly::F14FastMap<NodeCP, int32_t>& standaloneIndices() const {
    return standaloneIndices_;
  }

  std::vector<StandaloneStats>& standaloneStats() {
    return standaloneStats_;
  }

  const std::vector<StandaloneStats>& standaloneStats() const {
    return standaloneStats_;
  }

  CompileCtx* compileCtx() const {
    return compileCtx_;
  }

  /// Records that 'value' is consumed by more than one part of a multipart op
  /// expansion (e.g. the shared input of cumsum_head and cumsum_final), so it
  /// must not be released as a per-op freeable intermediate. Called by the
  /// split / variant lowerings while generating the parts. The set lives here
  /// (not on the transient CompileCtx) because it is consulted at execution
  /// time, when LaunchData decides which kernel outputs are freeable.
  void declareMultiplyReferencedInput(const nativert::Value* value);

  bool isMultiUseInput(nativert::ValueId id) const {
    return multiUseInputs_.count(id) != 0;
  }

  /// True if 'id' is a graph output value (or an element of a list-typed graph
  /// output). Such values escape the graph and must never be released as a
  /// per-op freeable intermediate.
  bool isGraphOutput(nativert::ValueId id) const {
    return graphOutputIds_.count(id) != 0;
  }

  /// Returns the ModelContext, or nullptr if none was provided.
  ModelContext* modelContext() const {
    return modelContext_;
  }

  folly::F14FastMap<NodeCP, NodeInfo>& nodeInfos() {
    return nodeInfos_;
  }

  std::string toString(Listing mode = kExprs) const;

  /// Takes ownership of a variant graph and returns a raw pointer.
  nativert::Graph* addVariantGraph(std::unique_ptr<nativert::Graph> graph) {
    variantGraphs_.push_back(std::move(graph));
    return variantGraphs_.back().get();
  }

  /// The variant graph currently being built by variantSubgraph, or nullptr.
  nativert::Graph* currentVariantGraph() const {
    return currentVariantGraph_;
  }

  void setCurrentVariantGraph(nativert::Graph* graph) {
    currentVariantGraph_ = graph;
  }

 private:
  struct OptimizeOnlyTag {};
  WaveGraph(nativert::Graph& graph, ValueTypes types, OptimizeOnlyTag);

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
  // newTensorValue/newScalarValue, keyed by value id.
  std::unordered_map<nativert::ValueId, c10::ScalarType> createdValueDtypes_;

  // Placeholder node used by duplicateValue to attach new Values.
  nativert::Node* placeholderNode_{nullptr};

  // For nodes that have a multikernel implementation, like multiblock
  // reduction, this gives the subgraph to substitute for the Node when
  // generating the multiblock case of a ProjectOperation.
  std::unordered_map<NodeCP, Subgraph> multiKernelVariants_;

  // Counter for generating unique value names via uniqueName().
  int32_t nextValueId_{0};

  // Counter for assigning sequence numbers to CompositeInvocations.
  int32_t nextCompositeInvocationId_{0};

  // Maps each actual standalone Node to a serial index (0-based).
  folly::F14FastMap<NodeCP, int32_t> standaloneIndices_;

  // Accumulated timing for each standalone, indexed by standaloneIndices_.
  std::vector<StandaloneStats> standaloneStats_;

  // ValueIds of outputs whose Metadata has shapeSetOnDevice or neededOnHost.
  std::unordered_set<nativert::ValueId> syncableValueIds_;

  // Graphs created by CompileCtx::variantSubgraph, owned here for lifetime.
  std::vector<std::unique_ptr<nativert::Graph>> variantGraphs_;

  // Set to the graph being built during variantSubgraph, nullptr otherwise.
  nativert::Graph* currentVariantGraph_{nullptr};

  // Cached Metadata lookups keyed by node pointer.
  folly::F14FastMap<NodeCP, NodeInfo> nodeInfos_;

  // Retained for recreating OpKernels after graph mutations.
  ModelContext* modelContext_;

  // Set during construction, cleared after.
  CompileCtx* compileCtx_{nullptr};

  // Values consumed by more than one part of a multipart op expansion; they
  // must never be freed as per-op intermediates. Populated at compile time via
  // declareMultiplyReferencedInput, read at execution time by LaunchData.
  std::unordered_set<nativert::ValueId> multiUseInputs_;

  // Graph output value ids (plus elements of list-typed outputs). Escaping
  // values that must never be freed as per-op intermediates. Populated at the
  // start of compile, read at execution time by LaunchData.
  std::unordered_set<nativert::ValueId> graphOutputIds_;

  // Alive during construction only. Retains visited set so multikernel
  // variant nodes reuse the main-graph pass.
  std::unique_ptr<Optimizer> optimizer_;

  // Pool of reusable ExecutionState objects.
  std::mutex statePoolMutex_;
  std::vector<std::unique_ptr<ExecutionState>> statePool_;
};

nativert::Graph* variantNodeGraph(WaveGraph* waveGraph);

nativert::Value* newVariantTensorValue(
    nativert::Node* node,
    WaveGraph* waveGraph,
    std::string_view name,
    c10::ScalarType dtype);

nativert::Value* newVariantScalarValue(
    nativert::Node* node,
    WaveGraph* waveGraph,
    std::string_view name,
    c10::ScalarType dtype);

void copyOriginalOutputs(
    nativert::Node* node,
    NodeCP original,
    WaveGraph* waveGraph);

} // namespace torch::wave
