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

#include "velox/core/PlanNode.h"

namespace facebook::velox::exec::fixedpoint {

using core::PlanNode;
using core::PlanNodeId;
using core::PlanNodePtr;

/// Declares a named persistent state entry that survives across iterations.
///
/// Subclasses define the storage type and access pattern.  The fixed point
/// operator stores these by shared_ptr and does not interpret the contents —
/// operators in the body plans access the state through type-specific
/// mechanisms.
class StateDeclaration {
 public:
  virtual ~StateDeclaration() = default;

  const std::string& name() const {
    return name_;
  }

  /// Optional plan that produces the initial value of this state entry.
  /// Runs once in Phase 1.  nullptr means the state starts empty.
  const PlanNodePtr& initialPlan() const {
    return initialPlan_;
  }

 protected:
  StateDeclaration(std::string name, PlanNodePtr initialPlan)
      : name_{std::move(name)}, initialPlan_{std::move(initialPlan)} {}

 private:
  std::string name_;
  PlanNodePtr initialPlan_;
};

using StateDeclarationPtr = std::shared_ptr<const StateDeclaration>;

/// Vector state: row batches stored as std::vector<RowVectorPtr>.
///
/// Read via StateSourceNode, write via StateSinkNode.  Imperative operators
/// can extract raw pointers via FlatVector<T>::mutableRawValues().  An entry is
/// either replace-mode (default -- each write overwrites it) or append-mode
/// (writes accumulate across iterations); whether to accumulate is a property
/// of the entry, declared here, not of the StateSink that writes it.
class VectorStateDeclaration : public StateDeclaration {
 public:
  VectorStateDeclaration(
      std::string name,
      RowTypePtr schema,
      PlanNodePtr initialPlan = nullptr,
      bool append = false)
      : StateDeclaration{std::move(name), std::move(initialPlan)},
        schema_{std::move(schema)},
        append_{append} {}

  const RowTypePtr& schema() const {
    return schema_;
  }

  /// When true, a StateSink writing this entry appends its rows rather than
  /// replacing the entry, so it accumulates across iterations (e.g. a recursive
  /// CTE's UNION ALL).  The fixed point's output is then the entry's final
  /// contents.
  bool append() const {
    return append_;
  }

 private:
  RowTypePtr schema_;

  // Whether writes to this entry append (accumulate) or replace (default).
  bool append_;
};

/// HashTable state: BaseHashTable for key-value lookups or key-only
/// membership checks.
///
/// Operators access directly (not via StateSourceNode/StateSinkNode).
/// Extensible: additional StateDeclaration subclasses (e.g., for FAISS
/// index) can be added in the future.
class HashTableStateDeclaration : public StateDeclaration {
 public:
  HashTableStateDeclaration(
      std::string name,
      RowTypePtr schema,
      std::vector<std::string> keyColumns,
      PlanNodePtr initialPlan = nullptr)
      : StateDeclaration{std::move(name), std::move(initialPlan)},
        schema_{std::move(schema)},
        keyColumns_{std::move(keyColumns)} {}

  /// Full row schema (key + value columns).
  const RowTypePtr& schema() const {
    return schema_;
  }

  /// Columns that form the hash table key.
  const std::vector<std::string>& keyColumns() const {
    return keyColumns_;
  }

 private:
  RowTypePtr schema_;
  std::vector<std::string> keyColumns_;
};

/// Convergence checking for the fixed point iteration.
///
/// Expresses "aggregate + predicate over a state entry" as a plan.  After each
/// iteration the framework runs `plan` and inspects its single BOOLEAN output
/// column (named `convergedColumn`); a true value stops the iteration.  The
/// plan begins with a StateSourceNode reading the convergence state entry and
/// typically takes the shape:
///
///   StateSource(entry) -> Aggregation(global) -> Project(predicate AS
///   converged)
///
/// Unifies all convergence modes:
///
///   count(1) == 0       -> false until empty   (VLP: empty frontier)
///   bool_or(isTarget)   -> target found        (SHORTEST)
///   sum(active) == 0    -> no changes          (CC)
///   max(delta) < eps    -> centroids settled   (KMeans)
///
/// When the convergence state entry is empty the plan may emit zero rows; the
/// framework treats "no output row" as converged (the canonical empty-frontier
/// terminal state).  maxIterations (a ConvergenceConfig field below) is always
/// active as a safety bound.
///
/// Multi-worker contract: each worker evaluates `plan` over its own local
/// shard, so for a shuffling fixed point (N > 1) the verdict must be globally
/// consistent -- every worker must reach the same value on the same iteration,
/// or lockstep breaks and the shuffle deadlocks.  Making it consistent is the
/// plan's responsibility: synchronize the convergence-deciding state across
/// workers through the body's shuffle (e.g. replicate it), so each worker's
/// local read agrees.  The framework deliberately performs no cross-worker
/// reduction (that would add an all-reduce shuffle).  A null plan means "never
/// converge" (the loop is bounded only by maxIterations) and is always safe.
struct ConvergenceConfig {
  /// Plan producing a single BOOLEAN column.  Starts with a StateSourceNode.
  /// Null means never converge (loop bounded by maxIterations).
  PlanNodePtr plan;

  /// Name of the BOOLEAN output column of `plan` that signals convergence.
  std::string convergedColumn{"converged"};

  /// Maximum iterations the loop runs -- always active as a safety bound, and
  /// the sole bound when `plan` is null (a fixed-count loop).
  int32_t maxIterations;

  /// When true (the default), the fixed point fails if it runs all
  /// `maxIterations` without converging.  Requires a convergence `plan`
  /// (FixedPointNode validates this) -- a null plan never converges, so set
  /// this false for a fixed-count loop.
  bool errorWhenMaxIterationReached{true};
};

/// Orchestrates iterative computation over persistent state.
///
/// Holds a list of state declarations, a list of plans to run each iteration,
/// a convergence config, and a max iteration bound.  The execution model is
/// three-phase:
///
///   Phase 1: Initialize persistent state (run each state's initial plan once).
///   Phase 2: Iterate — run plans sequentially as sub-tasks each iteration.
///   Phase 3: Output — produce final result from persistent state.
///
/// Plans are run as separate sub-tasks because Exchange is a SourceOperator
/// that must be in a separate Task from PartitionedOutput.  The convention is:
///
///   - The first plan starts with StateSourceNode.
///   - The last plan ends with StateSinkNode.
///   - Every non-last plan ends with PartitionedOutput.
///   - Every non-first plan starts with Exchange.
///   - For in-process use (no exchange), the list has a single plan.
///
/// A shuffling fixed point (first plan ends with PartitionedOutput) runs as N
/// peer top-level tasks — the workers — the same FixedPointNode with distinct
/// task destinations 0..N-1 (the worker / shard index); N is that
/// PartitionedOutput's partition count.  The coordinator wires the workers'
/// shuffle topology by adding remote splits to the parent tasks (this node is a
/// split source): a split's taskId is a peer worker's parent task id, naming a
/// worker this one shuffles with.  The sub-tasks inherit the topology — a
/// producer sub-task's id is derived from its parent task id, so a worker that
/// knows a peer's parent id (from a split) can address that peer's
/// per-iteration producer.  The topology is the coordinator's choice
/// (all-to-all, 1-to-1, etc.), carried by splits, not by the plan.
class FixedPointNode : public PlanNode {
 public:
  FixedPointNode(
      PlanNodeId id,
      std::vector<StateDeclarationPtr> stateDeclarations,
      std::vector<PlanNodePtr> plans,
      ConvergenceConfig convergenceConfig,
      std::string outputStateEntry,
      RowTypePtr outputType)
      : PlanNode{std::move(id)},
        stateDeclarations_{std::move(stateDeclarations)},
        plans_{std::move(plans)},
        convergenceConfig_{std::move(convergenceConfig)},
        outputStateEntry_{std::move(outputStateEntry)},
        outputType_{std::move(outputType)} {
    validatePlans();
    // errorWhenMaxIterationReached fails the loop when maxIterations is reached
    // without converging; that is only meaningful with a convergence plan (a
    // null plan never converges, so it would always fail).
    VELOX_USER_CHECK(
        !convergenceConfig_.errorWhenMaxIterationReached ||
            convergenceConfig_.plan != nullptr,
        "FixedPointNode: errorWhenMaxIterationReached requires a convergence "
        "plan; set it false for a fixed-count loop with no convergence plan");
  }

  const RowTypePtr& outputType() const override {
    return outputType_;
  }

  /// Returns empty — plans are sub-tasks, not pipeline sources.
  const std::vector<PlanNodePtr>& sources() const override {
    return kEmptySources;
  }

  std::string_view name() const override {
    return "FixedPoint";
  }

  const std::vector<StateDeclarationPtr>& stateDeclarations() const {
    return stateDeclarations_;
  }

  const std::vector<PlanNodePtr>& plans() const {
    return plans_;
  }

  int32_t maxIterations() const {
    return convergenceConfig_.maxIterations;
  }

  const ConvergenceConfig& convergenceConfig() const {
    return convergenceConfig_;
  }

  /// The state entry whose final contents this node emits.  Output shaping
  /// (e.g. a recursive CTE's UNION ALL of every iteration) is expressed by the
  /// plan accumulating into this entry -- e.g. an append-mode StateSink -- not
  /// by a framework output mode.
  const std::string& outputStateEntry() const {
    return outputStateEntry_;
  }

  /// A fixed point is a split source when the coordinator must assign it
  /// splits: the body shuffles (first plan ends with PartitionedOutput, so the
  /// coordinator names peer workers), or an initial plan reads from a split
  /// source (a TableScan's file splits, or an Exchange's upstream task).  A
  /// fully local fixed point needs no splits.
  bool requiresSplits() const override;

 protected:
  void addDetails(std::stringstream& stream) const override {
    stream << "maxIterations: " << convergenceConfig_.maxIterations
           << ", plans: " << plans_.size();
  }

 private:
  static inline const std::vector<PlanNodePtr> kEmptySources{};

  // Validates the sub-plan chaining convention (see the class comment): plans
  // are joined only by shuffle, so the first reads state, the last writes
  // state, and adjacent plans are linked by PartitionedOutput -> Exchange.
  // Throws a VeloxUserError if violated.  Defined in the .cpp, which sees the
  // complete StateSource / StateSink node types it inspects.
  void validatePlans() const;

  std::vector<StateDeclarationPtr> stateDeclarations_;

  // Plans to run sequentially each iteration as separate sub-tasks.
  std::vector<PlanNodePtr> plans_;

  ConvergenceConfig convergenceConfig_;

  // Vector state entry whose final contents form this node's output.
  std::string outputStateEntry_;

  RowTypePtr outputType_;
};

/// Source operator that reads a Vector persistent state entry as row batches.
///
/// Always a leaf node (no sources).  Used as the first operator in a plan
/// within a FixedPointNode iteration.  Reads from the named state entry in the
/// parent task's persistent state.  Whether the read yields an append entry's
/// latest delta (the in-loop frontier) or its full accumulation (the fixed
/// point's output) is an execution decision made by the operator, not declared
/// here.
class StateSourceNode : public PlanNode {
 public:
  StateSourceNode(PlanNodeId id, std::string stateName, RowTypePtr outputType)
      : PlanNode{std::move(id)},
        stateName_{std::move(stateName)},
        outputType_{std::move(outputType)} {}

  const RowTypePtr& outputType() const override {
    return outputType_;
  }

  const std::vector<PlanNodePtr>& sources() const override {
    return kEmptySources;
  }

  std::string_view name() const override {
    return "StateSource";
  }

  const std::string& stateName() const {
    return stateName_;
  }

 protected:
  void addDetails(std::stringstream& stream) const override {
    stream << "state: " << stateName_;
  }

 private:
  static inline const std::vector<PlanNodePtr> kEmptySources{};

  // Name of the persistent state entry to read from.
  std::string stateName_;

  RowTypePtr outputType_;
};

/// Sink operator that writes row batches to a Vector persistent state entry.
///
/// Has exactly one source.  Used as the last operator in a plan within a
/// FixedPointNode iteration.  Collects all input rows and writes them to the
/// named entry, appending (accumulating across iterations) or replacing per the
/// entry's VectorStateDeclaration::append() flag -- whether to accumulate is a
/// property of the entry, not of the write.
class StateSinkNode : public PlanNode {
 public:
  StateSinkNode(PlanNodeId id, std::string stateName, PlanNodePtr source)
      : PlanNode{std::move(id)},
        stateName_{std::move(stateName)},
        sources_{std::move(source)} {}

  const RowTypePtr& outputType() const override {
    return sources_[0]->outputType();
  }

  const std::vector<PlanNodePtr>& sources() const override {
    return sources_;
  }

  std::string_view name() const override {
    return "StateSink";
  }

  const std::string& stateName() const {
    return stateName_;
  }

 protected:
  void addDetails(std::stringstream& stream) const override {
    stream << "state: " << stateName_;
  }

 private:
  // Name of the persistent state entry to write to.
  std::string stateName_;

  std::vector<PlanNodePtr> sources_;
};

/// Inner hash join of the input (probe) against a HashTable persistent state
/// entry that is built once and reused across iterations (hash-table reuse).
/// Output rows are the probe input columns followed by the hash table's
/// dependent (payload) columns.  The probe input's join key columns must occupy
/// the same channels as the build key columns (keys-first on both sides).
///
/// Only inner join is supported: probe rows with no match are dropped (this is
/// what gives VLP / recursive CTE its empty-frontier termination).  Supporting
/// other join types would require a JoinType parameter here, emitting null
/// build columns on misses in the operator, and build-side probed-flag/dedup
/// handling for semi/anti joins.
class StateHashJoinNode : public PlanNode {
 public:
  StateHashJoinNode(
      PlanNodeId id,
      std::string stateName,
      std::vector<std::string> probeKeys,
      RowTypePtr outputType,
      PlanNodePtr source)
      : PlanNode{std::move(id)},
        stateName_{std::move(stateName)},
        probeKeys_{std::move(probeKeys)},
        outputType_{std::move(outputType)},
        sources_{std::move(source)} {}

  const RowTypePtr& outputType() const override {
    return outputType_;
  }

  const std::vector<PlanNodePtr>& sources() const override {
    return sources_;
  }

  std::string_view name() const override {
    return "StateHashJoin";
  }

  const std::string& stateName() const {
    return stateName_;
  }

  const std::vector<std::string>& probeKeys() const {
    return probeKeys_;
  }

 protected:
  void addDetails(std::stringstream& stream) const override {
    stream << "state: " << stateName_;
  }

 private:
  // Name of the HashTable persistent state entry to probe.
  std::string stateName_;

  // Probe-side join key column names.
  std::vector<std::string> probeKeys_;

  RowTypePtr outputType_;

  std::vector<PlanNodePtr> sources_;
};

} // namespace facebook::velox::exec::fixedpoint
