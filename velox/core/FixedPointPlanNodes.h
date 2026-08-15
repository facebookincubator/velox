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

namespace facebook::velox::core {

/// Declares a named persistent state entry that survives across iterations.
///
/// Subclasses define the storage type and access pattern.  The fixed point
/// operator stores these by shared_ptr and does not interpret the contents —
/// operators in the body plans access the state through type-specific
/// mechanisms.
class StateDeclaration : public ISerializable {
 public:
  const std::string& name() const {
    return name_;
  }

  /// Optional plan that produces the initial value of this state entry.
  /// Runs once in Phase 1.  nullptr means the state starts empty.
  const PlanNodePtr& initialPlan() const {
    return initialPlan_;
  }

  /// Serializes this declaration, including its concrete subtype so that
  /// deserialize() can reconstruct the right class.
  folly::dynamic serialize() const override = 0;

  /// Reconstructs the concrete StateDeclaration named by obj["type"].
  static std::shared_ptr<const StateDeclaration> deserialize(
      const folly::dynamic& obj,
      void* context);

  /// Renders this declaration's name and kind-specific properties for plan
  /// printing (FixedPointNode::addDetails).
  virtual std::string toString() const = 0;

 protected:
  StateDeclaration(std::string name, PlanNodePtr initialPlan)
      : name_{std::move(name)}, initialPlan_{std::move(initialPlan)} {}

  // Serializes the common name + initialPlan fields under the subtype tag
  // 'type'; subclasses extend the returned object with their own fields.
  folly::dynamic serializeBase(std::string_view type) const;

 private:
  std::string name_;
  PlanNodePtr initialPlan_;
};

using StateDeclarationPtr = std::shared_ptr<const StateDeclaration>;

/// Stores row batches as a std::vector<RowVectorPtr> persistent state entry.
///
/// Choose this when the entry is scanned or accumulated, or is the mutable loop
/// variable -- anything read sequentially rather than probed by key (a probed
/// index belongs in HashTableStateDeclaration).
///
/// Read via StateSourceNode; the framework writes the mutable output entry each
/// iteration.  Imperative operators can extract raw pointers via
/// FlatVector<T>::mutableRawValues().  An entry is either replace-mode (default
/// -- each write overwrites it) or append-mode (writes accumulate across
/// iterations); whether to accumulate is a property of the entry, declared
/// here, not of the write.
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

  /// When true, writing this entry appends its rows rather than replacing the
  /// entry, so it accumulates across iterations (e.g. a recursive CTE's UNION
  /// ALL).  The fixed point's output is then the entry's final contents.
  bool append() const {
    return append_;
  }

  folly::dynamic serialize() const override;

  std::string toString() const override;

 private:
  RowTypePtr schema_;

  // Whether writes to this entry append (accumulate) or replace (default).
  bool append_;
};

/// Fluent builder for a VectorStateDeclaration -- declares a vector state entry
/// without a raw make_shared, e.g.
///
///   VectorState("n", rowType, /*append=*/true).initial(seedPlan)
class VectorState {
 public:
  VectorState(std::string name, RowTypePtr schema, bool append = false)
      : name_{std::move(name)}, schema_{std::move(schema)}, append_{append} {}

  /// Sets the plan producing the entry's initial value (run once in Phase 1).
  VectorState& initial(PlanNodePtr initialPlan);

  /// Builds the declaration.  Implicit so the builder can be passed wherever a
  /// StateDeclarationPtr is expected (e.g. fixedPoint's state list).
  operator StateDeclarationPtr() const {
    return std::make_shared<VectorStateDeclaration>(
        name_, schema_, initialPlan_, append_);
  }

 private:
  std::string name_;
  RowTypePtr schema_;
  bool append_;
  PlanNodePtr initialPlan_;
};

/// Holds a BaseHashTable for key-value lookups or key-only membership
/// checks.
///
/// Choose this when the entry is probed by key each iteration (a built-once
/// index): declare its `keyColumns` and read it via StateHashJoinNode.  An
/// entry scanned or accumulated sequentially belongs in VectorStateDeclaration
/// instead.
///
/// Operators access directly (not via StateSourceNode).
/// Extensible: additional StateDeclaration subclasses (e.g., for FAISS
/// index) can be added in the future.
class HashTableStateDeclaration : public StateDeclaration {
 public:
  HashTableStateDeclaration(
      std::string name,
      RowTypePtr schema,
      std::vector<std::string> keyColumns,
      PlanNodePtr initialPlan = nullptr);

  /// Full row schema (key + value columns).
  const RowTypePtr& schema() const {
    return schema_;
  }

  /// Columns that form the hash table key.
  const std::vector<std::string>& keyColumns() const {
    return keyColumns_;
  }

  folly::dynamic serialize() const override;

  std::string toString() const override;

 private:
  RowTypePtr schema_;
  std::vector<std::string> keyColumns_;
};

/// Fluent builder for a HashTableStateDeclaration, e.g.
///
///   HashTableState("friends", schema, {"person"}).initial(friendsPlan)
class HashTableState {
 public:
  HashTableState(
      std::string name,
      RowTypePtr schema,
      std::vector<std::string> keyColumns)
      : name_{std::move(name)},
        schema_{std::move(schema)},
        keyColumns_{std::move(keyColumns)} {}

  /// Sets the plan producing the table's build rows (run once in Phase 1).
  HashTableState& initial(PlanNodePtr initialPlan);

  /// Builds the declaration.  Implicit so the builder can be passed wherever a
  /// StateDeclarationPtr is expected (e.g. fixedPoint's state list).
  operator StateDeclarationPtr() const {
    return std::make_shared<HashTableStateDeclaration>(
        name_, schema_, keyColumns_, initialPlan_);
  }

 private:
  std::string name_;
  RowTypePtr schema_;
  std::vector<std::string> keyColumns_;
  PlanNodePtr initialPlan_;
};

/// Configures convergence checking for the fixed point iteration.
///
/// Expresses "aggregate + predicate over a state entry" as a plan.  After each
/// iteration the framework runs `plan` and inspects its single BOOLEAN output
/// column; a true value stops the iteration.  The plan begins with a
/// StateSourceNode reading the convergence state entry and typically takes the
/// shape:
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
  /// Plan producing exactly one BOOLEAN output column.  Starts with a
  /// StateSourceNode.  The framework reads that single column after each
  /// iteration; a true value stops the loop.  Null means never converge (loop
  /// bounded by maxIterations).
  PlanNodePtr plan{nullptr};

  /// Maximum iterations the loop runs -- always active as a safety bound, and
  /// the sole bound when `plan` is null (a fixed-count loop).
  int32_t maxIterations{0};

  /// Decides what happens when `maxIterations` is reached before the
  /// convergence `plan` signals convergence.  When true (the default), the
  /// fixed point fails -- a guard against silently returning a non-converged
  /// result.  When false, the loop instead stops at the cap and returns the
  /// current approximate result without failing: best-effort convergence (run
  /// toward convergence, but accept the partial result if the cap is hit
  /// first), as in an early-stopped KMeans.  Requires a convergence `plan`
  /// (FixedPointNode validates this) -- a null plan never converges, so set
  /// this false for a fixed-count loop with no convergence plan, where the cap
  /// is the intended stopping point rather than a best-effort cutoff.
  bool errorWhenMaxIterationReached{true};

  /// Builds a fixed-count loop config: no convergence plan, bounded only by
  /// `maxIterations`, so it never errors on reaching the bound.  Use for loops
  /// with a known iteration count (e.g. Fibonacci, a bounded expansion).
  static ConvergenceConfig withMaxIterations(int32_t maxIterations);

  /// Builds a convergence-plan loop config: runs `plan` after each iteration
  /// and stops when its BOOLEAN output is true, failing if `maxIterations` is
  /// reached first.
  static ConvergenceConfig converging(PlanNodePtr plan, int32_t maxIterations);

  folly::dynamic serialize() const;

  static ConvergenceConfig deserialize(
      const folly::dynamic& obj,
      void* context);

  /// Renders the convergence properties for plan printing
  /// (FixedPointNode::addDetails): max iteration bound, whether reaching it is
  /// an error, and whether a convergence plan is present.
  std::string toString() const;
};

/// Orchestrates iterative computation over persistent state, expressing
/// recursive CTEs, variable-length graph paths, shortest path, connected
/// components, and KMeans/PageRank-style iteration as one node.  Holds a list
/// of StateDeclarations (named state that survives across iterations), a list
/// of body plans run sequentially as sub-tasks each iteration, and a
/// ConvergenceConfig (the termination test plus an iteration bound).  Exactly
/// one VectorStateDeclaration -- named by `outputStateEntry` -- is the mutable
/// loop variable: the framework writes each iteration's last plan output back
/// into it (appending or replacing per its declaration) and emits its final
/// contents as the node's output; every other state entry is seeded once and
/// read-only thereafter.
class FixedPointNode : public PlanNode {
 public:
  FixedPointNode(
      PlanNodeId id,
      std::vector<StateDeclarationPtr> stateDeclarations,
      std::vector<PlanNodePtr> plans,
      ConvergenceConfig convergenceConfig,
      std::string outputStateEntry);

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

  /// The mutable loop variable: the VectorStateDeclaration whose final contents
  /// this node emits and into which the framework writes each iteration's last
  /// plan output.  Output shaping (e.g. a recursive CTE's UNION ALL of every
  /// iteration) is expressed by declaring this entry append-mode, not by a
  /// framework output mode.
  const std::string& outputStateEntry() const {
    return outputStateEntry_;
  }

  /// A fixed point is a split source when the coordinator must assign it
  /// splits: the body shuffles (first plan ends with PartitionedOutput, so the
  /// coordinator names peer workers), or an initial plan reads from a split
  /// source (a TableScan's file splits, or an Exchange's upstream task).  A
  /// fully local fixed point needs no splits.
  bool requiresSplits() const override;

  folly::dynamic serialize() const override;

  static PlanNodePtr create(const folly::dynamic& obj, void* context);

 protected:
  void addDetails(std::stringstream& stream) const override;

 private:
  static inline const std::vector<PlanNodePtr> kEmptySources{};

  // Validates the sub-plan chaining convention (see the class comment): plans
  // are joined only by shuffle, so the first reads state, the last produces the
  // rows written back, and adjacent plans are linked by PartitionedOutput ->
  // Exchange.  Throws a VeloxUserError if violated.  Defined in the .cpp, which
  // sees the complete node types it inspects.
  void validatePlans() const;

  // Resolves every StateSource / StateHashJoin reference (in the body plans and
  // the convergence plan) to a declared state entry of the matching kind, and
  // checks that referenced schemas, probe-key arity, initial-plan and last-plan
  // output schemas, and the convergence-plan contract are consistent.  Throws a
  // VeloxUserError if violated.
  void resolveAndValidateStateReferences() const;

  std::vector<StateDeclarationPtr> stateDeclarations_;

  // Plans to run sequentially each iteration as separate sub-tasks.
  std::vector<PlanNodePtr> plans_;

  ConvergenceConfig convergenceConfig_;

  // Name of the mutable vector state entry this node writes and outputs.
  std::string outputStateEntry_;

  // Derived in the constructor from the schema of the VectorStateDeclaration
  // named by outputStateEntry_.
  RowTypePtr outputType_;
};

/// Reads a Vector persistent state entry as row batches into the pipeline.
///
/// Always a leaf node (no sources).  Used as the first operator in a plan
/// within a FixedPointNode iteration.  Reads from the named state entry in the
/// parent task's persistent state.  `delta` selects what an append entry
/// yields: true returns the latest delta -- the rows written in the most recent
/// iteration (the in-loop frontier); false returns the entry's full
/// accumulation.  For a replace entry the two are identical.
class StateSourceNode : public PlanNode {
 public:
  StateSourceNode(
      PlanNodeId id,
      std::string stateName,
      RowTypePtr outputType,
      bool delta)
      : PlanNode{std::move(id)},
        stateName_{std::move(stateName)},
        outputType_{std::move(outputType)},
        delta_{delta} {}

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

  /// When true, an in-loop read yields the append entry's latest delta (the
  /// frontier); when false, its full accumulation.  Immaterial for a replace
  /// entry, where the two coincide.
  bool delta() const {
    return delta_;
  }

  folly::dynamic serialize() const override;

  static PlanNodePtr create(const folly::dynamic& obj, void* context);

 protected:
  void addDetails(std::stringstream& stream) const override {
    stream << "state: " << stateName_ << (delta_ ? ", delta" : ", full");
  }

 private:
  static inline const std::vector<PlanNodePtr> kEmptySources{};

  // Name of the persistent state entry to read from.
  std::string stateName_;

  RowTypePtr outputType_;

  // Whether an in-loop read yields the append entry's latest delta (frontier)
  // or its full accumulation.  Immaterial for a replace entry.
  bool delta_;
};

/// Inner-joins the input (probe) against a HashTable persistent state entry
/// that is built once and reused across iterations (hash-table reuse). Output
/// rows are the probe input columns followed by the hash table's dependent
/// (payload) columns.  The probe input's join key columns must occupy the same
/// channels as the build key columns (keys-first on both sides).
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
      PlanNodePtr source);

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

  folly::dynamic serialize() const override;

  static PlanNodePtr create(const folly::dynamic& obj, void* context);

 protected:
  void addDetails(std::stringstream& stream) const override;

 private:
  // Name of the HashTable persistent state entry to probe.
  std::string stateName_;

  // Probe-side join key column names.
  std::vector<std::string> probeKeys_;

  RowTypePtr outputType_;

  std::vector<PlanNodePtr> sources_;
};

} // namespace facebook::velox::core
