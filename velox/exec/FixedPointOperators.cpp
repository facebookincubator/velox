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
#include "velox/exec/FixedPointOperators.h"

#include "velox/core/FixedPointPlanNodes.h"
#include "velox/exec/FixedPointLoop.h"
#include "velox/exec/HashTable.h"
#include "velox/exec/Operator.h"
#include "velox/exec/PersistentState.h"
#include "velox/exec/Task.h"
#include "velox/exec/VectorHasher.h"

namespace facebook::velox::exec {
namespace {

using core::FixedPointNode;
using core::StateHashJoinNode;
using core::StateSourceNode;

// Resolves the PersistentState holding entry 'name' for this sub-task.  The
// state belongs to the FixedPointLoop that created the sub-task and linked
// itself as the parent task.  When one fixed point is nested inside another's
// body, the enclosing loops are that task's own ancestors, so the search walks
// up the parent chain to the nearest one declaring 'name' -- innermost wins,
// as lexical scoping of the state names in the plan implies.
PersistentState* stateOf(
    const exec::OperatorCtx* operatorCtx,
    const std::string& name,
    bool hashTable) {
  auto* parent = operatorCtx->task()->parentFixedPoint();
  VELOX_CHECK_NOT_NULL(
      parent, "Fixed point sub-task has no FixedPointLoop parent");
  for (auto* loop = parent; loop != nullptr;
       loop = loop->owner()->parentFixedPoint()) {
    auto* state = loop->state();
    VELOX_CHECK_NOT_NULL(state, "FixedPointLoop has no persistent state");
    if (hashTable ? state->hasHashTable(name) : state->hasVector(name)) {
      return state;
    }
  }
  VELOX_FAIL(
      "No enclosing fixed point declares {} state entry '{}'",
      hashTable ? "hash table" : "vector",
      name);
}

// Emits the batches of a Vector persistent state entry, snapshotted at
// initialize() time so concurrent writes to the same entry do not affect this
// read.  A delta source reads what an iteration consumes -- an append entry's
// latest delta (the frontier) or a replace entry's full contents; a non-delta
// source reads the entry's whole contents (the fixed point's output, an append
// entry's whole accumulation).
class StateSourceOperator : public exec::SourceOperator {
 public:
  StateSourceOperator(
      int32_t operatorId,
      exec::DriverCtx* driverCtx,
      const core::StateSourceNodePtr& node)
      : SourceOperator(
            driverCtx,
            node->outputType(),
            operatorId,
            node->id(),
            "FixedPointStateSource"),
        stateName_{node->stateName()},
        delta_{node->delta()} {}

  void initialize() override {
    Operator::initialize();
    auto* state = stateOf(operatorCtx_.get(), stateName_, /*hashTable=*/false);
    batches_ =
        delta_ ? state->readVector(stateName_) : state->getVector(stateName_);
  }

  RowVectorPtr getOutput() override {
    while (current_ < batches_.size()) {
      auto batch = batches_[current_++];
      if (batch != nullptr && batch->size() > 0) {
        return batch;
      }
    }
    return nullptr;
  }

  exec::BlockingReason isBlocked(ContinueFuture* /* future */) override {
    return exec::BlockingReason::kNotBlocked;
  }

  bool isFinished() override {
    return current_ >= batches_.size();
  }

  void close() override {
    SourceOperator::close();
    // 'batches_' holds shared references to vectors allocated from the owning
    // FixedPointLoop's state pool.  Release them here rather than in the
    // destructor: the loop waits only on each sub-task's taskCompletionFuture,
    // which fires when drivers close, while this operator object can be
    // destroyed later on an executor thread, after the loop and its pool are
    // gone.  Releasing in close() keeps every state-pool reference inside the
    // loop's lifetime.
    batches_.clear();
  }

 private:
  const std::string stateName_;

  // Whether to read the per-iteration delta (an append entry's frontier) or the
  // entry's full contents (the output).
  const bool delta_;

  // Snapshot of the state entry taken in initialize().
  std::vector<RowVectorPtr> batches_;

  // Index of the next batch to emit.
  size_t current_{0};
};

// Inner-joins the probe input against a HashTable persistent state entry built
// once and reused across iterations.  Emits the probe columns (wrapped via the
// matched probe-row indices) followed by the hash table's dependent columns
// (extracted from the matched build rows).
class StateHashJoinOperator : public exec::Operator {
 public:
  StateHashJoinOperator(
      int32_t operatorId,
      exec::DriverCtx* driverCtx,
      const core::StateHashJoinNodePtr& node)
      : Operator(
            driverCtx,
            node->outputType(),
            operatorId,
            node->id(),
            "FixedPointStateHashJoin"),
        stateName_{node->stateName()},
        probeType_{node->sources()[0]->outputType()} {}

  void initialize() override {
    Operator::initialize();
    auto entry = stateOf(operatorCtx_.get(), stateName_, /*hashTable=*/true)
                     ->getHashTable(stateName_);
    VELOX_CHECK(
        entry.has_value(), "No hash table state registered: {}", stateName_);
    entry_ = std::move(entry);
    table_ = entry_->table.get();
    lookup_ = std::make_unique<exec::HashLookup>(table_->hashers(), pool());
  }

  bool needsInput() const override {
    return !noMoreInput_ && input_ == nullptr && probe_ == nullptr;
  }

  void addInput(RowVectorPtr input) override {
    input_ = std::move(input);
  }

  RowVectorPtr getOutput() override {
    if (probe_ == nullptr) {
      if (input_ == nullptr) {
        return nullptr;
      }
      // Start the next probe batch: look its rows up once, then page the
      // matches out over the calls that follow.
      probe_ = std::move(input_);
      SelectivityVector activeRows(probe_->size());
      lookup_->reset(probe_->size());
      table_->prepareForJoinProbe(
          *lookup_, probe_, activeRows, /*decodeAndRemoveNulls=*/true);
      if (lookup_->rows.empty()) {
        probe_ = nullptr;
        return nullptr;
      }
      lookup_->hits.resize(lookup_->rows.back() + 1);
      table_->joinProbe(*lookup_);
      iterator_.reset(*lookup_);
    }

    // One output vector per call, sized by the configured batch: a high-fanout
    // probe matches arbitrarily more rows than it has, so draining every match
    // into a single vector would ignore preferredOutputBatchRows entirely.
    const vector_size_t maxRows = outputBatchRows();
    indexBuffer_.resize(maxRows);
    rowBuffer_.resize(maxRows);
    const auto numMatches = table_->listJoinResults(
        iterator_,
        /*includeMisses=*/false,
        folly::Range(indexBuffer_.data(), maxRows),
        folly::Range(rowBuffer_.data(), maxRows),
        /*maxBytes=*/1 << 20);
    if (numMatches == 0) {
      // This probe batch is exhausted; the next call takes the next input.
      probe_ = nullptr;
      return nullptr;
    }
    auto output = makeOutput(probe_, numMatches);
    if (iterator_.atEnd()) {
      probe_ = nullptr;
    }
    return output;
  }

  exec::BlockingReason isBlocked(ContinueFuture* /* future */) override {
    return exec::BlockingReason::kNotBlocked;
  }

  bool isFinished() override {
    return noMoreInput_ && input_ == nullptr && probe_ == nullptr;
  }

  void close() override {
    Operator::close();
    // 'entry_' is a hash table built in the owning FixedPointLoop's state
    // pool, and 'input_'/'probe_' can also reference state-pool vectors when
    // this operator reads behind a StateSource.  Release them here rather
    // than in the destructor: the loop waits only on each sub-task's
    // taskCompletionFuture, which fires when drivers close, while this
    // operator object can be destroyed later on an executor thread, after
    // the loop and its pool are gone.  'lookup_' borrows the table's hashers,
    // so it is released before 'entry_'.
    input_ = nullptr;
    probe_ = nullptr;
    lookup_.reset();
    table_ = nullptr;
    entry_.reset();
  }

 private:
  // Assembles one output vector from the first 'numOutput' matches buffered in
  // indexBuffer_ / rowBuffer_: probe columns wrapped by the matched probe-row
  // indices, followed by the hash table's dependent columns extracted from the
  // matched build rows.
  RowVectorPtr makeOutput(const RowVectorPtr& probe, vector_size_t numOutput) {
    auto indices = AlignedBuffer::allocate<vector_size_t>(numOutput, pool());
    std::memcpy(
        indices->asMutable<vector_size_t>(),
        indexBuffer_.data(),
        numOutput * sizeof(vector_size_t));

    std::vector<VectorPtr> children;
    children.reserve(outputType_->size());
    for (auto channel = 0; channel < probeType_->size(); ++channel) {
      children.push_back(
          BaseVector::wrapInDictionary(
              /*nulls=*/nullptr, indices, numOutput, probe->childAt(channel)));
    }
    const auto numKeys = entry_->numKeys;
    const auto& buildType = entry_->buildType;
    for (auto column = numKeys; column < buildType->size(); ++column) {
      auto result =
          BaseVector::create(buildType->childAt(column), numOutput, pool());
      table_->extractColumn(
          folly::Range<char* const*>(rowBuffer_.data(), numOutput),
          column,
          result);
      children.push_back(std::move(result));
    }
    return std::make_shared<RowVector>(
        pool(), outputType_, /*nulls=*/nullptr, numOutput, std::move(children));
  }

  const std::string stateName_;
  const RowTypePtr probeType_;

  std::optional<HashTableEntry> entry_;
  exec::BaseHashTable* table_{nullptr};
  std::unique_ptr<exec::HashLookup> lookup_;

  // Probe batch awaiting lookup.
  RowVectorPtr input_;

  // Probe batch whose matches are being paged out, and the walk over them.
  // Both outlive a getOutput() call: one probe batch can yield many output
  // vectors.
  RowVectorPtr probe_;
  exec::BaseHashTable::JoinResultIterator iterator_{{}, 0, std::nullopt};

  // Matches listed by the current getOutput() call: probe-row indices and the
  // build rows they matched.
  std::vector<vector_size_t> indexBuffer_;
  std::vector<char*> rowBuffer_;
};

// The FixedPointNode itself runs as a FixedPointLoop composed into its
// task, not
// an Operator, so this translator only handles the per-iteration state
// operators that run inside its sub-tasks.
class FixedPointOperatorTranslator : public exec::Operator::PlanNodeTranslator {
 public:
  std::unique_ptr<exec::Operator> toOperator(
      exec::DriverCtx* ctx,
      int32_t id,
      const core::PlanNodePtr& node) override {
    if (auto fixedPoint =
            std::dynamic_pointer_cast<const FixedPointNode>(node)) {
      // A FixedPointNode that reaches a driver pipeline is the leaf of a
      // FixedPointLoop's trailing plan: read the loop's already-computed
      // output state (the enclosing FixedPointLoop populated it before
      // running the trailing plan). The output reads the entry's full contents
      // (an append entry's whole accumulation, not just its latest delta), so
      // the synthesized source is a non-delta read.
      auto stateSource = std::make_shared<StateSourceNode>(
          fixedPoint->id(),
          fixedPoint->outputStateEntry(),
          fixedPoint->outputType(),
          /*delta=*/false);
      return std::make_unique<StateSourceOperator>(id, ctx, stateSource);
    }
    if (auto source = std::dynamic_pointer_cast<const StateSourceNode>(node)) {
      // The node's delta flag selects an append entry's latest delta (frontier)
      // vs. its full accumulation; immaterial for a replace entry.
      return std::make_unique<StateSourceOperator>(id, ctx, source);
    }
    if (auto hashJoin =
            std::dynamic_pointer_cast<const StateHashJoinNode>(node)) {
      return std::make_unique<StateHashJoinOperator>(id, ctx, hashJoin);
    }
    return nullptr;
  }
};

} // namespace

void registerFixedPoint() {
  exec::Operator::registerOperator(
      std::make_unique<FixedPointOperatorTranslator>());
}

} // namespace facebook::velox::exec
