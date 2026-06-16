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

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <folly/futures/Future.h>

#include "velox/common/rpc/RPCTypes.h"
#include "velox/core/QueryConfig.h"
#include "velox/type/Type.h"
#include "velox/vector/BaseVector.h"
#include "velox/vector/FlatVector.h"
#include "velox/vector/SelectivityVector.h"

namespace facebook::velox::exec::rpc {

// Import core RPC types from velox/common/rpc into this namespace so that
// existing code in velox/expression/rpc can use them unqualified.
using velox::rpc::RPCRequest;
using velox::rpc::RPCResponse;
using velox::rpc::RPCStreamingMode;

/// Base interface for async RPC functions (business logic layer).
///
/// Lives in velox/expression/rpc/ because it is a function interface — it
/// defines what an RPC function is (signature, dispatch, response format),
/// analogous to VectorFunction in velox/expression/. Transport-layer types
/// (IRPCClient, RPCRequest, RPCResponse) live in velox/common/rpc/.
/// The execution operator (RPCOperator) that drives async dispatch lives
/// in velox/exec/rpc/.
///
/// AsyncRPCFunction owns the domain-specific logic: how to dispatch RPCs
/// from input rows (dispatchPerRow / accumulateBatch+flushBatch) and how
/// to interpret responses back into Velox vectors (buildOutput). The
/// function creates and holds its own transport/client internally.
///
/// Subclasses implement domain-specific dispatch and response handling
/// (e.g., LLM inference, embedding calls). The RPCOperator handles async
/// coordination: rate limiting, timeout wrapping, rowId assignment,
/// RPCState wiring, and passthrough columns.
///
/// Lifecycle (called by RPCOperator):
///   1. initialize(queryConfig, inputTypes, constantInputs) — create/cache
///      transport and RPC clients, inspect constant values (called once
///      during operator init).
///   2. dispatchPerRow(rows, args) — dispatch individual RPCs per row
///      OR accumulateBatch(rows, args) + flushBatch() — accumulate and
///      dispatch as a batch.
///   3. buildOutput() — convert RPC responses to output vectors.
class AsyncRPCFunction {
 public:
  virtual ~AsyncRPCFunction() = default;

  /// Initialize the RPC function with query configuration and constant
  /// arguments.
  /// Called by RPCOperator during initialize(), before any dispatch.
  /// Use this to create/cache RPC clients, read session properties, and
  /// inspect constant argument values (e.g., model name, options JSON).
  ///
  /// @param queryConfig Query configuration with session properties.
  /// @param inputTypes Types of each argument expression.
  /// @param constantInputs Constant values aligned with inputTypes.
  ///        Non-constant arguments are nullptr. Constant arguments are
  ///        single-element ConstantVectors.
  virtual void initialize(
      const core::QueryConfig& /*queryConfig*/,
      const std::vector<TypePtr>& /*inputTypes*/,
      const std::vector<VectorPtr>& /*constantInputs*/) {}

  /// Return the name of this RPC function.
  virtual std::string name() const = 0;

  /// Return the Velox type of the result column.
  virtual TypePtr resultType() const = 0;

  /// Return the service tier key for rate limiting.
  /// Empty string means "no tier configured — uses global default limit."
  virtual std::string tierKey() const {
    return "";
  }

  // ── PER_ROW mode ──────────────────────────────────────────────

  /// Dispatch individual RPCs for each active row.
  /// Returns one future per active row, keyed by originalRowIndex.
  ///
  /// The function:
  ///   1. Unpacks argument vectors (typed)
  ///   2. Builds typed requests directly
  ///   3. Dispatches via transport
  ///   4. Returns the futures
  ///
  /// Null-input rows: return an immediate RPCResponse with
  /// error="null_input".
  virtual std::vector<std::pair<vector_size_t, folly::SemiFuture<RPCResponse>>>
  dispatchPerRow(
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args) = 0;

  // ── BATCH mode ────────────────────────────────────────────────

  /// Accumulate rows for batch dispatch.
  /// Called by the operator on each addInput(). The function unpacks typed
  /// data from (rows, args) and stores it internally.
  ///
  /// Returns the original row indices for ALL processed rows (null and
  /// non-null). For null rows, the function stores a null marker
  /// internally. For non-null rows, the function stores typed data.
  ///
  /// The operator uses these indices for:
  /// 1. storeInputBatch(flattenedColumns, rowCount)
  /// 2. batchRowLocations_ — one entry per accumulated row
  /// 3. batchRowIds_ — assign one rowId per accumulated row
  virtual std::vector<vector_size_t> accumulateBatch(
      const SelectivityVector& /*rows*/,
      const std::vector<VectorPtr>& /*args*/) {
    VELOX_UNSUPPORTED(
        "accumulateBatch() not implemented for function '{}'", name());
  }

  /// Dispatch accumulated batch.
  /// Called by the operator at flush time (noMoreInput or threshold).
  /// The function builds the typed batch request from its internal
  /// accumulated state and dispatches it.
  ///
  /// @param maxRows Maximum number of rows to flush. 0 means flush all.
  /// Returns responses for the flushed rows. Null rows get
  /// RPCResponse{.error = "null_input"}. This keeps the operator
  /// completely agnostic to null handling in batch mode.
  virtual folly::SemiFuture<std::vector<RPCResponse>> flushBatch(
      int32_t /*maxRows*/) {
    VELOX_UNSUPPORTED("flushBatch() not implemented for function '{}'", name());
  }

  /// Convenience overload: flush all accumulated rows.
  virtual folly::SemiFuture<std::vector<RPCResponse>> flushBatch() {
    return flushBatch(0);
  }

  /// Number of rows accumulated so far (for threshold checks).
  /// Batch-capable functions MUST override this; the operator uses
  /// function_->pendingBatchSize() >= dispatchBatchSize_ to decide
  /// when to flush.
  virtual int32_t pendingBatchSize() const {
    return 0;
  }

  // ── Output ────────────────────────────────────────────────────

  /// Build output vector from completed responses.
  /// Default: VARCHAR FlatVector (errors → SQL NULL, success → string
  /// value). Override for non-VARCHAR return types (e.g., ARRAY(REAL)
  /// for embeddings) or custom result processing.
  virtual VectorPtr buildOutput(
      const std::vector<RPCResponse>& responses,
      memory::MemoryPool* pool) const {
    const auto numRows = static_cast<vector_size_t>(responses.size());
    auto result =
        BaseVector::create<FlatVector<StringView>>(VARCHAR(), numRows, pool);
    for (vector_size_t i = 0; i < numRows; ++i) {
      if (responses[i].hasError()) {
        result->setNull(i, true);
      } else {
        result->set(i, StringView(responses[i].result));
      }
    }
    return result;
  }

  // ── Congestion Control ───────────────────────────────────────

  /// Signal returned by evaluateCongestion() to indicate batch health.
  enum class CongestionSignal {
    /// Batch completed successfully — increase concurrency window.
    kSuccess,
    /// Batch had errors — decrease concurrency window.
    kError,
    /// No congestion evaluation — skip window adjustment.
    kNone,
  };

  /// Evaluate batch congestion from completed responses.
  /// Called by RPCOperator after a BATCH-mode batch completes.
  /// The function inspects responses and returns a signal that the
  /// operator maps to window adjustments (additive increase on
  /// kSuccess, multiplicative decrease on kError).
  /// Default: kNone (no congestion control).
  virtual CongestionSignal evaluateCongestion(
      const std::vector<RPCResponse>& /*responses*/) const {
    return CongestionSignal::kNone;
  }

  /// How much to increase the concurrency window on kSuccess.
  /// Override to tune recovery speed per client. Default: 2.
  virtual int64_t congestionRecoveryIncrement() const {
    return 2;
  }
};

} // namespace facebook::velox::exec::rpc
