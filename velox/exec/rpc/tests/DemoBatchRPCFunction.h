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

#include <unordered_set>

#include "velox/expression/FunctionSignature.h"
#include "velox/expression/rpc/AsyncRPCFunction.h"

namespace facebook::velox::exec::rpc {

/// Batch-capable AsyncRPCFunction for testing RPCOperator's BATCH mode.
///
/// Supports configurable response behaviors to exercise correctness
/// edge cases in the batch dispatch pipeline:
///   - In-order responses (default, happy path)
///   - Reversed responses (simulates backend returning out of order)
///   - Per-row failures (simulates partial batch failure)
///   - Whole-batch failure (simulates an operator-level batch timeout / RPC
///     error where the flush future itself fails, not individual rows)
///
/// Returns "Batch response for: <prompt>" for each non-null, non-failing row.
/// Null inputs produce RPCResponse{.error = "null_input"}.
/// Failing rows produce RPCResponse{.error = "simulated_failure"}.
/// With failWholeBatch=true, flushBatch() returns a FAILED future instead.
class DemoBatchRPCFunction : public AsyncRPCFunction {
 public:
  enum class ResponseOrder {
    kInOrder,
    kReversed,
  };

  explicit DemoBatchRPCFunction(
      ResponseOrder order = ResponseOrder::kInOrder,
      std::unordered_set<int32_t> failingRowIndices = {},
      bool failWholeBatch = false,
      bool failOnError = false,
      bool dropOneResponse = false);

  void initialize(
      const core::QueryConfig& queryConfig,
      const std::vector<TypePtr>& inputTypes,
      const std::vector<VectorPtr>& constantInputs) override;

  std::string name() const override {
    return "demo_batch_rpc";
  }

  TypePtr resultType() const override {
    return VARCHAR();
  }

  /// With failOnError=true, mimics the meta_ai_on_error='fail' policy: any
  /// errored response hard-fails the query (VELOX_USER_FAIL) instead of NULLing
  /// the row. Otherwise defers to the base (errors -> NULL).
  VectorPtr buildOutput(
      const std::vector<RPCResponse>& responses,
      memory::MemoryPool* pool) const override;

  std::vector<std::pair<vector_size_t, folly::SemiFuture<RPCResponse>>>
  dispatchPerRow(
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args) override;

  std::vector<vector_size_t> accumulateBatch(
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args) override;

  folly::SemiFuture<std::vector<RPCResponse>> flushBatch(
      int32_t maxRows) override;

  int32_t pendingBatchSize() const override;

  static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures();

 private:
  struct PendingRow {
    std::string prompt;
    bool isNull;
  };

  std::vector<PendingRow> pendingRows_;
  ResponseOrder responseOrder_;
  std::unordered_set<int32_t> failingRowIndices_;
  bool failWholeBatch_{false};
  bool failOnError_{false};
  bool dropOneResponse_{false};
  int32_t totalAccumulatedCount_{0};
};

} // namespace facebook::velox::exec::rpc
