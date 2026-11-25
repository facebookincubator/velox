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

#include <chrono>
#include <memory>
#include <string>
#include <vector>

#include "velox/expression/rpc/DeferredRpcBackend.h"

namespace facebook::velox::exec::rpc {

/// Executes a registered durable phase through a backend-owned binding.
class DeferredRpcPhaseFunction : public AsyncRPCFunction {
 public:
  /// Binds a registered name and phase to a fresh backend factory result.
  DeferredRpcPhaseFunction(
      std::string name,
      DeferredRpcPhase phase,
      const DeferredRpcBackendFactory& factory);
  DeferredRpcPhaseFunction(const DeferredRpcPhaseFunction&) = delete;
  DeferredRpcPhaseFunction& operator=(const DeferredRpcPhaseFunction&) = delete;
  DeferredRpcPhaseFunction(DeferredRpcPhaseFunction&&) = delete;
  DeferredRpcPhaseFunction& operator=(DeferredRpcPhaseFunction&&) = delete;
  ~DeferredRpcPhaseFunction() override;
  /// Binds receipt storage to the query allocator before decoding any input.
  void setMemoryPool(std::shared_ptr<memory::MemoryPool> pool) override;
  /// Releases pending inputs and cancels local work without dropping live
  /// state.
  void close() noexcept override;
  /// Initializes backend-owned configuration before validating its receipt
  /// type.
  void initialize(
      const core::QueryConfig& queryConfig,
      const std::vector<TypePtr>& inputTypes,
      const std::vector<VectorPtr>& constantInputs,
      RPCStreamingMode instruction) override;
  std::string name() const override;
  TypePtr resultType() const override;
  std::string admissionKey() const override;
  int64_t numRetriesAttempted() const override;
  int64_t configuredCeiling() const override;
  double configuredAdaptiveFloor() const override;
  std::chrono::milliseconds requestTimeout() const override;
  bool requiresRowInspectionBeforeAdmission() const override;
  /// Delegates original input validation before any submission admission.
  AdmissionPreparationResult prepareInputForAdmissionAndDispatch(
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args) override;
  /// Rejects per-row execution independently of the provider's capabilities.
  std::vector<std::pair<vector_size_t, folly::SemiFuture<RPCResponse>>>
  dispatchPerRow(
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args) override;
  /// Buffers original inputs for Submit and owned receipts for later phases.
  std::vector<vector_size_t> accumulateBatch(
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args) override;
  /// Preserves input cardinality while serving the bound lifecycle operation.
  folly::SemiFuture<std::vector<RPCResponse>> flushBatch(
      int32_t maxRows) override;
  int32_t pendingBatchSize() const override;
  /// Applies original submission limits; receipt phases preserve their inputs.
  int32_t maxRowsPerFlush(int32_t desired) const override;
  /// Delegates backend units, including partial groups of job receipts.
  int32_t admissionUnitsForBatch(int32_t numRows) const override;
  /// Encodes non-null intermediate receipts or builds final collection results.
  VectorPtr buildOutput(
      const std::vector<RPCResponse>& responses,
      memory::MemoryPool* pool) const override;
  /// Retains receipt payload types and delegates phase-aware congestion policy.
  CongestionSignal evaluateCongestion(
      const std::vector<RPCResponse>& responses) const override;

 private:
  // Keeps accounting available to live continuations without retaining the
  // owner.
  struct ReceiptAccounting;
  // Charges the pending and prepared queues to their allocator.
  struct PendingReceipts;
  // Retains dispatched receipt ownership until a phase continuation finishes.
  struct ReceiptOwners;

  // Initializes accounting for direct callers that supply an input vector.
  void ensureMemoryPool(const std::vector<VectorPtr>& args);
  // Encodes lifecycle outcomes while preserving their already-accounted
  // storage.
  static std::vector<RPCResponse> receiptResponses(
      ReceiptBatch receipts,
      const std::shared_ptr<ReceiptAccounting>& accounting);
  // Uses the registered name for diagnostics and function lookup.
  const std::string name_;
  // Selects the lifecycle operation; physical dispatch remains BATCH.
  const DeferredRpcPhase phase_;
  // Owns copied query configuration and all live asynchronous backend state.
  const std::shared_ptr<DeferredRpcBackend> backend_;
  // Allows close to detach the allocator before late continuations finish.
  std::shared_ptr<ReceiptAccounting> accounting_;
  // Holds decoded receipt queues, whose storage is charged before dispatch.
  std::unique_ptr<PendingReceipts> receipts_;
  // Prevents repeated cleanup and dispatch after cancellation.
  bool closed_{false};
};

} // namespace facebook::velox::exec::rpc
