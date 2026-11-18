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

#include <functional>

#include "velox/expression/rpc/JobReceipt.h"

namespace facebook::velox::exec::rpc {

/// Identifies the durable lifecycle operation served by a function instance.
enum class DeferredRpcPhase { kSubmit, kPoll, kCollect };

/// Adds durable job operations to the existing function preparation/output
/// hooks.
class DeferredRpcBackend : public AsyncRPCFunction {
 public:
  /// Rejects per-row dispatch for durable batch-only backends.
  std::vector<std::pair<vector_size_t, folly::SemiFuture<RPCResponse>>>
  dispatchPerRow(
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args) override;
  /// Returns a codec owned by this binding for the duration of live work.
  virtual const JobReceiptCodec& receiptCodec() const = 0;
  /// Acknowledges pending inputs and returns one receipt per original input
  /// slot, in pending-input order. The wrapper assigns positions by vector
  /// index; the backend must restore that order after correlating opaque IDs.
  virtual folly::SemiFuture<ReceiptBatch> submitPendingBatch(
      int32_t maxRows) = 0;
  /// Preserves membership, errors and the input ReceiptBatch vector order while
  /// waiting on accepted job handles. Backend job IDs do not define positions.
  virtual folly::SemiFuture<ReceiptBatch> poll(ReceiptBatch receipts) = 0;
  /// Returns one result/error per requested receipt. RPCResponse.rowId must
  /// cover local positions [0, receipts.size()) exactly once; vector order may
  /// differ. The backend owns correlation between opaque IDs and positions.
  virtual folly::SemiFuture<std::vector<RPCResponse>> collect(
      ReceiptBatch receipts) = 0;
  /// Counts current backend work using actual identities, including subsets.
  /// Returns zero when every receipt completes locally in the bound phase.
  virtual int32_t admissionUnitsForReceipts(
      const ReceiptBatch& receipts) const = 0;
  /// Stops local work; live continuations retain their state until completion.
  virtual void stop() noexcept = 0;
};

/// Creates a fresh phase binding; factories must not retain query credentials.
using DeferredRpcBackendFactory =
    std::function<std::shared_ptr<DeferredRpcBackend>(DeferredRpcPhase)>;

} // namespace facebook::velox::exec::rpc
