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

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "velox/expression/rpc/AsyncRPCFunction.h"

namespace facebook::velox::exec::rpc {

/// Describes an input's durable job outcome without defining backend identity.
class JobReceipt {
 public:
  /// Names the lifecycle outcomes carried between phases.
  enum class Outcome { kSubmitted, kReady, kFailed, kNull };

  virtual ~JobReceipt() = default;
  /// Returns this input's acknowledged or locally terminal outcome.
  virtual Outcome outcome() const = 0;
  /// Returns a typed cause for failed/null outcomes and kNone for success.
  virtual velox::rpc::RPCErrorKind errorKind() const = 0;
  /// Distinguishes a current backend outcome from a replayed or local result.
  /// This phase-local metadata is not part of the persisted receipt schema.
  virtual bool reportsBackendOutcome() const = 0;
  /// Returns a safe diagnostic suitable for the final row failure policy.
  virtual std::string_view error() const = 0;
  /// Returns the original epoch-millisecond enqueue time, when applicable.
  virtual std::optional<int64_t> enqueueTime() const = 0;
  /// Returns the original epoch-millisecond deadline, when applicable.
  virtual std::optional<int64_t> absoluteDeadline() const = 0;
  /// Reports owned receipt storage; the receipt must contain no credentials.
  virtual int64_t retainedBytes() const noexcept = 0;
};

using JobReceiptPtr = std::shared_ptr<const JobReceipt>;
using ReceiptBatch = std::vector<JobReceiptPtr>;

/// Owns a receipt behind the existing checked response payload interface.
struct JobReceiptPayload {
  /// Retains backend data until asynchronous consumers have finished.
  JobReceiptPtr receipt;
  /// Retains an existing query-pool charge alongside the receipt's storage.
  std::shared_ptr<const void> accountingOwner{};

  int64_t retainedBytes() const noexcept {
    return receipt && !accountingOwner ? receipt->retainedBytes() : 0;
  }
};

/// Encodes backend-owned receipts as the ROW schema declared by a registration.
class JobReceiptCodec {
 public:
  virtual ~JobReceiptCodec() = default;
  /// Declares the concrete vector schema selected by the registered call.
  virtual RowTypePtr receiptType() const = 0;
  /// Declares the same schema for signature registration.
  virtual std::string signature() const = 0;
  /// Validates and copies one non-null row into owned backend receipt data.
  virtual JobReceiptPtr decode(const VectorPtr& rows, vector_size_t row)
      const = 0;
  /// Returns one non-null receipt row for each input receipt.
  virtual VectorPtr encode(
      const ReceiptBatch& receipts,
      memory::MemoryPool* pool) const = 0;
  /// Creates a local null/failure receipt, applying backend-safe diagnostics.
  virtual JobReceiptPtr fromError(
      velox::rpc::RPCErrorKind kind,
      std::string_view message) const = 0;
};

} // namespace facebook::velox::exec::rpc
