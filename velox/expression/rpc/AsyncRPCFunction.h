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
#include <vector>

#include "velox/common/rpc/IRPCClient.h"
#include "velox/common/rpc/RPCTypes.h"
#include "velox/core/QueryConfig.h"
#include "velox/type/Type.h"
#include "velox/vector/BaseVector.h"
#include "velox/vector/SelectivityVector.h"

namespace facebook::velox::exec::rpc {

// Import core RPC types from velox/common/rpc into this namespace so that
// existing code in velox/expression/rpc can use them unqualified.
using velox::rpc::IRPCClient;
using velox::rpc::RPCRequest;
using velox::rpc::RPCResponse;
using velox::rpc::RPCStreamingMode;

/// Base interface for async RPC functions (business logic layer).
///
/// Lives in velox/expression/rpc/ because it is a function interface — it
/// defines what an RPC function is (signature, request/response format),
/// analogous to VectorFunction in velox/expression/. Transport-layer types
/// (IRPCClient, RPCRequest, RPCResponse) live in velox/common/rpc/.
/// The execution operator (RPCOperator) that drives async dispatch lives
/// in velox/exec/rpc/.
///
/// AsyncRPCFunction owns the domain-specific logic: how to convert input rows
/// into RPC requests (prepareRequests) and how to interpret responses back into
/// Velox vectors (buildOutput). It is decoupled from the transport layer —
/// IRPCClient (in velox/common/rpc/) handles the actual network communication.
///
/// Subclasses implement domain-specific request/response handling (e.g., LLM
/// inference, embedding calls). The RPCOperator handles async execution,
/// batching, and result collection.
///
/// Lifecycle (called by RPCOperator):
///   1. initialize(queryConfig, inputTypes, constantInputs) — create/cache RPC
///      clients, inspect argument types and constant values (called once during
///      operator init). constantInputs[i] is non-null when argument i is a
///      constant expression (e.g., a literal model name or JSON options
///      string).
///   2. prepareRequests() — convert input rows to RPC requests
///   3. getClient()->call() or getBatchClient()->callBatch() — dispatch RPCs
///   4. buildOutput() — convert RPC responses to output vectors
class AsyncRPCFunction {
 public:
  virtual ~AsyncRPCFunction() = default;

  /// Initialize the RPC function with query configuration and constant
  /// arguments.
  /// Called by RPCOperator during initialize(), before any dispatch.
  /// Use this to create/cache RPC clients, read session properties, and
  /// inspect constant argument values (e.g., model name, options JSON).
  ///
  /// Follows the same pattern as SimpleFunction::initialize() and the stateful
  /// VectorFunction factory, which receive argument types and constant values
  /// at init time.
  ///
  /// @param queryConfig Query configuration with session properties.
  /// @param inputTypes Types of each argument expression.
  /// @param constantInputs Constant values aligned with inputTypes.
  /// Non-constant
  ///        arguments are nullptr. Constant arguments are single-element
  ///        ConstantVectors. Matches the VectorFunctionArg convention.
  virtual void initialize(
      const core::QueryConfig& /*queryConfig*/,
      const std::vector<TypePtr>& /*inputTypes*/,
      const std::vector<VectorPtr>& /*constantInputs*/) {}

  /// Return the name of this RPC function.
  virtual std::string name() const = 0;

  /// Return the Velox type of the result column.
  virtual TypePtr resultType() const = 0;

  /// Prepare RPC requests from input rows.
  ///
  /// @param rows Active rows in the input batch.
  /// @param args Evaluated argument columns (e.g., prompt, model name).
  /// @return One RPCRequest per active non-null row. Each request's
  ///         originalRowIndex is set to the row's position in the input batch.
  virtual std::vector<RPCRequest> prepareRequests(
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args) const = 0;

  /// Build output vector from RPC responses.
  ///
  /// @param responses Completed RPC responses (may include errors).
  /// @param pool Memory pool for allocating the output vector.
  /// @return Vector matching resultType(), one element per response.
  virtual VectorPtr buildOutput(
      const std::vector<RPCResponse>& responses,
      memory::MemoryPool* pool) const = 0;

  /// Get the RPC client for individual call() dispatch.
  virtual std::shared_ptr<IRPCClient> getClient() const = 0;

  /// Get the RPC client for callBatch() dispatch.
  /// Default returns getClient(). Override for backends with separate
  /// batch endpoints.
  virtual std::shared_ptr<IRPCClient> getBatchClient() const {
    return getClient();
  }
};

} // namespace facebook::velox::exec::rpc
