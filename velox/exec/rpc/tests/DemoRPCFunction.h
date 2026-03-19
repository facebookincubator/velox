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

#include "velox/common/rpc/clients/MockRPCClient.h"
#include "velox/expression/FunctionSignature.h"
#include "velox/expression/rpc/AsyncRPCFunction.h"

namespace facebook::velox::exec::rpc {

using velox::rpc::MockRPCClient;

/// Demo AsyncRPCFunction that uses MockRPCClient for end-to-end testing.
///
/// Demonstrates the full AsyncRPCFunction lifecycle:
///   1. initialize() — creates and caches the MockRPCClient
///   2. dispatchPerRow() — dispatches per-row RPCs via MockRPCClient
///   3. buildOutput() — uses base class default (error→null, result→varchar)
///
/// Returns "Response for: <prompt>" for each input row. No external
/// dependencies — runs entirely in-process with simulated latency.
///
/// SQL usage:
///   SELECT demo_rpc('hello world')
///   -- Returns: "Response for: hello world"
class DemoAsyncRPCFunction : public AsyncRPCFunction {
 public:
  /// Initialize the mock client. Called by RPCOperator during init.
  void initialize(
      const core::QueryConfig& queryConfig,
      const std::vector<TypePtr>& inputTypes,
      const std::vector<VectorPtr>& constantInputs) override;

  std::string name() const override {
    return "demo_rpc";
  }

  TypePtr resultType() const override {
    return VARCHAR();
  }

  /// Dispatch individual RPCs for each active row via MockRPCClient.
  /// Null-input rows get an immediate RPCResponse with error="null_input".
  std::vector<std::pair<vector_size_t, folly::SemiFuture<RPCResponse>>>
  dispatchPerRow(
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args) override;

  /// SQL function signatures for registration.
  static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures();

 private:
  std::shared_ptr<MockRPCClient> client_;
};

} // namespace facebook::velox::exec::rpc
