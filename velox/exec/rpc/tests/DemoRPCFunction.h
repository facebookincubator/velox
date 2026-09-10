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

#include "velox/exec/rpc/tests/ResponseSimulator.h"
#include "velox/expression/FunctionSignature.h"
#include "velox/expression/rpc/AsyncRPCFunction.h"

namespace facebook::velox::exec::rpc {

/// End-to-end test function: echoes its input through the async dispatch path.
///
/// Exercises the full AsyncRPCFunction lifecycle — initialize(),
/// dispatchPerRow(), buildOutput() — with no external dependency, using
/// ResponseSimulator for latency and failure.
///
/// The returned value is derived from the input, so a test can tell a working
/// dispatch from one that dropped or misrouted a row. It is prefixed rather
/// than returned verbatim so that a value which bypassed the RPC path
/// entirely, and simply carried the input column through, is also
/// distinguishable.
///
/// SQL usage:
///   SELECT demo_rpc('hello world')
///   -- Returns: "demo: hello world"
class DemoAsyncRPCFunction : public AsyncRPCFunction {
 public:
  /// The concrete request: this function's backend needs the prompt, so the
  /// request carries it. The framework's own row bookkeeping is separate and
  /// lives in the operator.
  struct Request {
    std::string prompt;
  };

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

  /// Dispatch one simulated RPC per active row. Null-input rows short-circuit
  /// to an error response so buildOutput() produces SQL NULL.
  std::vector<std::pair<vector_size_t, folly::SemiFuture<RPCResponse>>>
  dispatchPerRow(
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args) override;

  VectorPtr buildOutput(
      const std::vector<RPCResponse>& responses,
      memory::MemoryPool* pool) const override {
    return buildTextOutput(responses, pool);
  }

  /// Test hook: a non-error response carrying the "OVERLOAD" sentinel is
  /// treated as congestion, so RPCOperatorTest can drive the shrink path.
  CongestionSignal evaluateCongestion(
      const std::vector<RPCResponse>& responses) const override;

  /// SQL function signatures for registration.
  static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures();

  test::ResponseSimulator* simulator() {
    return simulator_.get();
  }

 private:
  std::shared_ptr<test::ResponseSimulator> simulator_;
};

} // namespace facebook::velox::exec::rpc
