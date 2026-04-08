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

#include "velox/exec/rpc/tests/DemoRPCFunction.h"

namespace facebook::velox::exec::rpc {

void DemoAsyncRPCFunction::initialize(
    const core::QueryConfig& /*queryConfig*/,
    const std::vector<TypePtr>& /*inputTypes*/,
    const std::vector<VectorPtr>& /*constantInputs*/) {
  // Create and cache the mock client during initialization.
  client_ = std::make_shared<MockRPCClient>(
      std::chrono::milliseconds(1), // minimal latency
      0.0); // no errors
}

std::vector<std::pair<vector_size_t, folly::SemiFuture<RPCResponse>>>
DemoAsyncRPCFunction::dispatchPerRow(
    const SelectivityVector& rows,
    const std::vector<VectorPtr>& args) {
  std::vector<std::pair<vector_size_t, folly::SemiFuture<RPCResponse>>> results;

  if (args.empty()) {
    return results;
  }

  auto* promptVector = args[0]->as<SimpleVector<StringView>>();
  if (!promptVector) {
    return results;
  }

  rows.applyToSelected([&](vector_size_t row) {
    if (promptVector->isNullAt(row)) {
      // Null input → immediate error response.
      results.emplace_back(
          row,
          folly::makeSemiFuture<RPCResponse>(RPCResponse{
              .rowId = 0,
              .result = "",
              .metadata = {},
              .error = "null_input"}));
      return;
    }

    // Build RPCRequest for MockRPCClient (test utility still uses payload).
    RPCRequest request;
    request.payload = promptVector->valueAt(row).str();

    results.emplace_back(row, client_->call(request));
  });

  return results;
}

std::vector<std::shared_ptr<exec::FunctionSignature>>
DemoAsyncRPCFunction::signatures() {
  // 1-argument form: demo_rpc(prompt)
  auto sig = exec::FunctionSignatureBuilder()
                 .returnType("varchar")
                 .argumentType("varchar") // prompt
                 .build();

  return {std::move(sig)};
}

} // namespace facebook::velox::exec::rpc
