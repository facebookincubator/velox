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

#include "velox/exec/rpc/tests/DemoBatchRPCFunction.h"

#include <algorithm>

namespace facebook::velox::exec::rpc {

DemoBatchRPCFunction::DemoBatchRPCFunction(
    ResponseOrder order,
    std::unordered_set<int32_t> failingRowIndices,
    bool failWholeBatch,
    bool failOnError,
    bool dropOneResponse)
    : responseOrder_(order),
      failingRowIndices_(std::move(failingRowIndices)),
      failWholeBatch_(failWholeBatch),
      failOnError_(failOnError),
      dropOneResponse_(dropOneResponse) {}

VectorPtr DemoBatchRPCFunction::buildOutput(
    const std::vector<RPCResponse>& responses,
    memory::MemoryPool* pool) const {
  if (failOnError_) {
    for (const auto& r : responses) {
      if (r.hasError()) {
        VELOX_USER_FAIL("RPC call failed for row");
      }
    }
  }
  return AsyncRPCFunction::buildOutput(responses, pool);
}

void DemoBatchRPCFunction::initialize(
    const core::QueryConfig& /*queryConfig*/,
    const std::vector<TypePtr>& /*inputTypes*/,
    const std::vector<VectorPtr>& /*constantInputs*/) {}

std::vector<std::pair<vector_size_t, folly::SemiFuture<RPCResponse>>>
DemoBatchRPCFunction::dispatchPerRow(
    const SelectivityVector& /*rows*/,
    const std::vector<VectorPtr>& /*args*/) {
  VELOX_UNSUPPORTED(
      "DemoBatchRPCFunction is batch-only; use accumulateBatch/flushBatch");
}

std::vector<vector_size_t> DemoBatchRPCFunction::accumulateBatch(
    const SelectivityVector& rows,
    const std::vector<VectorPtr>& args) {
  std::vector<vector_size_t> indices;

  if (args.empty()) {
    return indices;
  }

  auto* promptVector = args[0]->as<SimpleVector<StringView>>();
  if (!promptVector) {
    return indices;
  }

  rows.applyToSelected([&](vector_size_t row) {
    indices.push_back(row);
    if (promptVector->isNullAt(row)) {
      pendingRows_.push_back({.prompt = "", .isNull = true});
    } else {
      pendingRows_.push_back(
          {.prompt = promptVector->valueAt(row).str(), .isNull = false});
    }
  });

  totalAccumulatedCount_ += static_cast<int32_t>(indices.size());
  return indices;
}

folly::SemiFuture<std::vector<RPCResponse>> DemoBatchRPCFunction::flushBatch(
    int32_t maxRows) {
  auto flushCount = maxRows > 0
      ? std::min(static_cast<int32_t>(pendingRows_.size()), maxRows)
      : static_cast<int32_t>(pendingRows_.size());

  std::vector<PendingRow> toFlush(
      pendingRows_.begin(), pendingRows_.begin() + flushCount);
  pendingRows_.erase(pendingRows_.begin(), pendingRows_.begin() + flushCount);

  // Simulate an operator-level batch failure (e.g. an RPC/batch timeout): the
  // whole flush future fails rather than returning per-row responses. Rows are
  // still consumed above so pendingBatchSize() drains and noMoreInput()
  // terminates.
  if (failWholeBatch_) {
    return folly::makeSemiFuture<std::vector<RPCResponse>>(
        std::runtime_error("simulated batch timeout"));
  }

  std::vector<RPCResponse> responses;
  responses.reserve(toFlush.size());

  // The index within this flush determines the "global accumulated index"
  // used to check failingRowIndices_. We use totalAccumulatedCount_ minus
  // the remaining pending rows to compute the starting offset.
  auto startOffset = totalAccumulatedCount_ -
      static_cast<int32_t>(pendingRows_.size()) - flushCount;

  for (int32_t i = 0; i < flushCount; ++i) {
    RPCResponse response;
    response.rowId = i;

    if (toFlush[i].isNull) {
      response.error = "null_input";
    } else if (failingRowIndices_.count(startOffset + i)) {
      response.error = "simulated_failure";
    } else {
      response.result = "Batch response for: " + toFlush[i].prompt;
    }
    responses.push_back(std::move(response));
  }

  if (responseOrder_ == ResponseOrder::kReversed) {
    std::reverse(responses.begin(), responses.end());
  }

  // Simulate a function-contract violation: return fewer responses than rows.
  // The operator's scatter must hard-fail on the count mismatch (not degrade).
  if (dropOneResponse_ && !responses.empty()) {
    responses.pop_back();
  }

  return folly::makeSemiFuture(std::move(responses));
}

int32_t DemoBatchRPCFunction::pendingBatchSize() const {
  return static_cast<int32_t>(pendingRows_.size());
}

std::vector<std::shared_ptr<exec::FunctionSignature>>
DemoBatchRPCFunction::signatures() {
  auto sig = exec::FunctionSignatureBuilder()
                 .returnType("varchar")
                 .argumentType("varchar")
                 .build();
  return {std::move(sig)};
}

} // namespace facebook::velox::exec::rpc
