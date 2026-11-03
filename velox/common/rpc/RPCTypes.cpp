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

#include "velox/common/rpc/RPCTypes.h"

namespace facebook::velox::rpc {

RpcPayload::RpcPayload(RpcPayload&& other) noexcept {
  moveFrom(other);
}

RpcPayload& RpcPayload::operator=(RpcPayload&& other) noexcept {
  if (this != &other) {
    reset();
    moveFrom(other);
  }
  return *this;
}

RpcPayload::~RpcPayload() {
  reset();
}

void RpcPayload::reset() noexcept {
  if (vtable_ != nullptr) {
    vtable_->destroy(static_cast<void*>(storage_));
    vtable_ = nullptr;
  }
}

void RpcPayload::moveFrom(RpcPayload& other) noexcept {
  if (other.vtable_ != nullptr) {
    other.vtable_->moveTo(
        static_cast<void*>(storage_), static_cast<void*>(other.storage_));
    vtable_ = other.vtable_;
    other.vtable_ = nullptr;
  }
}

RPCResponse
RPCResponse::failed(int64_t rowId, RPCErrorKind kind, std::string message) {
  RPCResponse response;
  response.rowId = rowId;
  response.setError(kind, std::move(message));
  return response;
}

void RPCResponse::setError(RPCErrorKind kind, std::string errorMessage) {
  VELOX_CHECK(
      kind != RPCErrorKind::kNone,
      "RPC response error kind must name a cause, not kNone");
  result_ = folly::makeUnexpected(RpcError{kind, std::move(errorMessage)});
  capacityRetry_.reset();
}

void RPCResponse::setCapacityRetry(CapacityRetry retry) {
  VELOX_CHECK(hasError(), "Capacity retry applies only to an error response");
  VELOX_CHECK_EQ(
      static_cast<int>(errorCategory(errorKind())),
      static_cast<int>(RPCErrorCategory::kOverload),
      "Capacity retry applies only to an overload error");
  VELOX_CHECK_GE(retry.retryAfter.count(), 0);
  if (retry.maxRetries.has_value()) {
    VELOX_CHECK_GT(
        retry.maxRetries.value(),
        0,
        "Capacity retry maxRetries must be positive");
    VELOX_CHECK_EQ(
        retry.minRetries,
        0,
        "Capacity retry minRetries applies only without maxRetries");
  } else {
    VELOX_CHECK(
        retry.budget.count() > 0 || retry.minRetries > 0,
        "Capacity retry without maxRetries needs a positive budget or minRetries");
  }
  VELOX_CHECK_GE(
      retry.minRetries, 0, "Capacity retry minRetries must not be negative");
  VELOX_CHECK_GE(
      retry.numInPlaceRetries,
      0,
      "Capacity retry numInPlaceRetries must not be negative");
  VELOX_CHECK_GE(retry.maxRetryAfter.count(), 0);
  capacityRetry_ = retry;
}

} // namespace facebook::velox::rpc
