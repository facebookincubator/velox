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

#include "velox/exec/rpc/tests/ResponseSimulator.h"

namespace facebook::velox::exec::rpc::test {

ResponseSimulator::ResponseSimulator(
    std::chrono::milliseconds latency,
    std::shared_ptr<folly::CPUThreadPoolExecutor> executor)
    : latency_(latency),
      ownedExecutor_(
          executor == nullptr
              ? std::make_shared<folly::CPUThreadPoolExecutor>(4)
              : nullptr),
      executor_(executor != nullptr ? executor : ownedExecutor_) {}

folly::SemiFuture<velox::rpc::RPCErrorKind> ResponseSimulator::nextCall() {
  const auto kind = outcomeFor(callCount_.fetch_add(1));
  return delayed(kind);
}

folly::SemiFuture<std::vector<velox::rpc::RPCErrorKind>>
ResponseSimulator::nextBatch(size_t numCalls) {
  const auto firstCall = callCount_.fetch_add(static_cast<int64_t>(numCalls));
  std::vector<velox::rpc::RPCErrorKind> kinds;
  kinds.reserve(numCalls);
  for (size_t i = 0; i < numCalls; ++i) {
    kinds.push_back(outcomeFor(firstCall + static_cast<int64_t>(i)));
  }
  return delayed(std::move(kinds));
}

int64_t ResponseSimulator::callCount() const {
  return callCount_.load();
}

void ResponseSimulator::resetCallCount() {
  callCount_.store(0);
}

void ResponseSimulator::setErrorBurst(const ErrorBurst& burst) {
  VELOX_CHECK(
      !burstInstalled_.load(std::memory_order_acquire) &&
          callCount_.load() == 0,
      "ResponseSimulator: error burst must be installed once, before the "
      "first call");
  errorBurst_ = burst;
  burstInstalled_.store(true, std::memory_order_release);
}

velox::rpc::RPCErrorKind ResponseSimulator::outcomeFor(int64_t ordinal) const {
  if (burstInstalled_.load(std::memory_order_acquire) &&
      ordinal >= errorBurst_.firstCall && ordinal < errorBurst_.lastCall) {
    return errorBurst_.errorKind;
  }
  return velox::rpc::RPCErrorKind::kNone;
}

} // namespace facebook::velox::exec::rpc::test
