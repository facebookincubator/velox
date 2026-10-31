/*
 * Copyright (c) Meta Platforms, Inc. and its affiliates.
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

#include "velox/exec/tests/utils/LocalFixedPoint.h"

#include <fmt/format.h>
#include <folly/executors/CPUThreadPoolExecutor.h>

#include "velox/exec/Exchange.h"
#include "velox/exec/tests/utils/LocalExchangeSource.h"

namespace facebook::velox::exec::test {
namespace {

// Returns the "local://" task id of the producer 'planNodeId' of the loop at
// 'address' in 'iteration'.
std::string producerId(
    const FixedPointOptions::LoopAddress& address,
    int32_t iteration,
    const std::string& planNodeId) {
  return fmt::format(
      "local://{}{}.it{}.{}",
      address.worker,
      LocalFixedPoint::nestingSuffix(address.nesting),
      iteration,
      planNodeId);
}

} // namespace

// static
const FixedPointOptions& LocalFixedPoint::options() {
  static const FixedPointOptions kOptions = [] {
    FixedPointOptions options;
    options.orchestrationExecutor = orchestrationExecutor();
    options.producerTaskId = producerId;
    options.producerSplit = [](const FixedPointOptions::LoopAddress& peer,
                               int32_t iteration,
                               const std::string& planNodeId)
        -> std::shared_ptr<connector::ConnectorSplit> {
      return std::make_shared<RemoteConnectorSplit>(
          LocalExchangeSource::subTaskLocation(
              producerId(peer, iteration, planNodeId), peer.worker));
    };
    options.subTaskId = [](const std::string& workerTaskId, int64_t counter) {
      return fmt::format("{}.sub{}", workerTaskId, counter);
    };
    return options;
  }();
  return kOptions;
}

// static
folly::Executor* LocalFixedPoint::orchestrationExecutor() {
  static const auto kExecutor =
      std::make_shared<folly::CPUThreadPoolExecutor>(1);
  return kExecutor.get();
}

// static
std::string LocalFixedPoint::nestingSuffix(
    const std::vector<FixedPointOptions::NestingStep>& nesting) {
  std::string suffix;
  for (const auto& step : nesting) {
    suffix += fmt::format("/{}.{}", step.planNodeId, step.iteration);
  }
  return suffix;
}

} // namespace facebook::velox::exec::test
