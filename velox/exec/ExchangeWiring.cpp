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

#include "velox/exec/ExchangeWiring.h"

#include <folly/container/F14Set.h>

#include "velox/exec/Task.h"

namespace facebook::velox::exec {

// static
void ExchangeWiring::updateOutputBuffers(
    Task& producer,
    const core::PlanNode& producerRoot,
    int32_t numConsumerTasks) {
  const auto* output = producerRoot.as<core::PartitionedOutputNode>();
  if (output != nullptr && (output->isBroadcast() || output->isArbitrary())) {
    producer.updateOutputBuffers(numConsumerTasks, /*noMoreBuffers=*/true);
  }
}

// static
void ExchangeWiring::addInputSplits(
    Task& task,
    const core::Stage& consumer,
    const std::function<int32_t(int32_t producerStageId)>& numProducerTasks,
    const std::function<Split(int32_t producerStageId, int32_t producer)>&
        split) {
  folly::F14FastSet<core::PlanNodeId> exchangeIds;
  for (const auto& input : consumer.inputs) {
    const auto numProducers = numProducerTasks(input.producerStageId);
    for (int32_t producer = 0; producer < numProducers; ++producer) {
      task.addSplit(input.exchangeId, split(input.producerStageId, producer));
    }
    exchangeIds.insert(input.exchangeId);
  }
  for (const auto& exchangeId : exchangeIds) {
    task.noMoreSplits(exchangeId);
  }
}

} // namespace facebook::velox::exec
