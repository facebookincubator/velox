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

#include "velox/experimental/ucx-exchange/UcxExchangeRegistration.h"

#include "velox/core/PlanNode.h"
#include "velox/exec/ExchangeTransportRegistry.h"
#include "velox/exec/OutputTransportRegistry.h"
#include "velox/experimental/ucx-exchange/UcxExchange.h"
#include "velox/experimental/ucx-exchange/UcxExchangeClient.h"
#include "velox/experimental/ucx-exchange/UcxOutputQueueManager.h"
#include "velox/experimental/ucx-exchange/UcxPartitionedOutput.h"

namespace facebook::velox::ucx_exchange {

void registerUcxTransports() {
  exec::OutputTransportRegistry::global().insert(
      std::string{core::TransportKind::kUcx},
      exec::OutputTransportEntry::make<UcxOutputQueueManager>(
          UcxOutputQueueManager::getInstanceRef(),
          [](int32_t operatorId,
             exec::DriverCtx* ctx,
             const std::shared_ptr<const core::PartitionedOutputNode>& node,
             bool /*eagerFlush*/,
             const std::shared_ptr<UcxOutputQueueManager>& manager)
              -> std::unique_ptr<exec::Operator> {
            // 'eagerFlush' is not honored: UcxPartitionedOutput batches by row
            // count (CudfConfig::kUcxPartitionedOutputBatchRows) because a
            // packed GPU table is the unit of transfer.
            return std::make_unique<UcxPartitionedOutput>(
                operatorId, ctx, node, manager);
          }),
      /*overwrite=*/true);

  exec::ExchangeTransportRegistry::global().insert(
      std::string{core::TransportKind::kUcx},
      exec::ExchangeTransportEntry::make<UcxExchangeClient>(
          [](const exec::ExchangeClientContext& context) {
            // UCX pages are GPU buffers allocated by RMM, not from a Velox
            // memory pool, and the sources are driven by the UCX progress
            // thread rather than by an executor, so 'context.pool',
            // 'context.executor' and the byte-based
            // 'context.maxExchangeBufferSize' /
            // 'context.minExchangeOutputBatchBytes' have no meaning here.
            // UcxExchangeClient bounds its queue by the number of packed
            // tables instead.
            return std::make_shared<UcxExchangeClient>(
                context.taskId, context.destination, context.numberOfConsumers);
          },
          [](int32_t operatorId,
             exec::DriverCtx* ctx,
             const std::shared_ptr<const core::ExchangeNode>& node,
             const std::shared_ptr<UcxExchangeClient>& client)
              -> std::unique_ptr<exec::Operator> {
            return std::make_unique<UcxExchange>(operatorId, ctx, node, client);
          },
          // A merge exchange receives over UCX exactly like a plain one: this
          // slot is non-null only to say that the transport can carry a merge
          // exchange at all, which is what exec::Task and LocalPlanner check.
          //
          // It deliberately does not build the sort that a merge needs.
          // UcxExchangeClient multiplexes every source into one queue, so the
          // per-source orderings exec::MergeExchange relies on are gone by the
          // time the rows arrive and the merged result has to be sorted
          // instead. That sort is described by UcxExchangeAdapter in the cuDF
          // driver-adaptation pass, where DriverFactory::replaceOperators
          // splices it in and renumbers the driver's operator ids.
          [](int32_t operatorId,
             exec::DriverCtx* ctx,
             const std::shared_ptr<const core::ExchangeNode>& node,
             const std::shared_ptr<UcxExchangeClient>& client)
              -> std::unique_ptr<exec::Operator> {
            VELOX_CHECK_NOT_NULL(
                std::dynamic_pointer_cast<const core::MergeExchangeNode>(node),
                "Expected a MergeExchangeNode, plan node: {}",
                node->id());
            return std::make_unique<UcxExchange>(operatorId, ctx, node, client);
          }),
      /*overwrite=*/true);
}

void unregisterUcxTransports() {
  // The UcxOutputQueueManager itself is a process-lifetime singleton, so this
  // is not about freeing it. It is about not leaving a resolvable transport
  // behind: unregisterCudf() drops the cuDF driver adapter and resets the cuDF
  // memory resources, so a plan naming kUcx afterwards would build UCX
  // operators over torn-down cuDF state. The registries are process-global, so
  // stale entries would also leak across tests.
  exec::OutputTransportRegistry::global().erase(
      std::string{core::TransportKind::kUcx});
  exec::ExchangeTransportRegistry::global().erase(
      std::string{core::TransportKind::kUcx});
}

} // namespace facebook::velox::ucx_exchange
