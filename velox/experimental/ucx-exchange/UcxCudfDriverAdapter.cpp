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
#include "velox/experimental/ucx-exchange/UcxCudfDriverAdapter.h"

#include <glog/logging.h>
#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>

#include "velox/core/ExchangeTransportType.h"
#include "velox/core/PlanNode.h"
#include "velox/exec/Driver.h"
#include "velox/exec/Exchange.h"
#include "velox/exec/Merge.h"
#include "velox/exec/PartitionedOutput.h"
#include "velox/exec/Task.h"
#include "velox/experimental/cudf/CudfConfig.h"
#include "velox/experimental/cudf/exec/CudfOrderBy.h"
#include "velox/experimental/ucx-exchange/Communicator.h"
#include "velox/experimental/ucx-exchange/UcxExchange.h"
#include "velox/experimental/ucx-exchange/UcxExchangeClient.h"
#include "velox/experimental/ucx-exchange/UcxPartitionedOutput.h"

namespace facebook::velox::ucx_exchange {

namespace {

constexpr const char* kAdapterLabel = "CudfUcx";

std::once_flag communicatorStartedFlag;

struct TaskPipelineKey {
  std::string taskId;
  int pipelineId;

  bool operator==(const TaskPipelineKey& other) const {
    return taskId == other.taskId && pipelineId == other.pipelineId;
  }
};

struct TaskPipelineKeyHash {
  size_t operator()(const TaskPipelineKey& key) const {
    return std::hash<std::string>{}(key.taskId) ^
        (static_cast<size_t>(key.pipelineId) << 1);
  }
};

std::mutex& exchangeClientMapMutex() {
  static std::mutex mutex;
  return mutex;
}

auto& exchangeClientMap() {
  static std::unordered_map<
      TaskPipelineKey,
      std::weak_ptr<UcxExchangeClient>,
      TaskPipelineKeyHash>
      map;
  return map;
}

std::atomic<bool> communicatorStarted{false};
std::once_flag communicatorNotStartedLoggedFlag;

bool canUseCudfUcxExchange() {
  if (communicatorStarted.load()) {
    return true;
  }

  std::call_once(communicatorNotStartedLoggedFlag, []() {
    LOG(WARNING)
        << "[CUDF-UCX] cudf.exchange is enabled, but the startup communicator "
           "is not running; keeping standard exchange operators";
  });
  return false;
}

core::PlanNodePtr findPlanNode(
    const exec::DriverFactory& factory,
    const core::PlanNodeId& id) {
  for (const auto& node : factory.planNodes) {
    if (node->id() == id) {
      return node;
    }
  }
  if (factory.consumerNode && factory.consumerNode->id() == id) {
    return factory.consumerNode;
  }
  return nullptr;
}

std::shared_ptr<UcxExchangeClient> getOrCreateExchangeClient(
    exec::Exchange* exchangeOp,
    const exec::Operator* op,
    exec::DriverCtx* ctx) {
  const TaskPipelineKey key{op->taskId(), ctx->pipelineId};
  std::lock_guard<std::mutex> lock(exchangeClientMapMutex());
  auto& clientMap = exchangeClientMap();
  for (auto it = clientMap.begin(); it != clientMap.end();) {
    if (it->second.expired()) {
      it = clientMap.erase(it);
    } else {
      ++it;
    }
  }

  auto it = clientMap.find(key);
  if (it != clientMap.end()) {
    if (auto existing = it->second.lock()) {
      exchangeOp->resetExchangeClient();
      return existing;
    }
  }

  auto veloxExchangeClient = exchangeOp->releaseExchangeClient();
  VELOX_CHECK_NOT_NULL(
      veloxExchangeClient, "Velox exchange client can't be null.");
  auto client = std::make_shared<UcxExchangeClient>(
      op->taskId(),
      veloxExchangeClient->getDestination(),
      veloxExchangeClient->getNumberOfConsumers(),
      ctx->task->queryCtx()->queryConfig().maxOutputBufferSize());
  clientMap[key] = client;
  return client;
}

bool adaptDriver(const exec::DriverFactory& factory, exec::Driver& driver) {
  const auto& config = cudf_velox::CudfConfig::getInstance();
  if (!config.enabled || !config.exchange) {
    return false;
  }

  auto* ctx = driver.driverCtx();
  auto operators = driver.operators();

  for (int32_t i = static_cast<int32_t>(operators.size()) - 1; i >= 0; --i) {
    auto* op = operators[i];
    if (!op || op->planNodeId().empty() || op->planNodeId() == "N/A") {
      continue;
    }

    if (dynamic_cast<exec::PartitionedOutput*>(op) != nullptr) {
      auto planNode = findPlanNode(factory, op->planNodeId());
      auto partitionedOutputNode =
          std::dynamic_pointer_cast<const core::PartitionedOutputNode>(
              planNode);
      if (!partitionedOutputNode ||
          ctx->task->queryCtx()->outputTransportType(op->planNodeId()) !=
              core::ExchangeTransportType::kUcx) {
        continue;
      }

      if (!canUseCudfUcxExchange()) {
        continue;
      }
      std::vector<std::unique_ptr<exec::Operator>> replacement;
      replacement.push_back(std::make_unique<UcxPartitionedOutput>(
          op->operatorId(), ctx, partitionedOutputNode, /*eagerFlush=*/false));
      [[maybe_unused]] auto replaced =
          factory.replaceOperators(driver, i, i + 1, std::move(replacement));
      VLOG(1) << "[CUDF-UCX] replacing PartitionedOutput at index " << i
              << " (planNodeId=" << op->planNodeId() << ")";
      continue;
    }

    if (dynamic_cast<exec::MergeExchange*>(op) != nullptr) {
      auto planNode = findPlanNode(factory, op->planNodeId());
      auto mergeExchangeNode =
          std::dynamic_pointer_cast<const core::MergeExchangeNode>(planNode);
      if (!mergeExchangeNode ||
          ctx->task->queryCtx()->inputTransportType(op->planNodeId()) !=
              core::ExchangeTransportType::kUcx) {
        continue;
      }

      if (!canUseCudfUcxExchange()) {
        continue;
      }
      std::vector<std::unique_ptr<exec::Operator>> replacement;
      replacement.push_back(
          std::make_unique<UcxExchange>(
              op->operatorId(), ctx, mergeExchangeNode, nullptr));
      replacement.push_back(
          std::make_unique<cudf_velox::CudfOrderBy>(
              op->operatorId(), ctx, mergeExchangeNode));
      [[maybe_unused]] auto replaced =
          factory.replaceOperators(driver, i, i + 1, std::move(replacement));
      VLOG(1) << "[CUDF-UCX] replacing MergeExchange at index " << i
              << " (planNodeId=" << op->planNodeId() << ")";
      continue;
    }

    if (dynamic_cast<exec::Exchange*>(op) != nullptr) {
      auto* exchangeOp = dynamic_cast<exec::Exchange*>(op);
      auto planNode = findPlanNode(factory, op->planNodeId());
      auto exchangeNode =
          std::dynamic_pointer_cast<const core::ExchangeNode>(planNode);
      if (!exchangeNode ||
          ctx->task->queryCtx()->inputTransportType(op->planNodeId()) !=
              core::ExchangeTransportType::kUcx) {
        continue;
      }

      if (!canUseCudfUcxExchange()) {
        continue;
      }
      std::vector<std::unique_ptr<exec::Operator>> replacement;
      replacement.push_back(std::make_unique<UcxExchange>(
              op->operatorId(),
              ctx,
              exchangeNode,
              getOrCreateExchangeClient(exchangeOp, op, ctx)));
      [[maybe_unused]] auto replaced =
          factory.replaceOperators(driver, i, i + 1, std::move(replacement));
      VLOG(1) << "[CUDF-UCX] replacing Exchange at index " << i
              << " (planNodeId=" << op->planNodeId() << ")";
      continue;
    }
  }

  // Return false intentionally: the cuDF adapter must still run after this
  // pass to replace compute operators around the UCX exchange boundary.
  return false;
}

std::atomic<bool> registered{false};

} // namespace

bool startCudfUcxExchange() {
  const auto& config = cudf_velox::CudfConfig::getInstance();
  if (!config.enabled || !config.exchange) {
    return false;
  }

  if (config.exchangeServerPort <= 0 || config.exchangeServerPort > 65535) {
    LOG(ERROR) << "[CUDF-UCX] Invalid cudf.exchange.server.port="
               << config.exchangeServerPort
               << "; keeping standard exchange operators";
    return false;
  }

  std::call_once(communicatorStartedFlag, [&config]() {
    const auto port = static_cast<uint16_t>(config.exchangeServerPort);
    LOG(INFO) << "[CUDF-UCX] Starting Communicator on port "
              << config.exchangeServerPort;
    auto communicator = Communicator::initAndGet(port, "");
    if (!communicator) {
      LOG(ERROR) << "[CUDF-UCX] Communicator::initAndGet failed";
      return;
    }
    std::thread([communicator = std::move(communicator)]() {
      communicator->run();
    }).detach();
    communicatorStarted.store(true);
  });

  return communicatorStarted.load();
}

bool cudfUcxExchangeStarted() {
  return communicatorStarted.load();
}

void registerCudfUcxDriverAdapter() {
  bool expected = false;
  if (!registered.compare_exchange_strong(expected, true)) {
    return;
  }

  exec::DriverAdapter adapter{
      std::string(kAdapterLabel),
      /*inspect=*/{},
      &adaptDriver};
  exec::DriverFactory::registerAdapter(std::move(adapter));
  LOG(INFO) << "[CUDF-UCX] DriverAdapter registered";
}

__attribute__((constructor)) static void cudfUcxAutoRegister() {
  registerCudfUcxDriverAdapter();
}

} // namespace facebook::velox::ucx_exchange
