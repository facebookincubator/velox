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
#include "velox/exec/DefaultOutputBufferManager.h"
#include "velox/core/PlanNode.h"
#include "velox/exec/OutputTransportRegistry.h"
#include "velox/exec/PartitionedOutput.h"
#include "velox/exec/Task.h"

namespace facebook::velox::exec {

// static
const std::shared_ptr<DefaultOutputBufferManager>&
DefaultOutputBufferManager::getInstanceRef() {
  return getInstanceRef(Options());
}

// static
const std::shared_ptr<DefaultOutputBufferManager>&
DefaultOutputBufferManager::getInstanceRef(const Options& options) {
  static const std::shared_ptr<DefaultOutputBufferManager> instance =
      std::make_shared<DefaultOutputBufferManager>(options);
  return instance;
}

// static
std::shared_ptr<OutputTransportEntry>
DefaultOutputBufferManager::makeDefaultTransportEntry() {
  return OutputTransportEntry::make<DefaultOutputBufferManager>(
      getInstanceRef(),
      [](int32_t operatorId,
         DriverCtx* ctx,
         const std::shared_ptr<const core::PartitionedOutputNode>& node,
         bool eagerFlush,
         const std::shared_ptr<DefaultOutputBufferManager>& manager)
          -> std::unique_ptr<Operator> {
        return std::make_unique<PartitionedOutput>(
            operatorId, ctx, node, eagerFlush, manager);
      });
}

std::shared_ptr<OutputBuffer> DefaultOutputBufferManager::getBuffer(
    const std::string& taskId) {
  return buffers_.withLock([&](auto& buffers) {
    auto it = buffers.find(taskId);
    VELOX_CHECK(
        it != buffers.end(), "Output buffers for task not found: {}", taskId);
    return it->second;
  });
}

std::shared_ptr<OutputBuffer> DefaultOutputBufferManager::getBufferIfExists(
    const std::string& taskId) {
  return buffers_.withLock([&](auto& buffers) {
    auto it = buffers.find(taskId);
    return it == buffers.end() ? nullptr : it->second;
  });
}

uint64_t DefaultOutputBufferManager::numBuffers() const {
  return buffers_.lock()->size();
}

bool DefaultOutputBufferManager::enqueue(
    const std::string& taskId,
    int destination,
    std::unique_ptr<SerializedPageBase> data,
    ContinueFuture* future) {
  return getBuffer(taskId)->enqueue(destination, std::move(data), future);
}

void DefaultOutputBufferManager::noMoreData(const std::string& taskId) {
  getBuffer(taskId)->noMoreData();
}

bool DefaultOutputBufferManager::isFinished(const std::string& taskId) {
  return getBuffer(taskId)->isFinished();
}

void DefaultOutputBufferManager::acknowledge(
    const std::string& taskId,
    int destination,
    int64_t sequence) {
  auto buffer =
      buffers_.withLock([&](auto& buffers) -> std::shared_ptr<OutputBuffer> {
        auto it = buffers.find(taskId);
        if (it == buffers.end()) {
          VLOG(1) << "Receiving ack for non-existent task " << taskId
                  << " destination " << destination << " sequence " << sequence;
          return nullptr;
        }
        return it->second;
      });
  if (buffer) {
    buffer->acknowledge(destination, sequence);
  }
}

void DefaultOutputBufferManager::deleteResults(
    const std::string& taskId,
    int destination) {
  auto buffer =
      buffers_.withLock([&](auto& buffers) -> std::shared_ptr<OutputBuffer> {
        if (auto it = buffers.find(taskId); it != buffers.end()) {
          return it->second;
        }
        // The consumer is done before the producer started.
        if (auto held = heldRequests_.find(taskId);
            held != heldRequests_.end()) {
          std::erase_if(held->second, [&](const auto& request) {
            return request.destination == destination;
          });
          if (held->second.empty()) {
            heldRequests_.erase(held);
          }
        }
        return nullptr;
      });
  if (buffer != nullptr) {
    buffer->deleteResults(destination);
  }
}

bool DefaultOutputBufferManager::getData(
    const std::string& taskId,
    int destination,
    uint64_t maxBytes,
    int64_t sequence,
    DataAvailableCallback notify,
    DataConsumerActiveCheckCallback activeCheck) {
  if (auto buffer = getBufferIfExists(taskId)) {
    buffer->getData(destination, maxBytes, sequence, notify, activeCheck);
    return true;
  }
  return false;
}

namespace {

template <typename HeldRequests>
void dropInactive(HeldRequests& heldRequests) {
  for (auto it = heldRequests.begin(); it != heldRequests.end();) {
    std::erase_if(
        it->second, [](const auto& request) { return !request.activeCheck(); });
    it = it->second.empty() ? heldRequests.erase(it) : std::next(it);
  }
}

} // namespace

void DefaultOutputBufferManager::getDataWhenCreated(
    const std::string& taskId,
    int destination,
    uint64_t maxBytes,
    int64_t sequence,
    DataAvailableCallback notify,
    DataConsumerActiveCheckCallback activeCheck) {
  VELOX_CHECK_NOT_NULL(activeCheck);
  auto buffer =
      buffers_.withLock([&](auto& buffers) -> std::shared_ptr<OutputBuffer> {
        if (auto it = buffers.find(taskId); it != buffers.end()) {
          return it->second;
        }
        dropInactive(heldRequests_);
        heldRequests_[taskId].push_back(
            {destination, maxBytes, sequence, notify, activeCheck});
        return nullptr;
      });
  if (buffer != nullptr) {
    buffer->getData(destination, maxBytes, sequence, notify, activeCheck);
  }
}

void DefaultOutputBufferManager::initializeTask(
    std::shared_ptr<Task> task,
    core::PartitionedOutputNode::Kind kind,
    int numDestinations,
    int numDrivers,
    const std::string& /*transportOptions*/) {
  const auto taskId = task->taskId();

  std::shared_ptr<OutputBuffer> buffer;
  std::vector<HeldRequest> heldRequests;
  buffers_.withLock([&](auto& buffers) {
    auto it = buffers.find(taskId);
    if (it == buffers.end()) {
      buffer = std::make_shared<OutputBuffer>(
          std::move(task), kind, numDestinations, numDrivers);
      buffers[taskId] = buffer;
    } else {
      VELOX_FAIL(
          "Registering an output buffer for pre-existing taskId {}", taskId);
    }
    if (auto held = heldRequests_.find(taskId); held != heldRequests_.end()) {
      heldRequests = std::move(held->second);
      heldRequests_.erase(held);
    }
  });
  for (auto& request : heldRequests) {
    if (request.activeCheck()) {
      buffer->getData(
          request.destination,
          request.maxBytes,
          request.sequence,
          std::move(request.notify),
          std::move(request.activeCheck));
    }
  }
}

bool DefaultOutputBufferManager::updateOutputBuffers(
    const std::string& taskId,
    int numDestinations,
    bool noMoreBuffers) {
  if (auto buffer = getBufferIfExists(taskId)) {
    buffer->updateOutputBuffers(numDestinations, noMoreBuffers);
    return true;
  }
  return false;
}

bool DefaultOutputBufferManager::updateNumDrivers(
    const std::string& taskId,
    uint32_t newNumDrivers) {
  if (auto buffer = getBufferIfExists(taskId)) {
    buffer->updateNumDrivers(newNumDrivers);
    return true;
  }
  return false;
}

void DefaultOutputBufferManager::removeTask(const std::string& taskId) {
  auto buffer =
      buffers_.withLock([&](auto& buffers) -> std::shared_ptr<OutputBuffer> {
        auto it = buffers.find(taskId);
        if (it == buffers.end()) {
          // Already removed.
          return nullptr;
        }
        auto taskBuffer = it->second;
        buffers.erase(taskId);
        return taskBuffer;
      });
  if (buffer != nullptr) {
    buffer->terminate();
  }
}

std::string DefaultOutputBufferManager::toString() {
  return buffers_.withLock([](const auto& buffers) {
    std::stringstream out;
    out << "[BufferManager:" << std::endl;
    for (const auto& pair : buffers) {
      out << pair.first << ": " << pair.second->toString() << std::endl;
    }
    out << "]";
    return out.str();
  });
}

std::string DefaultOutputBufferManager::toString(const std::string& taskId) {
  auto buffer = getBufferIfExists(taskId);
  if (buffer != nullptr) {
    return buffer->toString();
  }
  return "";
}

std::optional<double> DefaultOutputBufferManager::getUtilization(
    const std::string& taskId) {
  auto buffer = getBufferIfExists(taskId);
  if (buffer != nullptr) {
    return buffer->getUtilization();
  }
  return std::nullopt;
}

std::optional<bool> DefaultOutputBufferManager::isOverutilized(
    const std::string& taskId) {
  auto buffer = getBufferIfExists(taskId);
  if (buffer != nullptr) {
    return buffer->isOverutilized();
  }
  return std::nullopt;
}

std::optional<OutputBufferStats> DefaultOutputBufferManager::stats(
    const std::string& taskId) {
  auto buffer = getBufferIfExists(taskId);
  if (buffer != nullptr) {
    return buffer->stats();
  }
  return std::nullopt;
}

} // namespace facebook::velox::exec
