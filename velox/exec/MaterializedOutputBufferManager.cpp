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

#include "velox/exec/MaterializedOutputBufferManager.h"

#include <algorithm>

#include <fmt/format.h>

#include "velox/common/base/Exceptions.h"
#include "velox/exec/MaterializedPartitionedOutput.h"
#include "velox/exec/Task.h"

namespace facebook::velox::exec {

MaterializedOutputBufferManager::MaterializedOutputBufferManager(
    ExchangeSink::Factory sinkFactory,
    int64_t maxBufferedBytes,
    MaterializedOutputBatchConfig outputBatchConfig)
    : sinkFactory_(std::move(sinkFactory)),
      maxBufferedBytes_(maxBufferedBytes),
      outputBatchConfig_(outputBatchConfig) {
  VELOX_CHECK(sinkFactory_ != nullptr, "Exchange sink factory is null");
  VELOX_CHECK_GT(maxBufferedBytes_, 0);
  VELOX_CHECK_GT(outputBatchConfig_.minOutputBatchBytes, 0);
  VELOX_CHECK_GE(
      outputBatchConfig_.maxOutputBatchBytes,
      outputBatchConfig_.minOutputBatchBytes);
  VELOX_CHECK_GT(outputBatchConfig_.estimatedRowBytes, 0);
}

int64_t MaterializedOutputBufferManager::outputBatchSizeBytes(
    int32_t numDestinations) const {
  VELOX_CHECK_GT(numDestinations, 0);
  const auto scaledBatchBytes = outputBatchConfig_.estimatedRowBytes >
          outputBatchConfig_.maxOutputBatchBytes / numDestinations
      ? outputBatchConfig_.maxOutputBatchBytes
      : outputBatchConfig_.estimatedRowBytes * numDestinations;
  return std::clamp(
      scaledBatchBytes,
      outputBatchConfig_.minOutputBatchBytes,
      outputBatchConfig_.maxOutputBatchBytes);
}

std::shared_ptr<MaterializedOutputBuffer>
MaterializedOutputBufferManager::buffer(const std::string& taskId) const {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = buffers_.find(taskId);
  VELOX_CHECK(
      it != buffers_.end(),
      "Materialized output is not configured for task: {}",
      taskId);
  return it->second.buffer;
}

folly::F14FastMap<int32_t, std::string>
MaterializedOutputBufferManager::outputLocations(
    const std::string& taskId) const {
  std::lock_guard<std::mutex> lock(mutex_);
  if (auto it = buffers_.find(taskId); it != buffers_.end()) {
    return it->second.buffer->outputLocations();
  }
  if (auto it = completedOutputs_.find(taskId); it != completedOutputs_.end()) {
    return it->second;
  }
  return {};
}

std::shared_ptr<velox::exec::OutputTransportEntry>
MaterializedOutputBufferManager::transportEntry() {
  return velox::exec::OutputTransportEntry::make<
      MaterializedOutputBufferManager>(
      shared_from_this(),
      [](int32_t operatorId,
         velox::exec::DriverCtx* ctx,
         const std::shared_ptr<const velox::core::PartitionedOutputNode>& node,
         bool /*eagerFlush*/,
         const std::shared_ptr<MaterializedOutputBufferManager>& manager) {
        return std::make_unique<MaterializedPartitionedOutput>(
            operatorId, ctx, node, manager);
      });
}

void MaterializedOutputBufferManager::initializeTask(
    std::shared_ptr<velox::exec::Task> task,
    velox::core::PartitionedOutputNode::Kind kind,
    int numDestinations,
    int numDrivers,
    const std::string& transportOptions) {
  VELOX_CHECK(
      kind == velox::core::PartitionedOutputNode::Kind::kPartitioned ||
          kind == velox::core::PartitionedOutputNode::Kind::kBroadcast,
      "Unsupported materialized output kind: {}",
      kind);
  auto sinkPool = task->pool()->addLeafChild("materialized-output");
  auto sink = sinkFactory_(transportOptions, task->taskId(), sinkPool.get());
  VELOX_CHECK_NOT_NULL(sink, "Exchange sink factory returned null");
  auto taskBuffer = std::make_shared<MaterializedOutputBuffer>(
      numDestinations,
      std::move(sink),
      maxBufferedBytes_,
      MaterializedOutputBuffer::kDefaultDrainThreshold,
      sinkPool);
  taskBuffer->setNumDrivers(numDrivers);
  std::lock_guard<std::mutex> lock(mutex_);
  VELOX_CHECK(
      buffers_
          .emplace(
              task->taskId(),
              TaskBuffer{
                  .buffer = std::move(taskBuffer),
                  .sinkPool = std::move(sinkPool),
                  .maxBufferedBytes = maxBufferedBytes_})
          .second,
      "Materialized output is already configured for task: {}",
      task->taskId());
}

bool MaterializedOutputBufferManager::updateOutputBuffers(
    const std::string& taskId,
    int numDestinations,
    bool noMoreBuffers) {
  auto taskBuffer = buffer(taskId);
  VELOX_CHECK_EQ(taskBuffer->numPartitions(), numDestinations);
  VELOX_CHECK(noMoreBuffers);
  return true;
}

bool MaterializedOutputBufferManager::updateNumDrivers(
    const std::string& taskId,
    uint32_t newNumDrivers) {
  buffer(taskId)->setNumDrivers(newNumDrivers);
  return true;
}

void MaterializedOutputBufferManager::removeTask(const std::string& taskId) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = buffers_.find(taskId);
  if (it == buffers_.end()) {
    return;
  }
  completedOutputs_[taskId] = it->second.buffer->outputLocations();
  buffers_.erase(it);
}

std::optional<velox::exec::OutputBufferStats>
MaterializedOutputBufferManager::stats(const std::string& taskId) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = buffers_.find(taskId);
  if (it == buffers_.end()) {
    return std::nullopt;
  }
  velox::exec::OutputBufferStats stats;
  stats.bufferedBytes = it->second.buffer->bufferedBytes();
  return stats;
}

std::optional<double> MaterializedOutputBufferManager::getUtilization(
    const std::string& taskId) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = buffers_.find(taskId);
  if (it == buffers_.end()) {
    return std::nullopt;
  }
  return static_cast<double>(it->second.buffer->bufferedBytes()) /
      it->second.maxBufferedBytes;
}

std::optional<bool> MaterializedOutputBufferManager::isOverutilized(
    const std::string& taskId) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = buffers_.find(taskId);
  if (it == buffers_.end()) {
    return std::nullopt;
  }
  return it->second.buffer->isBufferFull();
}

std::string MaterializedOutputBufferManager::toString(
    const std::string& taskId) {
  auto taskBuffer = buffer(taskId);
  const auto partitionCount = taskBuffer->numPartitions();
  return fmt::format(
      "MaterializedOutputBuffer[task={}, partitionCount={}, bufferedBytes={}, "
      "maxBufferedBytes={}, partitionDrainThresholdBytes={}, "
      "highWatermarkBytes={}, lowWatermarkBytes={}, "
      "outputBatchSizeBytes={}, minOutputBatchBytes={}, "
      "maxOutputBatchBytes={}, estimatedRowBytes={}]",
      taskId,
      partitionCount,
      taskBuffer->bufferedBytes(),
      maxBufferedBytes_,
      taskBuffer->partitionDrainThreshold(),
      taskBuffer->highWatermarkBytes(),
      taskBuffer->lowWatermarkBytes(),
      outputBatchSizeBytes(partitionCount),
      outputBatchConfig_.minOutputBatchBytes,
      outputBatchConfig_.maxOutputBatchBytes,
      outputBatchConfig_.estimatedRowBytes);
}

} // namespace facebook::velox::exec
