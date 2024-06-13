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

#include "velox/exec/TaskOutput.h"

#include "velox/exec/OutputBufferManager.h"
#include "velox/exec/Task.h"

namespace facebook::velox::exec {
namespace {
void generateSplits(
    const std::unique_ptr<BatchVectorSerializer>& serializer,
    const VectorPtr& input,
    const IndexRange& inputRange,
    vector_size_t previousSize,
    vector_size_t targetSize,
    Scratch& scratch,
    std::vector<detail::Split>& splits) {
  if (inputRange.size == 1) {
    splits.push_back({inputRange, previousSize});
    return;
  }

  vector_size_t size;
  vector_size_t* sizePtr = &size;

  serializer->estimateSerializedSize(
      input, {&inputRange, 1}, &sizePtr, scratch);

  if (size <= targetSize || size == previousSize) {
    splits.push_back({inputRange, size});
    return;
  }

  IndexRange left{inputRange.begin, inputRange.size / 2};
  IndexRange right{
      inputRange.begin + inputRange.size / 2,
      inputRange.size - inputRange.size / 2};

  generateSplits(serializer, input, left, size, targetSize, scratch, splits);
  generateSplits(serializer, input, right, size, targetSize, scratch, splits);
}
} // namespace

TaskOutput::TaskOutput(
    int32_t operatorId,
    DriverCtx* ctx,
    const std::shared_ptr<const core::PartitionedOutputNode>& planNode)
    : Operator(
          ctx,
          planNode->outputType(),
          operatorId,
          planNode->id(),
          "TaskOutput"),
      outputChannels_(calculateOutputChannels(
          planNode->inputType(),
          planNode->outputType(),
          planNode->outputType())),
      bufferManager_(OutputBufferManager::getInstance()),
      // NOTE: 'bufferReleaseFn_' holds a reference on the associated task to
      // prevent it from deleting while there are output buffers being accessed
      // out of the partitioned output buffer manager such as in Prestissimo,
      // the http server holds the buffers while sending the data response.
      bufferReleaseFn_([task = operatorCtx_->task()]() {}),
      taskId_(operatorCtx_->taskId()) {
  serializer::presto::PrestoVectorSerde::PrestoOptions options;
  options.compressionKind =
      OutputBufferManager::getInstance().lock()->compressionKind();
  options.minCompressionRatio =
      OutputBufferManager::getInstance().lock()->minCompressionRatio();
  serializer_ = getVectorSerde()->createBatchSerializer(pool(), &options);
}

void TaskOutput::initializeOutput(const RowVectorPtr& input) {
  if (outputType_->size() == 0) {
    output_ = std::make_shared<RowVector>(
        input->pool(),
        outputType_,
        nullptr /*nulls*/,
        input->size(),
        std::vector<VectorPtr>{});
  } else if (outputChannels_.empty()) {
    output_ = input;
  } else {
    std::vector<VectorPtr> outputColumns;
    outputColumns.reserve(outputChannels_.size());
    for (auto i : outputChannels_) {
      outputColumns.push_back(input->childAt(i));
    }

    output_ = std::make_shared<RowVector>(
        input->pool(),
        outputType_,
        nullptr /*nulls*/,
        input->size(),
        outputColumns);
  }
}

void TaskOutput::addInput(RowVectorPtr input) {
  initializeOutput(input);

  splits_.clear();
  splitIdx_ = 0;

  // Limit serialized pages to ~1MB.
  static const vector_size_t kBaseTargetSize = 1 << 20;
  vector_size_t targetSizePct = 70 + (folly::Random::rand32(rng_) % 50);
  vector_size_t targetSize = (kBaseTargetSize * targetSizePct) / 100;

  generateSplits(
      serializer_,
      output_,
      {0, output_->size()},
      0,
      targetSize,
      scratch_,
      splits_);
}

RowVectorPtr TaskOutput::getOutput() {
  if (finished_) {
    return nullptr;
  }

  blockingReason_ = BlockingReason::kNotBlocked;
  auto bufferManager = bufferManager_.lock();
  VELOX_CHECK_NOT_NULL(
      bufferManager, "OutputBufferManager was already destructed");

  while (blockingReason_ == BlockingReason::kNotBlocked &&
         splitIdx_ < splits_.size()) {
    static constexpr vector_size_t kMinMessageSize = 128;

    const auto& split = splits_[splitIdx_++];
    auto listener = bufferManager->newListener();
    IOBufOutputStream stream(
        *pool(),
        listener.get(),
        std::max<vector_size_t>(kMinMessageSize, split.estimatedSize));
    const int64_t flushedRows = split.range.size;

    serializer_->serialize(output_, {&split.range, 1}, scratch_, &stream);

    const int64_t flushedBytes = stream.tellp();

    const bool blocked = bufferManager->enqueue(
        taskId_,
        0,
        std::make_unique<SerializedPage>(
            stream.getIOBuf(bufferReleaseFn_), nullptr, flushedRows),
        &future_);

    {
      auto lockedStats = stats_.wlock();
      lockedStats->addOutputVector(flushedBytes, flushedRows);
    }

    blockingReason_ = blocked ? BlockingReason::kWaitForConsumer
                              : BlockingReason::kNotBlocked;
  }

  if (blockingReason_ != BlockingReason::kNotBlocked) {
    // The input isn't fully processed yet.
    return nullptr;
  }

  if (noMoreInput_) {
    bufferManager->noMoreData(taskId_);
    finished_ = true;

    // Update the runtime stats with the serializer stats.
    const auto serializerStats = serializer_->runtimeStats();
    auto lockedStats = stats().wlock();
    for (auto& pair : serializerStats) {
      lockedStats->addRuntimeStat(pair.first, pair.second);
    }
  }
  // The input is fully processed, drop the reference to allow reuse.
  output_ = nullptr;
  return nullptr;
}

bool TaskOutput::isFinished() {
  return finished_;
}

} // namespace facebook::velox::exec
