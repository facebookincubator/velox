/*
 * Copyright (c) International Business Machines Corporation
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

#include "velox/exec/OptimizedPartitionedOutput.h"

#include "velox/exec/HashPartitionFunction.h"
#include "velox/exec/SerializedPage.h"
#include "velox/exec/Task.h"

namespace facebook::velox::exec {

OptimizedPartitionedOutput::OptimizedPartitionedOutput(
    int32_t operatorId,
    DriverCtx* ctx,
    const std::shared_ptr<const core::PartitionedOutputNode>& planNode)
    : Operator(
          ctx,
          planNode->outputType(),
          operatorId,
          planNode->id(),
          "OptimizedPartitionedOutput"),
      taskId_(operatorCtx_->taskId()),
      inputType_(planNode->inputType()),
      keyChannels_(toChannels(planNode->inputType(), planNode->keys())),
      outputChannels_(calculateOutputChannels(
          planNode->inputType(),
          planNode->outputType(),
          planNode->outputType())),
      numDestinations_(planNode->numPartitions()),
      replicateNullsAndAny_(planNode->isReplicateNullsAndAny()),
      bufferManager_(OutputBufferManager::getInstanceRef()),
      // NOTE: 'bufferReleaseFn_' holds a reference on the associated task to
      // prevent it from deleting while there are output buffers being accessed
      // out of the partitioned output buffer manager such as in Prestissimo,
      // the http server holds the buffers while sending the data response.
      bufferReleaseFn_([task = operatorCtx_->task()]() {}),
      maxOutputBufferBytes_(ctx->task->queryCtx()
                                ->queryConfig()
                                .maxPartitionedOutputBufferSize()),
      pool_(pool()),
      partitionFunction_(
          numDestinations_ == 1
              ? nullptr
              : planNode->partitionFunctionSpec().create(numDestinations_)) {
  if (!planNode->isPartitioned()) {
    VELOX_USER_CHECK_EQ(numDestinations_, 1);
  }
  if (numDestinations_ == 1) {
    VELOX_USER_CHECK(keyChannels_.empty());
  }

  serializer::presto::SerdeOpts options;
  options.compressionKind = common::stringToCompressionKind(
      operatorCtx_->driverCtx()->queryConfig().shuffleCompressionKind());
  options.minCompressionRatio = 0.8;

  serializer_ = std::make_unique<
      serializer::presto::PrestoIterativePartitioningSerializer>(
      inputType_, numDestinations_, options, pool_);
}

void OptimizedPartitionedOutput::addInput(RowVectorPtr input) {
  VELOX_USER_CHECK(
      !replicateNullsAndAny_,
      "replicateNullsAndAny is not yet supported by OptimizedPartitionedOutput");

  if (serializer_->bytesBuffered() + input->retainedSize() >=
      maxOutputBufferBytes_) {
    flush();
  }

  const auto numRows = input->size();
  partitions_.resize(numRows);

  if (numDestinations_ == 1) {
    std::fill(partitions_.begin(), partitions_.end(), 0u);
  } else {
    std::optional<uint32_t> partition =
        partitionFunction_->partition(*input, partitions_);
    if (partition.has_value()) {
      // All rows go to the same partition
      std::fill(partitions_.begin(), partitions_.end(), partition.value());
    }
  }

  serializer_->append(input, partitions_);

  auto lockedStats = stats_.wlock();
  ++numAppends_;
  lockedStats->addRuntimeStat("numAppends", RuntimeCounter(1));
}

bool OptimizedPartitionedOutput::needsInput() const {
  return blockingReason_ == BlockingReason::kNotBlocked;
}

RowVectorPtr OptimizedPartitionedOutput::getOutput() {
  if (finished_) {
    return nullptr;
  }

  blockingReason_ = BlockingReason::kNotBlocked;

  if (noMoreInput_ || serializer_->bytesBuffered() >= maxOutputBufferBytes_) {
    flush();
  }

  // If blocked, stop here. We avoid advancing operator state while blocked,
  // even if noMoreInput_ may already be true. The driver will resume and call
  // getOutput() again once the OutputBuffer has space.
  if (blockingReason_ != BlockingReason::kNotBlocked) {
    return nullptr;
  }

  if (noMoreInput_ && serializer_->bytesBuffered() == 0) {
    // TODO: merge serializer runtime stats into operator stats once
    // PrestoIterativePartitioningSerializer exposes runtimeStats().
    bufferManager_.lock()->noMoreData(operatorCtx_->task()->taskId());
    finished_ = true;
  }

  return nullptr;
}

BlockingReason OptimizedPartitionedOutput::isBlocked(ContinueFuture* future) {
  if (blockingReason_ != BlockingReason::kNotBlocked) {
    *future = std::move(future_);
    blockingReason_ = BlockingReason::kNotBlocked;
    return BlockingReason::kWaitForConsumer;
  }
  return BlockingReason::kNotBlocked;
}

bool OptimizedPartitionedOutput::isFinished() {
  return finished_;
}

void OptimizedPartitionedOutput::flush() {
  const auto flushedBytes = serializer_->bytesBuffered();
  const auto flushedRows = serializer_->rowsBuffered();

  // This will serialize all destinations and reset serializer_->bytesBuffered()
  // to 0.
  auto serializedIOBufs = serializer_->flush();
  auto bufferManager = bufferManager_.lock();
  VELOX_CHECK_NOT_NULL(
      bufferManager, "OutputBufferManager was already destructed");

  bool shouldBlock = false;
  ContinueFuture future = ContinueFuture::makeEmpty();
  for (auto& [destination, pageData] : serializedIOBufs) {
    // We will only pass the future to bufferManager->enqueue() for the first
    // blocked destination. This is to avoid unnecessary creation of
    // ContinueFuture objects for the remaining destinations.
    ContinueFuture* futurePtr = shouldBlock ? nullptr : &future;

    // Enqueue the data for each non-empty partition. Since the pageData is
    // already serialized, enqueueing them would not cause new memory
    // allocations. This will always move the pageData to the OutputBuffers no
    // matter if the OutputBuffer is blocked.
    bool blocked = bufferManager->enqueue(
        taskId_,
        static_cast<int>(destination),
        std::make_unique<PrestoSerializedPage>(
            std::move(pageData.first),
            [fn = bufferReleaseFn_](folly::IOBuf&) { fn(); },
            pageData.second),
        futurePtr);

    if (blocked && !shouldBlock) {
      blockingReason_ = BlockingReason::kWaitForConsumer;
      shouldBlock = true;
      future_ = std::move(future);
    }
  }

  auto lockedStats = stats_.wlock();
  lockedStats->addOutputVector(flushedBytes, flushedRows);
  if (flushedRows > 0) {
    ++numFlushes_;
    lockedStats->addRuntimeStat("numFlushes", RuntimeCounter(1));
  }
  if (shouldBlock) {
    ++numBlockedTimes_;
    lockedStats->addRuntimeStat("numBlockedTimes", RuntimeCounter(1));
  }
}

} // namespace facebook::velox::exec
