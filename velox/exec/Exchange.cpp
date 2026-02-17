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
#include "velox/exec/Exchange.h"
#include "velox/common/Casts.h"
#include "velox/common/serialization/Serializable.h"
#include "velox/exec/OperatorUtils.h"
#include "velox/exec/Task.h"
#include "velox/serializers/CompactRowSerializer.h"

namespace facebook::velox::exec {

folly::dynamic RemoteConnectorSplit::serialize() const {
  folly::dynamic obj = folly::dynamic::object;
  obj["name"] = "RemoteConnectorSplit";
  obj["taskId"] = taskId;
  return obj;
}

// static
std::shared_ptr<RemoteConnectorSplit> RemoteConnectorSplit::create(
    const folly::dynamic& obj) {
  const auto taskId = obj["taskId"].asString();
  return std::make_shared<RemoteConnectorSplit>(taskId);
}

// static
void RemoteConnectorSplit::registerSerDe() {
  auto& registry = DeserializationRegistryForSharedPtr();
  registry.Register("RemoteConnectorSplit", RemoteConnectorSplit::create);
}

namespace {
std::unique_ptr<folly::IOBuf> mergePages(
    std::vector<std::unique_ptr<SerializedPageBase>>& pages) {
  VELOX_CHECK(!pages.empty());
  std::unique_ptr<folly::IOBuf> mergedBufs;
  for (const auto& page : pages) {
    if (mergedBufs == nullptr) {
      mergedBufs = page->getIOBuf();
    } else {
      mergedBufs->appendToChain(page->getIOBuf());
    }
  }
  return mergedBufs;
}
} // namespace

Exchange::Exchange(
    int32_t operatorId,
    DriverCtx* driverCtx,
    const std::shared_ptr<const core::ExchangeNode>& exchangeNode,
    std::shared_ptr<ExchangeClient> exchangeClient,
    const std::string& operatorType)
    : SourceOperator(
          driverCtx,
          exchangeNode->outputType(),
          operatorId,
          exchangeNode->id(),
          operatorType),
      preferredOutputBatchBytes_{
          driverCtx->queryConfig().preferredOutputBatchBytes()},
      serdeKind_{exchangeNode->serdeKind()},
      serdeOptions_{getVectorSerdeOptions(
          common::stringToCompressionKind(operatorCtx_->driverCtx()
                                              ->queryConfig()
                                              .shuffleCompressionKind()),
          serdeKind_)},
      processSplits_{operatorCtx_->driverCtx()->driverId == 0},
      driverId_{driverCtx->driverId},
      exchangeClient_{std::move(exchangeClient)} {}

void Exchange::addRemoteTaskIds(std::vector<std::string>& remoteTaskIds) {
  std::shuffle(std::begin(remoteTaskIds), std::end(remoteTaskIds), rng_);
  for (const std::string& taskId : remoteTaskIds) {
    exchangeClient_->addRemoteTaskId(taskId);
  }
  stats_.wlock()->numSplits += remoteTaskIds.size();
}

void Exchange::getSplits(ContinueFuture* future) {
  if (!processSplits_) {
    return;
  }
  if (noMoreSplits_) {
    return;
  }
  std::vector<std::string> remoteTaskIds;
  for (;;) {
    exec::Split split;
    const auto reason = operatorCtx_->task()->getSplitOrFuture(
        operatorCtx_->driverCtx()->driverId,
        operatorCtx_->driverCtx()->splitGroupId,
        planNodeId(),
        /*maxPreloadSplits=*/0,
        /*preload=*/nullptr,
        split,
        *future);
    if (reason != BlockingReason::kNotBlocked) {
      addRemoteTaskIds(remoteTaskIds);
      return;
    }

    if (split.hasConnectorSplit()) {
      auto remoteSplit =
          checkedPointerCast<RemoteConnectorSplit>(split.connectorSplit);
      if (FOLLY_UNLIKELY(splitTracer_ != nullptr)) {
        splitTracer_->write(split);
      }
      remoteTaskIds.push_back(remoteSplit->taskId);
      continue;
    }

    addRemoteTaskIds(remoteTaskIds);
    exchangeClient_->noMoreRemoteTasks();
    noMoreSplits_ = true;
    if (atEnd_) {
      operatorCtx_->task()->multipleSplitsFinished(
          false, stats_.rlock()->numSplits, 0);
      recordExchangeClientStats();
    }
    return;
  }
}

BlockingReason Exchange::isBlocked(ContinueFuture* future) {
  if (!currentPages_.empty() || atEnd_) {
    return BlockingReason::kNotBlocked;
  }

  // Start fetching data right away. Do not wait for all splits to be available.
  if (!splitFuture_.valid()) {
    getSplits(&splitFuture_);
  }

  ContinueFuture dataFuture;
  currentPages_ = exchangeClient_->next(
      driverId_, preferredOutputBatchBytes_, &atEnd_, &dataFuture);
  if (!currentPages_.empty() || atEnd_) {
    if (atEnd_ && noMoreSplits_) {
      const auto numSplits = stats_.rlock()->numSplits;
      operatorCtx_->task()->multipleSplitsFinished(false, numSplits, 0);
    }
    recordExchangeClientStats();
    return BlockingReason::kNotBlocked;
  }

  // We have a dataFuture and we may also have a splitFuture_.

  if (splitFuture_.valid()) {
    // Block until data becomes available or more splits arrive.
    std::vector<ContinueFuture> futures;
    futures.push_back(std::move(splitFuture_));
    futures.push_back(std::move(dataFuture));
    *future = folly::collectAny(futures).unit();
    return BlockingReason::kWaitForSplit;
  }

  // Block until data becomes available.
  VELOX_CHECK(dataFuture.valid());
  *future = std::move(dataFuture);
  return BlockingReason::kWaitForProducer;
}

bool Exchange::isFinished() {
  return atEnd_ && currentPages_.empty();
}

RowVectorPtr Exchange::getOutput() {
  auto* serde = getSerde();
  if (serde->supportsAppendInDeserialize()) {
    return getOutputFromColumnarPages(serde);
  }
  return getOutputFromRowPages(serde);
}

RowVectorPtr Exchange::getOutputFromColumnarPages(VectorSerde* serde) {
  if (currentPages_.empty()) {
    return nullptr;
  }

  // Calculate target row count based on estimated row size, similar to
  // getOutputFromRowPages.
  // Start conservatively, then use estimates.
  const auto numRows = estimatedRowSize_.has_value()
      ? std::max(
            (preferredOutputBatchBytes_ / estimatedRowSize_.value()),
            kInitialOutputRows)
      : kInitialOutputRows;

  // Process pages one-by-one from currentPages_ pointed by columnarPageIdx_.
  // Within each page, deserialize vectors incrementally until we hit the target
  // batch size.
  uint64_t rawInputBytes = 0;
  vector_size_t resultOffset{0};

  // Should be either starting fresh or continuing from a previous partial page
  VELOX_CHECK(
      inputStream_ == nullptr || columnarPageIdx_ < currentPages_.size());

  // Iterate through pages
  while (columnarPageIdx_ < currentPages_.size()) {
    auto& page = currentPages_[columnarPageIdx_];

    if (!inputStream_) {
      // NOTE: 'rawInputBytes' only counts bytes from pages processed from the
      // beginning in this call. If processing resumes from the middle of a
      // page, that page's bytes are not counted. This ensures each page is
      // counted only once in 'rawInputBytes' across multiple calls.
      rawInputBytes += page->size();
      inputStream_ = page->prepareStreamForDeserialize();
    }

    // Inner loop: deserialize vectors from current page until batch is full
    // or page is exhausted.
    while (!inputStream_->atEnd() && resultOffset < numRows) {
      serde->deserialize(
          inputStream_.get(),
          pool(),
          outputType_,
          &result_,
          resultOffset,
          serdeOptions_.get());

      resultOffset = result_->size();
    }

    if (inputStream_->atEnd()) {
      // Page is fully consumed, free memory immediately, and move to the next.
      inputStream_ = nullptr;
      page.reset();
      ++columnarPageIdx_;
    }

    // Stop if accumulated enough rows for this batch.
    if (resultOffset >= numRows) {
      break;
    }
  }

  const auto numOutputRows = result_->size();
  VELOX_CHECK_GT(numOutputRows, 0);

  estimatedRowSize_ = std::max(
      result_->estimateFlatSize() / numOutputRows,
      estimatedRowSize_.value_or(1L));

  // If processed all pages, clear the vector and reset state.
  if (columnarPageIdx_ >= currentPages_.size()) {
    VELOX_CHECK_NULL(inputStream_);
    currentPages_.clear();
    columnarPageIdx_ = 0;
  }

  recordInputStats(rawInputBytes);
  return result_;
}

RowVectorPtr Exchange::getOutputFromRowPages(VectorSerde* serde) {
  uint64_t rawInputBytes{0};
  if (currentPages_.empty()) {
    VELOX_CHECK_NULL(inputStream_);
    VELOX_CHECK_NULL(rowIterator_);
    return nullptr;
  }

  if (inputStream_ == nullptr) {
    std::unique_ptr<folly::IOBuf> mergedBufs = mergePages(currentPages_);
    rawInputBytes += mergedBufs->computeChainDataLength();
    mergedRowPage_ =
        std::make_unique<PrestoSerializedPage>(std::move(mergedBufs));
    inputStream_ = mergedRowPage_->prepareStreamForDeserialize();
  }

  auto numRows = kInitialOutputRows;
  if (estimatedRowSize_.has_value()) {
    numRows = std::max(
        (preferredOutputBatchBytes_ / estimatedRowSize_.value()),
        kInitialOutputRows);
  }

  // Check if the serde supports batched deserialization
  serde->deserialize(
      inputStream_.get(),
      rowIterator_,
      numRows,
      outputType_,
      &result_,
      pool(),
      serdeOptions_.get());

  const auto numOutputRows = result_->size();
  VELOX_CHECK_GT(numOutputRows, 0);

  estimatedRowSize_ = std::max(
      result_->estimateFlatSize() / numOutputRows,
      estimatedRowSize_.value_or(1L));

  if (inputStream_->atEnd() && rowIterator_ == nullptr) {
    // only clear the input stream if we have reached the end of the row
    // iterator because row iterator may depend on input stream if serialized
    // rows are not compressed.
    inputStream_ = nullptr;
    mergedRowPage_ = nullptr;
    currentPages_.clear();
  }

  recordInputStats(rawInputBytes);
  return result_;
}

void Exchange::recordInputStats(uint64_t rawInputBytes) {
  auto lockedStats = stats_.wlock();
  lockedStats->rawInputBytes += rawInputBytes;
  lockedStats->rawInputPositions += result_->size();
  lockedStats->addInputVector(result_->estimateFlatSize(), result_->size());
}

void Exchange::close() {
  SourceOperator::close();
  currentPages_.clear();
  result_ = nullptr;

  // Clean up stateful deserialization state
  inputStream_ = nullptr;
  mergedRowPage_ = nullptr;
  rowIterator_ = nullptr;
  columnarPageIdx_ = 0;

  if (exchangeClient_) {
    recordExchangeClientStats();
    exchangeClient_->close();
  }
  exchangeClient_ = nullptr;
  {
    auto lockedStats = stats_.wlock();
    lockedStats->addRuntimeStat(
        Operator::kShuffleSerdeKind,
        RuntimeCounter(static_cast<int64_t>(serdeKind_)));
    lockedStats->addRuntimeStat(
        Operator::kShuffleCompressionKind,
        RuntimeCounter(static_cast<int64_t>(serdeOptions_->compressionKind)));
  }
}

void Exchange::recordExchangeClientStats() {
  if (!processSplits_) {
    return;
  }

  auto lockedStats = stats_.wlock();
  const auto exchangeClientStats = exchangeClient_->stats();
  for (const auto& [name, value] : exchangeClientStats) {
    lockedStats->runtimeStats.erase(name);
    lockedStats->runtimeStats.insert({name, value});
  }

  const auto iter = exchangeClientStats.find(Operator::kBackgroundCpuTimeNanos);
  if (iter != exchangeClientStats.end()) {
    const CpuWallTiming backgroundTiming{
        static_cast<uint64_t>(iter->second.count),
        0,
        static_cast<uint64_t>(iter->second.sum)};
    lockedStats->backgroundTiming.clear();
    lockedStats->backgroundTiming.add(backgroundTiming);
  }
}

VectorSerde* Exchange::getSerde() {
  return getNamedVectorSerde(serdeKind_);
}

} // namespace facebook::velox::exec
