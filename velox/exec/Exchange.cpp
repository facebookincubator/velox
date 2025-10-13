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
std::unique_ptr<VectorSerde::Options> getVectorSerdeOptions(
    const core::QueryConfig& queryConfig,
    VectorSerde::Kind kind) {
  std::unique_ptr<VectorSerde::Options> options =
      kind == VectorSerde::Kind::kPresto
      ? std::make_unique<serializer::presto::PrestoVectorSerde::PrestoOptions>()
      : std::make_unique<VectorSerde::Options>();
  options->compressionKind =
      common::stringToCompressionKind(queryConfig.shuffleCompressionKind());
  return options;
}

std::unique_ptr<folly::IOBuf> mergePages(
    std::vector<std::unique_ptr<SerializedPage>>& pages) {
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
          operatorCtx_->driverCtx()->queryConfig(),
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
        operatorCtx_->driverCtx()->splitGroupId, planNodeId(), split, *future);
    if (reason != BlockingReason::kNotBlocked) {
      addRemoteTaskIds(remoteTaskIds);
      return;
    }

    if (split.hasConnectorSplit()) {
      auto remoteSplit =
          checked_pointer_cast<RemoteConnectorSplit>(split.connectorSplit);
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
    uint64_t rawInputBytes{0};
    if (currentPages_.empty()) {
      return nullptr;
    }
    vector_size_t resultOffset = 0;
    for (const auto& page : currentPages_) {
      rawInputBytes += page->size();

      auto inputStream = page->prepareStreamForDeserialize();
      while (!inputStream->atEnd()) {
        serde->deserialize(
            inputStream.get(),
            pool(),
            outputType_,
            &result_,
            resultOffset,
            serdeOptions_.get());
        resultOffset = result_->size();
      }
    }
    currentPages_.clear();
    recordInputStats(rawInputBytes);
    return result_;
  }
  return getOutputFromRows(serde);
}

RowVectorPtr Exchange::getOutputFromRows(VectorSerde* serde) {
  uint64_t rawInputBytes{0};
  if (currentPages_.empty()) {
    VELOX_CHECK_NULL(rowInputStream_);
    VELOX_CHECK_NULL(rowIterator_);
    return nullptr;
  }

  if (rowInputStream_ == nullptr) {
    std::unique_ptr<folly::IOBuf> mergedBufs = mergePages(currentPages_);
    rawInputBytes += mergedBufs->computeChainDataLength();
    rowPages_ = std::make_unique<SerializedPage>(std::move(mergedBufs));
    rowInputStream_ = rowPages_->prepareStreamForDeserialize();
  }

  auto numRows = kInitialOutputRows;
  if (estimatedRowSize_.has_value()) {
    numRows = std::max(
        (preferredOutputBatchBytes_ / estimatedRowSize_.value()),
        kInitialOutputRows);
  }

  // Check if the serde supports batched deserialization
  serde->deserialize(
      rowInputStream_.get(),
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

  if (rowInputStream_->atEnd() && rowIterator_ == nullptr) {
    // only clear the input stream if we have reached the end of the row
    // iterator because row iterator may depend on input stream if serialized
    // rows are not compressed.
    rowInputStream_ = nullptr;
    rowPages_ = nullptr;
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

  const auto backgroundCpuTimeMs =
      exchangeClientStats.find(ExchangeClient::kBackgroundCpuTimeMs);
  if (backgroundCpuTimeMs != exchangeClientStats.end()) {
    const CpuWallTiming backgroundTiming{
        static_cast<uint64_t>(backgroundCpuTimeMs->second.count),
        0,
        static_cast<uint64_t>(backgroundCpuTimeMs->second.sum) *
            Timestamp::kNanosecondsInMillisecond};
    lockedStats->backgroundTiming.clear();
    lockedStats->backgroundTiming.add(backgroundTiming);
  }
}

VectorSerde* Exchange::getSerde() {
  return getNamedVectorSerde(serdeKind_);
}

} // namespace facebook::velox::exec
