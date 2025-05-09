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

#include "velox/exec/Merge.h"
#include "velox/common/testutil/TestValue.h"
#include "velox/exec/MergeBuffer.h"
#include "velox/exec/Task.h"

using facebook::velox::common::testutil::TestValue;

namespace facebook::velox::exec {
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
} // namespace

Merge::Merge(
    int32_t operatorId,
    DriverCtx* driverCtx,
    RowTypePtr outputType,
    const std::vector<std::shared_ptr<const core::FieldAccessTypedExpr>>&
        sortingKeys,
    const std::vector<core::SortOrder>& sortingOrders,
    const std::string& planNodeId,
    const std::string& operatorType)
    : SourceOperator(
          driverCtx,
          std::move(outputType),
          operatorId,
          planNodeId,
          operatorType),
      outputBatchSize_{outputBatchRows()} {
  auto numKeys = sortingKeys.size();
  sortingKeys_.reserve(numKeys);
  for (int i = 0; i < numKeys; ++i) {
    auto channel = exprToChannel(sortingKeys[i].get(), outputType_);
    VELOX_CHECK_NE(
        channel,
        kConstantChannel,
        "Merge doesn't allow constant grouping keys");
    sortingKeys_.emplace_back(
        channel,
        CompareFlags{
            sortingOrders[i].isNullsFirst(),
            sortingOrders[i].isAscending(),
            false});
  }
}

void Merge::initializeTreeOfLosers() {
  std::vector<std::unique_ptr<SourceStream>> sourceCursors;
  sourceCursors.reserve(sources_.size());
  for (auto& source : sources_) {
    sourceCursors.push_back(std::make_unique<SourceStream>(
        source.get(), sortingKeys_, outputBatchSize_));
  }

  // Save the pointers to cursors before moving these into the TreeOfLosers.
  streams_.reserve(sources_.size());
  for (auto& cursor : sourceCursors) {
    streams_.push_back(cursor.get());
  }

  treeOfLosers_ =
      std::make_unique<TreeOfLosers<SourceStream>>(std::move(sourceCursors));
}

BlockingReason Merge::isBlocked(ContinueFuture* future) {
  TestValue::adjust("facebook::velox::exec::Merge::isBlocked", this);

  const auto reason = addMergeSources(future);
  if (reason != BlockingReason::kNotBlocked) {
    return reason;
  }

  // NOTE: the task might terminate early which leaves empty sources. Once it
  // happens, we shall simply mark the merge operator as finished.
  if (sources_.empty()) {
    finished_ = true;
    return BlockingReason::kNotBlocked;
  }

  if (maxMergeSources_ < sources_.size()) {
    if (mergeBuffer_ == nullptr) {
      VELOX_CHECK(!finished_);
      const auto spillConfig = spillConfig_.has_value()
          ? spillConfig_
          : operatorCtx_->driverCtx()->makeSpillConfig(
                operatorCtx_->operatorId());
      VELOX_CHECK(spillConfig.has_value());
      mergeBuffer_ = std::make_unique<MergeBuffer>(
          outputType_,
          pool(),
          sortingKeys_,
          spillConfig_.has_value() ? &(spillConfig_.value()) : nullptr,
          &spillStats_);
    }

    VELOX_CHECK_NOT_NULL(mergeBuffer_);

    if (partialInputMerger_ == nullptr &&
        numStartedSources_ < sources_.size()) {
      // Plans merge sources for this run.
      std::vector<MergeSource*> currentSources;
      for (auto i = numStartedSources_; i <
           (std::min(sources_.size(), numStartedSources_ + maxMergeSources_));
           ++i) {
        currentSources.push_back(sources_.at(i).get());
      }

      // Initializes the input merger.
      std::vector<std::unique_ptr<SourceStream>> currentCursors;
      currentCursors.reserve(currentSources.size());
      for (auto* source : currentSources) {
        currentCursors.push_back(std::make_unique<SourceStream>(
            source, sortingKeys_, outputBatchSize_));
      }

      // mergeStreams_ and inputMerger_ are paired.
      VELOX_CHECK(partialMergeStream_.empty());
      for (auto& cursor : currentCursors) {
        partialMergeStream_.push_back(cursor.get());
      }
      partialInputMerger_ = std::make_unique<TreeOfLosers<SourceStream>>(
          std::move(currentCursors));

      // Start sources.
      for (auto& source : currentSources) {
        source->start();
      }
      numStartedSources_ += currentSources.size();
    }

    VELOX_USER_CHECK(
        (partialInputMerger_ == nullptr && partialMergeStream_.empty()) ||
        (partialInputMerger_ != nullptr && !partialMergeStream_.empty()));

    // Load data if needed.
    if (sourceBlockingFutures_.empty()) {
      for (auto& cursor : partialMergeStream_) {
        cursor->isBlocked(sourceBlockingFutures_);
      }
    }

    if (sourceBlockingFutures_.empty()) {
      return BlockingReason::kNotBlocked;
    }

    *future = std::move(sourceBlockingFutures_.back());
    sourceBlockingFutures_.pop_back();
    return BlockingReason::kWaitForProducer;
  }

  startSources();

  // No merging is needed if there is only one source.
  if (streams_.empty() && sources_.size() > 1) {
    initializeTreeOfLosers();
  }

  if (sourceBlockingFutures_.empty()) {
    for (auto& cursor : streams_) {
      cursor->isBlocked(sourceBlockingFutures_);
    }
  }

  if (sourceBlockingFutures_.empty()) {
    return BlockingReason::kNotBlocked;
  }

  *future = std::move(sourceBlockingFutures_.back());
  sourceBlockingFutures_.pop_back();
  return BlockingReason::kWaitForProducer;
}

void Merge::startSources() {
  VELOX_CHECK_LE(numStartedSources_, sources_.size());
  // Start the merge source once.
  if (numStartedSources_ >= sources_.size()) {
    return;
  }
  VELOX_CHECK_EQ(numStartedSources_, 0);
  VELOX_CHECK(streams_.empty());
  VELOX_CHECK(sourceBlockingFutures_.empty());
  // TODO: support lazy start for local merge with a large number of sources
  // to cap the memory usage.
  for (auto& source : sources_) {
    source->start();
  }
  numStartedSources_ = sources_.size();
}

bool Merge::isFinished() {
  return finished_;
}

RowVectorPtr Merge::getOutput() {
  if (finished_) {
    return nullptr;
  }

  if (mergeBuffer_ != nullptr) {
    if (partialInputMerger_ != nullptr) {
      if (!output_) {
        output_ = BaseVector::create<RowVector>(
            outputType_, outputBatchSize_, operatorCtx_->pool());
        for (auto& child : output_->children()) {
          child->resize(outputBatchSize_);
        }
      }

      for (;;) {
        const auto stream = partialInputMerger_->next();
        if (!stream) {
          partialInputMerger_ = nullptr;
          partialMergeStream_.clear();
          if (outputSize_ > 0) {
            output_->resize(outputSize_);
            mergeBuffer_->addInput(output_);
          }
          mergeBuffer_->finishSpill(numStartedSources_ >= sources_.size());
          return nullptr;
        }

        if (stream->setOutputRow(outputSize_)) {
          // The stream is at end of input batch. Need to copy out the rows
          // before fetching next batch in 'pop'.
          stream->copyToOutput(output_);
        }

        ++outputSize_;

        // Advance the stream.
        stream->pop(sourceBlockingFutures_);

        if (outputSize_ == outputBatchSize_) {
          // Copy out data from all sources.
          for (const auto& s : streams_) {
            s->copyToOutput(output_);
          }

          outputSize_ = 0;
          mergeBuffer_->addInput(std::move(output_));
          return nullptr;
        }

        if (!sourceBlockingFutures_.empty()) {
          return nullptr;
        }
      }
    }

    VELOX_CHECK(partialInputMerger_ == nullptr && partialMergeStream_.empty());
    output_ = mergeBuffer_->getOutput(outputBatchSize_);
    if (output_ == nullptr) {
      finished_ = true;
      return nullptr;
    }

    return std::move(output_);
  }

  VELOX_CHECK_EQ(numStartedSources_, sources_.size());

  // No merging is needed if there is only one source.
  if (sources_.size() == 1) {
    ContinueFuture future;
    RowVectorPtr data;
    auto reason = sources_[0]->next(data, &future);
    if (reason != BlockingReason::kNotBlocked) {
      sourceBlockingFutures_.emplace_back(std::move(future));
      return nullptr;
    }

    finished_ = data == nullptr;
    return data;
  }

  if (!output_) {
    output_ = BaseVector::create<RowVector>(
        outputType_, outputBatchSize_, operatorCtx_->pool());
    for (auto& child : output_->children()) {
      child->resize(outputBatchSize_);
    }
  }

  for (;;) {
    auto stream = treeOfLosers_->next();

    if (!stream) {
      finished_ = true;

      // Return nullptr if there is no data.
      if (outputSize_ == 0) {
        return nullptr;
      }

      output_->resize(outputSize_);
      return std::move(output_);
    }

    if (stream->setOutputRow(outputSize_)) {
      // The stream is at end of input batch. Need to copy out the rows before
      // fetching next batch in 'pop'.
      stream->copyToOutput(output_);
    }

    ++outputSize_;

    // Advance the stream.
    stream->pop(sourceBlockingFutures_);

    if (outputSize_ == outputBatchSize_) {
      // Copy out data from all sources.
      for (auto& s : streams_) {
        s->copyToOutput(output_);
      }

      outputSize_ = 0;
      return std::move(output_);
    }

    if (!sourceBlockingFutures_.empty()) {
      return nullptr;
    }
  }
}

void Merge::close() {
  for (auto& source : sources_) {
    source->close();
  }
}

bool SourceStream::operator<(const MergeStream& other) const {
  const auto& otherCursor = static_cast<const SourceStream&>(other);
  for (auto i = 0; i < sortingKeys_.size(); ++i) {
    const auto& [_, compareFlags] = sortingKeys_[i];
    VELOX_DCHECK(
        compareFlags.nullAsValue(), "not supported null handling mode");
    if (auto result = keyColumns_[i]
                          ->compare(
                              otherCursor.keyColumns_[i],
                              currentSourceRow_,
                              otherCursor.currentSourceRow_,
                              compareFlags)
                          .value()) {
      return result < 0;
    }
  }
  return false;
}

bool SourceStream::pop(std::vector<ContinueFuture>& futures) {
  ++currentSourceRow_;
  if (currentSourceRow_ == data_->size()) {
    // Make sure all current data has been copied out.
    VELOX_CHECK(!outputRows_.hasSelections());
    return fetchMoreData(futures);
  }

  return false;
}

void SourceStream::copyToOutput(RowVectorPtr& output) {
  outputRows_.updateBounds();

  if (!outputRows_.hasSelections()) {
    return;
  }

  vector_size_t sourceRow = firstSourceRow_;
  outputRows_.applyToSelected(
      [&](auto row) { sourceRows_[row] = sourceRow++; });

  for (auto i = 0; i < output->type()->size(); ++i) {
    output->childAt(i)->copy(
        data_->childAt(i).get(), outputRows_, sourceRows_.data());
  }

  outputRows_.clearAll();

  if (sourceRow == data_->size()) {
    firstSourceRow_ = 0;
  } else {
    firstSourceRow_ = sourceRow;
  }
}

bool SourceStream::fetchMoreData(std::vector<ContinueFuture>& futures) {
  ContinueFuture future;
  auto reason = source_->next(data_, &future);
  if (reason != BlockingReason::kNotBlocked) {
    needData_ = true;
    futures.emplace_back(std::move(future));
    return true;
  }

  atEnd_ = !data_ || data_->size() == 0;
  needData_ = false;
  currentSourceRow_ = 0;

  if (!atEnd_) {
    for (auto& child : data_->children()) {
      child = BaseVector::loadedVectorShared(child);
    }
    keyColumns_.clear();
    for (const auto& key : sortingKeys_) {
      keyColumns_.push_back(data_->childAt(key.first).get());
    }
  }
  return false;
}

LocalMerge::LocalMerge(
    int32_t operatorId,
    DriverCtx* driverCtx,
    const std::shared_ptr<const core::LocalMergeNode>& localMergeNode)
    : Merge(
          operatorId,
          driverCtx,
          localMergeNode->outputType(),
          localMergeNode->sortingKeys(),
          localMergeNode->sortingOrders(),
          localMergeNode->id(),
          "LocalMerge") {
  VELOX_CHECK_EQ(
      operatorCtx_->driverCtx()->driverId,
      0,
      "LocalMerge needs to run single-threaded");
}

BlockingReason LocalMerge::addMergeSources(ContinueFuture* /* future */) {
  if (sources_.empty()) {
    sources_ = operatorCtx_->task()->getLocalMergeSources(
        operatorCtx_->driverCtx()->splitGroupId, planNodeId());
  }
  return BlockingReason::kNotBlocked;
}

MergeExchange::MergeExchange(
    int32_t operatorId,
    DriverCtx* driverCtx,
    const std::shared_ptr<const core::MergeExchangeNode>& mergeExchangeNode)
    : Merge(
          operatorId,
          driverCtx,
          mergeExchangeNode->outputType(),
          mergeExchangeNode->sortingKeys(),
          mergeExchangeNode->sortingOrders(),
          mergeExchangeNode->id(),
          "MergeExchange"),
      serde_(getNamedVectorSerde(mergeExchangeNode->serdeKind())),
      serdeOptions_(getVectorSerdeOptions(
          driverCtx->queryConfig(),
          mergeExchangeNode->serdeKind())) {}

BlockingReason MergeExchange::addMergeSources(ContinueFuture* future) {
  if (operatorCtx_->driverCtx()->driverId != 0) {
    // When there are multiple pipelines, a single operator, the one from
    // pipeline 0, is responsible for merging pages.
    return BlockingReason::kNotBlocked;
  }
  if (noMoreSplits_) {
    return BlockingReason::kNotBlocked;
  }

  for (;;) {
    exec::Split split;
    auto reason = operatorCtx_->task()->getSplitOrFuture(
        operatorCtx_->driverCtx()->splitGroupId, planNodeId(), split, *future);
    if (reason != BlockingReason::kNotBlocked) {
      return reason;
    }

    if (split.hasConnectorSplit()) {
      auto remoteSplit =
          std::dynamic_pointer_cast<RemoteConnectorSplit>(split.connectorSplit);
      VELOX_CHECK_NOT_NULL(remoteSplit, "Wrong type of split");
      remoteSourceTaskIds_.push_back(remoteSplit->taskId);
      continue;
    }

    noMoreSplits_ = true;
    if (!remoteSourceTaskIds_.empty()) {
      const auto maxMergeExchangeBufferSize =
          operatorCtx_->driverCtx()->queryConfig().maxMergeExchangeBufferSize();
      const auto maxQueuedBytesPerSource = std::min<int64_t>(
          std::max<int64_t>(
              maxMergeExchangeBufferSize / remoteSourceTaskIds_.size(),
              MergeSource::kMaxQueuedBytesLowerLimit),
          MergeSource::kMaxQueuedBytesUpperLimit);
      for (uint32_t remoteSourceIndex = 0;
           remoteSourceIndex < remoteSourceTaskIds_.size();
           ++remoteSourceIndex) {
        auto* pool = operatorCtx_->task()->addMergeSourcePool(
            operatorCtx_->planNodeId(),
            operatorCtx_->driverCtx()->pipelineId,
            remoteSourceIndex);
        sources_.emplace_back(MergeSource::createMergeExchangeSource(
            this,
            remoteSourceTaskIds_[remoteSourceIndex],
            operatorCtx_->task()->destination(),
            maxQueuedBytesPerSource,
            pool,
            operatorCtx_->task()->queryCtx()->executor()));
      }
    }
    // TODO Delay this call until all input data has been processed.
    operatorCtx_->task()->multipleSplitsFinished(
        false, remoteSourceTaskIds_.size(), 0);
    return BlockingReason::kNotBlocked;
  }
}

void MergeExchange::close() {
  for (auto& source : sources_) {
    source->close();
  }
  Operator::close();
  {
    auto lockedStats = stats_.wlock();
    lockedStats->addRuntimeStat(
        Operator::kShuffleSerdeKind,
        RuntimeCounter(static_cast<int64_t>(serde_->kind())));
    lockedStats->addRuntimeStat(
        Operator::kShuffleCompressionKind,
        RuntimeCounter(static_cast<int64_t>(serdeOptions_->compressionKind)));
  }
}
} // namespace facebook::velox::exec
