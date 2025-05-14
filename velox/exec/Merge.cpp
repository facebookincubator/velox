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
#include "velox/exec/OperatorUtils.h"
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
    const std::string& operatorType,
    const std::optional<common::SpillConfig>& spillConfig)
    : SourceOperator(
          driverCtx,
          std::move(outputType),
          operatorId,
          planNodeId,
          operatorType,
          spillConfig),
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
    const auto& queryConfig = operatorCtx_->task()->queryCtx()->queryConfig();
    maxMergeSources_ = queryConfig.localMergeMaxMergeSources();
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
      mergeBuffer_ = std::make_unique<MergeBuffer>(outputType_, pool());
    }
    mergeBuffer_->maybeStartMoreSources(
        numStartedSources_,
        sourceBlockingFutures_,
        maxMergeSources_,
        outputBatchSize_,
        sources_,
        sortingKeys_);
  } else {
    startSources();

    // No merging is needed if there is only one source.
    if (streams_.empty() && sources_.size() > 1) {
      initializeTreeOfLosers();
    }

    if (sourceBlockingFutures_.empty()) {
      for (const auto& cursor : streams_) {
        cursor->isBlocked(sourceBlockingFutures_);
      }
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

void Merge::spill() {
  if (mergeSpiller_ == nullptr) {
    VELOX_CHECK(spillConfig_.has_value());
    mergeSpiller_ = std::make_unique<MergeSpiller>(
        outputType_,
        std::nullopt,
        HashBitRange{},
        sortingKeys_,
        &spillConfig_.value(),
        &spillStats_);
  }
  bool isLastRun = false;
  const auto output = mergeBuffer_->getOutputFromSource(
      outputBatchSize_, sourceBlockingFutures_, isLastRun);
  if (output != nullptr) {
    mergeBuffer_->addSpillRowsNum(output->size());
    mergeSpiller_->spill(SpillPartitionId{0}, output);
  }

  if (isLastRun) {
    SpillPartitionSet spillPartitionSet;
    mergeSpiller_->finishSpill(spillPartitionSet);
    mergeSpiller_ = nullptr;
    VELOX_CHECK_EQ(spillPartitionSet.size(), 1);
    const auto& spillFiles = spillPartitionSet.cbegin()->second->files();
    std::vector<std::unique_ptr<SpillReadFile>> spillReadFiles;
    spillReadFiles.reserve(spillFiles.size());
    for (const auto& spillFile : spillFiles) {
      spillReadFiles.emplace_back(SpillReadFile::create(
          spillFile, spillConfig_->readBufferSize, pool(), &spillStats_));
    }
    spillReadFilesGroup_.emplace_back(std::move(spillReadFiles));
  }
}

RowVectorPtr Merge::getOutput() {
  if (finished_) {
    return nullptr;
  }

  if (maxMergeSources_ < sources_.size()) {
    VELOX_CHECK_NOT_NULL(mergeBuffer_);
    if (mergeBuffer_->needsSpill()) {
      spill();
      return nullptr;
    }

    VELOX_CHECK(numStartedSources_ >= sources_.size());
    if (!spillReadFilesGroup_.empty()) {
      mergeBuffer_->createSpillMerger(std::move(spillReadFilesGroup_));
    }

    // Start to output final results.
    output_ = mergeBuffer_->getOutputFromSpill(outputBatchSize_);
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

RowVectorPtr MergeBuffer::getOutputFromSource(
    vector_size_t maxOutputRows,
    std::vector<ContinueFuture>& sourceBlockingFutures,
    bool& isLastRun) {
  VELOX_CHECK_NOT_NULL(sourceStreamMerger_);
  VELOX_CHECK_NULL(spillMerger_);
  if (!output_) {
    output_ = BaseVector::create<RowVector>(type_, maxOutputRows, pool_);
    for (auto& child : output_->children()) {
      child->resize(maxOutputRows);
    }
  }

  uint64_t outputSize = 0;
  for (;;) {
    const auto stream = sourceStreamMerger_->next();
    if (stream == nullptr) {
      if (outputSize > 0) {
        output_->resize(outputSize);
      }
      isLastRun = true;
      sourceStreamMerger_ = nullptr;
      sourceStreams_.clear();
      return std::move(output_);
    }

    if (stream->setOutputRow(outputSize)) {
      // The stream is at end of input batch. Need to copy out the rows
      // before fetching next batch in 'pop'.
      stream->copyToOutput(output_);
    }

    ++outputSize;

    // Advance the stream.
    stream->pop(sourceBlockingFutures);

    if (outputSize == maxOutputRows) {
      // Copy out data from all sources.
      for (const auto& s : sourceStreams_) {
        s->copyToOutput(output_);
      }

      return std::move(output_);
    }

    if (!sourceBlockingFutures.empty()) {
      return nullptr;
    }
  }
}

RowVectorPtr MergeBuffer::getOutputFromSpill(vector_size_t maxOutputRows) {
  VELOX_CHECK_NULL(sourceStreamMerger_);
  VELOX_CHECK_NOT_NULL(spillMerger_);
  // Finished.
  if (numOutputRows_ == numSpillRows_) {
    return nullptr;
  }

  VELOX_CHECK_GT(maxOutputRows, 0);
  VELOX_CHECK_GT(numSpillRows_, numOutputRows_);
  const vector_size_t batchSize =
      std::min<uint64_t>(numSpillRows_ - numOutputRows_, maxOutputRows);
  if (output_ != nullptr) {
    VectorPtr output = std::move(output_);
    BaseVector::prepareForReuse(output, batchSize);
    output_ = std::static_pointer_cast<RowVector>(output);
  } else {
    output_ = std::static_pointer_cast<RowVector>(
        BaseVector::create(type_, batchSize, pool_));
  }

  for (const auto& child : output_->children()) {
    child->resize(batchSize);
  }

  spillSources_.resize(batchSize);
  spillSourceRows_.resize(batchSize);

  VELOX_CHECK_GT(output_->size(), 0);
  VELOX_CHECK_LE(output_->size() + numOutputRows_, numSpillRows_);

  int32_t outputRow = 0;
  int32_t outputSize = 0;
  bool isEndOfBatch = false;
  while (outputRow + outputSize < output_->size()) {
    SpillMergeStream* stream = spillMerger_->next();
    VELOX_CHECK_NOT_NULL(stream);

    spillSources_[outputSize] = &stream->current();
    spillSourceRows_[outputSize] = stream->currentIndex(&isEndOfBatch);
    ++outputSize;
    if (FOLLY_UNLIKELY(isEndOfBatch)) {
      // The stream is at end of input batch. Need to copy out the rows before
      // fetching next batch in 'pop'.
      gatherCopy(
          output_.get(),
          outputRow,
          outputSize,
          spillSources_,
          spillSourceRows_,
          {});
      outputRow += outputSize;
      outputSize = 0;
    }
    // Advance the stream.
    stream->pop();
  }
  VELOX_CHECK_EQ(outputRow + outputSize, output_->size());

  if (FOLLY_LIKELY(outputSize != 0)) {
    gatherCopy(
        output_.get(),
        outputRow,
        outputSize,
        spillSources_,
        spillSourceRows_,
        {});
  }

  numOutputRows_ += output_->size();
  return std::move(output_);
}

void MergeBuffer::maybeStartMoreSources(
    size_t& numStartedSources,
    std::vector<ContinueFuture>& sourceBlockingFutures,
    uint32_t maxMergeSources,
    vector_size_t outputBatchSize,
    const std::vector<std::shared_ptr<MergeSource>>& sources,
    const std::vector<SpillSortKey>& sortingKeys) {
  if (sourceStreamMerger_ == nullptr && numStartedSources < sources.size()) {
    VELOX_CHECK(sourceStreams_.empty());
    // Plans merge sources for this run.
    std::vector<MergeSource*> currentSources;
    for (auto i = numStartedSources;
         i < (std::min(sources.size(), numStartedSources + maxMergeSources));
         ++i) {
      currentSources.push_back(sources[i].get());
    }

    // Initializes the input merger.
    std::vector<std::unique_ptr<SourceStream>> currentCursors;
    currentCursors.reserve(currentSources.size());
    for (auto* source : currentSources) {
      currentCursors.push_back(
          std::make_unique<SourceStream>(source, sortingKeys, outputBatchSize));
    }

    // mergeStreams_ and inputMerger_ are paired.
    VELOX_CHECK(sourceStreams_.empty());
    for (auto& cursor : currentCursors) {
      sourceStreams_.push_back(cursor.get());
    }
    sourceStreamMerger_ =
        std::make_unique<TreeOfLosers<SourceStream>>(std::move(currentCursors));

    // Start sources.
    for (const auto& source : currentSources) {
      source->start();
    }
    numStartedSources += currentSources.size();
  }

  if (sourceBlockingFutures.empty()) {
    for (const auto& cursor : sourceStreams_) {
      cursor->isBlocked(sourceBlockingFutures);
    }
  }
}

void MergeBuffer::createSpillMerger(
    std::vector<std::vector<std::unique_ptr<SpillReadFile>>>
        spillReadFilesGroup) {
  VELOX_CHECK_NULL(spillMerger_);
  std::vector<std::unique_ptr<SpillMergeStream>> streams;
  streams.reserve(spillReadFilesGroup.size());
  for (auto i = 0; i < spillReadFilesGroup.size(); ++i) {
    streams.push_back(ConcatFilesSpillMergeStream::create(
        i, std::move(spillReadFilesGroup[i])));
  }
  spillMerger_ =
      std::make_unique<TreeOfLosers<SpillMergeStream>>(std::move(streams));
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
          "LocalMerge",
          localMergeNode->canSpill(driverCtx->queryConfig())
              ? driverCtx->makeSpillConfig(operatorId)
              : std::nullopt) {
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
