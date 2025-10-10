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

#include "velox/exec/ExchangeAggregation.h"

#include "velox/exec/Task.h"

namespace facebook::velox::exec {

std::atomic<int64_t> deleteCount{0};

PartitionBucket::PartitionBucket(size_t bucketSize, int32_t driverCount)
    : bucketSize_{bucketSize},
      slots_(bucketSize),
      partitionCount_{0},
      memoryUsage_{0},
      driverCount_{driverCount},
      noMoreinputDriverCount_{0} {
  for (auto i = 0; i < bucketSize; ++i) {
    slots_[i].store(nullptr);
  }
}

bool PartitionBucket::tryAdd(Partitioned* partitioned) {
  size_t index;
  uint64_t random;
  auto size = partitioned->size;
  for (auto i = 0; i < 10; ++i) {
    random = folly::hardware_timestamp() / 1000;
    for (auto j = 0; j < 4; ++j) {
      index = (random + j) % bucketSize_;
      Partitioned* null = nullptr;
      if (slots_[index].compare_exchange_strong(null, partitioned)) {
        partitionCount_.fetch_add(1);
        memoryUsage_.fetch_add(size);
        return true;
      }
    }
  }
  return false;
}

Partitioned* FOLLY_NULLABLE PartitionBucket::getSlotAt(int32_t index) {
  auto* slot = slots_[index].load();
  if (slot != nullptr && slots_[index].compare_exchange_strong(slot, nullptr)) {
    partitionCount_.fetch_sub(1);
    memoryUsage_.fetch_sub(slot->size);
    return slot;
  }
  return nullptr;
}

void PartitionBucket::testingValidate() {
  for (auto& slot : slots_) {
    VELOX_CHECK(slot.load() == nullptr);
  }
}

ExchangeAggregation::ExchangeAggregation(
    int32_t operatorId,
    DriverCtx* ctx,
    const core::PartitionFunctionSpecPtr& partitionFunctionSpec,
    const std::shared_ptr<const core::AggregationNode>& planNode)
    : Operator(
          ctx,
          planNode->outputType(),
          operatorId,
          planNode->id(),
          "ExchangeAggregation",
          std::nullopt), // TODO: enable spilling.
      planNode_{planNode},
      driverId_{ctx->driverId},
      buckets_{ctx->task->getExchangeAggregationBuckets(
          ctx->splitGroupId,
          planNode->id())},
      numPartitions_{buckets_.size()},
      partitionFunction_{
          buckets_.size() == 1 ? nullptr
                               : partitionFunctionSpec->create(
                                     numPartitions_,
                                     /*localExchange=*/true)},
      memoryLimit_{ctx->queryConfig().maxLocalExchangeBufferSize()},
      isDistinct_{planNode->aggregates().empty()} {
  VELOX_CHECK(
      !planNode->groupingKeys().empty(), "Global aggregations is unsupported");
}

void ExchangeAggregation::initialize() {
  Operator::initialize();

  peers_ = operatorCtx_->task()->findPeerOperators(
      operatorCtx_->driverCtx()->pipelineId, this);

  VELOX_CHECK(pool()->trackUsage());

  const auto& inputType = planNode_->sources()[0]->outputType();
  std::vector<column_index_t> groupingKeyInputChannels;
  std::vector<column_index_t> groupingKeyOutputChannels;
  setupGroupingKeyChannelProjections(
      groupingKeyInputChannels, groupingKeyOutputChannels);

  auto hashers = createVectorHashers(inputType, groupingKeyInputChannels);
  const auto numHashers = hashers.size();

  std::vector<column_index_t> preGroupedChannels;
  preGroupedChannels.reserve(planNode_->preGroupedKeys().size());
  for (const auto& key : planNode_->preGroupedKeys()) {
    auto channel = exprToChannel(key.get(), inputType);
    preGroupedChannels.push_back(channel);
  }

  std::shared_ptr<core::ExpressionEvaluator> expressionEvaluator;
  std::vector<AggregateInfo> aggregateInfos = toAggregateInfo(
      *planNode_, *operatorCtx_, numHashers, expressionEvaluator);

  // Check that aggregate result type match the output type.
  for (auto i = 0; i < aggregateInfos.size(); i++) {
    const auto& aggResultType = aggregateInfos[i].function->resultType();
    const auto& expectedType = outputType_->childAt(numHashers + i);
    VELOX_CHECK(
        aggResultType->kindEquals(expectedType),
        "Unexpected result type for an aggregation: {}, expected {}, step {}",
        aggResultType->toString(),
        expectedType->toString(),
        core::AggregationNode::toName(planNode_->step()));
  }

  for (auto i = 0; i < hashers.size(); ++i) {
    identityProjections_.emplace_back(
        hashers[groupingKeyOutputChannels[i]]->channel(), i);
  }

  std::optional<column_index_t> groupIdChannel;
  if (planNode_->groupId().has_value()) {
    groupIdChannel =
        outputType_->getChildIdxIfExists(planNode_->groupId().value()->name());
    VELOX_CHECK(groupIdChannel.has_value());
  }

  groupingSet_ = std::make_unique<GroupingSet>(
      inputType,
      std::move(hashers),
      std::move(preGroupedChannels),
      std::move(groupingKeyOutputChannels),
      std::move(aggregateInfos),
      planNode_->ignoreNullKeys(),
      /*isPartialOutput_*/ false,
      isRawInput(planNode_->step()),
      planNode_->globalGroupingSets(),
      groupIdChannel,
      nullptr, // TODO: enable spilling.
      &nonReclaimableSection_,
      &operatorCtx_->driverCtx()->queryConfig(),
      operatorCtx_->pool(),
      spillStats_.get());

  planNode_.reset();
}

void ExchangeAggregation::setupGroupingKeyChannelProjections(
    std::vector<column_index_t>& groupingKeyInputChannels,
    std::vector<column_index_t>& groupingKeyOutputChannels) const {
  VELOX_CHECK(groupingKeyInputChannels.empty());
  VELOX_CHECK(groupingKeyOutputChannels.empty());

  const auto& inputType = planNode_->sources()[0]->outputType();
  const auto& groupingKeys = planNode_->groupingKeys();
  // The map from the grouping key output channel to the input channel.
  //
  // NOTE: grouping key output order is specified as 'groupingKeys' in
  // 'aggregationNode_'.
  std::vector<IdentityProjection> groupingKeyProjections;
  groupingKeyProjections.reserve(groupingKeys.size());
  for (auto i = 0; i < groupingKeys.size(); ++i) {
    groupingKeyProjections.emplace_back(
        exprToChannel(groupingKeys[i].get(), inputType), i);
  }

  groupingKeyInputChannels.reserve(groupingKeys.size());
  for (auto i = 0; i < groupingKeys.size(); ++i) {
    groupingKeyInputChannels.push_back(groupingKeyProjections[i].inputChannel);
  }

  groupingKeyOutputChannels.resize(groupingKeys.size());
  std::iota(
      groupingKeyOutputChannels.begin(), groupingKeyOutputChannels.end(), 0);
  return;
}

void ExchangeAggregation::prepareForInput(RowVectorPtr& input) {
  {
    auto lockedStats = stats_.wlock();
    lockedStats->addOutputVector(input->estimateFlatSize(), input->size());
  }

  // Lazy vectors must be loaded or processed to ensure the late materialized in
  // order.
  for (auto& child : input->children()) {
    child->loadedVector();
  }
}

Partitioned* ExchangeAggregation::processPartition(
    RowVectorPtr input,
    const std::vector<uint32_t>& partitions,
    uint64_t inputSize) {
  std::vector<uint64_t> counters(numPartitions_, 0);
  for (auto i = 0; i < partitions.size(); i++) {
    counters[partitions[i]]++;
  }
  std::vector<vector_size_t> offsets(numPartitions_ + 1, 0);
  for (auto i = 0; i < numPartitions_; i++) {
    offsets[i + 1] = offsets[i] + counters[i];
  }
  auto runningOffsets = offsets;

  auto sortedPartitions =
      AlignedBuffer::allocate<vector_size_t>(input->size(), pool());
  auto rawSortedPartitions = sortedPartitions->asMutable<vector_size_t>();
  for (auto i = 0; i < input->size(); i++) {
    auto& index = partitions[i];
    rawSortedPartitions[runningOffsets[index]++] = i;
  }

  return new Partitioned{
      input, std::move(sortedPartitions), std::move(offsets), inputSize};
}

void ExchangeAggregation::addWaitingListToBuckets() {
  auto numPartitions = waitingList_.size();
  while (numPartitions-- > 0) {
    auto& entry = waitingList_.front();
    auto i = entry.first;
    auto* partitioned = entry.second;
    waitingList_.pop();

    addOneToBucket(i, partitioned);
  }
}

void ExchangeAggregation::addInput(RowVectorPtr input) {
  addWaitingListToBuckets();

  prepareForInput(input);
  const auto inputSize = input->retainedSize();
  const auto singlePartition = numPartitions_ == 1
      ? 0
      : partitionFunction_->partition(*input, partitions_);
  Partitioned* partitioned;
  if (singlePartition.has_value()) {
    VELOX_CHECK_EQ(buckets_.size(), 1);
    partitioned = new Partitioned{input, nullptr, {}, inputSize};
    addOneToBucket(0, partitioned);
  } else {
    partitioned = processPartition(input, partitions_, inputSize);
    for (auto i = 0; i < numPartitions_; ++i) {
      addOneToBucket(i, partitioned);
    }
  }
}

RowVectorPtr ExchangeAggregation::wrapInDictionary(
    const Partitioned* partitioned,
    int32_t dirverId) {
  if (partitioned->partitions == nullptr) {
    // Single partition, no need to wrap in dictionary.
    return partitioned->input;
  }

  auto offset = partitioned->offsets[dirverId];
  auto size = partitioned->offsets[dirverId + 1] - offset;
  auto indices = Buffer::slice<vector_size_t>(
      partitioned->partitions, offset, size, pool());
  if (!partitionedVector_) {
    partitionedVector_ = std::make_shared<RowVector>(
        pool(),
        partitioned->input->type(),
        nullptr,
        size,
        std::vector<VectorPtr>(partitioned->input->childrenSize()));
  } else {
    partitionedVector_->unsafeResize(size);
  }

  for (auto i = 0; i < partitioned->input->childrenSize(); ++i) {
    auto& child = partitionedVector_->childAt(i);
    if (child && child->encoding() == VectorEncoding::Simple::DICTIONARY &&
        child.use_count() == 1) {
      child->BaseVector::resize(size);
      child->setWrapInfo(indices);
      child->setValueVector(partitioned->input->childAt(i));
    } else {
      child = BaseVector::wrapInDictionary(
          nullptr, indices, size, partitioned->input->childAt(i));
    }
  }
  return partitionedVector_;
}

RowVectorPtr ExchangeAggregation::getOutput() {
  if (aggregationFinished_) {
    return nullptr;
  } else if (exchangeFinished_) {
    auto vector = fillOutput();
    return vector;
  }

  addWaitingListToBuckets();

  VELOX_CHECK_LT(driverId_, buckets_.size());
  auto& bucket = buckets_[driverId_];
  auto isFinished = bucket->isFinished();
  if (isFinished) {
    exchangeFinished_ = true;
    return nullptr;
  }
  if (bucket->partitionCount() < bucket->bucketSize() * 0.5 &&
      bucket->memoryUsage() < memoryLimit_ && !bucket->needToConsume()) {
    if (distinctOutputs_.empty()) {
      return nullptr;
    } else {
      const auto output = distinctOutputs_.front();
      distinctOutputs_.pop();
      return output;
    }
  }

  for (auto i = 0; i < bucket->bucketSize(); ++i) {
    auto partition = bucket->getSlotAt(i);
    if (partition != nullptr) {
      auto partitionVector = wrapInDictionary(partition, driverId_);
      doAggregation(partitionVector);
      if (partition->accessCount.fetch_add(1) >= buckets_.size() - 1) {
        delete partition;
        deleteCount.fetch_add(1);
      }
    }
  }
  bucket->resetNeedToConsume();
  if (!distinctOutputs_.empty()) {
    const auto output = distinctOutputs_.front();
    distinctOutputs_.pop();
    return output;
  } else {
    return nullptr;
  }
}

void ExchangeAggregation::addOneToBucket(
    int32_t bucketIndex,
    Partitioned* partitioned) {
  auto succeed = buckets_[bucketIndex]->tryAdd(partitioned);
  if (!succeed) {
    buckets_[bucketIndex]->requestToConsume();
    auto partitionVector = wrapInDictionary(partitioned, bucketIndex);
    if (tryDoAggregation(bucketIndex, partitionVector)) {
      if (partitioned->accessCount.fetch_add(1) >= buckets_.size() - 1) {
        delete partitioned;
        deleteCount.fetch_add(1);
      }
    } else {
      waitingList_.emplace(bucketIndex, partitioned);
    }
  }
}

void ExchangeAggregation::doAggregationLocked(RowVectorPtr partitionVector) {
  groupingSet_->addInput(partitionVector, /*mayPushdown_*/ false);

  updateRuntimeStats();

  if (isDistinct_) {
    input_ = partitionVector;
    auto newDistincts = !groupingSet_->hashLookup().newGroups.empty();
    getDistinctOutput(newDistincts);
  }
}

bool ExchangeAggregation::tryDoAggregation(RowVectorPtr partitionVector) {
  if (mutex_.try_lock()) {
    doAggregationLocked(partitionVector);
    mutex_.unlock();
    return true;
  }
  return false;
}

void ExchangeAggregation::doAggregation(RowVectorPtr partitionVector) {
  mutex_.lock();
  doAggregationLocked(partitionVector);
  mutex_.unlock();
}

bool ExchangeAggregation::tryDoAggregation(
    int32_t driverId,
    RowVectorPtr partitionVector) {
  auto* exchangeAggregation =
      dynamic_cast<ExchangeAggregation*>(peers_[driverId]);
  return exchangeAggregation->tryDoAggregation(partitionVector);
}

void ExchangeAggregation::prepareOutput(vector_size_t size) {
  if (output_) {
    VectorPtr output = std::move(output_);
    BaseVector::prepareForReuse(output, size);
    output_ = std::static_pointer_cast<RowVector>(output);
  } else {
    output_ = std::static_pointer_cast<RowVector>(
        BaseVector::create(outputType_, size, pool()));
  }
}

void ExchangeAggregation::getDistinctOutput(bool newDistincts) {
  VELOX_CHECK(isDistinct_);
  VELOX_CHECK(!aggregationFinished_);

  if (newDistincts) {
    VELOX_CHECK_NOT_NULL(input_);

    auto& lookup = groupingSet_->hashLookup();
    const auto size = lookup.newGroups.size();
    BufferPtr indices = allocateIndices(size, operatorCtx_->pool());
    auto* indicesPtr = indices->asMutable<vector_size_t>();
    std::copy(lookup.newGroups.begin(), lookup.newGroups.end(), indicesPtr);
    auto output = Operator::fillOutput(size, indices);

    // Drop reference to input_ to make it singly-referenced at the producer and
    // allow for memory reuse.
    input_ = nullptr;

    distinctOutputs_.emplace(std::move(output));
  }

  if (noMoreInput_) {
    if (auto numRows = groupingSet_->numDefaultGlobalGroupingSetRows()) {
      prepareOutput(numRows.value());
      if (groupingSet_->getDefaultGlobalGroupingSetOutput(
              resultIterator_, output_)) {
        distinctOutputs_.emplace(std::move(output_));
      }
    }
  }
  return;
}

RowVectorPtr ExchangeAggregation::fillOutput() {
  if (isDistinct_) {
    if (distinctOutputs_.empty()) {
      // fillOutput() is only called when exchangeFinished_, so if
      // distinctOutputs_ is exhausited, the aggregation finishes too.
      aggregationFinished_ = true;
      return nullptr;
    } else {
      const auto output = distinctOutputs_.front();
      distinctOutputs_.pop();
      return output;
    }
  }

  const auto& queryConfig = operatorCtx_->driverCtx()->queryConfig();
  const auto maxOutputRows = outputBatchRows(estimatedOutputRowSize_);
  // Reuse output vectors if possible.
  prepareOutput(maxOutputRows);

  const bool hasData = groupingSet_->getOutput(
      maxOutputRows,
      queryConfig.preferredOutputBatchBytes(),
      resultIterator_,
      output_);
  if (!hasData) {
    resultIterator_.reset();
    aggregationFinished_ = true;
    return nullptr;
  }
  return output_;
}

void ExchangeAggregation::noMoreInput() {
  if (noMoreInput_ || !waitingList_.empty()) {
    return;
  }
  Operator::noMoreInput();
  for (auto& bucket : buckets_) {
    bucket->noMoreInput();
  }
}

bool ExchangeAggregation::isFinished() {
  return aggregationFinished_;
}

BlockingReason ExchangeAggregation::isBlocked(ContinueFuture* /*future*/) {
  return BlockingReason::kNotBlocked;
}

void ExchangeAggregation::updateRuntimeStats() {
  // Report range sizes and number of distinct values for the group-by keys.
  const auto& hashers = groupingSet_->hashLookup().hashers;
  uint64_t asRange{0};
  uint64_t asDistinct{0};
  const auto hashTableStats = groupingSet_->hashTableStats();

  auto lockedStats = stats_.wlock();
  auto& runtimeStats = lockedStats->runtimeStats;

  for (auto i = 0; i < hashers.size(); i++) {
    hashers[i]->cardinality(0, asRange, asDistinct);
    if (asRange != VectorHasher::kRangeTooLarge) {
      runtimeStats[fmt::format("rangeKey{}", i)] = RuntimeMetric(asRange);
    }
    if (asDistinct != VectorHasher::kRangeTooLarge) {
      runtimeStats[fmt::format("distinctKey{}", i)] = RuntimeMetric(asDistinct);
    }
  }

  runtimeStats[BaseHashTable::kCapacity] =
      RuntimeMetric(hashTableStats.capacity);
  runtimeStats[BaseHashTable::kNumRehashes] =
      RuntimeMetric(hashTableStats.numRehashes);
  runtimeStats[BaseHashTable::kNumDistinct] =
      RuntimeMetric(hashTableStats.numDistinct);
  runtimeStats[BaseHashTable::kNumTombstones] =
      RuntimeMetric(hashTableStats.numTombstones);
}

void ExchangeAggregation::close() {
  Operator::close();

  while (!waitingList_.empty()) {
    // waitingList_ can be non-empty if the query is aborted during execution.
    auto* partitioned = waitingList_.front().second;
    waitingList_.pop();
    if (partitioned->accessCount.fetch_add(1) >= buckets_.size() - 1) {
      delete partitioned;
    }
  }

  output_ = nullptr;
  partitionedVector_ = nullptr;
  groupingSet_.reset();
}

} // namespace facebook::velox::exec
