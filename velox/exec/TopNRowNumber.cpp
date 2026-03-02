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
#include "velox/exec/TopNRowNumber.h"

namespace facebook::velox::exec {

namespace {

#define RANK_FUNCTION_DISPATCH(TEMPLATE_FUNC, functionKind, ...)             \
  [&]() {                                                                    \
    switch (functionKind) {                                                  \
      case core::TopNRowNumberNode::RankFunction::kRowNumber: {              \
        return TEMPLATE_FUNC<                                                \
            core::TopNRowNumberNode::RankFunction::kRowNumber>(__VA_ARGS__); \
      }                                                                      \
      case core::TopNRowNumberNode::RankFunction::kRank: {                   \
        return TEMPLATE_FUNC<core::TopNRowNumberNode::RankFunction::kRank>(  \
            __VA_ARGS__);                                                    \
      }                                                                      \
      case core::TopNRowNumberNode::RankFunction::kDenseRank: {              \
        return TEMPLATE_FUNC<                                                \
            core::TopNRowNumberNode::RankFunction::kDenseRank>(__VA_ARGS__); \
      }                                                                      \
      default:                                                               \
        VELOX_FAIL(                                                          \
            "not a rank function kind: {}",                                  \
            core::TopNRowNumberNode::rankFunctionName(functionKind));        \
    }                                                                        \
  }()

std::vector<column_index_t> reorderInputChannels(
    const RowTypePtr& inputType,
    const std::vector<core::FieldAccessTypedExprPtr>& partitionKeys,
    const std::vector<core::FieldAccessTypedExprPtr>& sortingKeys) {
  const auto size = inputType->size();

  std::vector<column_index_t> channels;
  channels.reserve(size);

  std::unordered_set<std::string> keyNames;

  for (const auto& key : partitionKeys) {
    channels.push_back(exprToChannel(key.get(), inputType));
    keyNames.insert(key->name());
  }

  for (const auto& key : sortingKeys) {
    channels.push_back(exprToChannel(key.get(), inputType));
    keyNames.insert(key->name());
  }

  for (auto i = 0; i < size; ++i) {
    if (keyNames.count(inputType->nameOf(i)) == 0) {
      channels.push_back(i);
    }
  }

  return channels;
}

RowTypePtr reorderInputType(
    const RowTypePtr& inputType,
    const std::vector<column_index_t>& channels) {
  const auto size = inputType->size();

  VELOX_CHECK_EQ(size, channels.size());

  std::vector<std::string> names;
  names.reserve(size);

  std::vector<TypePtr> types;
  types.reserve(size);

  for (auto channel : channels) {
    names.push_back(inputType->nameOf(channel));
    types.push_back(inputType->childAt(channel));
  }

  return ROW(std::move(names), std::move(types));
}

std::vector<CompareFlags> makeSpillCompareFlags(
    int32_t numPartitionKeys,
    const std::vector<core::SortOrder>& sortingOrders) {
  std::vector<CompareFlags> compareFlags;
  compareFlags.reserve(numPartitionKeys + sortingOrders.size());

  for (auto i = 0; i < numPartitionKeys; ++i) {
    compareFlags.push_back({});
  }

  for (const auto& order : sortingOrders) {
    compareFlags.push_back(
        {order.isNullsFirst(), order.isAscending(), false /*equalsOnly*/});
  }

  return compareFlags;
}

// Returns a [start, end) slice of the 'types' vector.
std::vector<TypePtr>
slice(const std::vector<TypePtr>& types, int32_t start, int32_t end) {
  std::vector<TypePtr> result;
  result.reserve(end - start);
  for (auto i = start; i < end; ++i) {
    result.push_back(types[i]);
  }
  return result;
}
} // namespace

TopNRowNumber::TopNRowNumber(
    int32_t operatorId,
    DriverCtx* driverCtx,
    const std::shared_ptr<const core::TopNRowNumberNode>& node)
    : Operator(
          driverCtx,
          node->outputType(),
          operatorId,
          node->id(),
          "TopNRowNumber",
          node->canSpill(driverCtx->queryConfig())
              ? driverCtx->makeSpillConfig(operatorId, "TopNRowNumber")
              : std::nullopt),
      rankFunction_(node->rankFunction()),
      limit_{node->limit()},
      generateRowNumber_{node->generateRowNumber()},
      numPartitionKeys_{node->partitionKeys().size()},
      numSortingKeys_{node->sortingKeys().size()},
      inputChannels_{reorderInputChannels(
          node->inputType(),
          node->partitionKeys(),
          node->sortingKeys())},
      inputType_{reorderInputType(node->inputType(), inputChannels_)},
      spillCompareFlags_{
          makeSpillCompareFlags(numPartitionKeys_, node->sortingOrders())},
      abandonPartialMinRows_(
          driverCtx->queryConfig().abandonPartialTopNRowNumberMinRows()),
      abandonPartialMinPct_(
          driverCtx->queryConfig().abandonPartialTopNRowNumberMinPct()),
      data_(
          std::make_unique<RowContainer>(
              slice(inputType_->children(), 0, spillCompareFlags_.size()),
              slice(
                  inputType_->children(),
                  spillCompareFlags_.size(),
                  inputType_->size()),
              pool())),
      comparator_(
          inputType_,
          node->sortingKeys(),
          node->sortingOrders(),
          data_.get()),
      decodedVectors_(inputType_->size()) {
  const auto& keys = node->partitionKeys();
  const auto numKeys = keys.size();

  if (numKeys > 0) {
    Accumulator accumulator{
        true,
        sizeof(TopRows),
        false,
        1,
        nullptr,
        [](auto, auto) { VELOX_UNREACHABLE(); },
        [](auto) {}};

    table_ = std::make_unique<HashTable<false>>(
        createVectorHashers(node->inputType(), keys),
        std::vector<Accumulator>{accumulator},
        std::vector<TypePtr>{},
        false, // allowDuplicates
        false, // isJoinBuild
        false, // hasProbedFlag
        0, // minTableSizeForParallelJoinBuild
        pool());
    partitionOffset_ = table_->rows()->columnAt(numKeys).offset();
    lookup_ = std::make_unique<HashLookup>(table_->hashers(), pool());
  } else {
    allocator_ = std::make_unique<HashStringAllocator>(pool());
    singlePartition_ = std::make_unique<TopRows>(allocator_.get(), comparator_);
  }

  if (generateRowNumber_) {
    results_.resize(1);
  }
}

void TopNRowNumber::prepareInput(RowVectorPtr& input) {
  // Potential large memory usage site that might trigger arbitration. Make it
  // reclaimable because at this point it does not break the operator's state
  // atomicity.
  ReclaimableSectionGuard guard(this);
  for (auto i = 0; i < inputChannels_.size(); ++i) {
    decodedVectors_[i].decode(*input->childAt(inputChannels_[i]));
  }
}

void TopNRowNumber::addInput(RowVectorPtr input) {
  if (abandonedPartial_) {
    input_ = std::move(input);
    return;
  }

  const auto numInput = input->size();

  prepareInput(input);

  if (table_) {
    ensureInputFits(input);

    SelectivityVector rows(numInput);
    table_->prepareForGroupProbe(
        *lookup_, input, rows, BaseHashTable::kNoSpillInputStartPartitionBit);
    try {
      table_->groupProbe(
          *lookup_, BaseHashTable::kNoSpillInputStartPartitionBit);
    } catch (...) {
      // If groupProbe throws (e.g., due to OOM), we need to clean up the new
      // groups that were inserted but not yet initialized by
      // initializeNewPartitions(). Otherwise, close() will crash when trying to
      // destroy uninitialized TopRows structures.
      cleanupNewPartitions();
      throw;
    }

    // Initialize new partitions.
    initializeNewPartitions();

    // Process input rows. For each row, lookup the partition. If the highest
    // (top) rank in that partition is less than limit, add the new row.
    // Otherwise, check if row should replace an existing row or be discarded.
    RANK_FUNCTION_DISPATCH(processInputRowLoop, rankFunction_, numInput);

    // It is determined that the TopNRowNumber (as a partial) is not rejecting
    // enough input rows to make the duplicate detection worthwhile. Hence,
    // abandon the processing at this partial TopN and let the final TopN do
    // the processing.
    if (abandonPartialEarly()) {
      abandonedPartial_ = true;
      addRuntimeStat("abandonedPartial", RuntimeCounter(1));

      updateEstimatedOutputRowSize();
      outputBatchSize_ = outputBatchRows(estimatedOutputRowSize_);
      outputRows_.resize(outputBatchSize_);
    }
  } else {
    RANK_FUNCTION_DISPATCH(processInputRowLoop, rankFunction_, numInput);
  }
}

bool TopNRowNumber::abandonPartialEarly() const {
  if (table_ == nullptr || generateRowNumber_ || spiller_ != nullptr) {
    return false;
  }

  const auto numInput = stats_.rlock()->inputPositions;
  if (numInput < abandonPartialMinRows_) {
    return false;
  }

  const auto numOutput = data_->numRows();
  return (100 * numOutput / numInput) >= abandonPartialMinPct_;
}

void TopNRowNumber::initializeNewPartitions() {
  for (auto index : lookup_->newGroups) {
    new (lookup_->hits[index] + partitionOffset_)
        TopRows(table_->stringAllocator(), comparator_);
  }
}

void TopNRowNumber::cleanupNewPartitions() {
  std::vector<char*> newRows(lookup_->newGroups.size());
  for (auto i = 0; i < lookup_->newGroups.size(); ++i) {
    newRows[i] = lookup_->hits[lookup_->newGroups[i]];
  }
  table_->erase(folly::Range(newRows.data(), newRows.size()));
  lookup_->newGroups.clear();
}

template <>
char* TopNRowNumber::processRowWithinLimit<
    core::TopNRowNumberNode::RankFunction::kRank>(
    vector_size_t index,
    TopRows& partition) {
  // The topRanks queue is not filled yet.
  auto& topRows = partition.rows;
  if (topRows.empty()) {
    partition.topRank = 1;
  } else {
    // Rank assigns all peer rows the same rank, but the rank increments by
    // the number of peers when moving between peers. So when adding a new
    // row:
    // If row == top rank then top rank is unchanged.
    // If row < top rank then top rank += 1.
    // If row > top, then rank += number of peers of top rank.
    auto* topRow = topRows.top();
    auto result = comparator_.compare(decodedVectors_, index, topRow);
    if (result < 0) {
      partition.topRank += 1;
    } else if (result > 0) {
      partition.topRank += partition.numTopRankRows();
    }
  }
  return data_->newRow();
}

template <>
char* TopNRowNumber::processRowWithinLimit<
    core::TopNRowNumberNode::RankFunction::kDenseRank>(
    vector_size_t index,
    TopRows& partition) {
  // The topRanks queue is not filled yet.
  // dense_rank will add this row to its partition. But the top rank is
  // incremented only if the new row is not a peer of any other existing
  // row in the partition queue.
  if (!partition.isDuplicate(decodedVectors_, index)) {
    partition.topRank++;
  }
  return data_->newRow();
}

template <>
char* TopNRowNumber::processRowWithinLimit<
    core::TopNRowNumberNode::RankFunction::kRowNumber>(
    vector_size_t /*index*/,
    TopRows& partition) {
  // row_number accumulates the new row in the partition, and the top rank is
  // incremented by 1 as row_number increases by 1 at each new row.
  ++partition.topRank;
  return data_->newRow();
}

template <>
char* TopNRowNumber::processRowExceedingLimit<
    core::TopNRowNumberNode::RankFunction::kRank>(
    vector_size_t index,
    TopRows& partition) {
  auto& topRows = partition.rows;
  // The new row < top rank
  // For rank, the new row gets assigned its rank as per its position in the
  // queue. But the ranks of all subsequent rows increment by 1.
  // So we can remove the rows at the top rank as its rank > limit now.
  char* topRow = partition.removeTopRankRows();
  char* newRow = data_->initializeRow(topRow, /*reuse=*/true);
  if (topRows.empty()) {
    partition.topRank = 1;
  } else {
    // The new top rank value depends on the number of peers of the top ranking
    // row. If the current row also has the same value as the new top ranking
    // row then it has to be counted as a peer as well.
    auto numNewTopRankRows = partition.numTopRankRows();
    topRow = topRows.top();
    if (comparator_.compare(decodedVectors_, index, topRow) == 0) {
      partition.topRank = topRows.size() - numNewTopRankRows + 1;
    } else {
      partition.topRank = topRows.size() - numNewTopRankRows + 2;
    }
  }
  return newRow;
}

template <>
char* TopNRowNumber::processRowExceedingLimit<
    core::TopNRowNumberNode::RankFunction::kDenseRank>(
    vector_size_t index,
    TopRows& partition) {
  char* newRow = nullptr;
  // The new row < top rank
  // For dense_rank:
  // i) If the row is a peer of an existing row in the queue, then it
  // has the same rank as it. The ranks of other rows are unchanged. So its
  // only added to the queue.
  // ii) If the row is a distinct new value in the queue, then it is assigned
  // a rank as per its position, and the ranks of all subsequent rows += 1.
  // So the current top rank rows can be removed from the queue as their new
  // rank > limit.
  if (partition.isDuplicate(decodedVectors_, index)) {
    newRow = data_->newRow();
  } else {
    char* topRow = partition.removeTopRankRows();
    newRow = data_->initializeRow(topRow, /*reuse=*/true);
  }
  return newRow;
}

template <>
char* TopNRowNumber::processRowExceedingLimit<
    core::TopNRowNumberNode::RankFunction::kRowNumber>(
    vector_size_t /*index*/,
    TopRows& partition) {
  // The new row has rank < highest (aka top) rank at 'limit' function value.
  // For row_number, such rows are added to the accumulator queue and the
  // top rank row is popped out. The topRank remains the same.
  auto& topRows = partition.rows;
  char* topRow = topRows.top();
  topRows.pop();
  // Reuses the space of the popped row itself for the new row.
  return data_->initializeRow(topRow, true /* reuse */);
}

template <core::TopNRowNumberNode::RankFunction TRank>
void TopNRowNumber::processInputRow(vector_size_t index, TopRows& partition) {
  auto& topRows = partition.rows;

  char* newRow = nullptr;
  if (partition.topRank < limit_) {
    newRow = processRowWithinLimit<TRank>(index, partition);
  } else {
    // The partition has now accumulated >= limit rows. So the new rows can be
    // rejected or replace existing rows based on the order_by values.
    char* topRow = topRows.top();

    const auto result = comparator_.compare(decodedVectors_, index, topRow);
    if (result > 0) {
      // The new row is bigger than the top rank so far, so this row is ignored.
      return;
    }

    // This row has the same value as the top rank row. row_number rejects
    // such rows, but are added to the queue for rank and dense_rank. The top
    // rank remains unchanged.
    else if (result == 0) {
      if (rankFunction_ == core::TopNRowNumberNode::RankFunction::kRowNumber) {
        return;
      }
      newRow = data_->newRow();
    }

    else if (result < 0) {
      newRow = processRowExceedingLimit<TRank>(index, partition);
    }
  }

  for (auto col = 0; col < decodedVectors_.size(); ++col) {
    data_->store(decodedVectors_[col], index, newRow, col);
  }

  topRows.push(newRow);
}

template <core::TopNRowNumberNode::RankFunction TRank>
void TopNRowNumber::processInputRowLoop(vector_size_t numInput) {
  if (table_) {
    for (auto i = 0; i < numInput; ++i) {
      processInputRow<TRank>(i, partitionAt(lookup_->hits[i]));
    }
  } else {
    for (auto i = 0; i < numInput; ++i) {
      processInputRow<TRank>(i, *singlePartition_);
    }
  }
}

void TopNRowNumber::noMoreInput() {
  Operator::noMoreInput();

  updateEstimatedOutputRowSize();
  outputBatchSize_ = outputBatchRows(estimatedOutputRowSize_);

  if (spiller_ != nullptr) {
    // Spill remaining data to avoid running out of memory while sort-merging
    // spilled data.
    spill();

    VELOX_CHECK_NULL(merge_);
    SpillPartitionSet spillPartitionSet;
    spiller_->finishSpill(spillPartitionSet);
    VELOX_CHECK_EQ(spillPartitionSet.size(), 1);
    merge_ = spillPartitionSet.begin()->second->createOrderedReader(
        *spillConfig_, pool(), spillStats_.get());
  } else {
    outputRows_.resize(outputBatchSize_);
  }
}

void TopNRowNumber::updateEstimatedOutputRowSize() {
  const auto optionalRowSize = data_->estimateRowSize();
  if (!optionalRowSize.has_value()) {
    return;
  }

  auto rowSize = optionalRowSize.value();

  if (rowSize && generateRowNumber_) {
    rowSize += sizeof(int64_t);
  }

  if (!estimatedOutputRowSize_.has_value()) {
    estimatedOutputRowSize_ = rowSize;
  } else if (rowSize > estimatedOutputRowSize_.value()) {
    estimatedOutputRowSize_ = rowSize;
  }
}

// This function handles a special case when determining the starting
// rank value for the 'rank' function.
// If there are many peer rows for the highest rank, then topRank could
// oscillate between the two cases of topRank < limit and topRank > limit
// as rows are added
// E.g. If the input rows are 0, 0, 0, 5, 0, 0, 6 and we want rank <= 5, then
// at 0, 0, 0, 5 :
// topRows.pq - 0, 0, 0, 5 topRank -> 4
// 0 is added.
// topRows.pq - 0, 0, 0, 0, 5 topRank -> 5
// topRank = limit now.
// So when the next 0 is added, the last 5 is popped from TopRows and 0 is added
// topRows.pq - 0, 0, 0, 0, 0, topRank -> 1
// This makes topRank < 5 and so when 6 comes by, 6 is pushed
// topRows.pq - 0, 0, 0, 0, 0, 6 topRank -> 6
// So when doing getOutput, we need to adjust this case.
// Since topRank > limit, then the highest rank is popped and the
// topRank is adjusted as length(pq) - number_of_duplicates_of_new_top_row + 1.
vector_size_t TopNRowNumber::fixTopRank(TopRows& partition) {
  if (rankFunction_ == core::TopNRowNumberNode::RankFunction::kRank) {
    if (partition.topRank > limit_) {
      partition.removeTopRankRows();
      auto numNewTopRankRows = partition.numTopRankRows();
      partition.topRank = partition.rows.size() - numNewTopRankRows + 1;
    }
  }

  return partition.topRank;
}

TopNRowNumber::TopRows* TopNRowNumber::nextPartition() {
  auto setNextRankAndPeer = [&](TopRows& partition) {
    nextRank_ = fixTopRank(partition);
    numPeers_ = 1;
  };

  if (!table_) {
    if (!outputPartitionNumber_) {
      outputPartitionNumber_ = 0;
      setNextRankAndPeer(*singlePartition_);
      return singlePartition_.get();
    }
    return nullptr;
  }

  if (!outputPartitionNumber_) {
    numPartitions_ = table_->listAllRows(
        &partitionIt_,
        partitions_.size(),
        RowContainer::kUnlimited,
        partitions_.data());
    if (numPartitions_ == 0) {
      // No more partitions.
      return nullptr;
    }
    outputPartitionNumber_ = 0;
  } else {
    ++outputPartitionNumber_.value();
    if (outputPartitionNumber_ >= numPartitions_) {
      outputPartitionNumber_.reset();
      return nextPartition();
    }
  }

  auto partition = &partitionAt(partitions_[outputPartitionNumber_.value()]);
  setNextRankAndPeer(*partition);
  return partition;
}

template <core::TopNRowNumberNode::RankFunction TRank>
void TopNRowNumber::computeNextRankInMemory(
    TopRows& partition,
    vector_size_t outputIndex) {
  if constexpr (TRank == core::TopNRowNumberNode::RankFunction::kRowNumber) {
    nextRank_ -= 1;
    return;
  }

  // This is the logic for rank() and dense_rank().
  // If the next row is a peer of the current one, then the rank remains the
  // same.
  if (comparator_.compare(outputRows_[outputIndex], partition.rows.top()) ==
      0) {
    return;
  }

  // The new row is not a peer of the current one. So dense_rank drops the
  // rank by 1, but rank drops by the number of peers of the new top
  // row (new rank) in TopRows queue.
  if constexpr (TRank == core::TopNRowNumberNode::RankFunction::kDenseRank) {
    nextRank_ -= 1;
  } else {
    nextRank_ -= partition.numTopRankRows();
  }
}

template <core::TopNRowNumberNode::RankFunction TRank>
void TopNRowNumber::appendPartitionRows(
    TopRows& partition,
    vector_size_t numRows,
    vector_size_t outputOffset,
    FlatVector<int64_t>* rankValues) {
  // The partition.rows priority queue pops rows in order of reverse
  // ranks. Output rows based on nextRank_ and update it with each row.
  for (auto i = 0; i < numRows; ++i) {
    auto index = outputOffset + i;
    if (rankValues) {
      rankValues->set(index, nextRank_);
    }
    outputRows_[index] = partition.rows.top();
    partition.rows.pop();
    if (!partition.rows.empty()) {
      computeNextRankInMemory<TRank>(partition, index);
    }
  }
}

RowVectorPtr TopNRowNumber::getOutput() {
  if (finished_) {
    return nullptr;
  }

  if (abandonedPartial_) {
    if (input_ != nullptr) {
      auto output = std::move(input_);
      input_.reset();
      return output;
    }

    // There could be older rows accumulated in 'data_'.
    if (data_->numRows() > 0) {
      return getOutputFromMemory();
    }

    if (noMoreInput_) {
      finished_ = true;
    }

    // There is no data to return at this moment.
    return nullptr;
  }

  if (!noMoreInput_) {
    return nullptr;
  }

  // All the input data is received, so the operator can start producing
  // output.
  RowVectorPtr output;
  if (merge_ != nullptr) {
    output = RANK_FUNCTION_DISPATCH(getOutputFromSpill, rankFunction_);
  } else {
    output = getOutputFromMemory();
  }

  if (output == nullptr) {
    finished_ = true;
  }

  return output;
}

RowVectorPtr TopNRowNumber::getOutputFromMemory() {
  VELOX_CHECK_GT(outputBatchSize_, 0);

  // Loop over partitions and emit sorted rows along with row numbers.
  auto output =
      BaseVector::create<RowVector>(outputType_, outputBatchSize_, pool());
  FlatVector<int64_t>* rowNumbers = nullptr;
  if (generateRowNumber_) {
    rowNumbers = output->children().back()->as<FlatVector<int64_t>>();
  }

  vector_size_t offset = 0;
  // Continue to output as many remaining partitions as possible.
  while (offset < outputBatchSize_) {
    // Get the next partition if one is not available already and output it.
    if (!outputPartition_) {
      outputPartition_ = nextPartition();
      // There is nothing to output
      if (!outputPartition_) {
        break;
      }
    }

    const auto numOutputRowsLeft = outputBatchSize_ - offset;
    if (outputPartition_->rows.size() > numOutputRowsLeft) {
      // Output as many rows as possible.
      RANK_FUNCTION_DISPATCH(
          appendPartitionRows,
          rankFunction_,
          *outputPartition_,
          numOutputRowsLeft,
          offset,
          rowNumbers);
      offset += numOutputRowsLeft;
      break;
    }

    // Add all partition rows.
    const auto numPartitionRows = outputPartition_->rows.size();
    RANK_FUNCTION_DISPATCH(
        appendPartitionRows,
        rankFunction_,
        *outputPartition_,
        numPartitionRows,
        offset,
        rowNumbers);
    offset += numPartitionRows;
    outputPartition_ = nullptr;
  }

  if (offset == 0) {
    data_->clear();
    if (table_ != nullptr) {
      table_->clear(true);
    }
    pool()->release();
    return nullptr;
  }

  if (rowNumbers) {
    rowNumbers->resize(offset);
  }
  output->resize(offset);

  for (int i = 0; i < inputChannels_.size(); ++i) {
    data_->extractColumn(
        outputRows_.data(), offset, i, output->childAt(inputChannels_[i]));
  }

  return output;
}

bool TopNRowNumber::compareSpillRowColumns(
    const RowVectorPtr& output,
    vector_size_t index,
    const SpillMergeStream* next,
    vector_size_t startColumn,
    vector_size_t endColumn) {
  VELOX_CHECK_GT(index, 0);

  for (auto i = startColumn; i < endColumn; ++i) {
    if (!output->childAt(inputChannels_[i])
             ->equalValueAt(
                 next->current().childAt(i).get(),
                 index - 1,
                 next->currentIndex())) {
      return true;
    }
  }
  return false;
}

// Compares the partition keys for new partitions.
bool TopNRowNumber::isNewPartition(
    const RowVectorPtr& output,
    vector_size_t index,
    const SpillMergeStream* next) {
  return compareSpillRowColumns(output, index, next, 0, numPartitionKeys_);
}

// Compares the sorting keys for determining peers.
bool TopNRowNumber::isNewRank(
    const RowVectorPtr& output,
    vector_size_t index,
    const SpillMergeStream* next) {
  return compareSpillRowColumns(
      output,
      index,
      next,
      numPartitionKeys_,
      numPartitionKeys_ + numSortingKeys_);
}

template <core::TopNRowNumberNode::RankFunction TRank>
void TopNRowNumber::computeNextRankInSpill(
    const RowVectorPtr& output,
    vector_size_t index,
    const SpillMergeStream* next) {
  if (isNewPartition(output, index, next)) {
    nextRank_ = 1;
    numPeers_ = 1;
    return;
  }

  if constexpr (TRank == core::TopNRowNumberNode::RankFunction::kRowNumber) {
    nextRank_ += 1;
    return;
  }

  // The function is either rank or dense_rank.
  // This row belongs to the same partition as the previous row. However,
  // it should be determined if it is a peer row as well. If its a peer,
  // then increase numPeers_ but the rank remains unchanged.
  if (!isNewRank(output, index, next)) {
    numPeers_ += 1;
    return;
  }

  // The row is not a peer, so increment the rank and peers accordingly.
  if constexpr (TRank == core::TopNRowNumberNode::RankFunction::kDenseRank) {
    nextRank_ += 1;
    numPeers_ = 1;
    return;
  }

  // Rank function increments by number of peers.
  nextRank_ += numPeers_;
  numPeers_ = 1;
}

template <core::TopNRowNumberNode::RankFunction TRank>
void TopNRowNumber::setupNextOutput(const RowVectorPtr& output) {
  auto resetNextRankAndPeer = [this]() {
    nextRank_ = 1;
    numPeers_ = 1;
  };

  auto* lookAhead = merge_->next();
  if (lookAhead == nullptr) {
    resetNextRankAndPeer();
    return;
  }

  computeNextRankInSpill<TRank>(output, output->size(), lookAhead);
  if (nextRank_ <= limit_) {
    return;
  }

  // Skip remaining rows for this partition.
  lookAhead->pop();
  while (auto* next = merge_->next()) {
    if (isNewPartition(output, output->size(), next)) {
      resetNextRankAndPeer();
      return;
    }
    next->pop();
  }

  // This partition is the last partition.
  resetNextRankAndPeer();
}

template <core::TopNRowNumberNode::RankFunction TRank>
RowVectorPtr TopNRowNumber::getOutputFromSpill() {
  VELOX_CHECK_NOT_NULL(merge_);

  // merge_->next() produces data sorted by partition keys, then sorting keys.
  // All rows from the same partition will appear together.
  // We'll identify partition boundaries by comparing partition keys of the
  // current row with the previous row. When new partition starts, we'll reset
  // nextRank_ and numPeers_. Once rank reaches the 'limit_', we'll start
  // dropping rows until the next partition starts.
  // We'll emit output every time we accumulate 'outputBatchSize_' rows.
  auto output =
      BaseVector::create<RowVector>(outputType_, outputBatchSize_, pool());
  FlatVector<int64_t>* rankValues = nullptr;
  if (generateRowNumber_) {
    rankValues = output->children().back()->as<FlatVector<int64_t>>();
  }

  // Index of the next row to append to output.
  vector_size_t index = 0;
  VELOX_CHECK_LE(nextRank_, limit_);
  for (;;) {
    auto next = merge_->next();
    if (next == nullptr) {
      break;
    }

    if (index > 0) {
      computeNextRankInSpill<TRank>(output, index, next);
    }

    // Copy this row to the output buffer if this partition has
    // < limit_ rows output.
    if (nextRank_ <= limit_) {
      for (auto i = 0; i < inputChannels_.size(); ++i) {
        output->childAt(inputChannels_[i])
            ->copy(
                next->current().childAt(i).get(),
                index,
                next->currentIndex(),
                1);
      }

      if (rankValues) {
        rankValues->set(index, nextRank_);
      }
      ++index;
    }

    // Pop this row from the spill.
    next->pop();

    if (index == outputBatchSize_) {
      // This is the last row for this output batch.
      // Prepare the next batch :
      // i) If 'limit_' is reached for this partition, then skip the rows
      // until the next partition.
      // ii) If the next row is from a new partition, then reset nextRank_.
      setupNextOutput<TRank>(output);
      return output;
    }
  }

  // At this point, all rows are read from the spill merge stream.
  // (Note : The previous loop returns directly when the output buffer
  // is filled).
  if (index > 0) {
    output->resize(index);
  } else {
    output = nullptr;
  }

  finished_ = true;
  return output;
}

bool TopNRowNumber::isFinished() {
  return finished_;
}

void TopNRowNumber::close() {
  Operator::close();

  SCOPE_EXIT {
    table_.reset();
    singlePartition_.reset();
    data_.reset();
    allocator_.reset();
  };

  if (table_ == nullptr) {
    return;
  }

  partitionIt_.reset();
  partitions_.resize(1'000);
  while (auto numPartitions = table_->listAllRows(
             &partitionIt_,
             partitions_.size(),
             RowContainer::kUnlimited,
             partitions_.data())) {
    for (auto i = 0; i < numPartitions; ++i) {
      std::destroy_at(
          reinterpret_cast<TopRows*>(partitions_[i] + partitionOffset_));
    }
  }
}

void TopNRowNumber::reclaim(
    uint64_t /*targetBytes*/,
    memory::MemoryReclaimer::Stats& stats) {
  VELOX_CHECK(canReclaim());
  VELOX_CHECK(!nonReclaimableSection_);

  if (data_->numRows() == 0) {
    // Nothing to spill.
    return;
  }

  if (noMoreInput_) {
    ++stats.numNonReclaimableAttempts;
    // TODO Add support for spilling after noMoreInput().
    LOG(WARNING)
        << "Can't reclaim from topNRowNumber operator which has started producing output: "
        << pool()->name() << ", usage: " << succinctBytes(pool()->usedBytes())
        << ", reservation: " << succinctBytes(pool()->reservedBytes());
    return;
  }

  if (abandonedPartial_) {
    return;
  }

  spill();
}

void TopNRowNumber::ensureInputFits(const RowVectorPtr& input) {
  if (!spillEnabled()) {
    // Spilling is disabled.
    return;
  }

  if (data_->numRows() == 0) {
    // Nothing to spill.
    return;
  }

  // Test-only spill path.
  if (testingTriggerSpill(pool()->name())) {
    spill();
    return;
  }

  auto [freeRows, outOfLineFreeBytes] = data_->freeSpace();
  const auto outOfLineBytes =
      data_->stringAllocator().retainedSize() - outOfLineFreeBytes;
  const auto outOfLineBytesPerRow = outOfLineBytes / data_->numRows();

  const auto currentUsage = pool()->usedBytes();
  const auto minReservationBytes =
      currentUsage * spillConfig_->minSpillableReservationPct / 100;
  const auto availableReservationBytes = pool()->availableReservation();
  const auto tableIncrementBytes = table_->hashTableSizeIncrease(input->size());
  const auto incrementBytes =
      data_->sizeIncrement(
          input->size(), outOfLineBytesPerRow * input->size()) +
      tableIncrementBytes;

  // First to check if we have sufficient minimal memory reservation.
  if (availableReservationBytes >= minReservationBytes) {
    if ((tableIncrementBytes == 0) && (freeRows > input->size()) &&
        (outOfLineBytes == 0 ||
         outOfLineFreeBytes >= outOfLineBytesPerRow * input->size())) {
      // Enough free rows for input rows and enough variable length free
      // space.
      return;
    }
  }

  // Check if we can increase reservation. The increment is the largest of
  // twice the maximum increment from this input and
  // 'spillableReservationGrowthPct_' of the current memory usage.
  const auto targetIncrementBytes = std::max<int64_t>(
      incrementBytes * 2,
      currentUsage * spillConfig_->spillableReservationGrowthPct / 100);
  {
    ReclaimableSectionGuard guard(this);
    if (pool()->maybeReserve(targetIncrementBytes)) {
      return;
    }
  }

  LOG(WARNING) << "Failed to reserve " << succinctBytes(targetIncrementBytes)
               << " for memory pool " << pool()->name()
               << ", usage: " << succinctBytes(pool()->usedBytes())
               << ", reservation: " << succinctBytes(pool()->reservedBytes());
}

void TopNRowNumber::spill() {
  if (spiller_ == nullptr) {
    setupSpiller();
  }

  updateEstimatedOutputRowSize();

  spiller_->spill();
  table_->clear(true);
  data_->clear();
  pool()->release();
}

void TopNRowNumber::setupSpiller() {
  VELOX_CHECK_NULL(spiller_);
  VELOX_CHECK(spillConfig_.has_value());
  const auto sortingKeys = SpillState::makeSortingKeys(spillCompareFlags_);
  spiller_ = std::make_unique<SortInputSpiller>(
      data_.get(),
      inputType_,
      sortingKeys,
      &spillConfig_.value(),
      spillStats_.get());
}

// Using the underlying vector of the priority queue for the algorithms to
// check duplicates and count the number of top rank rows. This makes the
// algorithms O(n). There could be other approaches to make the
// algorithms O(1), but would trade memory efficiency.
namespace {
template <class T, class S, class C>
S& PriorityQueueVector(std::priority_queue<T, S, C>& q) {
  struct PrivateQueue : private std::priority_queue<T, S, C> {
    static S& Container(std::priority_queue<T, S, C>& q) {
      return q.*&PrivateQueue::c;
    }
  };
  return PrivateQueue::Container(q);
}
} // namespace

char* TopNRowNumber::TopRows::removeTopRankRows() {
  VELOX_CHECK(!rows.empty());

  char* topRow = rows.top();
  rows.pop();

  while (!rows.empty()) {
    char* newTopRow = rows.top();
    if (rowComparator.compare(topRow, newTopRow) != 0) {
      return topRow;
    }
    rows.pop();
  }
  return topRow;
}

vector_size_t TopNRowNumber::TopRows::numTopRankRows() {
  VELOX_CHECK(!rows.empty());

  tempTopRankRows.clear();
  SCOPE_EXIT {
    tempTopRankRows.clear();
  };
  auto popAndSaveTopRow = [&]() {
    tempTopRankRows.push_back(rows.top());
    rows.pop();
  };

  char* topRow = rows.top();
  popAndSaveTopRow();
  while (!rows.empty()) {
    if (rowComparator.compare(topRow, rows.top()) == 0) {
      popAndSaveTopRow();
    } else {
      break;
    }
  }

  vector_size_t numTopRows = tempTopRankRows.size();
  // Re-insert all rows with the top rank row.
  for (char* row : tempTopRankRows) {
    rows.push(row);
  }
  return numTopRows;
}

bool TopNRowNumber::TopRows::isDuplicate(
    const std::vector<DecodedVector>& decodedVectors,
    vector_size_t index) {
  const std::vector<char*, StlAllocator<char*>> partitionRowsVector =
      PriorityQueueVector(rows);
  for (const char* row : partitionRowsVector) {
    if (rowComparator.compare(decodedVectors, index, row) == 0) {
      return true;
    }
  }
  return false;
}

} // namespace facebook::velox::exec
