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

#include "velox/exec/JoinTableBuilder.h"

#include <algorithm>

#include "velox/exec/HashJoinBridge.h"
#include "velox/exec/OperatorUtils.h"

namespace facebook::velox::exec {

JoinTableBuilder::JoinTableBuilder(core::JoinType joinType, Options options)
    : joinType_(joinType),
      options_(std::move(options)),
      dropDuplicates_(core::canDropDuplicates(joinType_, options_.withFilter)),
      keyChannelMap_(options_.keyChannels.size()) {
  VELOX_CHECK_NOT_NULL(options_.inputType);
  const auto& inputType = options_.inputType;

  const auto& keyChannels = options_.keyChannels;
  const auto numKeys = keyChannels.size();
  for (auto i = 0; i < numKeys; ++i) {
    VELOX_CHECK_LT(keyChannels[i], inputType->size());
    keyChannelMap_[keyChannels[i]] = i;
  }

  const auto probedFlagChannel = options_.probedFlagChannel;
  if (probedFlagChannel.has_value()) {
    VELOX_CHECK_LT(probedFlagChannel.value(), inputType->size());
    VELOX_CHECK(
        inputType->childAt(probedFlagChannel.value())->isBoolean(),
        "The probed flag column must be boolean: {}",
        inputType->childAt(probedFlagChannel.value())->toString());
    VELOX_CHECK(
        !keyChannelMap_.contains(probedFlagChannel.value()),
        "The probed flag column can not be a join key: {}",
        probedFlagChannel.value());
  }

  // Identify the non-key build side columns and make a decoder for each.
  if (!dropDuplicates_) {
    // The number of join keys (numKeys) may be greater than the number of
    // input columns (inputType->size()), which makes 'numDependents' negative
    // and unusable for 'reserve'. This happens when we join different probe
    // side keys with the same build side key: SELECT * FROM t LEFT JOIN u ON
    // t.k1 = u.k AND t.k2 = u.k.
    const int32_t numDependents = inputType->size() - numKeys;
    if (numDependents > 0) {
      dependentChannels_.reserve(numDependents);
      decoders_.reserve(numDependents);
    }
    for (auto i = 0; i < inputType->size(); ++i) {
      if (keyChannelMap_.find(i) == keyChannelMap_.end() &&
          i != probedFlagChannel) {
        dependentChannels_.emplace_back(i);
        decoders_.emplace_back(std::make_unique<DecodedVector>());
      }
    }
  }

  // The same layout as 'hashJoinTableType()', less the probed flag column.
  tableInputChannels_ = keyChannels;
  tableInputChannels_.insert(
      tableInputChannels_.end(),
      dependentChannels_.begin(),
      dependentChannels_.end());
  std::vector<std::string> names;
  std::vector<TypePtr> types;
  names.reserve(tableInputChannels_.size());
  types.reserve(tableInputChannels_.size());
  for (const auto channel : tableInputChannels_) {
    names.emplace_back(inputType->nameOf(channel));
    types.emplace_back(inputType->childAt(channel));
  }
  tableType_ = ROW(std::move(names), std::move(types));
}

void JoinTableBuilder::initialize(
    memory::MemoryPool* tablePool,
    memory::MemoryPool* auxiliaryPool,
    const AntiJoinFilterInfo& filterInfo) {
  VELOX_CHECK_NOT_NULL(tablePool);
  VELOX_CHECK_NOT_NULL(auxiliaryPool);
  VELOX_CHECK_NULL(tablePool_, "JoinTableBuilder is already initialized");

  tablePool_ = tablePool;
  auxiliaryPool_ = auxiliaryPool;
  filterPropagatesNulls_ = filterInfo.propagatesNulls;

  setupTable();

  if (isAntiJoin(joinType_) && options_.withFilter && filterPropagatesNulls_) {
    setupFilterChannels(filterInfo);
  }
}

void JoinTableBuilder::setupTable() {
  VELOX_CHECK_NOT_NULL(tablePool_, "JoinTableBuilder is not initialized");
  VELOX_CHECK_NULL(table_);

  const auto& keyChannels = options_.keyChannels;
  const auto numKeys = keyChannels.size();
  std::vector<std::unique_ptr<VectorHasher>> keyHashers;
  keyHashers.reserve(numKeys);
  for (auto i = 0; i < numKeys; ++i) {
    keyHashers.emplace_back(
        VectorHasher::create(tableType_->childAt(i), keyChannels[i]));
  }

  const auto numDependents = tableType_->size() - numKeys;
  std::vector<TypePtr> dependentTypes;
  dependentTypes.reserve(numDependents);
  for (auto i = numKeys; i < tableType_->size(); ++i) {
    dependentTypes.emplace_back(tableType_->childAt(i));
  }

  if (isRightJoin(joinType_) || isFullJoin(joinType_) ||
      isRightSemiProjectJoin(joinType_) || isRightAntiJoin(joinType_)) {
    // Do not ignore null keys. kRightAnti must retain null keys: a null-keyed
    // build row never matches and is always returned.
    table_ = HashTable<false>::createForJoin(
        std::move(keyHashers),
        dependentTypes,
        true, // allowDuplicates
        true, // hasProbedFlag
        false, // hasCountFlag
        options_.minTableRowsForParallelJoinBuild,
        tablePool_);
  } else {
    // Right semi join needs to tag build rows that were probed.
    const bool needProbedFlag = isRightSemiFilterJoin(joinType_);
    const bool hasCountFlag = core::isCountingJoin(joinType_);
    if (options_.nullAsValue ||
        isLeftNullAwareJoinWithFilter(
            joinType_, options_.nullAware, options_.withFilter)) {
      // We need to check null key rows in build side in case of null-aware anti
      // or left semi project join with filter set.
      table_ = HashTable<false>::createForJoin(
          std::move(keyHashers),
          dependentTypes,
          !dropDuplicates_, // allowDuplicates
          needProbedFlag, // hasProbedFlag
          hasCountFlag,
          options_.minTableRowsForParallelJoinBuild,
          tablePool_);
    } else {
      // Ignore null keys
      table_ = HashTable<true>::createForJoin(
          std::move(keyHashers),
          dependentTypes,
          !dropDuplicates_, // allowDuplicates
          needProbedFlag, // hasProbedFlag
          hasCountFlag,
          options_.minTableRowsForParallelJoinBuild,
          tablePool_,
          options_.bloomFilterPushdownMaxSize);
    }
  }
  analyzeKeys_ = table_->hashMode() != BaseHashTable::HashMode::kHash;

  if (options_.abandonHashBuildDedupMinPct == 0 &&
      !core::isCountingJoin(joinType_)) {
    // Building a HashTable without duplicates is disabled if
    // abandonHashBuildDedupMinPct is 0. Counting joins always require dedup.
    abandonHashBuildDedup_ = true;
    table_->setAllowDuplicates(true);
    return;
  }
  // Only create HashLookup when dedup is enabled.
  lookup_ = std::make_unique<HashLookup>(table_->hashers(), auxiliaryPool_);
}

std::unique_ptr<BaseHashTable> JoinTableBuilder::takeTable() {
  lookup_.reset();
  return std::move(table_);
}

void JoinTableBuilder::setupFilterChannels(
    const AntiJoinFilterInfo& filterInfo) {
  VELOX_DCHECK(
      std::is_sorted(dependentChannels_.begin(), dependentChannels_.end()));
  VELOX_DCHECK(
      std::is_sorted(
          filterInfo.inputChannels.begin(), filterInfo.inputChannels.end()));

  for (const auto channel : filterInfo.inputChannels) {
    const auto keyIter = keyChannelMap_.find(channel);
    if (keyIter != keyChannelMap_.end()) {
      keyFilterChannels_.push_back(keyIter->second);
      continue;
    }
    const auto dependentIter = std::lower_bound(
        dependentChannels_.begin(), dependentChannels_.end(), channel);
    if (dependentIter == dependentChannels_.end() ||
        *dependentIter != channel) {
      // Not a build side column, e.g. a probe side column referenced by the
      // filter.
      continue;
    }
    dependentFilterChannels_.push_back(
        dependentIter - dependentChannels_.begin());
  }
}

void JoinTableBuilder::removeInputRowsForAntiJoinFilter() {
  bool changed = false;
  auto* rawActiveRows = activeRows_.asMutableRange().bits();
  auto removeNulls = [&](DecodedVector& decoded) {
    if (decoded.mayHaveNulls()) {
      changed = true;
      // NOTE: the true value of a raw null bit indicates non-null so we AND
      // 'rawActiveRows' with the raw bit.
      bits::andBits(
          rawActiveRows, decoded.nulls(&activeRows_), 0, activeRows_.end());
    }
  };
  for (const auto channel : keyFilterChannels_) {
    removeNulls(table_->hashers()[channel]->decodedVector());
  }
  for (const auto channel : dependentFilterChannels_) {
    removeNulls(*decoders_[channel]);
  }
  if (changed) {
    activeRows_.updateBounds();
  }
}

bool JoinTableBuilder::abandonHashBuildDedupEarly(int64_t numDistinct) const {
  VELOX_CHECK(dropDuplicates_);
  return numHashInputRows_ > options_.abandonHashBuildDedupMinRows &&
      100 * numDistinct / numHashInputRows_ >=
      options_.abandonHashBuildDedupMinPct;
}

void JoinTableBuilder::abandonHashBuildDedup() {
  // The hash table is no longer directly constructed in addInput. The data
  // that was previously inserted into the hash table is already in the
  // RowContainer.
  if (options_.onDedupAbandoned != nullptr) {
    options_.onDedupAbandoned();
  }
  abandonHashBuildDedup_ = true;
  table_->setAllowDuplicates(true);
  lookup_.reset();
}

bool JoinTableBuilder::addInput(const RowVectorPtr& input) {
  VELOX_CHECK_NOT_NULL(table_, "JoinTableBuilder is not initialized");

  decodeKeys(input);
  if (!processNullKeys()) {
    return false;
  }
  decodeDependents(input);

  if (options_.beforeInsertRows != nullptr && activeRows_.hasSelections()) {
    options_.beforeInsertRows(input, activeRows_);
    activeRows_.updateBounds();
  }

  insertRows(input);
  return true;
}

void JoinTableBuilder::decodeKeys(const RowVectorPtr& input) {
  activeRows_.resize(input->size());
  activeRows_.setAll();

  auto& hashers = table_->hashers();
  for (auto i = 0; i < hashers.size(); ++i) {
    auto* key = input->childAt(hashers[i]->channel())->loadedVector();
    hashers[i]->decode(*key, activeRows_);
  }
}

bool JoinTableBuilder::processNullKeys() {
  const auto& hashers = table_->hashers();
  const auto numInput = activeRows_.size();

  vector_size_t numNullKeyRows{0};
  if (!isRightJoin(joinType_) && !isFullJoin(joinType_) &&
      !isRightSemiProjectJoin(joinType_) && !isRightAntiJoin(joinType_) &&
      !options_.nullAsValue &&
      !isLeftNullAwareJoinWithFilter(
          joinType_, options_.nullAware, options_.withFilter)) {
    deselectRowsWithNulls(hashers, activeRows_);
    numNullKeyRows = numInput - activeRows_.countSelected();
  } else {
    // The join retains the rows with a null key, so count them on a copy.
    nonNullKeyRows_ = activeRows_;
    deselectRowsWithNulls(hashers, nonNullKeyRows_);
    numNullKeyRows = numInput - nonNullKeyRows_.countSelected();
  }
  numNullKeyRows_ += numNullKeyRows;
  if (options_.nullAware && numNullKeyRows > 0) {
    joinHasNullKeys_ = true;
  }

  // Null-aware anti join with no extra filter returns no rows if build side
  // has nulls in join keys. Hence, we can stop processing on first null.
  return !(
      isAntiJoin(joinType_) && options_.nullAware && joinHasNullKeys_ &&
      !options_.withFilter);
}

void JoinTableBuilder::decodeDependents(const RowVectorPtr& input) {
  for (auto i = 0; i < dependentChannels_.size(); ++i) {
    decoders_[i]->decode(
        *input->childAt(dependentChannels_[i])->loadedVector(), activeRows_);
  }

  if (isAntiJoin(joinType_) && options_.withFilter && filterPropagatesNulls_) {
    removeInputRowsForAntiJoinFilter();
  }
}

void JoinTableBuilder::insertRows(const RowVectorPtr& input) {
  if (!activeRows_.hasSelections()) {
    return;
  }

  if (dropDuplicates_ && !abandonHashBuildDedup_) {
    // Counting joins must not abandon dedup - accurate counts are required.
    VELOX_CHECK_NOT_NULL(lookup_);
    const bool abandonEarly = !core::isCountingJoin(joinType_) &&
        abandonHashBuildDedupEarly(table_->numDistinct());
    if (!abandonEarly) {
      numHashInputRows_ += activeRows_.countSelected();
      table_->prepareForGroupProbe(
          *lookup_,
          input,
          activeRows_,
          BaseHashTable::kNoSpillInputStartPartitionBit);
      if (lookup_->rows.empty()) {
        return;
      }
      table_->groupProbe(
          *lookup_, BaseHashTable::kNoSpillInputStartPartitionBit);

      // For counting joins, increment the count for duplicate rows.
      // New rows are initialized with count = 1 by initializeRow.
      // Increment count for all rows, then decrement for new rows to
      // correct the over-counting.
      if (core::isCountingJoin(joinType_)) {
        auto* rows = table_->rows();
        for (const auto row : lookup_->rows) {
          rows->incrementCount(lookup_->hits[row]);
        }
        for (const auto newRow : lookup_->newGroups) {
          rows->decrementCount(lookup_->hits[newRow]);
        }
      }
      return;
    }
    abandonHashBuildDedup();
  }

  if (analyzeKeys_ && hashes_.size() < activeRows_.end()) {
    hashes_.resize(activeRows_.end());
  }

  // As long as analyzeKeys is true, we keep running the keys through
  // the Vectorhashers so that we get a possible mapping of the keys
  // to small ints for array or normalized key. When mayUseValueIds is
  // false for the first time we stop. We do not retain the value ids
  // since the final ones will only be known after all data is
  // received.
  auto& hashers = table_->hashers();
  for (auto& hasher : hashers) {
    // TODO: Load only for active rows, except if right/full outer join.
    if (analyzeKeys_) {
      hasher->computeValueIds(activeRows_, hashes_);
      analyzeKeys_ = hasher->mayUseValueIds();
    }
  }

  const FlatVector<bool>* probedFlags{nullptr};
  if (options_.probedFlagChannel.has_value()) {
    probedFlags = input->childAt(options_.probedFlagChannel.value())
                      ->asFlatVector<bool>();
    VELOX_CHECK_NOT_NULL(probedFlags, "The probed flag column must be flat");
  }

  auto* rows = table_->rows();
  const auto nextOffset = rows->nextOffset();
  activeRows_.applyToSelected([&](auto rowIndex) {
    char* newRow = rows->newRow();
    if (nextOffset) {
      *reinterpret_cast<char**>(newRow + nextOffset) = nullptr;
    }
    // Store the columns for each row in sequence. At probe time
    // strings of the row will probably be in consecutive places, so
    // reading one will prime the cache for the next.
    for (auto i = 0; i < hashers.size(); ++i) {
      rows->store(hashers[i]->decodedVector(), rowIndex, newRow, i);
    }
    for (auto i = 0; i < dependentChannels_.size(); ++i) {
      rows->store(*decoders_[i], rowIndex, newRow, i + hashers.size());
    }
    if (probedFlags != nullptr) {
      VELOX_CHECK(!probedFlags->isNullAt(rowIndex));
      if (probedFlags->valueAt(rowIndex)) {
        rows->setProbedFlag(&newRow, 1);
      }
    }
  });
}

} // namespace facebook::velox::exec
