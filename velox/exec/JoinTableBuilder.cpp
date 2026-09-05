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
#include <numeric>

#include "velox/exec/HashJoinBridge.h"
#include "velox/exec/Operator.h"
#include "velox/exec/OperatorUtils.h"

namespace facebook::velox::exec {

JoinTableBuilder::JoinTableBuilder(Options options)
    : options_(std::move(options)),
      dropDuplicates_(
          core::canDropDuplicates(options_.joinType, options_.withFilter)),
      keyChannelMap_(options_.joinKeys.size()) {
  VELOX_CHECK_NOT_NULL(options_.inputType);

  const auto numKeys = options_.joinKeys.size();
  keyChannels_.reserve(numKeys);
  for (auto i = 0; i < numKeys; ++i) {
    const auto channel =
        exprToChannel(options_.joinKeys[i].get(), options_.inputType);
    keyChannelMap_[channel] = i;
    keyChannels_.emplace_back(channel);
  }

  // Identify the non-key build side columns and make a decoder for each.
  if (!dropDuplicates_) {
    const int32_t numDependents = options_.inputType->size() - numKeys;
    if (numDependents > 0) {
      // Number of join keys (numKeys) may be less then number of input columns
      // (inputType->size()). In this case numDependents is negative and cannot
      // be used to call 'reserve'. This happens when we join different probe
      // side keys with the same build side key: SELECT * FROM t LEFT JOIN u ON
      // t.k1 = u.k AND t.k2 = u.k.
      dependentChannels_.reserve(numDependents);
      decoders_.reserve(numDependents);
    }
    for (auto i = 0; i < options_.inputType->size(); ++i) {
      if (keyChannelMap_.find(i) == keyChannelMap_.end()) {
        dependentChannels_.emplace_back(i);
        decoders_.emplace_back(std::make_unique<DecodedVector>());
      }
    }
  }

  tableType_ =
      hashJoinTableType(options_.joinKeys, options_.inputType, dropDuplicates_);
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

  if (isAntiJoin(options_.joinType) && options_.withFilter &&
      filterPropagatesNulls_) {
    setupFilterChannels(filterInfo);
  }
}

void JoinTableBuilder::setupTable() {
  VELOX_CHECK_NOT_NULL(tablePool_, "JoinTableBuilder is not initialized");
  VELOX_CHECK_NULL(table_);

  const auto numKeys = keyChannels_.size();
  std::vector<std::unique_ptr<VectorHasher>> keyHashers;
  keyHashers.reserve(numKeys);
  for (auto i = 0; i < numKeys; ++i) {
    keyHashers.emplace_back(
        VectorHasher::create(tableType_->childAt(i), keyChannels_[i]));
  }

  const auto numDependents = tableType_->size() - numKeys;
  std::vector<TypePtr> dependentTypes;
  dependentTypes.reserve(numDependents);
  for (auto i = numKeys; i < tableType_->size(); ++i) {
    dependentTypes.emplace_back(tableType_->childAt(i));
  }

  const auto joinType = options_.joinType;
  if (isRightJoin(joinType) || isFullJoin(joinType) ||
      isRightSemiProjectJoin(joinType) || isRightAntiJoin(joinType)) {
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
    const bool needProbedFlag = isRightSemiFilterJoin(joinType);
    const bool hasCountFlag = core::isCountingJoin(joinType);
    if (options_.nullAsValue ||
        isLeftNullAwareJoinWithFilter(
            joinType, options_.nullAware, options_.withFilter)) {
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
      !core::isCountingJoin(joinType)) {
    // Building a HashTable without duplicates is disabled if
    // abandonHashBuildDedupMinPct is 0. Counting joins always require dedup.
    abandonHashBuildDedup_ = true;
    table_->setAllowDuplicates(true);
    return;
  }
  // Only create HashLookup when dedup is enabled.
  lookup_ = std::make_unique<HashLookup>(table_->hashers(), auxiliaryPool_);
}

void JoinTableBuilder::resetForSpillInput() {
  table_.reset();
  lookup_.reset();

  // Reset the key and dependent channels as the spilled data columns have
  // already been ordered.
  std::iota(keyChannels_.begin(), keyChannels_.end(), 0);
  std::iota(
      dependentChannels_.begin(),
      dependentChannels_.end(),
      keyChannels_.size());

  setupTable();
  numHashInputRows_ = 0;
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
  decodeKeys(input);
  if (!processNullKeys()) {
    return false;
  }
  decodeDependents(input);
  insertRows(input);
  return true;
}

void JoinTableBuilder::decodeKeys(const RowVectorPtr& input) {
  VELOX_CHECK_NOT_NULL(table_, "JoinTableBuilder is not initialized");

  activeRows_.resize(input->size());
  activeRows_.setAll();

  auto& hashers = table_->hashers();
  for (auto i = 0; i < hashers.size(); ++i) {
    auto* key = input->childAt(hashers[i]->channel())->loadedVector();
    hashers[i]->decode(*key, activeRows_);
  }
}

bool JoinTableBuilder::processNullKeys() {
  const auto joinType = options_.joinType;
  auto& hashers = table_->hashers();

  if (!isRightJoin(joinType) && !isFullJoin(joinType) &&
      !isRightSemiProjectJoin(joinType) && !isRightAntiJoin(joinType) &&
      !options_.nullAsValue &&
      !isLeftNullAwareJoinWithFilter(
          joinType, options_.nullAware, options_.withFilter)) {
    const auto numInput = activeRows_.size();
    deselectRowsWithNulls(hashers, activeRows_);
    if (options_.nullAware && !joinHasNullKeys_ &&
        activeRows_.countSelected() < numInput) {
      joinHasNullKeys_ = true;
    }
  } else if (options_.nullAware && !joinHasNullKeys_) {
    for (auto& hasher : hashers) {
      auto& decoded = hasher->decodedVector();
      if (decoded.mayHaveNulls()) {
        auto* nulls = decoded.nulls(&activeRows_);
        if (nulls && bits::countNulls(nulls, 0, activeRows_.end()) > 0) {
          joinHasNullKeys_ = true;
          break;
        }
      }
    }
  }

  // Null-aware anti join with no extra filter returns no rows if build side
  // has nulls in join keys. Hence, we can stop processing on first null.
  return !(
      isAntiJoin(joinType) && options_.nullAware && joinHasNullKeys_ &&
      !options_.withFilter);
}

void JoinTableBuilder::decodeDependents(const RowVectorPtr& input) {
  for (auto i = 0; i < dependentChannels_.size(); ++i) {
    decoders_[i]->decode(
        *input->childAt(dependentChannels_[i])->loadedVector(), activeRows_);
  }

  if (isAntiJoin(options_.joinType) && options_.withFilter &&
      filterPropagatesNulls_) {
    removeInputRowsForAntiJoinFilter();
  }
}

void JoinTableBuilder::insertRows(
    const RowVectorPtr& input,
    const FlatVector<bool>* spillProbedFlags) {
  if (!activeRows_.hasSelections()) {
    return;
  }

  if (dropDuplicates_ && !abandonHashBuildDedup_) {
    // Counting joins must not abandon dedup - accurate counts are required.
    VELOX_CHECK_NOT_NULL(lookup_);
    const bool abandonEarly = !core::isCountingJoin(options_.joinType) &&
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
      if (core::isCountingJoin(options_.joinType)) {
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
    if (spillProbedFlags != nullptr) {
      VELOX_CHECK(!spillProbedFlags->isNullAt(rowIndex));
      if (spillProbedFlags->valueAt(rowIndex)) {
        rows->setProbedFlag(&newRow, 1);
      }
    }
  });
}

} // namespace facebook::velox::exec
