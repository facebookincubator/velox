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

#include "velox/exec/HashTableBuilder.h"
#include "velox/exec/OperatorUtils.h"

namespace facebook::velox::exec {
namespace {
RowTypePtr hashJoinTableType(
    const std::vector<facebook::velox::core::FieldAccessTypedExprPtr>& joinKeys,
    const RowTypePtr& inputType) {
  const auto numKeys = joinKeys.size();

  std::vector<std::string> names;
  names.reserve(inputType->size());
  std::vector<TypePtr> types;
  types.reserve(inputType->size());
  std::unordered_set<uint32_t> keyChannelSet;
  keyChannelSet.reserve(inputType->size());

  for (int i = 0; i < numKeys; ++i) {
    auto& key = joinKeys[i];
    auto channel = exprToChannel(key.get(), inputType);
    keyChannelSet.insert(channel);
    names.emplace_back(inputType->nameOf(channel));
    types.emplace_back(inputType->childAt(channel));
  }

  for (auto i = 0; i < inputType->size(); ++i) {
    if (keyChannelSet.find(i) == keyChannelSet.end()) {
      names.emplace_back(inputType->nameOf(i));
      types.emplace_back(inputType->childAt(i));
    }
  }

  return ROW(std::move(names), std::move(types));
}

bool isLeftNullAwareJoinWithFilter(
    core::JoinType joinType,
    bool nullAware,
    bool withFilter) {
  return (isAntiJoin(joinType) || isLeftSemiProjectJoin(joinType) ||
          isLeftSemiFilterJoin(joinType)) &&
      nullAware && withFilter;
}
} // namespace

HashTableBuilder::HashTableBuilder(
    core::JoinType joinType,
    bool nullAware,
    bool withFilter,
    const std::vector<facebook::velox::core::FieldAccessTypedExprPtr>& joinKeys,
    const RowTypePtr& inputType,
    memory::MemoryPool* pool)
    : joinType_{joinType},
      nullAware_{nullAware},
      withFilter_(withFilter),
      keyChannelMap_(joinKeys.size()),
      inputType_(inputType),
      pool_(pool) {
  const auto numKeys = joinKeys.size();
  keyChannels_.reserve(numKeys);

  for (int i = 0; i < numKeys; ++i) {
    auto& key = joinKeys[i];
    auto channel = exprToChannel(key.get(), inputType_);
    keyChannelMap_[channel] = i;
    keyChannels_.emplace_back(channel);
  }

  // Identify the non-key build side columns and make a decoder for each.
  const int32_t numDependents = inputType_->size() - numKeys;
  if (numDependents > 0) {
    // Number of join keys (numKeys) may be less then number of input columns
    // (inputType->size()). In this case numDependents is negative and cannot be
    // used to call 'reserve'. This happens when we join different probe side
    // keys with the same build side key: SELECT * FROM t LEFT JOIN u ON t.k1 =
    // u.k AND t.k2 = u.k.
    dependentChannels_.reserve(numDependents);
    decoders_.reserve(numDependents);
  }
  for (auto i = 0; i < inputType->size(); ++i) {
    if (keyChannelMap_.find(i) == keyChannelMap_.end()) {
      dependentChannels_.emplace_back(i);
      decoders_.emplace_back(std::make_unique<DecodedVector>());
    }
  }

  tableType_ = hashJoinTableType(joinKeys, inputType);
  setupTable();
}

// Invoked to set up hash table to build.
void HashTableBuilder::setupTable() {
  VELOX_CHECK_NULL(table_);

  const auto numKeys = keyChannels_.size();
  std::vector<std::unique_ptr<VectorHasher>> keyHashers;
  keyHashers.reserve(numKeys);
  for (vector_size_t i = 0; i < numKeys; ++i) {
    keyHashers.emplace_back(
        VectorHasher::create(tableType_->childAt(i), keyChannels_[i]));
  }

  const auto numDependents = tableType_->size() - numKeys;
  std::vector<TypePtr> dependentTypes;
  dependentTypes.reserve(numDependents);
  for (int i = numKeys; i < tableType_->size(); ++i) {
    dependentTypes.emplace_back(tableType_->childAt(i));
  }
  if (isRightJoin(joinType_) || isFullJoin(joinType_) ||
      isRightSemiProjectJoin(joinType_)) {
    // Do not ignore null keys.
    table_ = HashTable<false>::createForJoin(
        std::move(keyHashers),
        dependentTypes,
        true, // allowDuplicates
        true, // hasProbedFlag
        1'000, // operatorCtx_->driverCtx()->queryConfig().minTableRowsForParallelJoinBuild()
        pool_,
        true);
  } else {
    // (Left) semi and anti join with no extra filter only needs to know whether
    // there is a match. Hence, no need to store entries with duplicate keys.
    const bool dropDuplicates = !withFilter_ &&
        (isLeftSemiFilterJoin(joinType_) || isLeftSemiProjectJoin(joinType_) ||
         isAntiJoin(joinType_));
    // Right semi join needs to tag build rows that were probed.
    const bool needProbedFlag = isRightSemiFilterJoin(joinType_);
    if (isLeftNullAwareJoinWithFilter(joinType_, nullAware_, withFilter_)) {
      // We need to check null key rows in build side in case of null-aware anti
      // or left semi project join with filter set.
      table_ = HashTable<false>::createForJoin(
          std::move(keyHashers),
          dependentTypes,
          !dropDuplicates, // allowDuplicates
          needProbedFlag, // hasProbedFlag
          1'000, // operatorCtx_->driverCtx()->queryConfig().minTableRowsForParallelJoinBuild()
          pool_,
          true);
    } else {
      // Ignore null keys
      table_ = HashTable<true>::createForJoin(
          std::move(keyHashers),
          dependentTypes,
          !dropDuplicates, // allowDuplicates
          needProbedFlag, // hasProbedFlag
          1'000, // operatorCtx_->driverCtx()->queryConfig().minTableRowsForParallelJoinBuild()
          pool_,
          true);
    }
  }
  analyzeKeys_ = table_->hashMode() != BaseHashTable::HashMode::kHash;
}

void HashTableBuilder::addInput(RowVectorPtr input) {
  activeRows_.resize(input->size());
  activeRows_.setAll();

  auto& hashers = table_->hashers();

  for (auto i = 0; i < hashers.size(); ++i) {
    auto key = input->childAt(hashers[i]->channel())->loadedVector();
    hashers[i]->decode(*key, activeRows_);
  }

  deselectRowsWithNulls(hashers, activeRows_);
  activeRows_.setAll();

  if (!isRightJoin(joinType_) && !isFullJoin(joinType_) &&
      !isRightSemiProjectJoin(joinType_) &&
      !isLeftNullAwareJoinWithFilter(joinType_, nullAware_, withFilter_)) {
    deselectRowsWithNulls(hashers, activeRows_);
    if (nullAware_ && !joinHasNullKeys_ &&
        activeRows_.countSelected() < input->size()) {
      joinHasNullKeys_ = true;
    }
  } else if (nullAware_ && !joinHasNullKeys_) {
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

  for (auto i = 0; i < dependentChannels_.size(); ++i) {
    decoders_[i]->decode(
        *input->childAt(dependentChannels_[i])->loadedVector(), activeRows_);
  }

  if (!activeRows_.hasSelections()) {
    return;
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
  for (auto& hasher : hashers) {
    // TODO: Load only for active rows, except if right/full outer join.
    if (analyzeKeys_) {
      hasher->computeValueIds(activeRows_, hashes_);
      analyzeKeys_ = hasher->mayUseValueIds();
    }
  }
  auto rows = table_->rows();

  activeRows_.applyToSelected([&](auto rowIndex) {
    char* newRow = rows->newRow();
    // Store the columns for each row in sequence. At probe time
    // strings of the row will probably be in consecutive places, so
    // reading one will prime the cache for the next.
    for (auto i = 0; i < hashers.size(); ++i) {
      rows->store(hashers[i]->decodedVector(), rowIndex, newRow, i);
    }
    for (auto i = 0; i < dependentChannels_.size(); ++i) {
      rows->store(*decoders_[i], rowIndex, newRow, i + hashers.size());
    }
  });
}

} // namespace facebook::velox::exec
