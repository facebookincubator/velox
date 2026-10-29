/*
 * Copyright (c) Meta Platforms, Inc. and its affiliates.
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

#include "velox/exec/VectorSearchJoinBuild.h"

#include "velox/exec/OperatorType.h"
#include "velox/exec/Task.h"
#include "velox/exec/VectorSearchJoinBridge.h"

namespace facebook::velox::exec {

namespace {

// Hashes the keys of 'row'. Equal keys give equal hashes whatever their
// encoding.
uint64_t hashKeys(
    const std::vector<const BaseVector*>& keys,
    vector_size_t row) {
  uint64_t hash = 0;
  for (const auto* key : keys) {
    hash = bits::hashMix(hash, key->hashValueAt(row));
  }
  return hash;
}

} // namespace

VectorSearchJoinTable::VectorSearchJoinTable(
    RowVectorPtr rows,
    column_index_t numKeys,
    std::vector<Index> indexes,
    std::vector<VectorPtr> packedColumns)
    : rows_{std::move(rows)},
      numKeys_{numKeys},
      packedColumns_{std::move(packedColumns)},
      indexes_{std::move(indexes)} {
  VELOX_CHECK_EQ(rows_->size(), indexes_.size());
  keys_.reserve(numKeys_);
  for (column_index_t i = 0; i < numKeys_; ++i) {
    keys_.push_back(rows_->childAt(i).get());
  }
  for (vector_size_t row = 0; row < rows_->size(); ++row) {
    rowsByHash_[hashKeys(keys_, row)].push_back(row);
  }
  if (!indexes_.empty()) {
    dimension_ = indexes_[0]->dimension();
  }
  for (const auto& index : indexes_) {
    VELOX_USER_CHECK_EQ(
        index->dimension(),
        dimension_,
        "Vector search join indexes must have the same dimension");
  }
}

void VectorSearchJoinTable::forEachMatch(
    const std::vector<const BaseVector*>& probeKeys,
    vector_size_t row,
    folly::FunctionRef<void(vector_size_t buildRow)> onMatch) const {
  const auto it = rowsByHash_.find(hashKeys(probeKeys, row));
  if (it == rowsByHash_.end()) {
    return;
  }
  for (const auto buildRow : it->second) {
    bool equal = true;
    for (column_index_t i = 0; i < numKeys_ && equal; ++i) {
      equal = probeKeys[i]->equalValueAt(keys_[i], row, buildRow);
    }
    if (equal) {
      onMatch(buildRow);
    }
  }
}

int64_t VectorSearchJoinTable::numResults(vector_size_t row, int64_t k) const {
  return std::min(k, indexes_[row]->size());
}

void VectorSearchJoinTable::search(
    vector_size_t row,
    const common::vectorindex::Embeddings& queries,
    int64_t numResults,
    std::span<int64_t> ids,
    std::span<float> scores) const {
  indexes_[row]->search(queries, numResults, ids, scores);
}

VectorSearchJoinBuild::VectorSearchJoinBuild(
    int32_t operatorId,
    DriverCtx* driverCtx,
    const core::VectorSearchJoinNodePtr& joinNode)
    : Operator(
          driverCtx,
          nullptr,
          operatorId,
          joinNode->id(),
          OperatorType::kVectorSearchJoinBuild),
      metric_{joinNode->metric()},
      searchOptions_{common::vectorindex::LocalVectorIndexOptions::parseSearch(
          {joinNode->searchOptions().begin(),
           joinNode->searchOptions().end()})} {
  const auto& buildType = joinNode->sources()[1]->outputType();
  const auto& outputType = joinNode->outputType();
  channels_.reserve(joinNode->rightKeys().size() + outputType->size());
  for (const auto& key : joinNode->rightKeys()) {
    channels_.push_back(buildType->getChildIdx(key->name()));
  }
  numKeys_ = channels_.size();
  for (const auto& name : outputType->names()) {
    if (const auto channel = buildType->getChildIdxIfExists(name)) {
      channels_.push_back(*channel);
    }
  }
  std::vector<TypePtr> types;
  types.reserve(channels_.size());
  for (const auto channel : channels_) {
    types.push_back(buildType->childAt(channel));
  }
  rowType_ = ROW(std::move(types));
  indexChannel_ = buildType->getChildIdx(joinNode->buildIndex()->name());
}

void VectorSearchJoinBuild::addInput(RowVectorPtr input) {
  for (auto& child : input->children()) {
    child->loadedVector();
  }
  DecodedVector index(*input->childAt(indexChannel_));
  std::vector<const BaseVector*> keys;
  keys.reserve(numKeys_);
  for (column_index_t i = 0; i < numKeys_; ++i) {
    keys.push_back(input->childAt(channels_[i]).get());
  }

  std::vector<vector_size_t> rows;
  for (vector_size_t row = 0; row < input->size(); ++row) {
    if (index.isNullAt(row) ||
        std::any_of(keys.begin(), keys.end(), [&](const auto* key) {
          return key->isNullAt(row);
        })) {
      continue;
    }
    const auto bytes = index.valueAt<StringView>(row);
    const std::string_view packed(bytes.data(), bytes.size());
    indexes_.push_back(
        std::make_unique<common::vectorindex::LocalVectorIndex>(
            packed, metric_, searchOptions_));
    rows.push_back(row);
  }
  if (rows.empty()) {
    return;
  }
  packedColumns_.push_back(input->childAt(indexChannel_));

  std::vector<VectorPtr> children;
  children.reserve(channels_.size());
  for (const auto channel : channels_) {
    children.push_back(input->childAt(channel));
  }
  const auto projected = std::make_shared<RowVector>(
      pool(), rowType_, nullptr, input->size(), std::move(children));
  std::vector<BaseVector::CopyRange> ranges;
  ranges.reserve(rows.size());
  for (vector_size_t i = 0; i < rows.size(); ++i) {
    ranges.push_back({rows[i], i, 1});
  }
  auto selected = BaseVector::create<RowVector>(rowType_, rows.size(), pool());
  selected->copyRanges(projected.get(), ranges);
  rows_.push_back(std::move(selected));
}

RowVectorPtr VectorSearchJoinBuild::mergeRows() const {
  vector_size_t numRows = 0;
  for (const auto& rows : rows_) {
    numRows += rows->size();
  }
  auto merged = BaseVector::create<RowVector>(rowType_, numRows, pool());
  vector_size_t offset = 0;
  for (const auto& rows : rows_) {
    merged->copy(rows.get(), offset, 0, rows->size());
    offset += rows->size();
  }
  return merged;
}

void VectorSearchJoinBuild::close() {
  rows_.clear();
  indexes_.clear();
  packedColumns_.clear();
  Operator::close();
}

BlockingReason VectorSearchJoinBuild::isBlocked(ContinueFuture* future) {
  if (!future_.valid()) {
    return BlockingReason::kNotBlocked;
  }
  *future = std::move(future_);
  return BlockingReason::kWaitForJoinBuild;
}

void VectorSearchJoinBuild::noMoreInput() {
  Operator::noMoreInput();
  std::vector<ContinuePromise> promises;
  std::vector<std::shared_ptr<Operator>> peerOperators;
  // The last Driver to hit VectorSearchJoinBuild::noMoreInput gathers the
  // rows and indexes of all build Drivers and hands them to the probe side.
  if (!operatorCtx_->allPeersArrived(&future_, promises, peerOperators)) {
    return;
  }

  {
    SCOPE_EXIT {
      // Realize the promises so that the other Drivers (which were not
      // the last to finish) can continue from the barrier and finish.
      peerOperators.clear();
      for (auto& promise : promises) {
        promise.setValue();
      }
    };

    for (const auto& peer : peerOperators) {
      auto* build = peer->as<VectorSearchJoinBuild>();
      VELOX_CHECK_NOT_NULL(build);
      rows_.insert(
          rows_.end(),
          std::make_move_iterator(build->rows_.begin()),
          std::make_move_iterator(build->rows_.end()));
      indexes_.insert(
          indexes_.end(),
          std::make_move_iterator(build->indexes_.begin()),
          std::make_move_iterator(build->indexes_.end()));
      packedColumns_.insert(
          packedColumns_.end(),
          std::make_move_iterator(build->packedColumns_.begin()),
          std::make_move_iterator(build->packedColumns_.end()));
    }
  }

  auto table = std::make_shared<const VectorSearchJoinTable>(
      mergeRows(), numKeys_, std::move(indexes_), std::move(packedColumns_));
  rows_.clear();
  operatorCtx_->task()
      ->getVectorSearchJoinBridge(
          operatorCtx_->driverCtx()->splitGroupId, planNodeId())
      ->setData(std::move(table));
}

bool VectorSearchJoinBuild::isFinished() {
  return !future_.valid() && noMoreInput_;
}

} // namespace facebook::velox::exec
