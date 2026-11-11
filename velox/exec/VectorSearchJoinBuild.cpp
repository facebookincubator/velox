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

std::unique_ptr<Operator> BaseVectorSearchJoinBuild::create(
    int32_t operatorId,
    DriverCtx* driverCtx,
    const core::VectorSearchJoinNodePtr& joinNode) {
  if (joinNode->isIndexOnBuild()) {
    return std::make_unique<VectorSearchJoinIndexBuild>(
        operatorId, driverCtx, joinNode);
  }
  return std::make_unique<VectorSearchJoinQueryBuild>(
      operatorId, driverCtx, joinNode);
}

BaseVectorSearchJoinBuild::BaseVectorSearchJoinBuild(
    int32_t operatorId,
    DriverCtx* driverCtx,
    const core::VectorSearchJoinNodePtr& joinNode,
    const core::FieldAccessTypedExprPtr& value,
    bool keepUnmatchableRows)
    : Operator(
          driverCtx,
          nullptr,
          operatorId,
          joinNode->id(),
          OperatorType::kVectorSearchJoinBuild),
      keepUnmatchableRows_{keepUnmatchableRows},
      minTableRowsForParallelJoinBuild_{
          driverCtx->queryConfig().minTableRowsForParallelJoinBuild()},
      vectorHasherMaxNumDistinct_{
          driverCtx->queryConfig().joinBuildVectorHasherMaxNumDistinct()},
      numKeys_{static_cast<column_index_t>(joinNode->rightKeys().size())},
      hashers_{VectorHasher::createForKeys(
          joinNode->buildType(),
          joinNode->rightKeys())} {
  const auto& buildType = joinNode->buildType();
  for (const auto& name : joinNode->outputType()->names()) {
    if (const auto channel = buildType->getChildIdxIfExists(name)) {
      dependentChannels_.push_back(*channel);
      dependentTypes_.push_back(buildType->childAt(*channel));
    }
  }
  valueChannel_ = buildType->getChildIdx(value->name());
  decodedDependents_.resize(dependentChannels_.size());
}

void BaseVectorSearchJoinBuild::createTable(int32_t valueSize) {
  // The probed flag marks the query rows of a right join that find embeddings.
  if (numKeys_ == 0) {
    rows_ = std::make_unique<RowContainer>(
        std::vector<TypePtr>{},
        std::vector<Accumulator>{},
        dependentTypes_,
        pool(),
        RowContainer::Options{
            .hasProbedFlag = keepUnmatchableRows_,
            .rawDependentSize = valueSize,
        });
    return;
  }

  table_ = HashTable<true>::createForJoin(
      std::move(hashers_),
      dependentTypes_,
      pool(),
      {
          .allowDuplicates = true,
          .hasProbedFlag = keepUnmatchableRows_,
          .minTableSizeForParallelJoinBuild = minTableRowsForParallelJoinBuild_,
          .rawDependentSize = valueSize,
      });
  analyzeKeys_ = table_->hashMode() != BaseHashTable::HashMode::kHash;
}

void BaseVectorSearchJoinBuild::addInput(RowVectorPtr input) {
  for (auto& child : input->children()) {
    child->loadedVector();
  }
  const auto numRows = input->size();
  for (column_index_t i = 0; i < dependentChannels_.size(); ++i) {
    decodedDependents_[i].decode(*input->childAt(dependentChannels_[i]));
  }
  const auto& valueColumn = input->childAt(valueChannel_);
  values_.decode(*valueColumn);
  addBatch(valueColumn);

  // A row with a null value or a null key cannot match.
  SelectivityVector rows(numRows);
  if (const auto* nulls = values_.nulls()) {
    rows.deselectNulls(nulls, 0, numRows);
  }
  for (auto& hasher : keyHashers()) {
    hasher->decode(*input->childAt(hasher->channel()), rows);
    if (const auto* nulls = hasher->decodedVector().nulls(&rows)) {
      rows.deselectNulls(nulls, 0, numRows);
    }
  }

  if (keepUnmatchableRows_ && rows.countSelected() < numRows) {
    SelectivityVector unmatchable(numRows);
    unmatchable.deselect(rows);
    addUnmatchableRows(unmatchable);
  }
  if (!rows.hasSelections()) {
    return;
  }

  if (table_ == nullptr && rows_ == nullptr) {
    createTable(valueSize());
  }
  if (analyzeKeys_) {
    analyzeKeys_ = table_->analyzeKeys(rows);
  }

  auto* tableRows = table_ != nullptr ? table_->rows() : rows_.get();
  const auto& hashers = keyHashers();
  rows.applyToSelected([&](vector_size_t row) {
    char* newRow = tableRows->newRow();
    for (column_index_t i = 0; i < numKeys_; ++i) {
      tableRows->store(hashers[i]->decodedVector(), row, newRow, i);
    }
    for (column_index_t i = 0; i < decodedDependents_.size(); ++i) {
      tableRows->store(decodedDependents_[i], row, newRow, numKeys_ + i);
    }
    storeValue(values_, row, tableRows->rawDependent(newRow));
  });
}

void BaseVectorSearchJoinBuild::addUnmatchableRows(
    const SelectivityVector& rows) {
  if (unmatchableRows_ == nullptr) {
    unmatchableRows_ = std::make_unique<RowContainer>(
        std::vector<TypePtr>{}, dependentTypes_, pool());
  }
  rows.applyToSelected([&](vector_size_t row) {
    char* newRow = unmatchableRows_->newRow();
    for (column_index_t i = 0; i < decodedDependents_.size(); ++i) {
      unmatchableRows_->store(decodedDependents_[i], row, newRow, i);
    }
  });
}

void BaseVectorSearchJoinBuild::close() {
  table_.reset();
  rows_.reset();
  unmatchableRows_.reset();
  Operator::close();
}

BlockingReason BaseVectorSearchJoinBuild::isBlocked(ContinueFuture* future) {
  if (!future_.valid()) {
    return BlockingReason::kNotBlocked;
  }
  *future = std::move(future_);
  return BlockingReason::kWaitForJoinBuild;
}

void BaseVectorSearchJoinBuild::noMoreInput() {
  Operator::noMoreInput();
  std::vector<ContinuePromise> promises;
  std::vector<std::shared_ptr<Operator>> peerOperators;
  // The last Driver to hit BaseVectorSearchJoinBuild::noMoreInput merges the
  // tables of all build Drivers and hands the result to the probe side.
  if (!operatorCtx_->allPeersArrived(&future_, promises, peerOperators)) {
    return;
  }

  std::vector<std::unique_ptr<BaseHashTable>> otherTables;
  std::vector<std::unique_ptr<RowContainer>> rows;
  if (rows_ != nullptr) {
    rows.push_back(std::move(rows_));
  }
  std::vector<std::unique_ptr<RowContainer>> unmatchableRows;
  if (unmatchableRows_ != nullptr) {
    unmatchableRows.push_back(std::move(unmatchableRows_));
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
      auto* build = peer->as<BaseVectorSearchJoinBuild>();
      VELOX_CHECK_NOT_NULL(build);
      updateDimension(build->dimension_);
      takeFromPeer(*build);
      if (build->table_ != nullptr) {
        otherTables.push_back(std::move(build->table_));
      }
      if (build->rows_ != nullptr) {
        rows.push_back(std::move(build->rows_));
      }
      if (build->unmatchableRows_ != nullptr) {
        unmatchableRows.push_back(std::move(build->unmatchableRows_));
      }
    }
  }

  if (table_ == nullptr && !otherTables.empty()) {
    table_ = std::move(otherTables.back());
    otherTables.pop_back();
  }
  if (table_ != nullptr) {
    auto* executor = otherTables.empty()
        ? nullptr
        : operatorCtx_->task()->queryCtx()->executor();
    table_->prepareJoinTable(
        std::move(otherTables),
        BaseHashTable::kNoSpillInputStartPartitionBit,
        vectorHasherMaxNumDistinct_,
        /*dropDuplicates=*/false,
        executor);
  }
  auto table =
      makeTable(std::move(table_), std::move(rows), std::move(unmatchableRows));
  operatorCtx_->task()
      ->getVectorSearchJoinBridge(splitGroupId(), planNodeId())
      ->setData(std::move(table));
}

void BaseVectorSearchJoinBuild::updateDimension(int32_t dimension) {
  if (dimension == 0) {
    return;
  }
  if (dimension_ == 0) {
    dimension_ = dimension;
  }
  VELOX_USER_CHECK_EQ(dimension, dimension_, "{}", dimensionError());
}

bool BaseVectorSearchJoinBuild::isFinished() {
  return !future_.valid() && noMoreInput_;
}

VectorSearchJoinIndexBuild::VectorSearchJoinIndexBuild(
    int32_t operatorId,
    DriverCtx* driverCtx,
    const core::VectorSearchJoinNodePtr& joinNode)
    : BaseVectorSearchJoinBuild(
          operatorId,
          driverCtx,
          joinNode,
          joinNode->index(),
          /*keepUnmatchableRows=*/false) {
  const auto formats =
      driverCtx->task->queryCtx()
          ->registry<common::vectorindex::LocalVectorIndexFormats>(
              common::vectorindex::LocalVectorIndexFormat::kRegistryKey);
  VELOX_CHECK_NOT_NULL(formats, "QueryCtx has no local vector index formats");
  reader_ = formats->makeReader(
      joinNode->format(), joinNode->metric(), joinNode->searchOptions());
}

void VectorSearchJoinIndexBuild::addBatch(const VectorPtr& values) {
  packedIndexes_ = values;
}

int32_t VectorSearchJoinIndexBuild::valueSize() const {
  return sizeof(common::vectorindex::LocalVectorIndex*);
}

void VectorSearchJoinIndexBuild::storeValue(
    const DecodedVector& values,
    vector_size_t row,
    std::span<char> bytes) {
  const auto packed = values.valueAt<StringView>(row);
  auto index = reader_->open(packed);
  updateDimension(index->dimension());
  // The raw dependent is 8-byte aligned.
  *reinterpret_cast<common::vectorindex::LocalVectorIndex**>(bytes.data()) =
      index.get();
  // The index refers to the bytes of the input it is opened from.
  indexes_.push_back({packedIndexes_, std::move(index)});
}

std::string_view VectorSearchJoinIndexBuild::dimensionError() const {
  return "Vector search join indexes must have the same dimension";
}

void VectorSearchJoinIndexBuild::takeFromPeer(BaseVectorSearchJoinBuild& peer) {
  auto& build = dynamic_cast<VectorSearchJoinIndexBuild&>(peer);
  indexes_.insert(
      indexes_.end(),
      std::make_move_iterator(build.indexes_.begin()),
      std::make_move_iterator(build.indexes_.end()));
}

std::shared_ptr<const VectorSearchJoinTable>
VectorSearchJoinIndexBuild::makeTable(
    std::unique_ptr<BaseHashTable> hashTable,
    std::vector<std::unique_ptr<RowContainer>> rows,
    std::vector<std::unique_ptr<RowContainer>> /*unmatchableRows*/) {
  return std::make_shared<const VectorSearchJoinIndexTable>(
      std::move(hashTable), std::move(rows), dimension_, std::move(indexes_));
}

void VectorSearchJoinIndexBuild::close() {
  BaseVectorSearchJoinBuild::close();
  indexes_.clear();
  packedIndexes_.reset();
}

VectorSearchJoinQueryBuild::VectorSearchJoinQueryBuild(
    int32_t operatorId,
    DriverCtx* driverCtx,
    const core::VectorSearchJoinNodePtr& joinNode)
    : BaseVectorSearchJoinBuild(
          operatorId,
          driverCtx,
          joinNode,
          joinNode->embedding(),
          joinNode->joinType() == core::JoinType::kRight),
      metric_{joinNode->metric()} {}

void VectorSearchJoinQueryBuild::addBatch(const VectorPtr& values) {
  embeddings_.emplace(*values, metric_);
  updateDimension(embeddings_->dimension());
}

int32_t VectorSearchJoinQueryBuild::valueSize() const {
  return dimension_ * sizeof(float);
}

void VectorSearchJoinQueryBuild::storeValue(
    const DecodedVector& /*values*/,
    vector_size_t row,
    std::span<char> bytes) {
  embeddings_->copyNormalized(
      row,
      {reinterpret_cast<float*>(bytes.data()),
       static_cast<size_t>(dimension_)});
}

std::string_view VectorSearchJoinQueryBuild::dimensionError() const {
  return "Query embeddings must have the same length";
}

std::shared_ptr<const VectorSearchJoinTable>
VectorSearchJoinQueryBuild::makeTable(
    std::unique_ptr<BaseHashTable> hashTable,
    std::vector<std::unique_ptr<RowContainer>> rows,
    std::vector<std::unique_ptr<RowContainer>> unmatchableRows) {
  return std::make_shared<const VectorSearchJoinQueryTable>(
      std::move(hashTable),
      std::move(rows),
      dimension_,
      std::move(unmatchableRows));
}

} // namespace facebook::velox::exec
