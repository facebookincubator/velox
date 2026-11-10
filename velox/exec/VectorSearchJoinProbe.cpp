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

#include "velox/exec/VectorSearchJoinProbe.h"

#include "velox/common/vectorindex/Embeddings.h"
#include "velox/exec/OperatorType.h"
#include "velox/exec/OperatorUtils.h"
#include "velox/exec/Task.h"
#include "velox/exec/VectorSearchJoinBridge.h"
#include "velox/exec/VectorSearchJoinEmbeddings.h"
#include "velox/vector/FlatVector.h"

namespace facebook::velox::exec {

std::unique_ptr<Operator> BaseVectorSearchJoinProbe::create(
    int32_t operatorId,
    DriverCtx* driverCtx,
    const core::VectorSearchJoinNodePtr& joinNode) {
  if (joinNode->isIndexOnBuild()) {
    return std::make_unique<VectorSearchJoinQueryProbe>(
        operatorId, driverCtx, joinNode);
  }
  return std::make_unique<VectorSearchJoinIndexProbe>(
      operatorId, driverCtx, joinNode);
}

BaseVectorSearchJoinProbe::BaseVectorSearchJoinProbe(
    int32_t operatorId,
    DriverCtx* driverCtx,
    const core::VectorSearchJoinNodePtr& joinNode)
    : Operator(
          driverCtx,
          joinNode->outputType(),
          operatorId,
          joinNode->id(),
          OperatorType::kVectorSearchJoinProbe),
      isOuterJoin_{joinNode->joinType() != core::JoinType::kInner},
      matchProbeRows_{pool()},
      matchBuildRows_{pool()},
      k_{joinNode->k()},
      preservesProbeRows_{joinNode->joinType() == core::JoinType::kLeft},
      buildRows_{pool()} {
  const auto& probeType = joinNode->probeType();
  const auto& buildType = joinNode->buildType();
  hashers_ = VectorHasher::createForKeys(probeType, joinNode->leftKeys());

  column_index_t buildColumn = 0;
  const auto& outputType = joinNode->outputType();
  for (column_index_t i = 0; i < outputType->size(); ++i) {
    const auto& name = outputType->nameOf(i);
    if (name == joinNode->idName()) {
      idChannel_ = i;
    } else if (name == joinNode->scoreName()) {
      scoreChannel_ = i;
    } else if (const auto channel = probeType->getChildIdxIfExists(name)) {
      probeProjections_.emplace_back(*channel, i);
    } else {
      VELOX_CHECK(buildType->containsChild(name));
      buildProjections_.emplace_back(buildColumn++, i);
    }
  }
}

namespace {
// Returns the estimated bytes of the build columns 'buildProjections' of an
// output row.
uint64_t estimateBuildRowSize(
    const VectorSearchJoinTable& table,
    const std::vector<IdentityProjection>& buildProjections) {
  const auto& containers = table.rows();
  if (containers.empty()) {
    return 0;
  }
  uint64_t numRows = 0;
  for (const auto* container : containers) {
    numRows += container->numRows();
  }
  uint64_t rowSize = 0;
  const auto& columnTypes = containers.front()->columnTypes();
  for (const auto& projection : buildProjections) {
    const auto column = table.numKeys() + projection.inputChannel;
    if (columnTypes[column]->isFixedWidth()) {
      rowSize += containers.front()->fixedSizeAt(column);
    } else {
      rowSize +=
          RowContainer::columnStats(containers, column).avgBytes(numRows);
    }
  }
  return rowSize;
}
} // namespace

BlockingReason BaseVectorSearchJoinProbe::isBlocked(ContinueFuture* future) {
  if (table_ != nullptr) {
    return BlockingReason::kNotBlocked;
  }
  auto table = operatorCtx_->task()
                   ->getVectorSearchJoinBridge(splitGroupId(), planNodeId())
                   ->dataOrFuture(future);
  if (!table.has_value()) {
    return BlockingReason::kWaitForJoinBuild;
  }
  table_ = std::move(table.value());
  buildRowSize_ = estimateBuildRowSize(*table_, buildProjections_);
  if (table_->hashTable() != nullptr) {
    lookup_ = std::make_unique<HashLookup>(hashers_, pool());
    joinResults_ = std::make_unique<BaseHashTable::JoinResultIterator>(
        std::vector<vector_size_t>{}, 0, std::nullopt);
  }
  if (!table_->rows().empty()) {
    setResultRows(table_->rows().front(), table_->numKeys());
  }
  setTable();
  if (!preservesProbeRows_) {
    if (table_->rows().empty()) {
      // No probe row matches, so none is needed.
      noMoreInput();
    } else if (
        table_->hashTable() != nullptr &&
        table_->hashTable()->hashMode() != BaseHashTable::HashMode::kHash) {
      // Outside kHash mode, the hashers have seen the keys of all build rows.
      pushdownDynamicFilters();
    }
  }
  return BlockingReason::kNotBlocked;
}

void BaseVectorSearchJoinProbe::pushdownDynamicFilters() {
  std::vector<column_index_t> keyChannels;
  keyChannels.reserve(hashers_.size());
  for (const auto& hasher : hashers_) {
    keyChannels.push_back(hasher->channel());
  }
  const auto& hashers = table_->hashTable()->hashers();
  operatorCtx_->driverCtx()->driver->pushdownFilters(
      this, keyChannels, [&](column_index_t key) {
        return hashers[key]->getFilter(/*nullAllowed=*/false);
      });
}

bool BaseVectorSearchJoinProbe::lookup(SelectivityVector& rows) {
  if (table_->rows().empty() || !rows.hasSelections()) {
    return false;
  }

  auto* hashTable = table_->hashTable();
  if (hashTable == nullptr) {
    crossProbeRows_.resize(rows.countSelected());
    vector_size_t i = 0;
    rows.applyToSelected(
        [&](vector_size_t row) { crossProbeRows_[i++] = row; });
    nextCrossProbeRow_ = 0;
    nextCrossContainer_ = 0;
    nextCrossRow_.reset();
    return true;
  }

  hashTable->prepareForJoinProbe(*lookup_, input_, rows, true);
  if (lookup_->rows.empty()) {
    return false;
  }

  lookup_->hits.resize(lookup_->rows.back() + 1);
  hashTable->joinProbe(*lookup_);
  joinResults_->reset(*lookup_);
  return true;
}

int32_t BaseVectorSearchJoinProbe::nextMatches(int32_t maxMatches) {
  matchProbeRows_.resize(maxMatches);
  matchBuildRows_.resize(maxMatches);
  if (table_->hashTable() == nullptr) {
    return nextCrossMatches(maxMatches);
  }
  if (joinResults_->atEnd()) {
    return 0;
  }
  return table_->hashTable()->listJoinResults(
      *joinResults_,
      /*includeMisses=*/false,
      std::span(matchProbeRows_.data(), maxMatches),
      std::span(matchBuildRows_.data(), maxMatches),
      RowContainer::kUnlimited);
}

int32_t BaseVectorSearchJoinProbe::nextCrossMatches(int32_t maxMatches) {
  const auto& containers = table_->rows();
  int32_t numMatches = 0;
  while (numMatches < maxMatches &&
         nextCrossProbeRow_ < crossProbeRows_.size()) {
    const auto numRows = containers[nextCrossContainer_]->listRows(
        &nextCrossRow_,
        std::span(
            matchBuildRows_.data() + numMatches, maxMatches - numMatches));
    std::fill_n(
        matchProbeRows_.data() + numMatches,
        numRows,
        crossProbeRows_[nextCrossProbeRow_]);
    numMatches += numRows;
    if (numMatches < maxMatches) {
      // The container is done.
      nextCrossRow_.reset();
      if (++nextCrossContainer_ == containers.size()) {
        nextCrossContainer_ = 0;
        ++nextCrossProbeRow_;
      }
    }
  }
  return numMatches;
}

void BaseVectorSearchJoinProbe::setMaxOutputRows() {
  uint64_t rowSize = sizeof(int64_t) + sizeof(float) + buildRowSize_;
  if (input_ != nullptr) {
    for (const auto& projection : probeProjections_) {
      const auto& column = *input_->childAt(projection.inputChannel);
      if (column.size() > 0) {
        rowSize += column.estimateFlatSize() / column.size();
      }
    }
  }
  maxOutputRows_ = outputBatchRows(rowSize);
}

float* BaseVectorSearchJoinProbe::searchQueries(
    size_t numQueries,
    size_t dimension) {
  float* rawSearchQueries;
  BaseVector::ensureBuffer<float>(
      numQueries * dimension, pool(), &searchQueries_, &rawSearchQueries);
  return rawSearchQueries;
}

void BaseVectorSearchJoinProbe::ensureOutput() {
  if (ids_ != nullptr) {
    if (outputCapacity_ >= maxSearchK_) {
      return;
    }
    VELOX_CHECK_EQ(numResults_, 0);
  }
  outputCapacity_ = std::max<int64_t>(maxOutputRows_, maxSearchK_);
  probeRows_ = allocateIndices(outputCapacity_, pool());
  buildRows_.resize(outputCapacity_);
  ids_ = BaseVector::create<FlatVector<int64_t>>(
      BIGINT(), outputCapacity_, pool());
  scores_ =
      BaseVector::create<FlatVector<float>>(REAL(), outputCapacity_, pool());
}

vector_size_t BaseVectorSearchJoinProbe::searchK(
    const common::vectorindex::LocalVectorIndex& index) const {
  return std::min<int64_t>(
      {k_, index.size(), std::numeric_limits<vector_size_t>::max()});
}

size_t BaseVectorSearchJoinProbe::maxQueries(
    const common::vectorindex::LocalVectorIndex& index) {
  const auto k = searchK(index);
  maxSearchK_ = std::max(maxSearchK_, k);
  if (numResults_ > 0 && outputCapacity_ - numResults_ < k) {
    return 0;
  }
  ensureOutput();
  return (outputCapacity_ - numResults_) / k;
}

void BaseVectorSearchJoinProbe::search(
    const common::vectorindex::LocalVectorIndex& index,
    size_t numQueries,
    folly::FunctionRef<std::pair<vector_size_t, char*>(size_t query)>
        resultRows) {
  const size_t k = searchK(index);
  const size_t numEntries = numQueries * k;
  VELOX_CHECK_LE(numResults_ + numEntries, outputCapacity_);
  auto* rawIds = ids_->mutableRawValues();
  auto* rawScores = scores_->mutableRawValues();
  index.search(
      common::vectorindex::Embeddings(
          searchQueries_->as<float>(), numQueries, index.dimension()),
      k,
      {rawIds + numResults_, numEntries},
      {rawScores + numResults_, numEntries});

  auto* rawProbeRows = probeRows_->asMutable<vector_size_t>();
  const auto first = numResults_;
  for (size_t i = 0; i < numQueries; ++i) {
    // Padding fills the entries after a query's last result.
    const auto begin = first + i * k;
    size_t numFound = 0;
    while (numFound < k &&
           rawIds[begin + numFound] !=
               common::vectorindex::LocalVectorIndex::kPaddingId) {
      ++numFound;
    }
    if (numFound == 0) {
      continue;
    }
    const auto [probeRow, buildRow] = resultRows(i);
    std::fill_n(rawProbeRows + numResults_, numFound, probeRow);
    std::fill_n(buildRows_.data() + numResults_, numFound, buildRow);
    // The results move toward the front, so a forward copy is safe.
    std::copy(rawIds + begin, rawIds + begin + numFound, rawIds + numResults_);
    std::copy(
        rawScores + begin,
        rawScores + begin + numFound,
        rawScores + numResults_);
    numResults_ += numFound;
  }
}

void BaseVectorSearchJoinProbe::addMiss(vector_size_t probeRow) {
  ensureOutput();
  VELOX_CHECK(!isOutputFull());
  probeRows_->asMutable<vector_size_t>()[numResults_] = probeRow;
  buildRows_[numResults_] = nullptr;
  ids_->setNull(numResults_, true);
  scores_->setNull(numResults_, true);
  ++numResults_;
}

std::span<char*> BaseVectorSearchJoinProbe::missRows() {
  ensureOutput();
  return {buildRows_.data() + numResults_, buildRows_.data() + outputCapacity_};
}

void BaseVectorSearchJoinProbe::addMisses(vector_size_t numRows) {
  VELOX_CHECK_LE(numResults_ + numRows, outputCapacity_);
  bits::fillBits(
      ids_->mutableRawNulls(), numResults_, numResults_ + numRows, bits::kNull);
  bits::fillBits(
      scores_->mutableRawNulls(),
      numResults_,
      numResults_ + numRows,
      bits::kNull);
  numResults_ += numRows;
}

RowVectorPtr BaseVectorSearchJoinProbe::getOutput() {
  if (table_ == nullptr || !fillResults()) {
    return nullptr;
  }
  return makeOutput();
}

RowVectorPtr BaseVectorSearchJoinProbe::makeOutput() {
  const auto numResults = numResults_;
  std::vector<VectorPtr> children(outputType_->size());

  if (input_ == nullptr) {
    for (const auto& projection : probeProjections_) {
      children[projection.outputChannel] = BaseVector::createNullConstant(
          outputType_->childAt(projection.outputChannel), numResults, pool());
    }
  } else {
    projectChildren(
        children, input_, probeProjections_, numResults, probeRows_);
  }

  for (const auto& projection : buildProjections_) {
    const auto& type = outputType_->childAt(projection.outputChannel);
    if (resultRows_ == nullptr) {
      children[projection.outputChannel] =
          BaseVector::createNullConstant(type, numResults, pool());
      continue;
    }
    auto column = BaseVector::create(type, numResults, pool());
    resultRows_->extractColumn(
        buildRows_.data(),
        numResults,
        firstResultColumn_ + projection.inputChannel,
        column);
    children[projection.outputChannel] = std::move(column);
  }

  if (idChannel_.has_value()) {
    ids_->resize(numResults);
    children[*idChannel_] = std::move(ids_);
  }
  if (scoreChannel_.has_value()) {
    scores_->resize(numResults);
    children[*scoreChannel_] = std::move(scores_);
  }
  ids_.reset();
  scores_.reset();
  probeRows_.reset();
  numResults_ = 0;
  return std::make_shared<RowVector>(
      pool(), outputType_, nullptr, numResults, std::move(children));
}

void BaseVectorSearchJoinProbe::close() {
  joinResults_.reset();
  lookup_.reset();
  table_.reset();
  ids_.reset();
  scores_.reset();
  probeRows_.reset();
  Operator::close();
}

VectorSearchJoinQueryProbe::VectorSearchJoinQueryProbe(
    int32_t operatorId,
    DriverCtx* driverCtx,
    const core::VectorSearchJoinNodePtr& joinNode)
    : BaseVectorSearchJoinProbe(operatorId, driverCtx, joinNode),
      metric_{joinNode->metric()},
      embeddingChannel_{
          joinNode->probeType()->getChildIdx(joinNode->embedding()->name())},
      matches_{memory::StlAllocator<std::pair<char*, vector_size_t>>(*pool())} {
}

void VectorSearchJoinQueryProbe::setTable() {
  indexTable_ = dynamic_cast<const VectorSearchJoinIndexTable*>(table_.get());
  VELOX_CHECK_NOT_NULL(indexTable_);
  dimension_ = indexTable_->dimension();
}

void VectorSearchJoinQueryProbe::addInput(RowVectorPtr input) {
  for (auto& child : input->children()) {
    child->loadedVector();
  }
  input_ = std::move(input);
  matches_.clear();
  nextMatch_ = 0;
  if (isOuterJoin_) {
    const auto numRows = input_->size();
    uint64_t* rawHasResults;
    BaseVector::ensureBuffer<bool>(
        numRows, pool(), &hasResults_, &rawHasResults);
    bits::fillBits(rawHasResults, 0, numRows, false);
  }
  nextLeftJoinRow_ = 0;
  setMaxOutputRows();

  prepareSearches();

  // An inner join outputs nothing for a batch with nothing to search.
  if (!isOuterJoin_ && matches_.empty()) {
    input_.reset();
  }
}

void VectorSearchJoinQueryProbe::prepareSearches() {
  const auto numRows = input_->size();
  // Every query is checked, whether it matches an index or not, as the
  // queries are when they are on the build side.
  const VectorSearchJoinEmbeddings embeddings(
      *input_->childAt(embeddingChannel_), metric_);
  if (embeddings.dimension() != 0 && dimension_ != 0) {
    VELOX_USER_CHECK_EQ(
        static_cast<size_t>(embeddings.dimension()),
        dimension_,
        "Query embedding length must match the vector index's");
  }

  SelectivityVector rows(numRows);
  embeddings.deselectNulls(rows);

  if (!lookup(rows)) {
    return;
  }

  for (auto numMatches = nextMatches(maxOutputRows_); numMatches > 0;
       numMatches = nextMatches(maxOutputRows_)) {
    for (int32_t i = 0; i < numMatches; ++i) {
      matches_.emplace_back(matchBuildRows_[i], matchProbeRows_[i]);
    }
  }
  std::sort(matches_.begin(), matches_.end());

  float* rawQueries;
  BaseVector::ensureBuffer<float>(
      numRows * dimension_, pool(), &queries_, &rawQueries);
  SelectivityVector matched(numRows, false);
  for (const auto& [buildRow, row] : matches_) {
    if (!matched.isValid(row)) {
      matched.setValid(row, true);
      embeddings.copyNormalized(
          row, {rawQueries + row * dimension_, dimension_});
    }
  }
}

bool VectorSearchJoinQueryProbe::fillResults() {
  if (input_ == nullptr) {
    return false;
  }
  auto* rawHasResults =
      isOuterJoin_ ? hasResults_->asMutable<uint64_t>() : nullptr;
  const auto* rawQueries = queries_ ? queries_->as<float>() : nullptr;
  while (nextMatch_ < matches_.size()) {
    // One search takes the next probe rows that match the same index row.
    char* const buildRow = matches_[nextMatch_].first;
    const auto& index = indexTable_->index(buildRow);
    const auto maxQueries = this->maxQueries(index);
    if (maxQueries == 0) {
      break;
    }
    size_t end = nextMatch_ + 1;
    while (end < matches_.size() && end - nextMatch_ < maxQueries &&
           matches_[end].first == buildRow) {
      ++end;
    }
    const auto numQueries = end - nextMatch_;
    auto* rawSearchQueries = searchQueries(numQueries, dimension_);
    for (size_t i = 0; i < numQueries; ++i) {
      std::copy_n(
          rawQueries + matches_[nextMatch_ + i].second * dimension_,
          dimension_,
          rawSearchQueries + i * dimension_);
    }
    const auto first = nextMatch_;
    search(index, numQueries, [&](size_t query) {
      const auto probeRow = matches_[first + query].second;
      if (rawHasResults != nullptr) {
        bits::setBit(rawHasResults, probeRow);
      }
      return std::pair{probeRow, buildRow};
    });
    nextMatch_ = end;
  }

  if (isOuterJoin_ && nextMatch_ == matches_.size()) {
    while (!isOutputFull() && nextLeftJoinRow_ < input_->size()) {
      if (!bits::isBitSet(rawHasResults, nextLeftJoinRow_)) {
        addMiss(nextLeftJoinRow_);
      }
      ++nextLeftJoinRow_;
    }
  }
  if (numResults() == 0) {
    input_.reset();
    return false;
  }
  return true;
}

void VectorSearchJoinQueryProbe::close() {
  matches_.clear();
  BaseVectorSearchJoinProbe::close();
}

VectorSearchJoinIndexProbe::VectorSearchJoinIndexProbe(
    int32_t operatorId,
    DriverCtx* driverCtx,
    const core::VectorSearchJoinNodePtr& joinNode)
    : BaseVectorSearchJoinProbe(operatorId, driverCtx, joinNode),
      indexChannel_{
          joinNode->probeType()->getChildIdx(joinNode->index()->name())} {
  const auto formats =
      driverCtx->task->queryCtx()
          ->registry<common::vectorindex::LocalVectorIndexFormats>(
              common::vectorindex::LocalVectorIndexFormat::kRegistryKey);
  VELOX_CHECK_NOT_NULL(formats, "QueryCtx has no local vector index formats");
  reader_ = formats->makeReader(
      joinNode->format(), joinNode->metric(), joinNode->searchOptions());
}

void VectorSearchJoinIndexProbe::setTable() {
  queryTable_ = dynamic_cast<const VectorSearchJoinQueryTable*>(table_.get());
  VELOX_CHECK_NOT_NULL(queryTable_);
}

BlockingReason VectorSearchJoinIndexProbe::isBlocked(ContinueFuture* future) {
  if (future_.valid()) {
    *future = std::move(future_);
    return BlockingReason::kWaitForJoinProbe;
  }
  return BaseVectorSearchJoinProbe::isBlocked(future);
}

void VectorSearchJoinIndexProbe::addInput(RowVectorPtr input) {
  for (auto& child : input->children()) {
    child->loadedVector();
  }
  input_ = std::move(input);
  numMatches_ = 0;
  nextMatch_ = 0;
  index_.reset();
  indexRow_ = kNoRow;
  setMaxOutputRows();

  const auto numRows = input_->size();
  packedIndexes_.decode(*input_->childAt(indexChannel_));
  SelectivityVector rows(numRows);
  if (const auto* nulls = packedIndexes_.nulls()) {
    rows.deselectNulls(nulls, 0, numRows);
  }
  if (!lookup(rows)) {
    input_.reset();
  }
}

const common::vectorindex::LocalVectorIndex&
VectorSearchJoinIndexProbe::probeIndex(vector_size_t probeRow) {
  if (probeRow != indexRow_) {
    const auto packed = packedIndexes_.valueAt<StringView>(probeRow);
    index_ = reader_->open(packed);
    indexRow_ = probeRow;
    VELOX_USER_CHECK_EQ(
        index_->dimension(),
        queryTable_->dimension(),
        "Query embedding length must match the vector index's");
  }
  return *index_;
}

void VectorSearchJoinIndexProbe::search(
    const common::vectorindex::LocalVectorIndex& index,
    vector_size_t probeRow,
    RowSpan buildRows) {
  const auto dimension = queryTable_->dimension();
  auto* rawSearchQueries = searchQueries(buildRows.size(), dimension);
  for (size_t i = 0; i < buildRows.size(); ++i) {
    std::copy_n(
        queryTable_->query(buildRows[i]).data(),
        dimension,
        rawSearchQueries + i * dimension);
  }
  auto* rows = queryTable_->rows().front();
  BaseVectorSearchJoinProbe::search(index, buildRows.size(), [&](size_t query) {
    char* buildRow = buildRows[query];
    if (isOuterJoin_) {
      rows->setProbedFlag({&buildRow, 1});
    }
    return std::pair{probeRow, buildRow};
  });
}

bool VectorSearchJoinIndexProbe::fillResults() {
  if (input_ != nullptr) {
    for (;;) {
      if (nextMatch_ == numMatches_) {
        numMatches_ = nextMatches(maxOutputRows_);
        nextMatch_ = 0;
        if (numMatches_ == 0) {
          break;
        }
      }
      // One search takes the next query rows that match the same probe row.
      const auto probeRow = matchProbeRows_[nextMatch_];
      const auto& index = probeIndex(probeRow);
      const auto maxQueries = this->maxQueries(index);
      if (maxQueries == 0) {
        break;
      }
      int32_t end = nextMatch_ + 1;
      while (end < numMatches_ && end - nextMatch_ < maxQueries &&
             matchProbeRows_[end] == probeRow) {
        ++end;
      }
      search(
          index,
          probeRow,
          RowSpan(matchBuildRows_.data() + nextMatch_, end - nextMatch_));
      nextMatch_ = end;
    }
    if (numResults() > 0) {
      return true;
    }
    input_.reset();
    index_.reset();
  }
  if (!noMoreInput_ || !isOuterJoin_ || !lastProbe()) {
    return false;
  }
  return fillUnmatchedResults();
}

bool VectorSearchJoinIndexProbe::fillUnmatchedResults() {
  const auto rows = missRows();
  const auto& tableRows = queryTable_->rows();
  while (notProbedContainer_ < tableRows.size()) {
    auto* container = tableRows[notProbedContainer_];
    const auto numRows =
        container->listRows<RowContainer::ProbedFilter::kNotProbed>(
            &notProbedRows_, rows, RowContainer::kUnlimited);
    if (numRows > 0) {
      setResultRows(container, queryTable_->numKeys());
      addMisses(numRows);
      return true;
    }
    ++notProbedContainer_;
    notProbedRows_.reset();
  }
  const auto& containers = queryTable_->unmatchableRows();
  while (unmatchableContainer_ < containers.size()) {
    const auto& container = containers[unmatchableContainer_];
    const auto numRows = container->listRows(&unmatchableRows_, rows);
    if (numRows > 0) {
      setResultRows(container.get(), 0);
      addMisses(numRows);
      return true;
    }
    ++unmatchableContainer_;
    unmatchableRows_.reset();
  }
  unmatchedDone_ = true;
  return false;
}

bool VectorSearchJoinIndexProbe::lastProbe() {
  if (barrierPassed_) {
    return lastProbe_;
  }
  barrierPassed_ = true;
  std::vector<ContinuePromise> promises;
  std::vector<std::shared_ptr<Operator>> peerOperators;
  if (!operatorCtx_->allPeersArrived(&future_, promises, peerOperators)) {
    return false;
  }
  // The peers have marked their matches. Let them finish.
  peerOperators.clear();
  for (auto& promise : promises) {
    promise.setValue();
  }
  lastProbe_ = true;
  setMaxOutputRows();
  return true;
}

bool VectorSearchJoinIndexProbe::isFinished() {
  if (!noMoreInput_ || input_ != nullptr || future_.valid()) {
    return false;
  }
  if (!isOuterJoin_) {
    return true;
  }
  return barrierPassed_ && (!lastProbe_ || unmatchedDone_);
}

void VectorSearchJoinIndexProbe::close() {
  index_.reset();
  BaseVectorSearchJoinProbe::close();
}

} // namespace facebook::velox::exec
