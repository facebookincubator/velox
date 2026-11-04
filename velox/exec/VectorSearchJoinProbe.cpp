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

#include "velox/exec/OperatorType.h"
#include "velox/exec/OperatorUtils.h"
#include "velox/exec/Task.h"
#include "velox/exec/VectorSearchJoinBridge.h"
#include "velox/vector/FlatVector.h"

namespace facebook::velox::exec {

VectorSearchJoinProbe::VectorSearchJoinProbe(
    int32_t operatorId,
    DriverCtx* driverCtx,
    const core::VectorSearchJoinNodePtr& joinNode)
    : Operator(
          driverCtx,
          joinNode->outputType(),
          operatorId,
          joinNode->id(),
          OperatorType::kVectorSearchJoinProbe),
      isLeftJoin_{joinNode->joinType() == core::JoinType::kLeft},
      metric_{joinNode->metric()},
      k_{joinNode->k()},
      results_{memory::StlAllocator<Result>(*pool())} {
  const auto& probeType = joinNode->sources()[0]->outputType();
  const auto& buildType = joinNode->sources()[1]->outputType();
  for (const auto& key : joinNode->leftKeys()) {
    keyChannels_.push_back(probeType->getChildIdx(key->name()));
  }
  embeddingChannel_ =
      probeType->getChildIdx(joinNode->probeEmbedding()->name());

  // The table's rows hold the keys, then the build columns of the output in
  // order.
  column_index_t buildChannel = keyChannels_.size();
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
      buildProjections_.emplace_back(buildChannel++, i);
    }
  }
}

BlockingReason VectorSearchJoinProbe::isBlocked(ContinueFuture* future) {
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
  dimension_ = table_->dimension();
  return BlockingReason::kNotBlocked;
}

void VectorSearchJoinProbe::addInput(RowVectorPtr input) {
  for (auto& child : input->children()) {
    child->loadedVector();
  }
  input_ = std::move(input);

  searches_.clear();
  results_.clear();
  numOutputResults_ = 0;
  nextSearch_ = 0;
  nextProbeRow_ = 0;

  if (isLeftJoin_) {
    const auto numRows = input_->size();
    uint64_t* rawHasResults;
    BaseVector::ensureBuffer<bool>(
        numRows, pool(), &hasResults_, &rawHasResults);
    bits::fillBits(rawHasResults, 0, numRows, false);
  }
  nextLeftJoinRow_ = 0;
  maxOutputRows_ = outputBatchRows(estimateOutputRowSize());

  prepareSearches();

  // An inner join outputs nothing for a batch with nothing to search.
  if (!isLeftJoin_ && searches_.empty()) {
    input_.reset();
  }
}

uint64_t VectorSearchJoinProbe::estimateOutputRowSize() const {
  uint64_t size = sizeof(int64_t) + sizeof(float);
  const auto addColumnSize = [&](const BaseVector& column) {
    if (column.size() > 0) {
      size += column.estimateFlatSize() / column.size();
    }
  };
  for (const auto& projection : probeProjections_) {
    addColumnSize(*input_->childAt(projection.inputChannel));
  }
  for (const auto& projection : buildProjections_) {
    addColumnSize(*table_->rows()->childAt(projection.inputChannel));
  }
  return size;
}

void VectorSearchJoinProbe::prepareSearches() {
  const auto numRows = input_->size();
  const DecodedVector embeddings(*input_->childAt(embeddingChannel_));
  // A null constant embedding has no array vector under it.
  if (embeddings.isConstantMapping() && embeddings.isNullAt(0)) {
    return;
  }
  const auto* arrays = embeddings.base()->asChecked<ArrayVector>();
  // Flat elements make each query one null check and one copy.
  auto elements = arrays->elements();
  BaseVector::flattenVector(elements);
  const auto* flatElements = elements->asFlatVector<float>();
  float* rawQueries;
  BaseVector::ensureBuffer<float>(
      numRows * dimension_, pool(), &queries_, &rawQueries);

  std::vector<const BaseVector*> keys;
  keys.reserve(keyChannels_.size());
  for (const auto channel : keyChannels_) {
    keys.push_back(input_->childAt(channel).get());
  }

  folly::F14FastMap<vector_size_t, size_t> searchIndexes;
  for (vector_size_t row = 0; row < numRows; ++row) {
    if (embeddings.isNullAt(row)) {
      continue;
    }
    bool matched = false;
    table_->forEachMatch(keys, row, [&](vector_size_t buildRow) {
      const auto [it, inserted] =
          searchIndexes.emplace(buildRow, searches_.size());
      if (inserted) {
        searches_.push_back({buildRow, {}});
      }
      searches_[it->second].probeRows.push_back(row);
      matched = true;
    });
    if (matched) {
      prepareQuery(
          *arrays,
          embeddings.index(row),
          *flatElements,
          {rawQueries + row * dimension_, dimension_});
    }
  }
}

void VectorSearchJoinProbe::prepareQuery(
    const ArrayVector& arrays,
    vector_size_t index,
    const FlatVector<float>& elements,
    std::span<float> query) const {
  VELOX_USER_CHECK_EQ(
      arrays.sizeAt(index),
      dimension_,
      "Query embedding length must match the vector index's");

  const auto offset = arrays.offsetAt(index);
  VELOX_USER_CHECK(
      elements.rawNulls() == nullptr ||
          bits::isAllSet(elements.rawNulls(), offset, offset + dimension_),
      "Query embedding must not contain nulls");
  std::copy_n(elements.rawValues() + offset, dimension_, query.data());
  common::vectorindex::EmbeddingNormalizer::normalize(metric_, query);
}

void VectorSearchJoinProbe::search(
    vector_size_t buildRow,
    std::span<const vector_size_t> probeRows) {
  const size_t numQueries = probeRows.size();
  float* rawSearchQueries;
  BaseVector::ensureBuffer<float>(
      numQueries * dimension_, pool(), &searchQueries_, &rawSearchQueries);
  const auto* rawQueries = queries_->as<float>();
  for (size_t i = 0; i < numQueries; ++i) {
    std::copy_n(
        rawQueries + probeRows[i] * dimension_,
        dimension_,
        rawSearchQueries + i * dimension_);
  }

  const size_t k = table_->numResults(buildRow, k_);
  const size_t numEntries = numQueries * k;
  int64_t* rawIds;
  float* rawScores;
  BaseVector::ensureBuffer<int64_t>(numEntries, pool(), &searchIds_, &rawIds);
  BaseVector::ensureBuffer<float>(
      numEntries, pool(), &searchScores_, &rawScores);
  table_->search(
      buildRow,
      common::vectorindex::Embeddings(rawSearchQueries, numQueries, dimension_),
      k,
      {rawIds, numEntries},
      {rawScores, numEntries});
  auto* rawHasResults =
      isLeftJoin_ ? hasResults_->asMutable<uint64_t>() : nullptr;
  for (size_t i = 0; i < numQueries; ++i) {
    for (size_t j = 0; j < k; ++j) {
      const auto id = rawIds[i * k + j];
      // Padding fills the entries after a query's last result.
      if (id == common::vectorindex::LocalVectorIndex::kPaddingId) {
        break;
      }
      results_.push_back({probeRows[i], buildRow, id, rawScores[i * k + j]});
      if (rawHasResults != nullptr) {
        bits::setBit(rawHasResults, probeRows[i]);
      }
    }
  }
}

bool VectorSearchJoinProbe::fillResults() {
  results_.clear();
  numOutputResults_ = 0;
  // The ids and scores of one search, 'k' per query, must fit in a buffer.
  const size_t maxQueries =
      std::max<int64_t>(1, std::numeric_limits<vector_size_t>::max() / k_);
  while (results_.size() < maxOutputRows_ && nextSearch_ < searches_.size()) {
    // One search takes up to an output batch of probe rows, whatever 'k' is,
    // so that FAISS searches many queries at once.
    const auto& [buildRow, probeRows] = searches_[nextSearch_];
    const auto numProbeRows = std::min(
        {probeRows.size() - nextProbeRow_, maxOutputRows_, maxQueries});
    search(
        buildRow,
        std::span<const vector_size_t>(
            probeRows.data() + nextProbeRow_, numProbeRows));
    nextProbeRow_ += numProbeRows;
    if (nextProbeRow_ == probeRows.size()) {
      ++nextSearch_;
      nextProbeRow_ = 0;
    }
  }

  if (isLeftJoin_ && nextSearch_ == searches_.size()) {
    const auto* rawHasResults = hasResults_->as<uint64_t>();
    while (results_.size() < maxOutputRows_ &&
           nextLeftJoinRow_ < input_->size()) {
      if (!bits::isBitSet(rawHasResults, nextLeftJoinRow_)) {
        results_.push_back({nextLeftJoinRow_, std::nullopt, 0, 0});
      }
      ++nextLeftJoinRow_;
    }
  }
  return !results_.empty();
}

RowVectorPtr VectorSearchJoinProbe::getOutput() {
  if (input_ == nullptr) {
    return nullptr;
  }
  while (numOutputResults_ == results_.size()) {
    if (!fillResults()) {
      input_.reset();
      return nullptr;
    }
  }
  const vector_size_t numResults =
      std::min(maxOutputRows_, results_.size() - numOutputResults_);
  auto output = makeOutput(numOutputResults_, numResults);
  numOutputResults_ += numResults;
  return output;
}

RowVectorPtr VectorSearchJoinProbe::makeOutput(
    size_t start,
    vector_size_t numResults) {
  auto probeIndices = allocateIndices(numResults, pool());
  auto* rawProbeIndices = probeIndices->asMutable<vector_size_t>();
  auto buildIndices = allocateIndices(numResults, pool());
  auto* rawBuildIndices = buildIndices->asMutable<vector_size_t>();
  BufferPtr buildNulls;
  FlatVectorPtr<int64_t> ids;
  if (idChannel_.has_value()) {
    ids = BaseVector::create<FlatVector<int64_t>>(BIGINT(), numResults, pool());
  }
  FlatVectorPtr<float> scores;
  if (scoreChannel_.has_value()) {
    scores = BaseVector::create<FlatVector<float>>(REAL(), numResults, pool());
  }
  for (vector_size_t i = 0; i < numResults; ++i) {
    const auto& result = results_[start + i];
    rawProbeIndices[i] = result.probeRow;
    if (result.buildRow.has_value()) {
      rawBuildIndices[i] = *result.buildRow;
      if (ids) {
        ids->set(i, result.id);
      }
      if (scores) {
        scores->set(i, result.score);
      }
      continue;
    }
    if (buildNulls == nullptr) {
      buildNulls = allocateNulls(numResults, pool());
    }
    bits::setNull(buildNulls->asMutable<uint64_t>(), i);
    rawBuildIndices[i] = 0;
    if (ids) {
      ids->setNull(i, true);
    }
    if (scores) {
      scores->setNull(i, true);
    }
  }

  std::vector<VectorPtr> children(outputType_->size());
  projectChildren(
      children, input_, probeProjections_, numResults, probeIndices);
  for (const auto& projection : buildProjections_) {
    children[projection.outputChannel] = wrapChild(
        numResults,
        buildIndices,
        table_->rows()->childAt(projection.inputChannel),
        buildNulls);
  }
  if (idChannel_.has_value()) {
    children[*idChannel_] = std::move(ids);
  }
  if (scoreChannel_.has_value()) {
    children[*scoreChannel_] = std::move(scores);
  }
  return std::make_shared<RowVector>(
      pool(), outputType_, nullptr, numResults, std::move(children));
}

void VectorSearchJoinProbe::close() {
  table_.reset();
  searches_.clear();
  results_.clear();
  Operator::close();
}

} // namespace facebook::velox::exec
