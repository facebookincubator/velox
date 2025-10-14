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

#include "velox/exec/NonMaterializedSortBuffer.h"

#include "velox/exec/MemoryReclaimer.h"
#include "velox/exec/Spiller.h"
#include "velox/expression/VectorReaders.h"

namespace facebook::velox::exec {

NonMaterializedSortBuffer::NonMaterializedSortBuffer(
    const RowTypePtr& inputType,
    const std::vector<column_index_t>& sortColumnIndices,
    const std::vector<CompareFlags>& sortCompareFlags,
    velox::memory::MemoryPool* pool,
    tsan_atomic<bool>* nonReclaimableSection,
    common::PrefixSortConfig prefixSortConfig,
    const common::SpillConfig* spillConfig,
    folly::Synchronized<velox::common::SpillStats>* spillStats)
    : SortBufferBase(
          inputType,
          sortColumnIndices,
          sortCompareFlags,
          pool,
          nonReclaimableSection,
          prefixSortConfig,
          /*spillConfig*/ nullptr,
          /*spillStats*/ nullptr),
      sortingKeys_(
          SpillState::makeSortingKeys(sortColumnIndices, sortCompareFlags)) {
  // Sorted key columns.
  std::vector<TypePtr> sortedColumnTypes;
  sortedColumnTypes.reserve(sortColumnIndices.size());
  for (column_index_t i = 0; i < sortColumnIndices.size(); ++i) {
    columnMap_.emplace_back(IdentityProjection(i, sortColumnIndices[i]));
    sortedColumnTypes.emplace_back(
        inputType_->childAt(sortColumnIndices.at(i)));
  }

  // Vector index and row index columns.
  const auto numSortKeys = columnMap_.size();
  for (auto i = 0; i < indexType_->size(); ++i) {
    indexColumnMap_.emplace_back(numSortKeys + i, i);
  }
  data_ = std::make_unique<RowContainer>(
      sortedColumnTypes, indexType_->children(), pool_);
}

NonMaterializedSortBuffer::~NonMaterializedSortBuffer() {
  inputs_.clear();
  rowIndices_.reset();
  pool_->release();
}

void NonMaterializedSortBuffer::addInput(const VectorPtr& input) {
  velox::common::testutil::TestValue::adjust(
      "facebook::velox::exec::HybridSortBuffer::addInput", this);
  VELOX_CHECK(!noMoreInput_);
  const SelectivityVector allRows(input->size());
  std::vector<char*> rows(input->size());
  for (int row = 0; row < input->size(); ++row) {
    rows[row] = data_->newRow();
  }

  // Stores the sort key columns.
  auto* inputRow = input->asChecked<RowVector>();
  for (const auto& columnProjection : columnMap_) {
    DecodedVector decoded(
        *inputRow->childAt(columnProjection.outputChannel), allRows);
    data_->store(
        decoded,
        folly::Range(rows.data(), input->size()),
        columnProjection.inputChannel);
  }

  // Stores the vector indices column.
  inputs_.push_back(std::static_pointer_cast<RowVector>(input));
  const auto vectorIndex = std::make_shared<ConstantVector<int64_t>>(
      pool(),
      input->size(),
      /*isNull*/ false,
      BIGINT(),
      inputs_.size() - 1);
  DecodedVector decoded;
  decoded.decode(*vectorIndex, allRows);
  auto indexColumnChannel = columnMap_.size();
  data_->store(
      decoded, folly::Range(rows.data(), input->size()), indexColumnChannel++);

  // Stores the row indices column.
  if (FOLLY_UNLIKELY(rowIndices_ == nullptr)) {
    rowIndices_ = BaseVector::create<FlatVector<int64_t>>(
        BIGINT(), input->size(), pool());
  }
  BaseVector::prepareForReuse(rowIndices_, input->size());
  const auto rowIndices = rowIndices_->asUnchecked<FlatVector<int64_t>>();
  for (int64_t i = 0; i < input->size(); ++i) {
    rowIndices->mutableRawValues()[i] = i;
  }
  decoded.decode(*rowIndices, allRows);
  data_->store(
      decoded, folly::Range(rows.data(), input->size()), indexColumnChannel);

  numInputRows_ += allRows.size();
  numInputBytes_ += input->retainedSize();
}

void NonMaterializedSortBuffer::noMoreInput() {
  velox::common::testutil::TestValue::adjust(
      "facebook::velox::exec::HybridSortBuffer::noMoreInput", this);
  VELOX_CHECK(!noMoreInput_);
  noMoreInput_ = true;

  // No data.
  if (numInputRows_ == 0) {
    return;
  }
  estimatedOutputRowSize_ = numInputBytes_ / numInputRows_;

  VELOX_CHECK_EQ(numInputRows_, data_->numRows());
  sortInput(numInputRows_);

  // Releases the unused memory reservation after procesing input.
  pool_->release();
}

int64_t NonMaterializedSortBuffer::estimateFlatInputBytes(
    const VectorPtr& input) const {
  int64_t flatInputBytes{0};
  const auto inputRowVector = input->asUnchecked<RowVector>();
  for (const auto identity : columnMap_) {
    flatInputBytes +=
        inputRowVector->childAt(identity.outputChannel)->estimateFlatSize();
  }
  flatInputBytes += indexColumnMap_.size() * sizeof(int64_t) * input->size();
  return flatInputBytes;
}

int64_t NonMaterializedSortBuffer::estimateIncrementalBytes(
    const VectorPtr& input,
    uint64_t outOfLineBytes,
    int64_t flatInputBytes) const {
  return data_->sizeIncrement(
             input->size(), outOfLineBytes ? flatInputBytes : 0) +
      input->retainedSize();
}

std::optional<uint64_t> NonMaterializedSortBuffer::estimateOutputRowSize()
    const {
  return estimatedOutputRowSize_;
}

void NonMaterializedSortBuffer::prepareOutputVector(
    RowVectorPtr& output,
    const RowTypePtr& outputType,
    vector_size_t outputBatchSize) const {
  if (output != nullptr) {
    VectorPtr vector = std::move(output);
    BaseVector::prepareForReuse(vector, outputBatchSize);
    output = std::static_pointer_cast<RowVector>(vector);
  } else {
    output = std::static_pointer_cast<RowVector>(
        BaseVector::create(outputType, outputBatchSize, pool_));
  }

  for (const auto& child : output->children()) {
    child->resize(outputBatchSize);
  }
}

void NonMaterializedSortBuffer::prepareOutput(vector_size_t batchSize) {
  prepareOutputVector(output_, inputType_, batchSize);
  prepareOutputVector(indexOutput_, indexType_, batchSize);
  VELOX_CHECK_GT(output_->size(), 0);
  VELOX_CHECK_LE(output_->size() + numOutputRows_, numInputRows_);
}

void NonMaterializedSortBuffer::gatherCopyOutput() {
  SCOPE_EXIT {
    numOutputRows_ += output_->size();
  };
  VELOX_DCHECK_EQ(numInputRows_, sortedRows_.size());
  for (const auto& columnProjection : indexColumnMap_) {
    data_->extractColumn(
        sortedRows_.data() + numOutputRows_,
        indexOutput_->size(),
        columnProjection.inputChannel,
        indexOutput_->childAt(columnProjection.outputChannel));
  }

  // Extracts vector indices.
  std::vector<const RowVector*> sourceVectors;
  sourceVectors.reserve(indexOutput_->size());
  const auto* vectorIndices =
      indexOutput_->childAt(0)->asChecked<FlatVector<int64_t>>();
  for (auto i = 0; i < indexOutput_->size(); ++i) {
    sourceVectors.push_back(inputs_[vectorIndices->rawValues()[i]].get());
  }

  // Extracts row indices.
  std::vector<vector_size_t> sourceRowIndices;
  sourceRowIndices.reserve(indexOutput_->size());
  const auto* rowIndices =
      indexOutput_->childAt(1)->asChecked<FlatVector<int64_t>>();
  for (auto i = 0; i < indexOutput_->size(); ++i) {
    sourceRowIndices.push_back(rowIndices->rawValues()[i]);
  }

  gatherCopy(
      output_.get(), 0, output_->size(), sourceVectors, sourceRowIndices);
}

RowVectorPtr NonMaterializedSortBuffer::getOutput(vector_size_t maxOutputRows) {
  SCOPE_EXIT {
    pool_->release();
  };

  VELOX_CHECK(noMoreInput_);
  if (numOutputRows_ == numInputRows_) {
    inputs_.clear();
    data_->clear();
    return nullptr;
  }

  VELOX_CHECK_GT(maxOutputRows, 0);
  VELOX_CHECK_GT(numInputRows_, numOutputRows_);
  const vector_size_t batchSize =
      std::min<uint64_t>(numInputRows_ - numOutputRows_, maxOutputRows);

  prepareOutput(batchSize);
  gatherCopyOutput();

  return output_;
}
} // namespace facebook::velox::exec
