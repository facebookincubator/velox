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

#include "velox/exec/HybridSortBuffer.h"

#include "velox/exec/MemoryReclaimer.h"
#include "velox/exec/Spiller.h"
#include "velox/expression/VectorReaders.h"

namespace facebook::velox::exec {

HybridSortBuffer::HybridSortBuffer(
    const RowTypePtr& input,
    const std::vector<column_index_t>& sortColumnIndices,
    const std::vector<CompareFlags>& sortCompareFlags,
    velox::memory::MemoryPool* pool,
    tsan_atomic<bool>* nonReclaimableSection,
    common::PrefixSortConfig prefixSortConfig,
    const common::SpillConfig* spillConfig,
    folly::Synchronized<velox::common::SpillStats>* spillStats)
    : input_(input),
      sortingKeys_(
          SpillState::makeSortingKeys(sortColumnIndices, sortCompareFlags)),
      sortCompareFlags_(sortCompareFlags),
      pool_(pool),
      nonReclaimableSection_(nonReclaimableSection),
      prefixSortConfig_(prefixSortConfig),
      spillConfig_(spillConfig),
      spillStats_(spillStats),
      sortedRows_(0, memory::StlAllocator<char*>(*pool)) {
  VELOX_CHECK_GE(input_->children().size(), sortCompareFlags_.size());
  VELOX_CHECK_GT(sortCompareFlags_.size(), 0);
  VELOX_CHECK_EQ(sortColumnIndices.size(), sortCompareFlags_.size());
  VELOX_CHECK_NOT_NULL(nonReclaimableSection_);

  // Sorted key columns.
  std::vector<TypePtr> sortedColumnTypes;
  sortedColumnTypes.reserve(sortColumnIndices.size());
  for (column_index_t i = 0; i < sortColumnIndices.size(); ++i) {
    columnMap_.emplace_back(IdentityProjection(i, sortColumnIndices.at(i)));
    sortedColumnTypes.emplace_back(input_->childAt(sortColumnIndices.at(i)));
  }

  // Vector index and row index columns.
  const auto numSortKeys = columnMap_.size();
  for (auto i = 0; i < indexType_->size(); ++i) {
    indexColumnMap_.emplace_back(numSortKeys + i, i);
  }
  data_ = std::make_unique<RowContainer>(
      sortedColumnTypes, indexType_->children(), pool_);
}

HybridSortBuffer::~HybridSortBuffer() {
  inputs_.clear();
  pool_->release();
}

void HybridSortBuffer::addInput(const VectorPtr& input) {
  velox::common::testutil::TestValue::adjust(
      "facebook::velox::exec::HybridSortBuffer::addInput", this);
  VELOX_CHECK(!noMoreInput_);
  ensureInputFits(input);

  VELOX_CHECK_EQ(input->encoding(), VectorEncoding::Simple::ROW);
  const SelectivityVector allRows(input->size());
  std::vector<char*> rows(input->size());
  for (int row = 0; row < input->size(); ++row) {
    rows[row] = data_->newRow();
  }

  // Stores the sort key columns.
  auto* inputRow = input->as<RowVector>();
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
      false, // isNull
      BIGINT(),
      inputs_.size() - 1);
  DecodedVector decoded;
  decoded.decode(*vectorIndex, allRows);
  const auto numSortKeys = columnMap_.size();
  data_->store(decoded, folly::Range(rows.data(), input->size()), numSortKeys);

  // Stores the row indices column.
  const auto rowIndex =
      BaseVector::create<FlatVector<int64_t>>(BIGINT(), input->size(), pool());
  for (int64_t i = 0; i < input->size(); ++i) {
    rowIndex->set(i, i);
  }
  decoded.decode(*rowIndex, allRows);
  data_->store(
      decoded, folly::Range(rows.data(), input->size()), numSortKeys + 1);

  numInputRows_ += allRows.size();
  numInputBytes_ += input->retainedSize();
}

void HybridSortBuffer::sortInput(uint64_t numRows) {
  sortedRows_.resize(numRows);
  RowContainerIterator iter;
  data_->listRows(&iter, numRows, sortedRows_.data());
  PrefixSort::sort(
      data_.get(), sortCompareFlags_, prefixSortConfig_, pool_, sortedRows_);
}

void HybridSortBuffer::noMoreInput() {
  velox::common::testutil::TestValue::adjust(
      "facebook::velox::exec::HybridSortBuffer::noMoreInput", this);
  VELOX_CHECK(!noMoreInput_);
  VELOX_CHECK_NULL(outputSpiller_);
  // It may trigger spill, make sure it's triggered before noMoreInput_ is set.
  ensureSortFits();

  noMoreInput_ = true;

  // No data.
  if (numInputRows_ == 0) {
    return;
  }
  estimatedOutputRowSize_ = numInputBytes_ / numInputRows_;

  if (inputSpiller_ == nullptr) {
    VELOX_CHECK_EQ(numInputRows_, data_->numRows());
    sortInput(numInputRows_);
  } else {
    // Spill the remaining in-memory state to disk if spilling has been
    // triggered on this sort buffer. This is to simplify query OOM prevention
    // when producing output as we don't support to spill during that stage as
    // for now.
    spill();
  }

  // Releases the unused memory reservation after procesing input.
  pool_->release();
}

RowVectorPtr HybridSortBuffer::getOutput(vector_size_t maxOutputRows) {
  SCOPE_EXIT {
    pool_->release();
  };

  VELOX_CHECK(noMoreInput_);

  // TODO: Track the vectors in 'inputs_' and evict any vector once all of its
  // rows have been copied to the output.
  if (numOutputRows_ == numInputRows_) {
    inputs_.clear();
    data_->clear();
    return nullptr;
  }
  VELOX_CHECK_GT(maxOutputRows, 0);
  VELOX_CHECK_GT(numInputRows_, numOutputRows_);
  const vector_size_t batchSize =
      std::min<uint64_t>(numInputRows_ - numOutputRows_, maxOutputRows);

  ensureOutputFits(batchSize);
  prepareOutput(batchSize);

  if (hasSpilled()) {
    getOutputWithSpill();
  } else {
    getOutputWithoutSpill();
  }

  return output_;
}

bool HybridSortBuffer::hasSpilled() const {
  if (inputSpiller_ != nullptr) {
    VELOX_CHECK_NULL(outputSpiller_);
    return true;
  }
  return outputSpiller_ != nullptr;
}

void HybridSortBuffer::spill() {
  VELOX_CHECK_NOT_NULL(
      spillConfig_,
      "spill config is null when HybridSortBuffer spill is called");

  // Check if sort buffer is empty or not, and skip spill if it is empty.
  if (data_->numRows() == 0) {
    return;
  }

  if (sortedRows_.empty()) {
    spillInput();
  } else {
    spillOutput();
  }
}

std::optional<uint64_t> HybridSortBuffer::estimateOutputRowSize() const {
  return estimatedOutputRowSize_;
}

void HybridSortBuffer::ensureInputFits(const VectorPtr& input) {
  // Check if spilling is enabled or not.
  if (spillConfig_ == nullptr) {
    return;
  }

  const int64_t numRows = data_->numRows();
  if (numRows == 0) {
    // 'data_' is empty. Nothing to spill.
    return;
  }

  auto [freeRows, outOfLineFreeBytes] = data_->freeSpace();
  const auto outOfLineBytes =
      data_->stringAllocator().retainedSize() - outOfLineFreeBytes;
  int64_t flatInputBytes{0};
  const auto inputRowVector = input->asUnchecked<RowVector>();
  for (const auto identity : columnMap_) {
    flatInputBytes +=
        inputRowVector->childAt(identity.outputChannel)->estimateFlatSize();
  }
  flatInputBytes += indexColumnMap_.size() * sizeof(int64_t) * input->size();

  // Test-only spill path.
  if (numRows > 0 && testingTriggerSpill(pool_->name())) {
    spill();
    return;
  }

  const auto currentMemoryUsage = pool_->usedBytes();
  const auto minReservationBytes =
      currentMemoryUsage * spillConfig_->minSpillableReservationPct / 100;
  const auto availableReservationBytes = pool_->availableReservation();
  const int64_t estimatedIncrementalBytes =
      data_->sizeIncrement(input->size(), outOfLineBytes ? flatInputBytes : 0) +
      input->retainedSize();
  if (availableReservationBytes > minReservationBytes) {
    // If we have enough free rows for input rows and enough variable length
    // free space for the vector's flat size, no need for spilling.
    if (freeRows > input->size() &&
        (outOfLineBytes == 0 || outOfLineFreeBytes >= flatInputBytes)) {
      return;
    }

    // If the current available reservation in memory pool is 2X the
    // estimatedIncrementalBytes, no need to spill.
    if (availableReservationBytes > 2 * estimatedIncrementalBytes) {
      return;
    }
  }

  // Try reserving targetIncrementBytes more in memory pool, if succeed, no
  // need to spill.
  const auto targetIncrementBytes = std::max<int64_t>(
      estimatedIncrementalBytes * 2,
      currentMemoryUsage * spillConfig_->spillableReservationGrowthPct / 100);
  {
    memory::ReclaimableSectionGuard guard(nonReclaimableSection_);
    if (pool_->maybeReserve(targetIncrementBytes)) {
      return;
    }
  }
  LOG(WARNING) << "Failed to reserve " << succinctBytes(targetIncrementBytes)
               << " for memory pool " << pool()->name()
               << ", usage: " << succinctBytes(pool()->usedBytes())
               << ", reservation: " << succinctBytes(pool()->reservedBytes());
}

void HybridSortBuffer::ensureOutputFits(vector_size_t batchSize) {
  VELOX_CHECK_GT(batchSize, 0);
  // Check if spilling is enabled or not.
  if (spillConfig_ == nullptr) {
    return;
  }

  // Test-only spill path.
  if (testingTriggerSpill(pool_->name())) {
    spill();
    return;
  }

  if (!estimatedOutputRowSize_.has_value() || hasSpilled()) {
    return;
  }

  const uint64_t outputBufferSizeToReserve =
      estimatedOutputRowSize_.value() * batchSize * 1.2;
  {
    memory::ReclaimableSectionGuard guard(nonReclaimableSection_);
    if (pool_->maybeReserve(outputBufferSizeToReserve)) {
      return;
    }
  }
  LOG(WARNING) << "Failed to reserve "
               << succinctBytes(outputBufferSizeToReserve)
               << " for memory pool " << pool_->name()
               << ", usage: " << succinctBytes(pool_->usedBytes())
               << ", reservation: " << succinctBytes(pool_->reservedBytes());
}

void HybridSortBuffer::ensureSortFits() {
  // Check if spilling is enabled or not.
  if (spillConfig_ == nullptr) {
    return;
  }

  // Test-only spill path.
  if (testingTriggerSpill(pool_->name())) {
    spill();
    return;
  }

  if (numInputRows_ == 0 || inputSpiller_ != nullptr) {
    return;
  }

  // The memory for std::vector sorted rows and prefix sort required buffer.
  const auto numBytesToReserve =
      numInputRows_ * sizeof(char*) +
      PrefixSort::maxRequiredBytes(
          data_.get(), sortCompareFlags_, prefixSortConfig_, pool_);
  {
    memory::ReclaimableSectionGuard guard(nonReclaimableSection_);
    if (pool_->maybeReserve(numBytesToReserve)) {
      return;
    }
  }

  LOG(WARNING) << fmt::format(
      "Failed to reserve {} for memory pool {}, usage: {}, reservation: {}",
      succinctBytes(numBytesToReserve),
      pool_->name(),
      succinctBytes(pool_->usedBytes()),
      succinctBytes(pool_->reservedBytes()));
}

void HybridSortBuffer::runSpill(
    NoRowContainerSpiller* spiller,
    int64_t numInputs,
    uint64_t offset) const {
  RowVectorPtr output;
  RowVectorPtr indexOutput;
  int64_t numOutputs{0};
  constexpr int32_t kTargetBatchRows = 64;
  while (numOutputs < numInputs) {
    int64_t batchSize =
        std::min<int64_t>(kTargetBatchRows, numInputs - numOutputs);
    prepareOutputVector(output, input_, batchSize);
    prepareOutputVector(indexOutput, indexType_, batchSize);
    gatherCopyOutput(output, indexOutput, sortedRows_, offset);
    VELOX_CHECK_EQ(batchSize, output->size());
    numOutputs += batchSize;
    offset += batchSize;
    spiller->spill(SpillPartitionId{0}, output);
  }
  VELOX_CHECK_EQ(numOutputs, numInputs);
}

void HybridSortBuffer::spillInput() {
  VELOX_CHECK_LT(!!(inputSpiller_ == nullptr) + !!noMoreInput_, 2);
  inputSpiller_ = std::make_unique<MergeSpiller>(
      input_,
      std::nullopt,
      HashBitRange{},
      sortingKeys_,
      spillConfig_,
      spillStats_);

  sortInput(data_->numRows());
  runSpill(inputSpiller_.get(), data_->numRows(), 0);
  finishInputSpill();
  inputs_.clear();
  data_->clear();
  sortedRows_.clear();
  sortedRows_.shrink_to_fit();
}

void HybridSortBuffer::spillOutput() {
  if (hasSpilled()) {
    // Already spilled.
    return;
  }
  if (numOutputRows_ == sortedRows_.size()) {
    // All the output has been produced.
    return;
  }

  outputSpiller_ = std::make_unique<NoRowContainerSpiller>(
      input_, std::nullopt, HashBitRange{}, spillConfig_, spillStats_);
  runSpill(
      outputSpiller_.get(), numInputRows_ - numOutputRows_, numOutputRows_);
  inputs_.clear();
  data_->clear();
  sortedRows_.clear();
  sortedRows_.shrink_to_fit();
  // Finish right after spilling as the output spiller only spills at most
  // once.
  finishOutputSpill();
}

void HybridSortBuffer::prepareOutputVector(
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

void HybridSortBuffer::prepareOutput(vector_size_t batchSize) {
  prepareOutputVector(output_, input_, batchSize);
  prepareOutputVector(indexOutput_, indexType_, batchSize);

  if (hasSpilled()) {
    spillSources_.resize(batchSize);
    spillSourceRows_.resize(batchSize);
    prepareOutputWithSpill();
  }

  VELOX_CHECK_GT(output_->size(), 0);
  VELOX_CHECK_LE(output_->size() + numOutputRows_, numInputRows_);
}

void HybridSortBuffer::gatherCopyOutput(
    const RowVectorPtr& output,
    const RowVectorPtr& indexOutput,
    const std::vector<char*, memory::StlAllocator<char*>>& sortedRows,
    uint64_t offset) const {
  for (const auto& columnProjection : indexColumnMap_) {
    data_->extractColumn(
        sortedRows.data() + offset,
        indexOutput->size(),
        columnProjection.inputChannel,
        indexOutput->childAt(columnProjection.outputChannel));
  }

  std::vector<const RowVector*> sources;
  sources.reserve(indexOutput->size());
  std::vector<vector_size_t> sourceIndices;
  sourceIndices.reserve(indexOutput->size());

  // Extracts vector indices.
  const SelectivityVector rows{indexOutput->size()};
  DecodedVector decoded;
  decoded.decode(*indexOutput->childAt(0), rows);
  const VectorReader<int64_t> vectorIndexReader(&decoded);
  rows.applyToSelected([&](vector_size_t row) {
    const auto index = vectorIndexReader.readNullFree(row);
    sources.push_back(inputs_[index].get());
  });

  // Extracts row indices.
  decoded.decode(*indexOutput->childAt(1), rows);
  const VectorReader<int64_t> rowIndexReader(&decoded);
  rows.applyToSelected([&](vector_size_t row) {
    const auto index = rowIndexReader.readNullFree(row);
    sourceIndices.push_back(index);
  });

  gatherCopy(output.get(), 0, output->size(), sources, sourceIndices);
}

void HybridSortBuffer::getOutputWithoutSpill() {
  VELOX_DCHECK_EQ(numInputRows_, sortedRows_.size());
  gatherCopyOutput(output_, indexOutput_, sortedRows_, numOutputRows_);
  numOutputRows_ += output_->size();
}

void HybridSortBuffer::getOutputWithSpill() {
  if (spillMerger_ != nullptr) {
    VELOX_DCHECK_EQ(sortedRows_.size(), 0);

    int32_t outputRow = 0;
    int32_t outputSize = 0;
    bool isEndOfBatch = false;
    while (outputRow + outputSize < output_->size()) {
      SpillMergeStream* stream = spillMerger_->next();
      VELOX_CHECK_NOT_NULL(stream);

      spillSources_[outputSize] = &stream->current();
      spillSourceRows_[outputSize] = stream->currentIndex(&isEndOfBatch);
      ++outputSize;
      if (FOLLY_UNLIKELY(isEndOfBatch)) {
        // The stream is at end of input batch. Need to copy out the rows before
        // fetching next batch in 'pop'.
        gatherCopy(
            output_.get(),
            outputRow,
            outputSize,
            spillSources_,
            spillSourceRows_,
            {});
        outputRow += outputSize;
        outputSize = 0;
      }
      // Advance the stream.
      stream->pop();
    }
    VELOX_CHECK_EQ(outputRow + outputSize, output_->size());

    if (FOLLY_LIKELY(outputSize != 0)) {
      gatherCopy(
          output_.get(),
          outputRow,
          outputSize,
          spillSources_,
          spillSourceRows_,
          {});
    }
  } else {
    VELOX_CHECK_NOT_NULL(batchStreamReader_);
    RowVectorPtr output;
    batchStreamReader_->nextBatch(output);
    output_ = std::move(output);
  }

  numOutputRows_ += output_->size();
}

void HybridSortBuffer::finishInputSpill() {
  VELOX_CHECK_NULL(spillMerger_);
  SpillPartitionSet spillPartitionSet;
  VELOX_CHECK_NOT_NULL(inputSpiller_);
  VELOX_CHECK_NULL(outputSpiller_);
  VELOX_CHECK(!inputSpiller_->finalized());
  inputSpiller_->finishSpill(spillPartitionSet);
  VELOX_CHECK_EQ(spillPartitionSet.size(), 1);
  inputSpillFileGroups_.push_back(spillPartitionSet.begin()->second->files());
}

void HybridSortBuffer::finishOutputSpill() {
  VELOX_CHECK_NULL(spillMerger_);
  VELOX_CHECK(outputSpillPartitionSet_.empty());
  VELOX_CHECK_NULL(inputSpiller_);
  VELOX_CHECK_NOT_NULL(outputSpiller_);
  VELOX_CHECK(!outputSpiller_->finalized());
  outputSpiller_->finishSpill(outputSpillPartitionSet_);
  VELOX_CHECK_EQ(outputSpillPartitionSet_.size(), 1);
}

void HybridSortBuffer::prepareOutputWithSpill() {
  VELOX_CHECK(hasSpilled());
  if (inputSpiller_ != nullptr) {
    if (spillMerger_ != nullptr) {
      VELOX_CHECK(inputSpillFileGroups_.empty());
      return;
    }

    std::vector<std::unique_ptr<SpillMergeStream>> spillStreams;
    int index = 0;
    for (const auto& spillFiles : inputSpillFileGroups_) {
      std::vector<std::unique_ptr<SpillReadFile>> spillReadFiles;
      spillReadFiles.reserve(spillFiles.size());
      for (const auto& spillFile : spillFiles) {
        spillReadFiles.emplace_back(
            SpillReadFile::create(
                spillFile, spillConfig_->readBufferSize, pool_, spillStats_));
      }
      auto stream = ConcatFilesSpillMergeStream::create(
          index++, std::move(spillReadFiles));
      spillStreams.push_back(std::move(stream));
    }
    inputSpillFileGroups_.clear();
    spillMerger_ = std::make_unique<TreeOfLosers<SpillMergeStream>>(
        std::move(spillStreams));
  } else {
    VELOX_CHECK_NOT_NULL(outputSpiller_);
    if (batchStreamReader_ != nullptr) {
      VELOX_CHECK(outputSpillPartitionSet_.empty());
      return;
    }
    VELOX_CHECK_EQ(outputSpillPartitionSet_.size(), 1);
    batchStreamReader_ =
        outputSpillPartitionSet_.begin()->second->createUnorderedReader(
            spillConfig_->readBufferSize, pool(), spillStats_);
  }
  outputSpillPartitionSet_.clear();
}
} // namespace facebook::velox::exec
