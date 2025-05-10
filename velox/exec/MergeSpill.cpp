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

#include "velox/exec/MergeSpill.h"

namespace facebook::velox::exec {
namespace {
const auto kSpillPartitionId = SpillPartitionId{0};
}

MergeSpiller::MergeSpiller(
    const RowTypePtr& type,
    velox::memory::MemoryPool* const pool,
    const std::vector<std::pair<column_index_t, CompareFlags>>& sortingKeys,
    const common::SpillConfig* spillConfig,
    folly::Synchronized<velox::common::SpillStats>* spillStats)
    : type_(type),
      pool_(pool),
      sortingKeys_(sortingKeys),
      spillConfig_(spillConfig),
      spillStats_(spillStats),
      inputSpiller_(std::make_unique<NoRowContainerSpiller>(
          type_,
          std::nullopt,
          HashBitRange{},
          sortingKeys_,
          spillConfig_,
          spillStats_)) {}

void MergeSpiller::addInput(const RowVectorPtr& vector) {
  numSpillRows_ += vector->size();
  // Ensure vector are lazy loaded before spilling.
  for (auto i = 0; i < vector->childrenSize(); ++i) {
    vector->childAt(i)->loadedVector();
  }
  inputSpiller_->spill(kSpillPartitionId, vector);
}

void MergeSpiller::finishSpill(bool lastRun) {
  VELOX_CHECK_NOT_NULL(inputSpiller_);
  SpillPartitionSet partitionSet;
  inputSpiller_->finishSpill(partitionSet);
  VELOX_CHECK_EQ(partitionSet.size(), 1);
  spillFilesLists_.emplace_back(partitionSet.cbegin()->second->files());
  if (!lastRun) {
    inputSpiller_ = std::make_unique<NoRowContainerSpiller>(
        type_, std::nullopt, HashBitRange{}, sortingKeys_, spillConfig_, spillStats_);
  }
}

uint64_t MergeSpiller::numSpillRows() const {
  return numSpillRows_;
}

std::vector<SpillFiles> MergeSpiller::spillFiles() const {
  return spillFilesLists_;
}

MergeBuffer::MergeBuffer(
    const RowTypePtr& type,
    velox::memory::MemoryPool* const pool,
    const std::vector<SpillFiles>& spillFilesList,
    uint64_t numInputRows,
    uint64_t readBufferSize,
    folly::Synchronized<velox::common::SpillStats>* spillStats)
    : type_(type),
      pool_(pool),
      numInputRows_(numInputRows),
      spillMerger_(
          createSortMergeReader(spillFilesList, readBufferSize, spillStats)) {}

RowVectorPtr MergeBuffer::getOutput(vector_size_t maxOutputRows) {
  // Finished.
  if (numOutputRows_ == numInputRows_) {
    return nullptr;
  }
  VELOX_CHECK_GT(maxOutputRows, 0);
  VELOX_CHECK_GT(numInputRows_, numOutputRows_);
  const vector_size_t batchSize =
      std::min<uint64_t>(numInputRows_ - numOutputRows_, maxOutputRows);
  prepareOutput(batchSize);
  getOutputInternal();
  return output_;
}

void MergeBuffer::prepareOutput(vector_size_t batchSize) {
  if (output_ != nullptr) {
    VectorPtr output = std::move(output_);
    BaseVector::prepareForReuse(output, batchSize);
    output_ = std::static_pointer_cast<RowVector>(output);
  } else {
    output_ = std::static_pointer_cast<RowVector>(
        BaseVector::create(type_, batchSize, pool_));
  }

  for (const auto& child : output_->children()) {
    child->resize(batchSize);
  }

  spillSources_.resize(batchSize);
  spillSourceRows_.resize(batchSize);

  VELOX_CHECK_GT(output_->size(), 0);
  VELOX_CHECK_LE(output_->size() + numOutputRows_, numInputRows_);
}

void MergeBuffer::getOutputInternal() {
  VELOX_CHECK_NOT_NULL(spillMerger_);
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
  numOutputRows_ += output_->size();
}

std::unique_ptr<TreeOfLosers<SpillMergeStream>>
MergeBuffer::createSortMergeReader(
    const std::vector<SpillFiles>& spillFilesLists,
    uint64_t readBufferSize,
    folly::Synchronized<velox::common::SpillStats>* spillStats) const {
  std::vector<std::unique_ptr<SpillMergeStream>> streams;
  streams.reserve(spillFilesLists.size());
  for (size_t id = 0; id < spillFilesLists.size(); ++id) {
    auto& spillFiles = spillFilesLists[id];
    // TODO: Lazy open spill read file in 'SortedFileSpillStream'.
    std::vector<std::unique_ptr<SpillReadFile>> spillReadFiles;
    spillReadFiles.reserve(spillFiles.size());
    for (const auto& spillFile : spillFiles) {
      spillReadFiles.emplace_back(
          SpillReadFile::create(spillFile, readBufferSize, pool_, spillStats));
    }
    streams.push_back(
        ConcatFilesSpillMergeStream::create(id, std::move(spillReadFiles)));
  }
  return std::make_unique<TreeOfLosers<SpillMergeStream>>(std::move(streams));
}

} // namespace facebook::velox::exec
