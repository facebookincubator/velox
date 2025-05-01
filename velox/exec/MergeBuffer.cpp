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

#include "velox/exec/MergeBuffer.h"

#include <utility>
#include "velox/exec/MemoryReclaimer.h"
#include "velox/exec/OperatorUtils.h"

namespace facebook::velox::exec {
namespace {
const auto kSpillPartitionId = SpillPartitionId{0};
}

MergeBuffer::MergeBuffer(
    RowTypePtr type,
    velox::memory::MemoryPool* const pool,
    common::SpillConfig spillConfig,
    folly::Synchronized<velox::common::SpillStats>* spillStats)
    : type_(std::move(type)),
      pool_(pool),
      spillConfig_(std::move(spillConfig)),
      spillStats_(spillStats),
      inputSpiller_(std::make_unique<NoRowContainerSpiller>(
          type_,
          std::nullopt,
          HashBitRange{},
          &spillConfig_,
          spillStats_)) {}

void MergeBuffer::addInput(RowVectorPtr input) {
  numInputRows_ += input->size();
  // Ensure vector are lazy loaded before spilling.
  for (auto i = 0; i < input->childrenSize(); ++i) {
    input->childAt(i)->loadedVector();
  }
  inputSpiller_->spill(kSpillPartitionId, input);
}

void MergeBuffer::noMoreInput() {
  VELOX_CHECK(!noMoreInput_);
  noMoreInput_ = true;
  finishSpill();
}

RowVectorPtr MergeBuffer::getOutput(vector_size_t maxOutputRows) {
  VELOX_CHECK(noMoreInput_);

  if (numOutputRows_ == numInputRows_) {
    return nullptr;
  }
  VELOX_CHECK_GT(maxOutputRows, 0);
  VELOX_CHECK_GT(numInputRows_, numOutputRows_);
  const vector_size_t batchSize =
      std::min<uint64_t>(numInputRows_ - numOutputRows_, maxOutputRows);
  prepareOutput(batchSize);
  getOutputWithSpill();
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

  for (auto& child : output_->children()) {
    child->resize(batchSize);
  }

  spillSources_.resize(batchSize);
  spillSourceRows_.resize(batchSize);
  prepareOutputWithSpill();

  VELOX_CHECK_GT(output_->size(), 0);
  VELOX_CHECK_LE(output_->size() + numOutputRows_, numInputRows_);
}

void MergeBuffer::getOutputWithSpill() {
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

void MergeBuffer::finishSpill() {
  VELOX_CHECK_NULL(spillMerger_);
  VELOX_CHECK(!inputSpiller_->finalized());
  inputSpiller_->finishSpill(spillPartitionSet_);
  VELOX_CHECK_EQ(spillPartitionSet_.size(), 1);
}
} // namespace facebook::velox::exec
