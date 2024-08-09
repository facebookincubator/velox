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

#include "velox/exec/MergingVector.h"
#include "velox/exec/OperatorUtils.h"

namespace facebook::velox::exec {

MergingVector::MergingVector(
    int32_t preferredOutputBatchRows,
    int64_t preferredOutputBatchBytes,
    int32_t maxMergingBatchRows,
    int64_t maxMergingBatchBytes,
    bool preserveOrder,
    bool reuseBuffer,
    velox::memory::MemoryPool* pool)
    : preferredOutputBatchRows_(preferredOutputBatchRows),
      preferredOutputBatchBytes_(preferredOutputBatchBytes),
      maxMergingBatchRows_(
          std::min(maxMergingBatchRows, preferredOutputBatchRows)),
      maxMergingBatchBytes_(
          std::min(maxMergingBatchBytes, preferredOutputBatchBytes)),
      preserveOrder_(preserveOrder),
      reuseBuffer_(reuseBuffer),
      pool_(pool) {}

RowVectorPtr MergingVector::getOutput(bool noMoreInput) {
  if (noMoreInput) {
    flush();
  }

  if (outputQueue_.empty()) {
    return nullptr;
  }

  const auto output = outputQueue_.front();
  outputQueue_.pop();
  return output;
}

void MergingVector::addInput(RowVectorPtr& input) {
  // To prevent consuming more memory, make sure that there is no ready-to-go
  // vectors.
  VELOX_CHECK(outputQueue_.empty());

  if (nullptr == input) {
    return;
  }

  const auto numInput = input->size();
  if (numInput == 0) {
    return;
  }

  uint64_t bytesInput;
  // Avoid memory copying for pages that are big enough
  if (numInput >= maxMergingBatchRows_ ||
      (bytesInput = input->estimateFlatSize()) >= maxMergingBatchBytes_) {
    if (preserveOrder_) {
      // In order to preserve the order of the rows, we need to flush first.
      flush();
    }
    outputQueue_.push(std::move(input));
    return;
  }

  buffer(input, bytesInput);
}

void MergingVector::reuseIfNeed(RowVectorPtr& reuse) {
  // Only reuse the RowVectorPtr when it was created by MergingVector.
  if (reuseBuffer_ && createdBuffers_.find(reuse) != createdBuffers_.end()) {
    reuseBuffers_.push_back(std::move(reuse));
  }
}

void MergingVector::buffer(RowVectorPtr& vector, uint64_t bytesInput) {
  const auto numInput = vector->size();

  // The initial row count of buffer_ is preferredOutputBatchRows_. In order to
  // prevent memory copy by resize() the buffer_, the row count of buffer_
  // should not exceed the initial row count.
  if (numBufferRows_ + numInput > preferredOutputBatchRows_ ||
      bufferBytes_ + bytesInput > preferredOutputBatchBytes_) {
    flush();
  }

  if (isFirstVector_) {
    isFirstVector_ = false;
    lastVector_ = std::move(vector);
    lastVector_->loadedVector();
    numBufferRows_ += numInput;
    bufferBytes_ += bytesInput;
    return;
  }

  if (!buffer_) {
    if (reuseBuffers_.size() > 0) {
      buffer_ = reuseBuffers_.back();
      reuseBuffers_.pop_back();
    } else {
      buffer_ = BaseVector::create<RowVector>(
          vector->type(), preferredOutputBatchRows_, pool_);
      if (reuseBuffer_) {
        createdBuffers_.insert(buffer_);
        VELOX_CHECK(createdBuffers_.size() <= MAX_REUSE_BUFFER_SIZE);
      }
    }

    mergeVector(buffer_, lastVector_, 0, lastVector_->size());
    lastVector_ = nullptr;
  }
  mergeVector(buffer_, vector, numBufferRows_, numInput);

  numBufferRows_ += numInput;
  bufferBytes_ += bytesInput;
}

void MergingVector::mergeVector(
    RowVectorPtr& dest,
    const RowVectorPtr& src,
    int32_t destRows,
    int32_t srcRows) {
  dest->resize(destRows + srcRows);
  const auto& children = src->children();
  for (auto i = 0; i < children.size(); i++) {
    const auto& vectorPtr = dest->childAt(i);
    vectorPtr->copy(children[i].get(), destRows, 0, srcRows);
  }
}

void MergingVector::flush() {
  if (lastVector_) {
    outputQueue_.push(lastVector_);
    lastVector_ = nullptr;
  }

  if (buffer_) {
    outputQueue_.push(buffer_);
    buffer_ = nullptr;
  }
  isFirstVector_ = true;
  numBufferRows_ = 0;
  bufferBytes_ = 0;
}
} // namespace facebook::velox::exec
