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

#include "velox/exec/MergingVectorInput.h"
#include "velox/exec/OperatorUtils.h"

namespace facebook::velox::exec {

MergingVectorInput::MergingVectorInput(
    velox::memory::MemoryPool* pool,
    int64_t preferredOutputBatchBytes,
    int32_t preferredOutputBatchRows,
    int32_t minOutputRows)
    : pool_(pool),
      preferredOutputBatchBytes_(preferredOutputBatchBytes),
      preferredOutputBatchRows_(preferredOutputBatchRows),
      minOutputRows_(minOutputRows) {}

RowVectorPtr MergingVectorInput::getVector(bool noMoreInput) {
  if (noMoreInput) {
    flush();
  }
  if (outputQueue_.size() > 0) {
    const auto res = outputQueue_.front();
    outputQueue_.pop();
    return res;
  }
  return nullptr;
}

void MergingVectorInput::addVector(RowVectorPtr vector) {
  if (nullptr == vector) {
    return;
  }

  const auto numInput = vector->size();
  if (numInput == 0) {
    return;
  }

  // Avoid memory copying for pages that are big enough
  if (numInput >= minOutputRows_) {
    // In order to preserve the order of the rows, we need to flush first.
    flush();
    outputQueue_.push(std::move(vector));
    return;
  }

  buffer(std::move(vector));
}

void MergingVectorInput::buffer(RowVectorPtr vector) {
  const auto numInput = vector->size();

  // In order to prevent vector.resize() cause memory copy, the initial value
  // of bufferInputs_ row count is preferredOutputBatchRows_ and cannot exceed
  // preferredOutputBatchRows_.
  if (numBufferRows_ + numInput > preferredOutputBatchRows_) {
    flush();
  }

  if (!bufferInputs_) {
    bufferInputs_ = BaseVector::create<RowVector>(
        vector->type(), preferredOutputBatchRows_, pool_);
  }

  mergeVector(bufferInputs_, vector, numBufferRows_, numInput);

  numBufferRows_ += numInput;
  bufferBytes_ += vector->estimateFlatSize();

  if (numBufferRows_ >= preferredOutputBatchRows_ ||
      bufferBytes_ >= preferredOutputBatchBytes_) {
    flush();
  }
}

void MergingVectorInput::mergeVector(
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

void MergingVectorInput::flush() {
  if (numBufferRows_ > 0) {
    outputQueue_.push(bufferInputs_);
    bufferInputs_ = nullptr;
    numBufferRows_ = 0;
    bufferBytes_ = 0;
  }
}
} // namespace facebook::velox::exec
