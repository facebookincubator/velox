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

#include "velox/exec/MergingVectorOutput.h"

namespace facebook::velox::exec {

MergingVectorOutput::MergingVectorOutput(
    velox::memory::MemoryPool* pool,
    int64_t preferredOutputBatchBytes,
    int32_t preferredOutputBatchRows,
    int64_t minOutputBatchBytes,
    int32_t minOutputBatchRows)
    : pool_(pool),
      preferredOutputBatchBytes_(preferredOutputBatchBytes),
      preferredOutputBatchRows_(preferredOutputBatchRows),
      minOutputBatchBytes_(minOutputBatchBytes),
      minOutputBatchRows_(minOutputBatchRows) {}

RowVectorPtr MergingVectorOutput::getOutput() {
  if (noMoreInput_) {
    flush();
  }
  if (outputQueue_.size() > 0) {
    const auto res = outputQueue_.front();
    outputQueue_.pop();
    return res;
  }
  return nullptr;
}

void MergingVectorOutput::addVector(RowVectorPtr vector) {
  const auto inputRows = vector->size();
  if (inputRows == 0) {
    return;
  }

  // Avoid memory copying for pages that are big enough
  if (vector->estimateFlatSize() >= minOutputBatchBytes_ ||
      inputRows >= minOutputBatchRows_) {
    flush();
    outputQueue_.push(std::move(vector));
    return;
  }

  buffer(std::move(vector));
}

void MergingVectorOutput::buffer(RowVectorPtr vector) {
  const auto inputRows = vector->size();
  const auto& childrens = vector->children();

  // In order to prevent vector.resize() cause memory copy, the initial value
  // of bufferInputs_ row count is preferredOutputBatchRows_ and cannot exceed
  // preferredOutputBatchRows_.
  if (bufferRows_ + inputRows > preferredOutputBatchRows_) {
    flush();
  }

  if (!bufferInputs_) {
    bufferInputs_ = BaseVector::create<RowVector>(
        vector->type(), preferredOutputBatchRows_, pool_);
    for (auto i = 0; i < childrens.size(); i++) {
      const auto& vectorPtr = bufferInputs_->childAt(i);
      vectorPtr->resize(bufferRows_ + inputRows);
    }
  }

  bufferInputs_->resize(bufferRows_ + inputRows);
  for (auto i = 0; i < childrens.size(); i++) {
    const auto& vectorPtr = bufferInputs_->childAt(i);
    vectorPtr->resize(bufferRows_ + inputRows);
    vectorPtr->copy(childrens[i].get(), bufferRows_, 0, inputRows);
  }

  bufferRows_ += inputRows;
  bufferBytes_ += vector->estimateFlatSize();

  if (bufferBytes_ >= preferredOutputBatchBytes_ ||
      bufferRows_ >= preferredOutputBatchRows_) {
    flush();
  }
}

void MergingVectorOutput::flush() {
  if (bufferRows_ > 0) {
    outputQueue_.push(bufferInputs_);
    bufferInputs_ = nullptr;
    bufferRows_ = 0;
    bufferBytes_ = 0;
  }
}
} // namespace facebook::velox::exec
