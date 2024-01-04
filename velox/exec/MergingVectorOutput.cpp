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
#include "velox/exec/OperatorUtils.h"

namespace facebook::velox::exec {

MergingVectorOutput::MergingVectorOutput(
    velox::memory::MemoryPool* pool,
    int64_t preferredOutputBatchBytes,
    int32_t preferredOutputBatchRows,
    int32_t minOutputBatchRows)
    : pool_(pool),
      preferredOutputBatchBytes_(preferredOutputBatchBytes),
      preferredOutputBatchRows_(preferredOutputBatchRows),
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
  const auto numInput = vector->size();
  if (numInput == 0) {
    return;
  }

  // Avoid memory copying for pages that are big enough
  if (numInput >= minOutputBatchRows_) {
    outputQueue_.push(std::move(vector));
    return;
  }

  // Currently, we only merge flat and dictionary vector, whether merging other
  // encodings should depend on benchmark or adaptivity.
  for (auto children : vector->children()) {
    if (!VectorEncoding::isFlat(children->encoding()) &&
        !VectorEncoding::isDictionary(children->encoding())) {
      outputQueue_.push(std::move(vector));
      return;
    }
  }

  buffer(std::move(vector));
}

void MergingVectorOutput::buffer(RowVectorPtr vector) {
  const auto numInput = vector->size();

  // In order to prevent vector.resize() cause memory copy, the initial value
  // of bufferInputs_ row count is preferredOutputBatchRows_ and cannot exceed
  // preferredOutputBatchRows_.
  if (bufferRows_ + numInput > preferredOutputBatchRows_) {
    flush();
  }

  if (!bufferInputs_) {
    bufferInputs_ = BaseVector::create<RowVector>(
        vector->type(), preferredOutputBatchRows_, pool_);
  }

  mergeVector(bufferInputs_, vector, bufferRows_, numInput);

  bufferRows_ += numInput;
  bufferBytes_ += vector->estimateFlatSize();

  if (bufferRows_ >= preferredOutputBatchRows_ ||
      bufferBytes_ >= preferredOutputBatchBytes_) {
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
