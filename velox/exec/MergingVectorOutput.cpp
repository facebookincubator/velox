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

RowVectorPtr MergingVectorOutput::getOutput(bool noMoreInput) {
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

bool MergingVectorOutput::canMerge(VectorPtr vector) {
  if (vector->encoding() == VectorEncoding::Simple::LAZY) {
    auto lazyVector = vector->as<const LazyVector>();
    if (!lazyVector->isLoaded()) {
      return false;
    }
  }

  if (vector->encoding() == VectorEncoding::Simple::ROW) {
    auto rowVector = vector->as<const RowVector>();
    for (auto children : rowVector->children()) {
      if (!canMerge(children)) {
        return false;
      }
    }
  }

  if (vector->encoding() == VectorEncoding::Simple::MAP) {
    auto mapVector = vector->as<const MapVector>();
    if (!canMerge(mapVector->mapKeys()) || !canMerge(mapVector->mapValues())) {
      return false;
    }
  }

  if (vector->encoding() == VectorEncoding::Simple::ARRAY) {
    auto arrayVector = vector->as<const ArrayVector>();
    return canMerge(arrayVector->elements());
  }

  if (vector->encoding() == VectorEncoding::Simple::DICTIONARY ||
      vector->encoding() == VectorEncoding::Simple::CONSTANT ||
      vector->encoding() == VectorEncoding::Simple::SEQUENCE) {
    return canMerge(vector->valueVector());
  }
}

void MergingVectorOutput::addVector(RowVectorPtr vector) {
  const auto numInput = vector->size();
  if (numInput == 0) {
    return;
  }

  // Avoid memory copying for pages that are big enough
  if (numInput >= minOutputBatchRows_) {
    // In order to preserve the order of the rows, we need to flush first.
    flush();
    outputQueue_.push(std::move(vector));
    return;
  }

  // we don't merge lazy encoding.
  if (canMerge(vector)) {
    outputQueue_.push(std::move(vector));
    return;
  }

  buffer(std::move(vector));
}

void MergingVectorOutput::buffer(RowVectorPtr vector) {
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

void MergingVectorOutput::flush() {
  if (numBufferRows_ > 0) {
    outputQueue_.push(bufferInputs_);
    bufferInputs_ = nullptr;
    numBufferRows_ = 0;
    bufferBytes_ = 0;
  }
}
} // namespace facebook::velox::exec
