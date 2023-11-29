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

#pragma once

#include "velox/type/Type.h"
#include "velox/vector/ComplexVector.h"

namespace facebook::velox::exec {

class MergingVectorOutput {
 public:
  MergingVectorOutput(
      velox::memory::MemoryPool* pool,
      int64_t preferredOutputBatchBytes,
      int32_t preferredOutputRows,
      int64_t minOutputBatchBytes,
      int32_t minOutputBatchRows);

  void addVector(RowVectorPtr input);

  RowVectorPtr getOutput();

  bool needsInput() const {
    return !noMoreInput_ && outputQueue_.empty();
  }

  void noMoreInput() {
    noMoreInput_ = true;
  }

  bool isFinished() {
    return noMoreInput_ && bufferRows_ == 0 && outputQueue_.empty();
  }

 private:
  // Push the buffer vector to the queue, and reset bufferInputs_,
  // bufferBytes_ and bufferRows_.
  void flush();

  // If the input vector is small enough, copy it to the buffer vector.
  void buffer(RowVectorPtr input);

  velox::memory::MemoryPool* pool_;

  // The preferred output bytes of the bufferInputs_.
  int64_t preferredOutputBatchBytes_;

  // The preferred output row count of the bufferInputs_.
  int32_t preferredOutputBatchRows_;

  // If the input vector bytes is larger than minOutputBatchBytes_, flush it to
  // the outputQueue_ directly.
  int64_t minOutputBatchBytes_;

  // If the input vector row count is larger than minOutputBatchBytes_, flush
  // it to the outputQueue_ directly.
  int32_t minOutputBatchRows_;

  // The RowVectorPtr queue where bufferInputs_ flush to.
  std::queue<RowVectorPtr> outputQueue_;

  // The vector buffer the small input vector.
  RowVectorPtr bufferInputs_;

  // The buffer row count in bufferInputs_.
  int32_t bufferRows_ = 0;

  // The buffer bytes in bufferInputs_.
  int64_t bufferBytes_ = 0;

  bool noMoreInput_ = false;
};
} // namespace facebook::velox::exec
