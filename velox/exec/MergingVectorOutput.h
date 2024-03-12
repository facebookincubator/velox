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
      int32_t minOutputBatchRows);

  /// Adds input vector. It will be merged into a big vector if smaller than
  /// minOutputBatchRows.
  /// @param input input vector.
  void addVector(RowVectorPtr input);

  /// Returns a RowVector after merging, or return nullptr if we haven't
  /// accumulated enough just yet.
  /// @return RowVector or nullptr.
  /// @param noMoreInput whether more data is expected to be added via
  /// addVector.
  RowVectorPtr getOutput(bool noMoreInput);

 private:
  // Push the buffer vector to the queue, and reset bufferInputs_,
  // bufferBytes_ and numBufferRows_.
  void flush();

  // If the input vector is small enough, copy it to the buffer vector.
  void buffer(RowVectorPtr input);

  bool canMerge(VectorPtr vector);

  velox::memory::MemoryPool* pool_;

  // The preferred output bytes of the bufferInputs_.
  const int64_t preferredOutputBatchBytes_;

  // The preferred output row count of the bufferInputs_.
  const int32_t preferredOutputBatchRows_;

  // If the input vector row count is larger than minOutputBatchBytes_, flush
  // it to the outputQueue_ directly.
  const int32_t minOutputBatchRows_;

  // The RowVectorPtr queue where bufferInputs_ flush to.
  std::queue<RowVectorPtr> outputQueue_;

  // The vector buffer the small input vector.
  RowVectorPtr bufferInputs_;

  // The buffer row count in bufferInputs_.
  int32_t numBufferRows_ = 0;

  // The buffer bytes in bufferInputs_.
  int64_t bufferBytes_ = 0;
};
} // namespace facebook::velox::exec
