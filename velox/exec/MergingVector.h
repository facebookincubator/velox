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

#include <set>

#include "velox/type/Type.h"
#include "velox/vector/ComplexVector.h"

namespace facebook::velox::exec {

/// MergingVector is used to merge input or output vectors for operators. it
/// only merge vectors that small than maxMergingBatchRows and
/// maxMergingBatchBytes. The merged vectors size will close to but smaller
/// than preferredOutputRows and preferredOutputBatchBytes.
class MergingVector {
 public:
  /// @param preferredOutputRows The preferred number rows of the merged
  /// vectors.
  /// @param preferredOutputBatchBytes The preferred size in bytes of the
  /// merged vectors.
  /// @param maxMergingBatchRows If the input vector row count is larger than
  /// maxMergingBatchRows, don't merge it to avoid large memory copy.
  /// @param maxMergingBatchBytes If the input vector bytes is larger than
  /// maxMergingBatchBytes, don't merge it to avoid large memory copy.
  /// @param preserveOrder Whether the merged vectors order need to preserve
  /// with the input order.
  /// @param reuseBuffer Whether reuse the buffer to reduce vector construction,
  /// notice only the operators that don't hold on to the input vector can
  /// reuse buffer.
  MergingVector(
      int32_t preferredOutputRows,
      int64_t preferredOutputBatchBytes,
      int32_t maxMergingBatchRows,
      int64_t maxMergingBatchBytes,
      bool preserveOrder,
      bool reuseBuffer,
      velox::memory::MemoryPool* pool);

  /// Add input vector. It will be merged into a big vector buffer if small
  /// enough. To prevent consuming more memory, the caller is supposed to first
  /// drain the buffer.
  void addInput(RowVectorPtr& input);

  /// Returns a RowVector after merged, or return nullptr if we haven't
  /// accumulated enough just yet.
  /// @param noMoreInput whether more data is expected to be added via
  /// addInput.
  /// @return RowVector or nullptr.
  RowVectorPtr getOutput(bool noMoreInput);

  /// Reuse the RowVectorPtr if the operator support, for example,
  /// HashAggregation will not hold the input vectors, so if we merge it's
  /// input, we can reuse the merged RowVectorPtr.
  void reuseIfNeed(RowVectorPtr& reuse);

 private:
  // Push the buffer vector to the queue, and reset buffer_,
  // bufferBytes_ and numBufferRows_.
  void flush();

  // If the input vector is small enough, copy it to the buffer vector.
  void buffer(RowVectorPtr& input, uint64_t bytesInput);

  // Merge the source row vector to dest row vector.
  void mergeVector(
      RowVectorPtr& dest,
      const RowVectorPtr& src,
      int32_t destRows,
      int32_t srcRows);

  // The preferred output row count of the buffer_.
  const int32_t preferredOutputBatchRows_;

  // The preferred output bytes of the buffer_.
  const int64_t preferredOutputBatchBytes_;

  // If the input vector row count is larger than maxMergingBatchRows_, flush
  // it to the outputQueue_ directly.
  const int32_t maxMergingBatchRows_;

  // If the input vector bytes is larger than maxMergingBatchBytes_, flush
  // it to the outputQueue_ directly.
  const int64_t maxMergingBatchBytes_;

  // Whether the merged vectors order need to preserve with the input order.
  const bool preserveOrder_ = true;

  // Whether reuse the buffer to reduce vector construction.
  const bool reuseBuffer_ = false;

  // Since we check the outputQueue_ should be empty() in addInput, the reuse
  // buffer size only need 2;
  const int32_t MAX_REUSE_BUFFER_SIZE{2};

  velox::memory::MemoryPool* const pool_;

  // The RowVectorPtr queue where buffer_ flush to.
  std::queue<RowVectorPtr> outputQueue_;

  // The vector buffer the small input vectors.
  RowVectorPtr buffer_;

  bool isFirstVector_ = true;

  // The last vector that lazy copy to buffer_.
  RowVectorPtr lastVector_;

  // The buffer row count in buffer_.
  int32_t numBufferRows_{0};

  // The buffer bytes in buffer_.
  int64_t bufferBytes_{0};

  // If reuseBuffers_ is not empty, prefer to get RowVectorPtr from it instead
  // of reconstructing a new one.
  std::vector<RowVectorPtr> reuseBuffers_;

  // If reuseBuffer is true, record the buffers create by MergingVector, used
  // to check the RowVectorPtr can be reused, the max size of createdBuffers_
  // should be less than MAX_REUSE_BUFFER_SIZE.
  std::set<RowVectorPtr> createdBuffers_;
};
} // namespace facebook::velox::exec
