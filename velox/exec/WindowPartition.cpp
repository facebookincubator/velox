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
#include "velox/exec/WindowPartition.h"

namespace facebook::velox::exec {

// Simple WindowPartition that builds over the RowContainer used for storing
// the input rows in the Window Operator. This works completely in-memory.
// TODO: This implementation will be revised for Spill to disk semantics.
WindowPartition::WindowPartition(
    const std::vector<exec::RowColumn>& columns,
    const std::vector<TypePtr>& argTypes,
    velox::memory::MemoryPool* pool)
    : columns_(columns), pool_(pool) {
  columnVectors_.reserve(columns.size());
  for (const auto& argType : argTypes) {
    columnVectors_.emplace_back(BaseVector::create(argType, 0, pool));
  }
}

void WindowPartition::resetPartition(const folly::Range<char**>& rows) {
  // TODO : Is this a copy of the folly::Range ? If this involves a copy
  // of the vector of row pointers, then we can just maintain the
  // vector as a member variable in the Window operator
  partition_ = rows;
  for (auto& argVector : columnVectors_) {
    argVector->resize(0);
  }
}

VectorPtr WindowPartition::argColumn(vector_size_t idx) const {
  if (columnVectors_[idx]->size() != partition_.size()) {
    columnVectors_[idx]->resize(partition_.size());
    exec::RowContainer::extractColumn(
        partition_.data(),
        partition_.size(),
        columns_[idx],
        columnVectors_[idx]);
  }
  return columnVectors_[idx];
}

void WindowPartition::extractColumnOffsets(
    vector_size_t idx,
    const BufferPtr& offsets,
    vector_size_t resultOffset,
    VectorPtr result) const {
  VELOX_NYI();
}
} // namespace facebook::velox::exec
