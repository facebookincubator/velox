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
    const std::vector<TypePtr>& argTypes)
    : columns_(columns) {}

void WindowPartition::resetPartition(const folly::Range<char**>& rows) {
  // TODO : Is this a copy of the folly::Range ? If this involves a copy
  // of the vector of row pointers, then we can just maintain the
  // vector as a member variable in the Window operator
  partition_ = rows;
}

void WindowPartition::extractColumn(
    vector_size_t idx,
    vector_size_t numRows,
    vector_size_t rowOffset,
    VectorPtr result) const {
  exec::RowContainer::extractColumn(
      partition_.data(), numRows, columns_[idx], result);
}

void WindowPartition::extractColumnOffsets(
    vector_size_t idx,
    const BufferPtr& offsets,
    vector_size_t resultOffset,
    VectorPtr result) const {
  VELOX_NYI();
}
} // namespace facebook::velox::exec
