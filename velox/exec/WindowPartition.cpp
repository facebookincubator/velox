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
// the input rows in this Window Operator.
WindowPartition::WindowPartition(
    std::vector<exec::RowColumn>& argColumns,
    const std::vector<TypePtr>& argTypes,
    velox::memory::MemoryPool* pool)
    : argColumns_(argColumns), pool_(pool) {
  argVectors_.reserve(argColumns.size());
  for (const auto& argType : argTypes) {
    argVectors_.emplace_back(BaseVector::create(argType, 0, pool));
  }
}

void WindowPartition::resetPartition(const folly::Range<char**>& rows) {
  // TODO : This is a copy of the folly::Range. If this involves a copy
  // of the vector of row pointers, then we can just maintain the
  // vector as a member variable in the Window operator
  partition_ = rows;
  for (auto& argVector : argVectors_) {
    argVector->resize(0);
  }
}

VectorPtr WindowPartition::argColumn(vector_size_t idx) const {
  if (argVectors_[idx]->size() != partition_.size()) {
    argVectors_[idx]->resize(partition_.size());
    exec::RowContainer::extractColumn(
        partition_.data(),
        partition_.size(),
        argColumns_[idx],
        argVectors_[idx]);
  }
  return argVectors_[idx];
}
} // namespace facebook::velox::exec
