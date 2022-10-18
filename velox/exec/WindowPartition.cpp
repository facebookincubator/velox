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

WindowPartition::WindowPartition(
    const std::vector<exec::RowColumn>& columns,
    const std::vector<TypePtr>& /* argTypes */)
    : columns_(columns) {}

void WindowPartition::resetPartition(const folly::Range<char**>& rows) {
  partition_ = rows;
}

void WindowPartition::extractColumn(
    int32_t columnIndex,
    folly::Range<const vector_size_t*> rowNumbers,
    vector_size_t resultOffset,
    const VectorPtr& result) const {
  RowContainer::extractColumn(
      partition_.data(),
      rowNumbers,
      columns_[columnIndex],
      resultOffset,
      result);
}

void WindowPartition::extractColumn(
    int32_t columnIndex,
    vector_size_t partitionOffset,
    vector_size_t numRows,
    vector_size_t resultOffset,
    const VectorPtr& result) const {
  RowContainer::extractColumn(
      partition_.data() + partitionOffset,
      numRows,
      columns_[columnIndex],
      resultOffset,
      result);
}

} // namespace facebook::velox::exec
