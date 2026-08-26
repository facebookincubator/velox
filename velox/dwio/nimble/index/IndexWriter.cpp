/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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
#include "velox/dwio/nimble/index/IndexWriter.h"

#include <set>

#include "velox/common/base/Nulls.h"
#include "velox/dwio/nimble/common/Exceptions.h"

namespace facebook::nimble::index {

// static
void IndexWriter::validateNoNullKeys(
    const velox::VectorPtr& input,
    const std::vector<velox::column_index_t>& keyColumnIndices) {
  const auto* rowVector = input->asChecked<velox::RowVector>();
  const auto& children = rowVector->children();
  for (const auto columnIndex : keyColumnIndices) {
    const auto& child = children[columnIndex];
    if (!child->mayHaveNulls()) {
      continue;
    }
    const auto* nulls = child->rawNulls();
    if (nulls == nullptr) {
      continue;
    }
    const auto nullCount = velox::bits::countNulls(nulls, 0, input->size());
    NIMBLE_USER_CHECK_EQ(
        nullCount,
        0,
        "Null value not allowed in index key column at index {}: found {} null(s)",
        columnIndex,
        nullCount);
  }
}

// static
void IndexWriter::validateKeyStreamEncodingLayout(
    const EncodingLayout& layout) {
  NIMBLE_USER_CHECK(
      layout.encodingType() == EncodingType::Prefix ||
          layout.encodingType() == EncodingType::Trivial,
      "Index key stream only supports Prefix or Trivial encoding, but got: {}",
      layout.encodingType());
}

// static
std::vector<velox::column_index_t> IndexWriter::getKeyColumnIndices(
    const std::vector<std::vector<std::string>>& columnSets,
    const velox::RowTypePtr& inputType) {
  std::set<velox::column_index_t> indexSet;
  std::vector<velox::column_index_t> indices;
  for (const auto& columns : columnSets) {
    for (const auto& columnName : columns) {
      const auto idx = inputType->getChildIdx(columnName);
      if (indexSet.insert(idx).second) {
        indices.emplace_back(idx);
      }
    }
  }
  return indices;
}

} // namespace facebook::nimble::index
