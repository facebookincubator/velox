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

#include <gtest/gtest.h>
#include <memory>
#include <utility>
#include <vector>

#include "velox/exec/FlatRowSerde.h"
#include "velox/exec/RowContainer.h"
#include "velox/exec/tests/utils/RowContainerTestBase.h"
#include "velox/vector/ComplexVector.h"
#include "velox/vector/DecodedVector.h"

namespace facebook::velox::exec::test {

/// Shared base class for flat-row-format test fixtures. Inherits pool(),
/// vector helpers, and serde registration from RowContainerTestBase and
/// VectorTestBase.
class FlatRowTestBase : public RowContainerTestBase {
 protected:
  /// Creates a RowContainer with the given key and dependent types.
  /// 'nullableKeys' controls whether key columns may be null. Intentionally
  /// shadows RowContainerTestBase::makeRowContainer, which builds a join-style
  /// container (isJoinBuild=true, hasProbedFlag=true) instead.
  std::unique_ptr<RowContainer> makeRowContainer(
      const std::vector<TypePtr>& keyTypes,
      const std::vector<TypePtr>& dependentTypes,
      bool nullableKeys) {
    return std::make_unique<RowContainer>(
        keyTypes,
        nullableKeys,
        std::vector<Accumulator>{},
        dependentTypes,
        /*hasNext=*/false,
        /*isJoinBuild=*/false,
        /*hasProbedFlag=*/false,
        /*hasCountFlag=*/false,
        /*hasNormalizedKey=*/false,
        /*useListRowIndex=*/false,
        pool());
  }

  /// Stores all rows from 'input' into 'container'. The first 'numKeys'
  /// columns are treated as key columns (columns 0 ... numKeys-1); the
  /// remaining columns are dependents. 'numKeys' must equal the number of key
  /// types used when the container was constructed.
  void storeRows(
      RowContainer* container,
      const RowVectorPtr& input,
      int32_t numKeys) {
    const int32_t numRows = static_cast<int32_t>(input->size());
    const int32_t numCols = static_cast<int32_t>(input->childrenSize());
    SelectivityVector allRows(numRows);
    std::vector<DecodedVector> decoded(numCols);
    for (int32_t col = 0; col < numCols; ++col) {
      decoded[col].decode(*input->childAt(col), allRows);
    }
    for (int32_t row = 0; row < numRows; ++row) {
      char* rowPtr = container->newRow();
      for (int32_t col = 0; col < numCols; ++col) {
        container->store(decoded[col], row, rowPtr, col);
      }
    }
  }

  /// Returns pointers to all live rows in 'container' in iteration order.
  std::vector<char*> listRows(RowContainer* container) {
    std::vector<char*> rows(container->numRows());
    RowContainerIterator iter;
    container->listRows(
        &iter,
        static_cast<int32_t>(rows.size()),
        RowContainer::kUnlimited,
        rows.data());
    return rows;
  }

  /// Serializes all rows in 'rows' into a single aligned buffer. Returns the
  /// backing buffer (which must remain alive as long as the returned row
  /// pointers are in use) and the per-row start pointers. Returns an empty
  /// pair when 'rows' is empty.
  std::pair<std::vector<char>, std::vector<char*>> serializeRows(
      const std::vector<char*>& rows,
      const RowContainerLayout& layout) {
    if (rows.empty()) {
      return {{}, {}};
    }
    size_t totalSize = 0;
    for (char* row : rows) {
      totalSize += layout.serializedSize(row);
    }
    const size_t alignment = static_cast<size_t>(layout.alignment);
    std::vector<char> buffer(totalSize + alignment * 2, 0);
    void* rawStart = buffer.data();
    size_t space = buffer.size();
    void* aligned = std::align(alignment, totalSize, rawStart, space);
    EXPECT_NE(aligned, nullptr);
    char* current = reinterpret_cast<char*>(aligned);
    const char* bufferEnd = buffer.data() + buffer.size();
    std::vector<char*> serializedRows;
    for (char* row : rows) {
      serializedRows.push_back(current);
      serializeRow(row, current, bufferEnd, layout);
    }
    return {std::move(buffer), std::move(serializedRows)};
  }
};

} // namespace facebook::velox::exec::test
