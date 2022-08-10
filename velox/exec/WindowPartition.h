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

#include "velox/exec/RowContainer.h"
#include "velox/vector/BaseVector.h"

namespace facebook::velox::exec {
class WindowPartition {
 public:
  WindowPartition(
      const std::vector<exec::RowColumn>& columns,
      const std::vector<TypePtr>& types);

  /// Extracts into the result VectorPtr values from column at index 'idx'
  /// for numRows starting at rowOffset in the partition.
  /// This API is useful for window functions that examine column values
  /// for their logic. Nulls at corresponding positions are copied.
  void extractColumn(
      vector_size_t idx,
      vector_size_t numRows,
      vector_size_t rowOffset,
      VectorPtr result) const;

  /// Returns the number of rows in the current WindowPartition.
  vector_size_t numRows() const {
    return partition_.size();
  }

  void resetPartition(const folly::Range<char**>& rows);

 private:
  // This is a copy of the input RowColumn objects that are used for
  // accessing the partition row columns. These RowColumn objects
  // index into the Window Operator RowContainer and can retrieve
  // the column values.
  // The order of these columns is the same as that of the input row
  // of the Window operator. The WindowFunctions know the
  // corresponding indexes of their input arguments into this vector.
  // They will request for column vector values at the respective index.
  std::vector<exec::RowColumn> columns_;

  // This folly::Range is for the partition rows iterator provided by the
  // Window operator. The pointers are to rows from a RowContainer owned
  // by the operator. We can assume these are valid values for the lifetime
  // of WindowPartition.
  folly::Range<char**> partition_;
};
} // namespace facebook::velox::exec
