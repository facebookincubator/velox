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
      std::vector<exec::RowColumn>& columns,
      const std::vector<TypePtr>& types,
      velox::memory::MemoryPool* pool);

  void resetPartition(const folly::Range<char**>& rows);

  // Retrieves the column at a particular index. The function
  // knows the index in the input columns list for the argument.
  VectorPtr argColumn(vector_size_t idx) const;

  // Extracts into the result VectorPtr (starting from resultOffset),
  // values from column at index 'idx' and at row offsets in 'offsets'
  // buffer. This API is useful for Value functions like NthValue
  // that are just extracting column values but are not interested in
  // them. Nulls at corresponding positions are copied.
  void extractColumnOffsets(
      vector_size_t idx,
      const BufferPtr& offsets,
      vector_size_t resultOffset,
      VectorPtr result) const;

  vector_size_t numRows() const {
    return partition_.size();
  }

 private:
  // This is a copy of the input RowColumn objects that are used for
  // accessing the partition row columns
  std::vector<exec::RowColumn> columns_;
  velox::memory::MemoryPool* pool_;

  // This folly::Range is for the partition rows iterator provided by the
  // Window operator. These pointers are from a RowContainer owned
  // by the operator.
  folly::Range<char**> partition_;

  // This is a vector of all the input column values
  // partition rows. These are used by functions for evaluation. These
  // are constructed only on request by the function.
  std::vector<VectorPtr> columnVectors_;
};
} // namespace facebook::velox::exec