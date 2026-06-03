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

#include "velox/core/PlanNode.h"
#include "velox/vector/BaseVector.h"

#include <utility>
#include <vector>

namespace facebook::velox::exec::window {

/// Copies a subset of columns from one row into self-contained one-row vectors
/// so the row can be compared against later rows after its source vector is
/// gone.
class RowColumnsSnapshot {
 public:
  /// Copies 'channels' values from 'input' row into self-contained vectors.
  void capture(
      const RowVectorPtr& input,
      vector_size_t row,
      const std::vector<column_index_t>& channels,
      memory::MemoryPool* pool);

  /// Clears the snapshot.
  void clear();

  /// Returns true if the snapshot holds a captured row.
  bool isValid() const {
    return valid_;
  }

  /// Returns true if the snapshot matches 'row' over 'keyInfo'.
  bool rowsEqual(
      const RowVectorPtr& input,
      vector_size_t row,
      const std::vector<std::pair<column_index_t, core::SortOrder>>& keyInfo,
      const std::vector<column_index_t>& inputChannels) const;

 private:
  // Returns the copied one-row vector for 'channel'.
  const VectorPtr& valueAt(column_index_t channel) const;

  // True if values_ contains copied values.
  bool valid_{false};

  // Original input channels represented by values_.
  std::vector<column_index_t> channels_;

  // One-row vectors holding copied values.
  std::vector<VectorPtr> values_;
};

} // namespace facebook::velox::exec::window
