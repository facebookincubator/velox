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

namespace facebook::velox::exec::detail {

// Points to a row in a retained input vector.
struct WindowPartitionRowReference {
  // Input vector that owns the referenced row.
  RowVectorPtr input;

  // Row number in 'input'.
  vector_size_t row;
};

// Owns copied key values from one row for later boundary comparisons.
class WindowPartitionKeyRowSnapshot {
 public:
  // Copies 'keyChannels' values from 'input' row into self-contained vectors.
  void capture(
      const RowVectorPtr& input,
      vector_size_t row,
      const std::vector<column_index_t>& keyChannels,
      memory::MemoryPool* pool);

  // Copies 'keyChannels' values from 'row' into self-contained vectors.
  void capture(
      const WindowPartitionRowReference& row,
      const std::vector<column_index_t>& keyChannels,
      memory::MemoryPool* pool);

  // Clears the snapshot.
  void clear();

  // Returns true if the snapshot contains a previous row.
  bool isValid() const {
    return valid_;
  }

  // Returns true if the snapshot matches 'row' over 'keyInfo'.
  bool rowsEqual(
      const RowVectorPtr& input,
      vector_size_t row,
      const std::vector<std::pair<column_index_t, core::SortOrder>>& keyInfo,
      const std::vector<column_index_t>& inputChannels) const;

  // Returns true if the snapshot matches 'row' over 'keyInfo'.
  bool rowsEqual(
      const WindowPartitionRowReference& row,
      const std::vector<std::pair<column_index_t, core::SortOrder>>& keyInfo,
      const std::vector<column_index_t>& inputChannels) const;

 private:
  // Returns the copied one-row vector for 'channel'.
  const VectorPtr& valueAt(column_index_t channel) const;

  // True if values_ contains copied key values.
  bool valid_{false};

  // Original input channels represented by values_.
  std::vector<column_index_t> channels_;

  // One-row vectors holding copied key values.
  std::vector<VectorPtr> values_;
};

// Builds the unique set of original input channels needed for key comparison.
class WindowPartitionKeyChannels {
 public:
  // Builds key channels for a single key list.
  static std::vector<column_index_t> create(
      const std::vector<std::pair<column_index_t, core::SortOrder>>& keyInfo,
      const std::vector<column_index_t>& inputChannels);

  // Builds key channels for two key lists.
  static std::vector<column_index_t> create(
      const std::vector<std::pair<column_index_t, core::SortOrder>>&
          firstKeyInfo,
      const std::vector<std::pair<column_index_t, core::SortOrder>>&
          secondKeyInfo,
      const std::vector<column_index_t>& inputChannels);

 private:
  // Adds 'channel' if it is not already present.
  static void appendUnique(
      std::vector<column_index_t>& channels,
      column_index_t channel);
};

} // namespace facebook::velox::exec::detail
