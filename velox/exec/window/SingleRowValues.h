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

#include <vector>

namespace facebook::velox::exec::window {

/// Stores copies of selected column values from a single row for later
/// comparison. Behaves like std::optional: it is empty after construction,
/// capture() populates it with a row, reset() clears it, and hasValue()
/// reports whether it currently holds a row.
class SingleRowValues {
 public:
  /// 'channels' are the input columns to copy and compare. 'pool' allocates the
  /// copied values and must outlive this object.
  SingleRowValues(
      std::vector<column_index_t> channels,
      memory::MemoryPool* pool);

  /// Copies the configured columns from 'input' row 'row'.
  void capture(const RowVectorPtr& input, vector_size_t row);

  /// Clears the stored values.
  void reset();

  /// Returns true if a row has been captured.
  bool hasValue() const {
    return hasValue_;
  }

  /// Returns true if the stored values equal 'input' row 'row' over the
  /// configured columns.
  bool equals(const RowVectorPtr& input, vector_size_t row) const;

 private:
  // Input columns copied and compared.
  const std::vector<column_index_t> channels_;

  // Pool used to allocate copied values. The owner must outlive this object.
  memory::MemoryPool* const pool_;

  // True if values_ holds a captured row.
  bool hasValue_{false};

  // Copied one-row values, one per entry in channels_.
  std::vector<VectorPtr> values_;
};

} // namespace facebook::velox::exec::window
