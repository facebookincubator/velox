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

#include "velox/connectors/paimon/PaimonRowKind.h"
#include "velox/connectors/paimon/PaimonTableHandle.h"
#include "velox/vector/ComplexVector.h"

namespace facebook::velox::connector::paimon {

/// Merges the rows of a key into one row by column, as the merge functions of
/// Paimon Java's aggregation engine do. Takes the rows of a key in merge order
/// and reads and writes the columns of row vectors by channel. Rows must stay
/// alive until write().
class PaimonMergeFunction {
 public:
  /// Creates the merge function for the columns of 'type' at 'channels' with
  /// the merge engine of 'options'. Throws a user error for options Paimon
  /// Java rejects.
  static std::unique_ptr<PaimonMergeFunction> create(
      const RowTypePtr& type,
      const std::vector<column_index_t>& channels,
      const PaimonTableHandle::MergeOptions& options);

  virtual ~PaimonMergeFunction() = default;

  /// Starts the next key.
  virtual void reset() = 0;

  /// Adds row 'row' of 'input', a row of kind 'kind'.
  virtual void
  add(const RowVector& input, vector_size_t row, PaimonRowKind kind) = 0;

  /// Returns the kind of the merged row. The reader drops a merged row that is
  /// a DELETE or UPDATE_BEFORE, as Paimon Java's DropDeleteReader does, so the
  /// key has no row in the output.
  virtual PaimonRowKind resultKind() const = 0;

  /// Writes the merged columns to row 'row' of 'output', whose children at the
  /// channels are flat vectors.
  virtual void write(vector_size_t row, RowVector& output) const = 0;
};

} // namespace facebook::velox::connector::paimon
