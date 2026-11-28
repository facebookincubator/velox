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

#include "velox/vector/BaseVector.h"

namespace facebook::velox::connector::paimon {

/// Merges one column over the rows of a key, as Paimon Java's FieldAggregator
/// subclasses do. The accumulator is null when a key starts. Functions that
/// pick an input value keep a reference to its row, so input vectors must stay
/// alive until write().
class PaimonFieldAggregator {
 public:
  /// Creates the aggregator of 'function' for column 'column' of type 'type'.
  /// 'ignoreRetract' makes DELETE and UPDATE_BEFORE rows keep the accumulator.
  /// Throws a user error for an unknown function or a type the function does
  /// not take.
  static std::unique_ptr<PaimonFieldAggregator> create(
      const std::string& function,
      const std::string& column,
      const TypePtr& type,
      bool ignoreRetract);

  virtual ~PaimonFieldAggregator() = default;

  /// Clears the accumulator and the state of the function for the next key.
  virtual void reset() = 0;

  /// Merges row 'row' of 'input' into the accumulator.
  virtual void add(const BaseVector& input, vector_size_t row) = 0;

  /// Retracts row 'row' of 'input', which comes from a DELETE or
  /// UPDATE_BEFORE row, from the accumulator. In a changelog, a retraction
  /// withdraws a row that an earlier change added. For example, when an order
  /// of 10 becomes 7, the changelog has an UPDATE_BEFORE row of 10 and an
  /// UPDATE_AFTER row of 7, and 'sum' goes from 10 to 0 to 7. Functions that
  /// cannot undo a row, such as 'max', fail unless the column ignores
  /// retractions.
  virtual void retract(const BaseVector& input, vector_size_t row) = 0;

  /// Replaces the accumulator with row 'row' of 'input' and keeps the state of
  /// the function, as Paimon Java does when a DELETE row replaces the merged
  /// row of its key.
  virtual void set(const BaseVector& input, vector_size_t row) = 0;

  /// Writes the accumulator to row 'row' of 'output', a flat vector of the
  /// column type.
  virtual void write(vector_size_t row, BaseVector& output) const = 0;
};

} // namespace facebook::velox::connector::paimon
