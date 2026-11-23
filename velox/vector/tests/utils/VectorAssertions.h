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

#include <set>

#include "velox/type/Variant.h"
#include "velox/vector/ComplexVector.h"

namespace facebook::velox::test {

/// Compares vectors in tests and reports mismatches as gtest failures.
///
/// Example:
///
///   VectorAssertions::equalOrdered(
///       duckDb.executeOrdered(sql, type), type, results, /*epsilon=*/false);
class VectorAssertions {
 public:
  /// Asserts that 'expected' and 'actual' have equivalent types and equal
  /// values at every position.
  static void equal(const VectorPtr& expected, const VectorPtr& actual);

  /// Asserts that 'expected' and 'actual' have equivalent types and equal
  /// values at the positions selected in 'rows'.
  static void equal(
      const VectorPtr& expected,
      const VectorPtr& actual,
      const SelectivityVector& rows);

  /// Returns the rows of 'vector', each as a list of its column values.
  static std::vector<std::vector<Variant>> toVariantRows(
      const RowVectorPtr& vector);

  /// Returns the rows of all 'vectors' as a multiset, each row as a list of its
  /// column values.
  static std::multiset<std::vector<Variant>> toMultiset(
      const std::vector<RowVectorPtr>& vectors);

  /// Asserts that 'actual' has a type equivalent to 'expectedType' and holds
  /// the 'expected' rows in the same order. The rows of 'actual' are read
  /// across its vectors in sequence, so any split into vectors compares the
  /// same. With 'epsilon', compares values with Variant::equalsWithEpsilon,
  /// which allows a small difference in REAL and DOUBLE values; otherwise
  /// exactly.
  ///
  /// A Variant holds only a TypeKind, not the full type, e.g. not DATE versus
  /// INTEGER or the precision of a DECIMAL. 'expectedType' is the row type of
  /// 'expected'.
  static void equalOrdered(
      const std::vector<std::vector<Variant>>& expected,
      const TypePtr& expectedType,
      const std::vector<RowVectorPtr>& actual,
      bool epsilon);

  /// Returns true if 'expected' and 'actual' have equivalent types and hold the
  /// same rows in any order; otherwise, reports the mismatch as a gtest
  /// failure.
  ///
  /// Floating-point values are compared with epsilon only in one case:
  ///   - With no floating-point columns, compares all values exactly.
  ///   - With floating-point columns, sorts both sets by the other columns. If
  ///     those columns form a unique key in both sets, compares the sorted rows
  ///     pairwise with Variant::equalsWithEpsilon. Aggregation results
  ///     typically qualify: one row per group.
  ///   - Otherwise, compares all values exactly.
  ///
  /// The epsilon comparator is not a strict weak ordering, so it cannot back a
  /// std::multiset. Matching arbitrary sets with epsilon would need an
  /// algorithm such as maximum bipartite matching. See
  /// https://github.com/facebookincubator/velox/issues/3493.
  static bool equalMultisets(
      const std::vector<RowVectorPtr>& expected,
      const std::vector<RowVectorPtr>& actual);

  /// Same as above for 'expected' rows of Variant values. 'expectedType' is
  /// their row type.
  static bool equalMultisets(
      const std::multiset<std::vector<Variant>>& expected,
      const TypePtr& expectedType,
      const std::vector<RowVectorPtr>& actual);

  /// Same as above for 'expected' and 'actual' rows of Variant values. Does not
  /// check types: the caller ensures that 'expected' and 'actual' have the same
  /// row type, 'type'. 'type' tells which columns are floating-point and how to
  /// print the values.
  static bool equalMultisets(
      const std::multiset<std::vector<Variant>>& expected,
      const std::multiset<std::vector<Variant>>& actual,
      const TypePtr& type);

  /// Asserts that 'actual' has a type equivalent to 'expectedType' and holds
  /// the 'expected' rows ordered by 'sortingKeys', where rows with equal keys
  /// may appear in any order. 'expected' must be ordered by 'sortingKeys'.
  /// Compares the rows of each key as in equalMultisets. 'expectedType' is the
  /// row type of 'expected' and 'sortingKeys' are indices of its columns.
  static void equalOrderedBy(
      const std::vector<std::vector<Variant>>& expected,
      const RowTypePtr& expectedType,
      const std::vector<RowVectorPtr>& actual,
      const std::vector<uint32_t>& sortingKeys);
};

} // namespace facebook::velox::test
