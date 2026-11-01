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

#include <optional>
#include <string>
#include <type_traits>
#include <vector>

#include <folly/container/F14Set.h>

#include "velox/common/base/CompareFlags.h"
#include "velox/functions/Udf.h"
#include "velox/functions/sparksql/JavaFloatingPoint.h"

namespace facebook::velox::functions::sparksql {

namespace detail {

// Selects the hash set for the top-level elements of an array of T. Floating
// point elements compare as Java's equals() does, as in Spark: all NaNs are
// equal, and -0.0 and 0.0 are different.
template <typename T, typename = void>
struct SparkElementSet {
  using type = folly::F14FastSet<T>;
};

template <typename T>
struct SparkElementSet<T, std::enable_if_t<std::is_floating_point_v<T>>> {
  using type = folly::
      F14FastSet<T, JavaFloatingPoint::Hash<T>, JavaFloatingPoint::Equals<T>>;
};

} // namespace detail

/// arrays_overlap(array(E), array(E)) -> boolean
///
/// Returns true if the two arrays have a non-null element in common. Otherwise
/// returns null if either array contains a null, and false if neither does. An
/// empty array never overlaps, so the result is false even when the other array
/// contains a null. A null array gives a null result.
///
/// Handles the element types that Spark compares with equals() and a hash set
/// (ArraysOverlap#fastEval): integers, booleans, decimals, dates, timestamps,
/// day-time and year-month intervals, floating point and strings. Also handles
/// binary, which Spark compares with its ordering in a nested loop
/// (bruteForceEval) because byte arrays lack value equals(); that ordering
/// finds two binary values equal exactly when their bytes are equal, as the
/// hash set here does. Like Spark, puts the elements of the smaller array in a
/// hash set and looks up the elements of the bigger one. When one argument is
/// constant, builds its set once, in initialize(), and looks up only the other
/// argument per row.
template <typename TExec, typename T>
struct ArraysOverlapFunction {
  VELOX_DEFINE_FUNCTION_TYPES(TExec);

  using Element = arg_type<T>;
  using ElementSet = typename detail::SparkElementSet<Element>::type;

  void initialize(
      const std::vector<TypePtr>& /*inputTypes*/,
      const core::QueryConfig& /*config*/,
      const arg_type<Array<T>>* left,
      const arg_type<Array<T>>* right) {
    // When both arguments are constant, the right one is used.
    const auto* constant = right != nullptr ? right : left;
    if (constant == nullptr) {
      return;
    }
    constantIsLeft_ = right == nullptr;
    constantIsEmpty_ = constant->size() == 0;
    constantSet_.emplace();
    // The strings own() copies are never more than the constant's elements, so
    // after this reservation the copies never move and their data stays put.
    ownedStrings_.reserve(constant->size());
    for (const auto& element : *constant) {
      if (element.has_value()) {
        constantSet_->insert(own(element.value()));
      } else {
        constantHasNull_ = true;
      }
    }
  }

  bool call(
      bool& out,
      const arg_type<Array<T>>& left,
      const arg_type<Array<T>>& right) {
    if (constantSet_.has_value()) {
      const auto& other = constantIsLeft_ ? right : left;
      if (constantIsEmpty_ || other.size() == 0) {
        out = false;
        return true;
      }
      return findOverlap(*constantSet_, constantHasNull_, other, out);
    }

    if (left.size() == 0 || right.size() == 0) {
      out = false;
      return true;
    }
    const bool leftIsSmaller = left.size() <= right.size();
    const auto& smaller = leftIsSmaller ? left : right;
    const auto& bigger = leftIsSmaller ? right : left;
    rowSet_.clear();
    bool hasNull = false;
    for (const auto& element : smaller) {
      if (element.has_value()) {
        rowSet_.insert(element.value());
      } else {
        hasNull = true;
      }
    }
    return findOverlap(rowSet_, hasNull, bigger, out);
  }

 private:
  // Sets 'out' to true if 'array' has a non-null element in 'set'. Otherwise
  // returns false, meaning a null result, if 'hasNull' is true or 'array' has a
  // null element, and sets 'out' to false if neither is the case.
  static bool findOverlap(
      const ElementSet& set,
      bool hasNull,
      const arg_type<Array<T>>& array,
      bool& out) {
    for (const auto& element : array) {
      if (!element.has_value()) {
        hasNull = true;
      } else if (set.contains(element.value())) {
        out = true;
        return true;
      }
    }
    if (hasNull) {
      return false;
    }
    out = false;
    return true;
  }

  // Returns a value that stays valid after initialize() returns. A string
  // longer than StringView's inline size points into the constant vector, so
  // its bytes are copied into ownedStrings_.
  Element own(const Element& value) {
    if constexpr (std::is_same_v<Element, StringView>) {
      if (!value.isInline()) {
        ownedStrings_.emplace_back(value.data(), value.size());
        const auto& copy = ownedStrings_.back();
        return StringView(copy.data(), static_cast<int32_t>(copy.size()));
      }
    }
    return value;
  }

  // Non-null elements of the constant argument. Empty optional when neither
  // argument is constant.
  std::optional<ElementSet> constantSet_;

  // True when the constant argument is the left one, so each row reads the
  // right one.
  bool constantIsLeft_{false};

  // True when the constant argument is an empty array, which overlaps nothing.
  bool constantIsEmpty_{false};

  // True when the constant argument contains a null.
  bool constantHasNull_{false};

  // Backing storage for the long strings in constantSet_, which must outlive
  // the constant vector seen by initialize().
  std::vector<std::string> ownedStrings_;

  // Elements of the smaller array of the current row when neither argument is
  // constant. Kept across rows so that its memory is reused.
  ElementSet rowSet_;
};

/// arrays_overlap(array(E), array(E)) -> boolean for arrays of structs and
/// arrays. Compares the elements in a nested loop, as Spark does for these
/// types (ArraysOverlap#bruteForceEval), with the same null handling as
/// ArraysOverlapFunction. Velox's equality with nulls as values matches Spark's
/// ordering for these types: NaNs are equal, -0.0 equals 0.0, and nulls inside
/// an element are equal. Maps are not orderable, so arrays of maps match no
/// signature, as in Spark.
template <typename TExec>
struct ArraysOverlapComplexFunction {
  VELOX_DEFINE_FUNCTION_TYPES(TExec);

  bool call(
      bool& out,
      const arg_type<Array<Orderable<T1>>>& left,
      const arg_type<Array<Orderable<T1>>>& right) {
    if (left.size() == 0 || right.size() == 0) {
      out = false;
      return true;
    }

    static constexpr CompareFlags kFlags =
        CompareFlags::equality(CompareFlags::NullHandlingMode::kNullAsValue);
    // A null in either array makes the result NULL when nothing matches. Check
    // the right array once rather than on every pass of the outer loop.
    bool hasNull = false;
    for (const auto& rightElement : right) {
      if (!rightElement.has_value()) {
        hasNull = true;
        break;
      }
    }
    for (const auto& leftElement : left) {
      if (!leftElement.has_value()) {
        hasNull = true;
        continue;
      }
      for (const auto& rightElement : right) {
        if (!rightElement.has_value()) {
          continue;
        }
        // With nulls as values, compare() always returns a value.
        if (leftElement.value().compare(rightElement.value(), kFlags).value() ==
            0) {
          out = true;
          return true;
        }
      }
    }
    if (hasNull) {
      return false;
    }
    out = false;
    return true;
  }
};

} // namespace facebook::velox::functions::sparksql
