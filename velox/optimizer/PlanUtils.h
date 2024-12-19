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

#include <folly/Range.h>
#include "velox/optimizer/QueryGraph.h"

namespace facebook::velox::optimizer {

template <typename T, typename U>
bool isSubset(const T& subset, const U& superset) {
  for (auto item : subset) {
    if (std::find(superset.begin(), superset.end(), item) == superset.end()) {
      return false;
    }
  }
  return true;
}

// Returns how many leading members of 'ordered' are covered by 'set'
template <typename Ordered, typename Set>
uint32_t prefixSize(Ordered ordered, Set set) {
  for (auto i = 0; i < ordered.size(); ++i) {
    if (std::find(set.begin(), set.end(), ordered[i]) == set.end()) {
      return i;
    }
  }
  return ordered.size();
}

// Replaces each element of 'set' that matches an element of 'originals' with
// the corresponding element of 'replaceWith'. Returns true if all elements were
// replaced.
template <typename Set, typename Old, typename New>
bool replace(Set& set, Old& originals, New replaceWith) {
  bool allReplaced = true;
  for (auto& element : set) {
    auto it = std::find(originals.begin(), originals.end(), element);
    if (it == originals.end()) {
      allReplaced = false;
      continue;
    }
    element = replaceWith[it - originals.begin()];
  }
  return allReplaced;
}

template <typename T, typename U>
void appendToVector(T& destination, U& source) {
  for (auto i : source) {
    destination.push_back(i);
  }
}

constexpr uint32_t kNotFound = ~0U;

/// Returns index of 'expr' in collection 'exprs'. kNotFound if not found.
/// Compares with equivalence classes, so that equal columns are
/// interchangeable.
template <typename V>
uint32_t position(const V& exprs, const Expr& expr) {
  for (auto i = 0; i < exprs.size(); ++i) {
    if (exprs[i]->sameOrEqual(expr)) {
      return i;
    }
  }
  return kNotFound;
}

/// Returns index of 'expr' in collection 'exprs'. kNotFound if not found.
/// Compares with equivalence classes, so that equal columns are
/// interchangeable. Applies 'getter' to each element of 'exprs' before
/// comparison.
template <typename V, typename Getter>
uint32_t position(const V& exprs, Getter getter, const Expr& expr) {
  for (auto i = 0; i < exprs.size(); ++i) {
    if (getter(exprs[i])->sameOrEqual(expr)) {
      return i;
    }
  }
  return kNotFound;
}

/// Prints a number with precision' digits followed by a scale letter (n, u, m,
/// , k, M, G T, P.
std::string succinctNumber(double value, int32_t precision = 2);

/// Returns the sum of the sizes of 'exprs'.
template <typename V>
float byteSize(const V& exprs) {
  float size = 0;
  for (auto& expr : exprs) {
    size += expr->value().byteSize();
  }
  return size;
}

template <typename Target, typename V, typename Func>
Target transform(const V& set, Func func) {
  Target result;
  for (auto& elt : set) {
    result.push_back(func(elt));
  }
  return result;
}

} // namespace facebook::velox::optimizer
