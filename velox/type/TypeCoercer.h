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

#include "velox/type/Type.h"

namespace facebook::velox {

using CallableCost = uint64_t;

// This indicates that coercion is not possible.
// In other words, it means we didn't find a valid implicit CAST,
// so the function signature won't match.
inline constexpr CallableCost kImpossibleCoercionCost =
    std::numeric_limits<CallableCost>::max();

/// Optional type coercion to bind a type to a signature.
/// If coercion is not possible, 'type' is nullptr
///  and 'cost' is kImpossibleCoercionCost.
/// If no coercion is necessary, 'type' is nullptr and 'cost' is zero.
/// Otherwise, 'type' is the resulting type after coercion
///  and 'cost' is greater than zero.
struct Coercion {
  /// Resulting type after coercion.
  /// If no coercion is necessary or possible, 'type' is nullptr.
  TypePtr type;
  CallableCost cost = 0;

  std::string toString() const {
    if (type == nullptr) {
      return "null";
    }

    return fmt::format("{} ({})", type->toString(), cost);
  }

  /// True if coercion is possible (including no coercion needed).
  /// False if coercion is needed but not possible.
  bool isPossible() const {
    return cost != kImpossibleCoercionCost;
  }

  explicit operator bool() const {
    return isPossible();
  }

  /// Returns overall cost of a list of coercions by adding up individual costs.
  /// Coercions must be possible (including no coercion needed).
  static CallableCost overallCost(const std::vector<Coercion>& coercions);

  /// Converts a list of valid Coercions into a list of TypePtr.
  /// Coercions must be possible (including no coercion needed).
  static void convert(
      const std::vector<Coercion>& from,
      std::vector<TypePtr>* to);
};

class TypeCoercer {
 public:
  /// Checks if the base of 'fromType' can be implicitly converted to a type
  /// with the given name.
  /// Only types without type parameters are supported.
  static Coercion coerceTypeBase(
      const TypePtr& fromType,
      const std::string& toTypeName);

  /// Checks if 'fromType' can be implicitly converted to 'toType'.
  ///
  /// @return "to" type and cost if conversion is possible.
  static Coercion coercible(const TypePtr& fromType, const TypePtr& toType);

  /// Returns least common type for 'a' and 'b', i.e. a type that both 'a' and
  /// 'b' are coercible to. Returns nullptr if no such type exists.
  ///
  /// When `a` and `b` are ROW types with different field names, the resulting
  /// ROW has empty field names for any positions where the corresponding field
  /// names do not match.
  static TypePtr leastCommonSuperType(const TypePtr& a, const TypePtr& b);
};

} // namespace facebook::velox
