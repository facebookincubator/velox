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

#include "velox/type/Cost.h"
#include "velox/type/Type.h"

namespace facebook::velox {

/// Type coercion necessary to bind a type to a signature.
struct Coercion {
  /// Resulting type after coercion.
  /// If no coercion is necessary or possible, 'type' is nullptr.
  TypePtr type;
  /// Cost of the coercion. Zero means no coercion is necessary.
  /// kCostInvalid means coercion is needed but not possible.
  Cost cost{0};

  std::string toString() const {
    if (type == nullptr) {
      return "null";
    }

    return fmt::format("{} ({})", type->toString(), cost);
  }

  /// False if coercion is needed but not possible.
  explicit operator bool() const {
    return cost != kInvalidCost;
  }

  /// Returns overall cost of a list of coercions by adding up individual costs.
  static Cost overallCost(const std::vector<Coercion>& coercions);

  /// Converts a list of valid Coercions into a list of TypePtr.
  static void convert(
      const std::vector<Coercion>& from,
      std::vector<TypePtr>* to);
};

class TypeCoercer {
 public:
  /// Checks if the base of 'fromType' can be implicitly converted to a type
  /// with the given name.
  /// Only types without type parameters are supported.
  ///
  /// @return "to" type and cost if conversion is possible.
  static Coercion coerceTypeBase(
      const TypePtr& fromType,
      const std::string& toTypeName);

  /// Checks if 'fromType' can be implicitly converted to 'toType'.
  ///
  /// @return "to" type and cost if conversion is possible.
  static Coercion coercible(const TypePtr& fromType, const TypePtr& toType);
};

} // namespace facebook::velox
