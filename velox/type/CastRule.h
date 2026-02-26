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

#include <fmt/format.h>

#include <cstdint>
#include <functional>
#include <memory>
#include <string>

namespace facebook::velox {

class Type;
using TypePtr = std::shared_ptr<const Type>;

/// Optional validator for type-specific compatibility checks
/// (e.g., DECIMAL precision/scale).
using CastValidator =
    std::function<bool(const TypePtr& from, const TypePtr& to)>;

/// Represents a cast rule between two types.
struct CastRule {
  std::string fromType;
  std::string toType;

  /// If true, cast can happen implicitly during type coercion.
  bool implicitAllowed;

  /// Lower cost = higher priority when multiple casts are possible.
  int32_t cost{0};

  /// Optional validator for parameter-aware type checks.
  CastValidator validator;

  /// If true, cast only changes type metadata without data transformation.
  bool typeOnlyCoercion{false};

  bool operator==(const CastRule& other) const {
    return fromType == other.fromType && toType == other.toType &&
        implicitAllowed == other.implicitAllowed && cost == other.cost &&
        typeOnlyCoercion == other.typeOnlyCoercion;
  }

  std::string toString() const {
    return fmt::format(
        "CastRule({} -> {}, implicit={}, cost={}, typeOnly={}, hasValidator={})",
        fromType,
        toType,
        implicitAllowed,
        cost,
        typeOnlyCoercion,
        validator != nullptr);
  }
};

} // namespace facebook::velox
