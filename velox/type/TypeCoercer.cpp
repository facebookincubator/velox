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
#include "velox/type/TypeCoercer.h"

namespace facebook::velox {

int64_t Coercion::overallCost(const std::vector<Coercion>& coercions) {
  int64_t cost = 0;
  for (const auto& coercion : coercions) {
    if (coercion.type != nullptr) {
      cost += coercion.cost;
    }
  }

  return cost;
}

namespace {

std::unordered_map<std::pair<std::string, std::string>, Coercion>
allowedCoercions() {
  std::unordered_map<std::pair<std::string, std::string>, Coercion> coercions;

  auto add = [&](const TypePtr& from, const std::vector<TypePtr>& to) {
    int32_t cost = 0;
    for (const auto& toType : to) {
      coercions.emplace(
          std::make_pair<std::string, std::string>(
              from->name(), toType->name()),
          Coercion{.type = toType, .cost = ++cost});
    }
  };

  add(TINYINT(), {SMALLINT(), INTEGER(), BIGINT(), REAL(), DOUBLE()});
  add(SMALLINT(), {INTEGER(), BIGINT(), REAL(), DOUBLE()});
  add(INTEGER(), {BIGINT(), REAL(), DOUBLE()});
  add(BIGINT(), {DOUBLE()});
  add(REAL(), {DOUBLE()});
  add(DATE(), {TIMESTAMP()});
  add(UNKNOWN(),
      {TINYINT(),
       BOOLEAN(),
       SMALLINT(),
       INTEGER(),
       BIGINT(),
       REAL(),
       DOUBLE(),
       VARCHAR(),
       VARBINARY()});
  // add(VARCHAR(),
  //     {VARCHAR()}); // Allow coercion between different VARCHAR lengths.

  return coercions;
}
} // namespace

// static
std::optional<Coercion> TypeCoercer::coerceTypeBase(
    const TypePtr& fromType,
    const std::string& toTypeName) {
  static const auto kAllowedCoercions = allowedCoercions();
  // Allow name coercion on name alone for primitive types without parameters
  // and non-primitive types.
  if (fromType->name() == toTypeName) {
    //&&
    //((fromType->isPrimitiveType() && fromType->parameters().empty()) ||
    // !fromType->isPrimitiveType())) {
    VLOG(0) << "Allow from " << fromType->toString() << " to type name "
            << toTypeName;
    return Coercion{.type = fromType, .cost = 0};
  }

  auto it = kAllowedCoercions.find({fromType->name(), toTypeName});
  if (it != kAllowedCoercions.end()) {
    return it->second;
  }

  return std::nullopt;
}

// static
std::optional<int32_t> TypeCoercer::coercible(
    const TypePtr& fromType,
    const TypePtr& toType) {
  if (fromType->isUnknown()) {
    if (toType->isUnknown()) {
      return 0;
    }
    return 1;
  }

  if (fromType->size() == 0) {
    if (auto coercion = TypeCoercer::coerceTypeBase(fromType, toType->name())) {
      return coercion->cost;
    }

    return std::nullopt;
  }

  if (fromType->name() != toType->name() ||
      fromType->size() != toType->size()) {
    return std::nullopt;
  }

  int32_t totalCost = 0;
  for (auto i = 0; i < fromType->size(); i++) {
    if (auto cost = coercible(fromType->childAt(i), toType->childAt(i))) {
      totalCost += cost.value();
    } else {
      return std::nullopt;
    }
  }

  return totalCost;
}

namespace {

bool isSameStringOrBinaryType(const TypePtr& fromType, const TypePtr& toType) {
  return (fromType->kind() == toType->kind()) &&
      (fromType->kind() == TypeKind::VARCHAR ||
       fromType->kind() == TypeKind::VARBINARY);
}

bool isDecimalTypeOnlyCoercion(const TypePtr& fromType, const TypePtr& toType) {
  if (!((fromType->isShortDecimal() || fromType->isLongDecimal()) &&
        (toType->isShortDecimal() || toType->isLongDecimal()))) {
    return false;
  }

  const auto [fromPrecision, fromScale] = getDecimalPrecisionScale(*fromType);
  const auto [toPrecision, toScale] = getDecimalPrecisionScale(*toType);

  // Type-only coercion for decimals requires:
  // 1. Same decimal subtype (both short or both long)
  // 2. Same scale
  // 3. Source precision <= result precision
  const bool sameDecimalSubtype =
      (fromType->isShortDecimal() && toType->isShortDecimal()) ||
      (fromType->isLongDecimal() && toType->isLongDecimal());
  const bool sameScale = fromScale == toScale;
  const bool sourcePrecisionIsLessOrEqualToResultPrecision =
      fromPrecision <= toPrecision;

  return sameDecimalSubtype && sameScale &&
      sourcePrecisionIsLessOrEqualToResultPrecision;
}

TypePtr leastCommonSuperRowType(const RowType& a, const RowType& b) {
  std::vector<std::string> childNames;
  childNames.reserve(a.size());

  const auto& aNames = a.names();
  const auto& bNames = b.names();

  for (auto i = 0; i < a.size(); i++) {
    if (aNames[i] == bNames[i]) {
      childNames.push_back(aNames[i]);
    } else {
      childNames.push_back("");
    }
  }

  std::vector<TypePtr> childTypes;
  childTypes.reserve(a.size());
  for (auto i = 0; i < a.size(); i++) {
    if (auto childType =
            TypeCoercer::leastCommonSuperType(a.childAt(i), b.childAt(i))) {
      childTypes.push_back(childType);
    } else {
      return nullptr;
    }
  }

  return ROW(std::move(childNames), std::move(childTypes));
}
} // namespace

// static
TypePtr TypeCoercer::leastCommonSuperType(const TypePtr& a, const TypePtr& b) {
  if (a->isUnknown()) {
    return b;
  }

  if (b->isUnknown()) {
    return a;
  }

  if (a->size() != b->size()) {
    return nullptr;
  }

  if (a->size() == 0) {
    if (TypeCoercer::coerceTypeBase(a, b->name())) {
      return b;
    }

    if (TypeCoercer::coerceTypeBase(b, a->name())) {
      return a;
    }

    return nullptr;
  }

  if (a->name() != b->name()) {
    return nullptr;
  }

  if (a->name() == TypeKindName::toName(TypeKind::ROW)) {
    return leastCommonSuperRowType(a->asRow(), b->asRow());
  }

  std::vector<TypeParameter> childTypes;
  childTypes.reserve(a->size());
  for (auto i = 0; i < a->size(); i++) {
    if (auto childType = leastCommonSuperType(a->childAt(i), b->childAt(i))) {
      childTypes.push_back(TypeParameter(childType));
    } else {
      return nullptr;
    }
  }

  return getType(a->name(), childTypes);
}

// static
bool TypeCoercer::isTypeOnlyCoercion(
    const TypePtr& fromType,
    const TypePtr& toType) {
  // If types are equal, it's always type-only coercion
  if (*fromType == *toType) {
    return true;
  }

  // Handle VARCHAR/VARBINARY type coercion (bounded <-> unbounded)
  if (isSameStringOrBinaryType(fromType, toType)) {
    return true;
  }

  // Handle DECIMAL type coercion
  if (isDecimalTypeOnlyCoercion(fromType, toType)) {
    return true;
  }

  // Handle covariant parametrized types (ARRAY, MAP, ROW)
  if (fromType->name() == toType->name() && fromType->size() > 0) {
    const auto fromTypeParams = fromType->parameters();
    const auto toTypeParams = toType->parameters();

    if (fromTypeParams.size() != toTypeParams.size()) {
      return false;
    }

    // Recursively check all type parameters
    return std::all_of(
        fromTypeParams.begin(),
        fromTypeParams.end(),
        [&toTypeParams, i = size_t{0}](const TypeParameter& fromParam) mutable {
          return isTypeOnlyCoercion(fromParam.type, toTypeParams[i++].type);
        });
  }

  // For all other cases, not a type-only coercion
  return false;
}

} // namespace facebook::velox
