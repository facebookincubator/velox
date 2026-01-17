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

CallableCost Coercion::overallCost(const std::vector<Coercion>& coercions) {
  CallableCost cost = 0;
  for (const auto& coercion : coercions) {
    VELOX_DCHECK(coercion);
    cost += coercion.cost;
  }
  return cost;
}

void Coercion::convert(
    const std::vector<Coercion>& from,
    std::vector<TypePtr>* to) {
  if (!to) {
    return;
  }
  to->clear();
  to->reserve(from.size());
  for (const auto& coercion : from) {
    VELOX_DCHECK(coercion);
    to->push_back(coercion.type);
  }
}

namespace {

// This is cost of CAST from UNKNOWN type to any other type.
// This is the lowest for any implicit CAST.
// Any other implicit CAST will have cost higher than this.
constexpr CallableCost kNullCoercionCost = 1;

std::unordered_map<std::pair<std::string, std::string>, Coercion>
allowedCoercions() {
  std::unordered_map<std::pair<std::string, std::string>, Coercion> coercions;

  auto add = [&](const TypePtr& from, const std::vector<TypePtr>& to) {
    auto cost = kNullCoercionCost;
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

  return coercions;
}

} // namespace

// static
Coercion TypeCoercer::coerceTypeBase(
    const TypePtr& fromType,
    const std::string& toTypeName) {
  if (fromType->name() == toTypeName) {
    return {{}, 0};
  }

  if (fromType == UNKNOWN()) {
    // Cast unknown to complex type in function is not supported yet
    return {getType(toTypeName, {}), kNullCoercionCost};
  }

  static const auto kAllowedCoercions = allowedCoercions();
  auto it = kAllowedCoercions.find({fromType->name(), toTypeName});
  if (it != kAllowedCoercions.end()) {
    return it->second;
  }

  return {{}, kImpossibleCoercionCost};
}

// static
Coercion TypeCoercer::coercible(
    const TypePtr& fromType,
    const TypePtr& toType) {
  if (fromType->equivalent(*toType)) {
    return {{}, 0};
  }

  if (fromType == UNKNOWN()) {
    return {toType, kNullCoercionCost};
  }

  if (fromType->size() != toType->size()) {
    return {{}, kImpossibleCoercionCost};
  }

  if (fromType->size() == 0) {
    return TypeCoercer::coerceTypeBase(fromType, toType->name());
  }

  if (fromType->name() != toType->name()) {
    return {{}, kImpossibleCoercionCost};
  }

  CallableCost cost = 0;
  for (size_t i = 0; i < fromType->size(); i++) {
    if (auto c = coercible(fromType->childAt(i), toType->childAt(i))) {
      cost += c.cost;
    } else {
      return {{}, kImpossibleCoercionCost};
    }
  }

  return {toType, cost};
}

namespace {

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
  if (!b) {
    return a;
  }

  if (!a) {
    return b;
  }

  if (a->equivalent(*b)) {
    return a;
  }

  if (b == UNKNOWN()) {
    return a;
  }

  if (a == UNKNOWN()) {
    return b;
  }

  if (a->size() != b->size()) {
    return nullptr;
  }

  if (a->size() == 0) {
    if (TypeCoercer::coerceTypeBase(b, a->name())) {
      return a;
    }

    if (TypeCoercer::coerceTypeBase(a, b->name())) {
      return b;
    }

    return nullptr;
  }

  if (a->name() != b->name()) {
    return nullptr;
  }

  if (a->isRow()) {
    return leastCommonSuperRowType(a->asRow(), b->asRow());
  }

  std::vector<TypeParameter> childTypes;
  childTypes.reserve(a->size());
  for (size_t i = 0; i < a->size(); ++i) {
    if (auto childType = leastCommonSuperType(a->childAt(i), b->childAt(i))) {
      childTypes.emplace_back(childType);
    } else {
      return nullptr;
    }
  }

  return getType(a->name(), childTypes);
}

} // namespace facebook::velox
