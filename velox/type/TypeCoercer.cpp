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
  static const auto kAllowedCoercions = allowedCoercions();
  if (fromType->name() == toTypeName) {
    return {{}, 0};
  }

  if (fromType == UNKNOWN()) {
    // Cast unknown to complex type in function is not supported yet
    return {getType(toTypeName, {}), kNullCoercionCost};
  }

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

} // namespace facebook::velox
