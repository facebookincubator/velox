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

#include "velox/type/CastRegistry.h"

#include <folly/synchronization/CallOnce.h>

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

// Registers implicit coercion rules for built-in types in the
// CastRulesRegistry. Idempotent — safe to call from multiple overloads.
// TODO: Migrate as part of phase 3 of Query-Scoped Registries.
void registerBuiltInCoercions() {
  static folly::once_flag flag;
  folly::call_once(flag, [] {
    auto add = [](const TypePtr& from, const std::vector<TypePtr>& toTypes) {
      std::vector<CastRule> rules;
      rules.reserve(toTypes.size());
      int32_t cost = 0;
      for (const auto& toType : toTypes) {
        rules.push_back(
            {from->name(),
             toType->name(),
             /*implicitAllowed=*/true,
             /*cost=*/++cost,
             /*validator=*/nullptr});
      }
      registerCastRules(rules);
    };

    add(TINYINT(), {SMALLINT(), INTEGER(), BIGINT(), REAL(), DOUBLE()});
    add(SMALLINT(), {INTEGER(), BIGINT(), REAL(), DOUBLE()});
    add(INTEGER(), {BIGINT(), REAL(), DOUBLE()});
    add(BIGINT(), {DOUBLE()});
    add(REAL(), {DOUBLE()});
    add(DECIMAL(1, 0), {REAL(), DOUBLE()});
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
  });
}

} // namespace

// static
std::optional<Coercion> TypeCoercer::coerceTypeBase(
    const TypePtr& fromType,
    const TypePtr& toType) {
  registerBuiltInCoercions();

  if (fromType->name() == toType->name()) {
    return Coercion{.type = fromType, .cost = 0};
  }

  // CastRulesRegistry is the single source of truth for all coercions
  // (built-in and custom).
  if (auto cost = CastRulesRegistry::instance().canCoerce(fromType, toType)) {
    return Coercion{.type = toType, .cost = *cost};
  }

  return std::nullopt;
}

// static
std::optional<Coercion> TypeCoercer::coerceTypeBase(
    const TypePtr& fromType,
    const std::string& toTypeName) {
  registerBuiltInCoercions();

  if (fromType->name() == toTypeName) {
    return Coercion{.type = fromType, .cost = 0};
  }

  // Look up by name first to avoid calling getType() for parametric types
  // whose factories throw with empty parameters (ARRAY, MAP, ROW, BIGINT_ENUM).
  if (fromType->size() == 0) {
    auto rule =
        CastRulesRegistry::instance().findRule(fromType->name(), toTypeName);
    if (!rule || !rule->implicitAllowed) {
      return std::nullopt;
    }
    auto toType = getType(toTypeName, {});
    if (!toType) {
      return std::nullopt;
    }
    if (rule->validator && !rule->validator(fromType, toType)) {
      return std::nullopt;
    }
    return Coercion{.type = std::move(toType), .cost = rule->cost};
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
    if (auto coercion = TypeCoercer::coerceTypeBase(fromType, toType)) {
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
    if (TypeCoercer::coerceTypeBase(a, b)) {
      return b;
    }

    if (TypeCoercer::coerceTypeBase(b, a)) {
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

} // namespace facebook::velox
