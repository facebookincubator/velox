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

std::vector<CoercionEntry> defaultRules() {
  std::vector<CoercionEntry> rules;

  auto add = [&](const TypePtr& from, const std::vector<TypePtr>& to) {
    int32_t cost = 0;
    for (const auto& toType : to) {
      rules.push_back({from, toType, ++cost});
    }
  };

  add(TINYINT(),
      {SMALLINT(), INTEGER(), BIGINT(), DECIMAL(3, 0), REAL(), DOUBLE()});
  add(SMALLINT(), {INTEGER(), BIGINT(), DECIMAL(5, 0), REAL(), DOUBLE()});
  add(INTEGER(), {BIGINT(), DECIMAL(10, 0), REAL(), DOUBLE()});
  add(BIGINT(), {DECIMAL(19, 0), DOUBLE()});
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

  return rules;
}

} // namespace

TypeCoercer::TypeCoercer(const std::vector<CoercionEntry>& rules) {
  // Tracks costs already used for a given source type to enforce uniqueness.
  std::unordered_map<std::string, std::unordered_set<int32_t>> costsByFrom;

  for (const auto& entry : rules) {
    VELOX_CHECK_NOT_NULL(entry.from, "CoercionEntry.from must not be null");
    VELOX_CHECK_NOT_NULL(entry.to, "CoercionEntry.to must not be null");

    // Built-in -> built-in only. Custom-type coercions go through
    // CastRulesRegistry and must not appear in a TypeCoercer rule set.
    //
    // Timing note: customTypeExists() checks the global custom-type
    // registry at construction time. Today's TypeCoercer instances are
    // lazy Meyers singletons (TypeCoercer::defaults(),
    // presto::typeCoercer()) initialized on first use, so by then all
    // relevant custom types are registered. If a future caller
    // constructs a TypeCoercer eagerly (e.g., during static
    // initialization), a rule with a name not yet registered as custom
    // could pass this check but later collide with that type. Keep
    // TypeCoercer construction lazy.
    VELOX_CHECK(
        !customTypeExists(entry.from->name()),
        "CoercionEntry.from must be a built-in type, got custom type {}",
        entry.from->name());
    VELOX_CHECK(
        !customTypeExists(entry.to->name()),
        "CoercionEntry.to must be a built-in type, got custom type {}",
        entry.to->name());

    if (entry.from->isDecimal()) {
      // DECIMAL -> DECIMAL is not honored: the same-name short-circuit at
      // the top of coerceTypeBase returns cost 0 before the rule lookup
      // runs. Reject these entries so users don't silently believe a
      // rule is in effect.
      VELOX_CHECK(
          !entry.to->isDecimal(),
          "DECIMAL -> DECIMAL coercion is not customizable via TypeCoercer. "
          "DECIMAL -> DECIMAL reconciliation is handled by the type system "
          "(LongDecimalType::commonSuperType). Rejecting rule {} -> {}.",
          entry.from->toString(),
          entry.to->toString());

      // Source DECIMAL must be the canonical placeholder DECIMAL(1, 0).
      // Source (p, s) is irrelevant at lookup -- the rule fires for any
      // DECIMAL(p, s) -- so allowing other values here would create a
      // false impression that callers can scope a rule to a particular
      // precision or scale.
      VELOX_CHECK(
          entry.from->equivalent(*DECIMAL(1, 0)),
          "Source DECIMAL in CoercionEntry must be DECIMAL(1, 0); got {}. "
          "TypeCoercer does not distinguish source decimals by (p, s).",
          entry.from->toString());
    }

    // Cost must be unique per source type so overload resolution has a
    // deterministic preference order.
    auto& seen = costsByFrom[entry.from->name()];
    VELOX_CHECK(
        seen.insert(entry.cost).second,
        "Duplicate cost {} for source type {} in TypeCoercer rule set",
        entry.cost,
        entry.from->name());

    // Reject duplicate (fromName, toName) pairs. This guards in particular
    // against the common DECIMAL footgun: all decimals share the same name,
    // so multiple {SOURCE, DECIMAL(p, s), cost} entries with different
    // (p, s) collide on the same map key. Only one DECIMAL rule per source
    // is supported (the minimum-width decimal -- see CoercionEntry doc).
    auto [it, inserted] = rules_.emplace(
        std::make_pair(entry.from->name(), entry.to->name()),
        Coercion{.type = entry.to, .cost = entry.cost});
    VELOX_CHECK(
        inserted,
        "Duplicate coercion rule {} -> {} in TypeCoercer rule set",
        entry.from->name(),
        entry.to->name());
  }
}

// static
const TypeCoercer& TypeCoercer::defaults() {
  static const TypeCoercer instance{defaultRules()};
  return instance;
}

std::optional<Coercion> TypeCoercer::coerceTypeBase(
    const TypePtr& fromType,
    const TypePtr& toType) const {
  if (fromType->name() == toType->name()) {
    return Coercion{.type = fromType, .cost = 0};
  }

  // Check this coercer's rule set first.
  auto it = rules_.find({fromType->name(), toType->name()});
  if (it != rules_.end()) {
    if (toType->isDecimal() && it->second.type->isDecimal()) {
      if (it->second.type->isShortDecimal()) {
        if (!it->second.type->asShortDecimal().isCoercibleTo(*toType)) {
          return std::nullopt;
        }
      } else {
        if (!it->second.type->asLongDecimal().isCoercibleTo(*toType)) {
          return std::nullopt;
        }
      }
      return Coercion{.type = toType, .cost = it->second.cost};
    }
    return it->second;
  }

  // Fall back to CastRulesRegistry for custom-type coercions. Custom-type
  // names are dialect-distinct in practice, so the global registry doesn't
  // cross-contaminate dialects.
  if (auto cost = CastRulesRegistry::instance().canCoerce(fromType, toType)) {
    return Coercion{.type = toType, .cost = *cost};
  }

  return std::nullopt;
}

std::optional<Coercion> TypeCoercer::coerceTypeBase(
    const TypePtr& fromType,
    const std::string& toTypeName) const {
  if (fromType->name() == toTypeName) {
    return Coercion{.type = fromType, .cost = 0};
  }

  // Check this coercer's rule set first.
  auto it = rules_.find({fromType->name(), toTypeName});
  if (it != rules_.end()) {
    return it->second;
  }

  // Fall back to CastRulesRegistry for custom type coercions. Skip
  // parameterized types -- we cannot construct the target type without knowing
  // its type parameters.
  // getCustomType() returns nullptr for built-in types. Callers must not
  // pass parametric custom type names (e.g., "BIGINT_ENUM") because their
  // factories throw when called with empty parameters. SignatureBinder
  // guards against this by checking typeSignature.parameters().empty().
  if (fromType->size() == 0 && fromType->parameters().empty()) {
    if (auto toType = getCustomType(toTypeName, {})) {
      if (auto cost =
              CastRulesRegistry::instance().canCoerce(fromType, toType)) {
        return Coercion{.type = std::move(toType), .cost = *cost};
      }
    }
  }

  return std::nullopt;
}

std::optional<int32_t> TypeCoercer::coercible(
    const TypePtr& fromType,
    const TypePtr& toType) const {
  if (fromType->isUnknown()) {
    if (toType->isUnknown()) {
      return 0;
    }
    return 1;
  }

  if (fromType->size() == 0) {
    if (auto coercion = coerceTypeBase(fromType, toType)) {
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

TypePtr leastCommonSuperRowType(
    const TypeCoercer& coercer,
    const RowType& a,
    const RowType& b) {
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
            coercer.leastCommonSuperType(a.childAt(i), b.childAt(i))) {
      childTypes.push_back(childType);
    } else {
      return nullptr;
    }
  }

  return ROW(std::move(childNames), std::move(childTypes));
}

} // namespace

TypePtr TypeCoercer::leastCommonSuperType(const TypePtr& a, const TypePtr& b)
    const {
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
    if (a->isDecimal() || b->isDecimal()) {
      if (auto result = LongDecimalType::commonSuperType(a, b)) {
        return result;
      }
    }

    if (coerceTypeBase(a, b)) {
      return b;
    }

    if (coerceTypeBase(b, a)) {
      return a;
    }

    return nullptr;
  }

  if (a->name() != b->name()) {
    return nullptr;
  }

  if (a->name() == TypeKindName::toName(TypeKind::ROW)) {
    return leastCommonSuperRowType(*this, a->asRow(), b->asRow());
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
