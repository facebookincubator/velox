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

#include "velox/type/CastRegistry.h"

namespace facebook::velox {

// static
CastRulesRegistry& CastRulesRegistry::instance() {
  static CastRulesRegistry registry;
  return registry;
}

void CastRulesRegistry::registerCastRules(const std::vector<CastRule>& rules) {
  auto lockedRules = rules_.wlock();

  for (const auto& rule : rules) {
    auto key = std::make_pair(rule.fromType, rule.toType);

    auto it = lockedRules->find(key);
    if (it != lockedRules->end()) {
      VELOX_CHECK(
          it->second == rule,
          "Conflicting CastRule for {} -> {}: existing={}, new={}",
          rule.fromType,
          rule.toType,
          it->second.toString(),
          rule.toString());
      continue;
    }

    (*lockedRules)[key] = rule;
  }
}

void CastRulesRegistry::unregisterCastRules(const std::string& typeName) {
  auto lockedRules = rules_.wlock();

  for (auto it = lockedRules->begin(); it != lockedRules->end();) {
    if (it->second.fromType == typeName || it->second.toType == typeName) {
      it = lockedRules->erase(it);
    } else {
      ++it;
    }
  }
}

bool CastRulesRegistry::canCast(const TypePtr& fromType, const TypePtr& toType)
    const {
  return castCostImpl(fromType, toType, /*requireImplicit=*/false).has_value();
}

std::optional<int32_t> CastRulesRegistry::canCoerce(
    const TypePtr& fromType,
    const TypePtr& toType) const {
  return castCostImpl(fromType, toType, /*requireImplicit=*/true);
}

std::optional<int32_t> CastRulesRegistry::castCostImpl(
    const TypePtr& fromType,
    const TypePtr& toType,
    bool requireImplicit) const {
  VELOX_CHECK_NOT_NULL(fromType, "fromType must not be null");
  VELOX_CHECK_NOT_NULL(toType, "toType must not be null");

  // Same type is always allowed with zero cost.
  if (fromType->equivalent(*toType)) {
    return 0;
  }

  const auto fromName = fromType->name();
  const auto toName = toType->name();

  // Try direct rule lookup first. This handles primitives and custom types
  // including those with children like IPPREFIX (which extends RowType).
  auto rule = findRule(fromName, toName);
  if (rule) {
    if (requireImplicit && !rule->implicitAllowed) {
      return std::nullopt;
    }
    if (rule->validator && !rule->validator(fromType, toType)) {
      return std::nullopt;
    }
    return requireImplicit ? rule->cost : 0;
  }

  // No explicit rule. For container types (ARRAY, MAP, ROW) with the same
  // base type, recursively check children and sum costs.
  if (fromName == toName && fromType->size() > 0) {
    if (fromType->size() != toType->size()) {
      return std::nullopt;
    }
    int32_t totalCost = 0;
    for (auto i = 0; i < fromType->size(); ++i) {
      auto childCost = castCostImpl(
          fromType->childAt(i), toType->childAt(i), requireImplicit);
      if (!childCost) {
        return std::nullopt;
      }
      totalCost += *childCost;
    }
    return totalCost;
  }

  // Different base types with children — not supported.
  return std::nullopt;
}

std::optional<CastRule> CastRulesRegistry::findRule(
    const std::string& fromTypeName,
    const std::string& toTypeName) const {
  auto lockedRules = rules_.rlock();
  auto it = lockedRules->find(std::make_pair(fromTypeName, toTypeName));
  if (it == lockedRules->end()) {
    return std::nullopt;
  }
  return it->second;
}

void CastRulesRegistry::clear() {
  rules_.wlock()->clear();
}

void registerCastRules(const std::vector<CastRule>& rules) {
  for (const auto& rule : rules) {
    // At least one side must be a registered custom type with a CastOperator,
    // otherwise the cast has no execution path.
    VELOX_CHECK(
        getCustomTypeCastOperator(rule.fromType) != nullptr ||
            getCustomTypeCastOperator(rule.toType) != nullptr,
        "CastRule {} -> {} requires at least one side to be a registered "
        "custom type with a CastOperator",
        rule.fromType,
        rule.toType);
  }
  CastRulesRegistry::instance().registerCastRules(rules);
}

void unregisterCastRules(const std::string& typeName) {
  CastRulesRegistry::instance().unregisterCastRules(typeName);
}

} // namespace facebook::velox
