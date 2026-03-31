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
  return canCastImpl(fromType, toType, /*requireImplicit=*/false);
}

bool CastRulesRegistry::canCoerce(
    const TypePtr& fromType,
    const TypePtr& toType) const {
  return canCastImpl(fromType, toType, /*requireImplicit=*/true);
}

bool CastRulesRegistry::canCastImpl(
    const TypePtr& fromType,
    const TypePtr& toType,
    bool requireImplicit) const {
  VELOX_CHECK_NOT_NULL(fromType, "fromType must not be null");
  VELOX_CHECK_NOT_NULL(toType, "toType must not be null");

  // Cast rules are only registered for non-parametric (leaf) types. Parametric
  // types (ARRAY, MAP, ROW, and custom types with children like IPPREFIX) are
  // handled by callers (TypeCoercer::coercible, leastCommonSuperType) which
  // recurse into children before reaching this method. Return false for
  // parametric types rather than throwing — this is a query function.
  if (fromType->size() != 0 || toType->size() != 0) {
    return false;
  }

  // Same type is always allowed.
  if (fromType->equivalent(*toType)) {
    return true;
  }

  auto rule = findRule(fromType->name(), toType->name());
  if (!rule) {
    return false;
  }
  if (requireImplicit && !rule->implicitAllowed) {
    return false;
  }
  if (rule->validator && !rule->validator(fromType, toType)) {
    return false;
  }
  return true;
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
