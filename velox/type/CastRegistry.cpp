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

#include <algorithm>

namespace facebook::velox {

// static
CastRegistry& CastRegistry::instance() {
  static CastRegistry registry;
  return registry;
}

void CastRegistry::registerCastRules(
    const std::string& customTypeName,
    const std::vector<CastRule>& rules) {
  std::lock_guard<std::mutex> lock(mutex_);

  for (const auto& rule : rules) {
    // Validate that the rule involves the custom type.
    VELOX_CHECK(
        rule.fromType == customTypeName || rule.toType == customTypeName,
        "CastRule must have '{}' as either fromType or toType, got: {} -> {}",
        customTypeName,
        rule.fromType,
        rule.toType);

    auto key = std::make_pair(rule.fromType, rule.toType);

    // Check for duplicate rules.
    if (rules_.find(key) != rules_.end()) {
      VELOX_CHECK(
          rules_[key] == rule,
          "Conflicting CastRule for {} -> {}: existing={}, new={}",
          rule.fromType,
          rule.toType,
          rules_[key].toString(),
          rule.toString());
      continue;
    }

    rules_[key] = rule;
    rulesByFromType_[rule.fromType].push_back(rule);
    rulesByToType_[rule.toType].push_back(rule);
  }

  // Sort indexes by cost.
  for (auto& [_, ruleVec] : rulesByFromType_) {
    std::sort(ruleVec.begin(), ruleVec.end(), [](const auto& a, const auto& b) {
      return a.cost < b.cost;
    });
  }
  for (auto& [_, ruleVec] : rulesByToType_) {
    std::sort(ruleVec.begin(), ruleVec.end(), [](const auto& a, const auto& b) {
      return a.cost < b.cost;
    });
  }
}

void CastRegistry::unregisterCastRules(const std::string& customTypeName) {
  std::lock_guard<std::mutex> lock(mutex_);

  // Remove from main map.
  for (auto it = rules_.begin(); it != rules_.end();) {
    if (it->second.fromType == customTypeName ||
        it->second.toType == customTypeName) {
      it = rules_.erase(it);
    } else {
      ++it;
    }
  }

  // Remove from indexes.
  rulesByFromType_.erase(customTypeName);
  rulesByToType_.erase(customTypeName);

  // Remove rules involving this type from other type's indexes.
  for (auto& [_, ruleVec] : rulesByFromType_) {
    ruleVec.erase(
        std::remove_if(
            ruleVec.begin(),
            ruleVec.end(),
            [&](const CastRule& r) { return r.toType == customTypeName; }),
        ruleVec.end());
  }
  for (auto& [_, ruleVec] : rulesByToType_) {
    ruleVec.erase(
        std::remove_if(
            ruleVec.begin(),
            ruleVec.end(),
            [&](const CastRule& r) { return r.fromType == customTypeName; }),
        ruleVec.end());
  }
}

bool CastRegistry::canCast(const TypePtr& fromType, const TypePtr& toType)
    const {
  return canCastImpl(fromType, toType, /*requireImplicit=*/false);
}

bool CastRegistry::canImplicitCast(
    const TypePtr& fromType,
    const TypePtr& toType) const {
  return canCastImpl(fromType, toType, /*requireImplicit=*/true);
}

bool CastRegistry::canCastImpl(
    const TypePtr& fromType,
    const TypePtr& toType,
    bool requireImplicit,
    int32_t* outCost) const {
  if (!fromType || !toType) {
    return false;
  }

  // Same type - always allowed with zero cost.
  if (fromType->equivalent(*toType)) {
    if (outCost) {
      *outCost = 0;
    }
    return true;
  }

  const auto fromName = fromType->name();
  const auto toName = toType->name();

  // For primitive types (no children), look up the rule directly.
  if (fromType->size() == 0 && toType->size() == 0) {
    auto rule = findRule(fromName, toName);
    if (!rule) {
      return false;
    }
    if (requireImplicit && !rule->implicitAllowed) {
      return false;
    }
    // If there's a validator, run it to check type-specific compatibility.
    if (rule->validator && !rule->validator(fromType, toType)) {
      return false;
    }
    if (outCost) {
      *outCost = rule->cost;
    }
    return true;
  }

  // For parametric types (ARRAY, MAP, ROW), the base type must match
  // and we recursively check the element types.
  if (fromName != toName) {
    // Different base types - check if there's a rule.
    auto rule = findRule(fromName, toName);
    if (!rule) {
      return false;
    }
    if (requireImplicit && !rule->implicitAllowed) {
      return false;
    }
    // If there's a validator, run it to check type-specific compatibility.
    if (rule->validator && !rule->validator(fromType, toType)) {
      return false;
    }
    if (outCost) {
      *outCost = rule->cost;
    }
    return true;
  }

  // Same base type with children - recursively check children.
  if (fromType->size() != toType->size()) {
    return false;
  }

  int32_t totalCost = 0;
  for (size_t i = 0; i < fromType->size(); ++i) {
    int32_t childCost = 0;
    if (!canCastImpl(
            fromType->childAt(i),
            toType->childAt(i),
            requireImplicit,
            &childCost)) {
      return false;
    }
    totalCost += childCost;
  }

  if (outCost) {
    *outCost = totalCost;
  }
  return true;
}

std::vector<CastRule> CastRegistry::getCastsFrom(
    const std::string& typeName) const {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = rulesByFromType_.find(typeName);
  if (it == rulesByFromType_.end()) {
    return {};
  }
  return it->second;
}

std::vector<CastRule> CastRegistry::getCastsTo(
    const std::string& typeName) const {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = rulesByToType_.find(typeName);
  if (it == rulesByToType_.end()) {
    return {};
  }
  return it->second;
}

std::optional<CastRule> CastRegistry::findRule(
    const std::string& fromTypeName,
    const std::string& toTypeName) const {
  std::lock_guard<std::mutex> lock(mutex_);
  auto key = std::make_pair(fromTypeName, toTypeName);
  auto it = rules_.find(key);
  if (it == rules_.end()) {
    return std::nullopt;
  }
  return it->second;
}

std::optional<int32_t> CastRegistry::getCastCost(
    const TypePtr& fromType,
    const TypePtr& toType) const {
  int32_t cost = 0;
  if (canCastImpl(fromType, toType, /*requireImplicit=*/false, &cost)) {
    return cost;
  }
  return std::nullopt;
}

TypeCompatibility CastRegistry::getTypeCompatibility(
    const TypePtr& fromType,
    const TypePtr& toType) const {
  if (!fromType || !toType) {
    return TypeCompatibility::incompatible();
  }

  // Same type - always compatible with zero cost.
  if (fromType->equivalent(*toType)) {
    return TypeCompatibility::sameType();
  }

  // UNKNOWN type (NULL literal) coerces to anything.
  if (isUnknownType(fromType)) {
    return TypeCompatibility::compatibleWith(
        /*cost=*/0, /*coercible=*/true, /*typeOnly=*/true);
  }

  // Check if cast is possible.
  int32_t cost = 0;
  bool canExplicit =
      canCastImpl(fromType, toType, /*requireImplicit=*/false, &cost);
  if (!canExplicit) {
    return TypeCompatibility::incompatible();
  }

  // Check if implicit coercion is allowed.
  bool canImplicit = canCastImpl(fromType, toType, /*requireImplicit=*/true);

  // Look up rule details for typeOnlyCoercion.
  auto rule = findRule(fromType->name(), toType->name());
  bool typeOnly = rule ? rule->typeOnlyCoercion : false;

  return TypeCompatibility::compatibleWith(cost, canImplicit, typeOnly);
}

std::optional<TypePtr> CastRegistry::getCommonSuperType(
    const TypePtr& typeA,
    const TypePtr& typeB) const {
  if (!typeA || !typeB) {
    return std::nullopt;
  }

  // Same type - return it.
  if (typeA->equivalent(*typeB)) {
    return typeA;
  }

  // UNKNOWN type (NULL literal) - the other type is the common super type.
  if (isUnknownType(typeA)) {
    return typeB;
  }
  if (isUnknownType(typeB)) {
    return typeA;
  }

  // Check if A can coerce to B.
  int32_t costAtoB = 0;
  bool aToB = canCastImpl(typeA, typeB, /*requireImplicit=*/true, &costAtoB);

  // Check if B can coerce to A.
  int32_t costBtoA = 0;
  bool bToA = canCastImpl(typeB, typeA, /*requireImplicit=*/true, &costBtoA);

  // If only one direction works, use that.
  if (aToB && !bToA) {
    return typeB;
  }
  if (bToA && !aToB) {
    return typeA;
  }

  // Both directions work - prefer lower cost.
  if (aToB && bToA) {
    return (costAtoB <= costBtoA) ? typeB : typeA;
  }

  // No implicit cast in either direction.
  return std::nullopt;
}

// static
bool CastRegistry::isUnknownType(const TypePtr& type) {
  return type && type->kind() == TypeKind::UNKNOWN;
}

void CastRegistry::clear() {
  std::lock_guard<std::mutex> lock(mutex_);
  rules_.clear();
  rulesByFromType_.clear();
  rulesByToType_.clear();
}

} // namespace facebook::velox
