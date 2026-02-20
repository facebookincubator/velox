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

#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <folly/Hash.h>

#include "velox/type/CastRule.h"
#include "velox/type/Type.h"

namespace facebook::velox {

/// Result of type compatibility check.
struct TypeCompatibility {
  bool compatible{false};
  bool coercible{false};
  int32_t cost{0};
  bool typeOnlyCoercion{false};

  static TypeCompatibility
  compatibleWith(int32_t cost, bool coercible = false, bool typeOnly = false) {
    return TypeCompatibility{
        .compatible = true,
        .coercible = coercible,
        .cost = cost,
        .typeOnlyCoercion = typeOnly};
  }

  static TypeCompatibility incompatible() {
    return TypeCompatibility{.compatible = false};
  }

  static TypeCompatibility sameType() {
    return TypeCompatibility{
        .compatible = true,
        .coercible = true,
        .cost = 0,
        .typeOnlyCoercion = true};
  }
};

/// Registry for cast rules between types. Provides lookup for determining
/// whether casts are supported and whether they can be implicit.
class CastRegistry {
 public:
  static CastRegistry& instance();

  /// Register cast rules. Each rule must involve the custom type.
  void registerCastRules(
      const std::string& customTypeName,
      const std::vector<CastRule>& rules);

  void unregisterCastRules(const std::string& customTypeName);

  /// Check if cast is supported (explicit or implicit).
  bool canCast(const TypePtr& fromType, const TypePtr& toType) const;

  /// Check if implicit coercion is allowed.
  bool canImplicitCast(const TypePtr& fromType, const TypePtr& toType) const;

  /// Get all rules where typeName is the source type, sorted by cost.
  std::vector<CastRule> getCastsFrom(const std::string& typeName) const;

  /// Get all rules where typeName is the target type, sorted by cost.
  std::vector<CastRule> getCastsTo(const std::string& typeName) const;

  std::optional<CastRule> findRule(
      const std::string& fromTypeName,
      const std::string& toTypeName) const;

  std::optional<int32_t> getCastCost(
      const TypePtr& fromType,
      const TypePtr& toType) const;

  TypeCompatibility getTypeCompatibility(
      const TypePtr& fromType,
      const TypePtr& toType) const;

  /// Find common super type for UNION/CASE expressions.
  std::optional<TypePtr> getCommonSuperType(
      const TypePtr& typeA,
      const TypePtr& typeB) const;

  /// UNKNOWN type (NULL literal) coerces to any type.
  static bool isUnknownType(const TypePtr& type);

  void clear();

 private:
  CastRegistry() = default;

  bool canCastImpl(
      const TypePtr& fromType,
      const TypePtr& toType,
      bool requireImplicit,
      int32_t* outCost = nullptr) const;

  struct PairHash {
    size_t operator()(const std::pair<std::string, std::string>& p) const {
      return folly::hash::hash_combine(
          std::hash<std::string>{}(p.first),
          std::hash<std::string>{}(p.second));
    }
  };

  std::unordered_map<std::pair<std::string, std::string>, CastRule, PairHash>
      rules_;
  std::unordered_map<std::string, std::vector<CastRule>> rulesByFromType_;
  std::unordered_map<std::string, std::vector<CastRule>> rulesByToType_;
  mutable std::mutex mutex_;
};

} // namespace facebook::velox
