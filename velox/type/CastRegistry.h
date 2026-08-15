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

#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <folly/Hash.h>
#include <folly/Synchronized.h>

#include "velox/type/CastRule.h"
#include "velox/type/Type.h"

namespace facebook::velox {

/// Registry for cast rules between types. Provides lookup for determining
/// whether casts are supported and whether they can be implicit (coercible).
///
/// Rules are registered via the registerCastRules() free function, typically
/// called alongside registerCustomType() in register*Type() functions.
/// This keeps rules co-located with the type while storing them in a global
/// registry for centralized querying.
class CastRulesRegistry {
 public:
  static CastRulesRegistry& instance();

  /// Register cast rules. Rules are validated for duplicates — if a rule
  /// for the same (fromType, toType) pair already exists, it must be
  /// identical, otherwise registration fails.
  void registerCastRules(const std::vector<CastRule>& rules);

  /// Unregister all rules involving the given type name.
  void unregisterCastRules(const std::string& typeName);

  /// Check if cast is supported (explicit or implicit). Handles container
  /// types (ARRAY, MAP, ROW) by recursively checking children.
  bool canCast(const TypePtr& fromType, const TypePtr& toType) const;

  /// Check if implicit coercion is allowed. Returns the coercion cost if
  /// allowed, or std::nullopt if not. For container types (ARRAY, MAP, ROW),
  /// returns the sum of children coercion costs.
  std::optional<int32_t> canCoerce(
      const TypePtr& fromType,
      const TypePtr& toType) const;

  /// Clear all registered rules. Used for testing.
  void clear();

 private:
  CastRulesRegistry() = default;

  // Unified implementation. When requireImplicit is true, returns the cost
  // of implicit coercion. When false, returns 0 for any supported cast.
  // Returns nullopt if the cast/coercion is not supported.
  std::optional<int32_t> castCostImpl(
      const TypePtr& fromType,
      const TypePtr& toType,
      bool requireImplicit) const;

  std::optional<CastRule> findRule(
      const std::string& fromTypeName,
      const std::string& toTypeName) const;

  struct PairHash {
    size_t operator()(const std::pair<std::string, std::string>& p) const {
      return folly::hash::hash_combine(
          std::hash<std::string>{}(p.first),
          std::hash<std::string>{}(p.second));
    }
  };

  // Maps (fromType, toType) pairs to their cast rules.
  folly::Synchronized<std::unordered_map<
      std::pair<std::string, std::string>,
      CastRule,
      PairHash>>
      rules_;
};

/// Register cast rules for custom types. Call this after registerCustomType()
/// in register*Type() functions. Validates that at least one type in each rule
/// has a registered CastOperator.
///
/// Example:
///   registerCastRules({
///       {"TIMESTAMP", "TIMESTAMP WITH TIME ZONE", true},
///       {"DATE", "TIMESTAMP WITH TIME ZONE", true},
///   });
void registerCastRules(const std::vector<CastRule>& rules);

/// Unregister all cast rules involving the given type name.
void unregisterCastRules(const std::string& typeName);

} // namespace facebook::velox
