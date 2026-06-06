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

#include <folly/hash/Hash.h>
#include "velox/type/Type.h"

namespace facebook::velox {

/// Type coercion necessary to bind a type to a signature.
struct Coercion {
  TypePtr type;
  int32_t cost{0};

  std::string toString() const {
    if (type == nullptr) {
      return "null";
    }

    return fmt::format("{} ({})", type->toString(), cost);
  }

  void reset() {
    type = nullptr;
    cost = 0;
  }

  /// Returns overall cost of a list of coercions by adding up individual costs.
  static int64_t overallCost(const std::vector<Coercion>& coercions);

  /// Returns an index of the lowest cost coercion in 'candidates' or nullptr if
  /// 'candidates' is empty or there is a tie.
  template <typename T>
  static std::optional<size_t> pickLowestCost(
      const std::vector<std::pair<std::vector<Coercion>, T>>& candidates) {
    if (candidates.empty()) {
      return std::nullopt;
    }

    if (candidates.size() == 1) {
      return 0;
    }

    std::vector<std::pair<size_t, int64_t>> costs;
    costs.reserve(candidates.size());
    for (auto i = 0; i < candidates.size(); ++i) {
      costs.emplace_back(i, overallCost(candidates[i].first));
    }

    std::sort(costs.begin(), costs.end(), [](const auto& a, const auto& b) {
      return a.second < b.second;
    });

    if (costs[0].second < costs[1].second) {
      return costs[0].first;
    }

    return std::nullopt;
  }
};

/// One entry in a TypeCoercer's rule set.
///
/// DECIMAL handling: rules are keyed by (from->name(), to->name()), and all
/// decimal types share name "DECIMAL". A coercer therefore holds at most one
/// DECIMAL rule per source type. The 'to' field on that rule must be the
/// minimum-width decimal that fits every value of the source type (e.g.,
/// TINYINT -> DECIMAL(3, 0)). Lookup at coerceTypeBase(from, target) widens
/// the stored minimum-width decimal to the caller's requested target if
/// isCoercibleTo allows; the rule's cost is preserved.
struct CoercionEntry {
  TypePtr from;
  TypePtr to;
  int32_t cost;
};

/// Type coercion rules. Each instance owns a complete rule set and
/// is immutable after construction. Velox ships a default instance
/// (TypeCoercer::defaults()) holding today's conservative built-in rules;
/// each SQL dialect can ship its own complete instance with rules that match
/// the dialect's semantics.
///
/// Coercion is a planning-time concern: by the time a Velox Task is
/// constructed, every type mismatch is already an explicit Cast node. Runtime
/// expression evaluators do not consult TypeCoercer.
///
/// Customization scope. The rule set determines coercion between primitive
/// types (TINYINT, SMALLINT, INTEGER, BIGINT, REAL, DOUBLE, BOOLEAN,
/// VARCHAR, VARBINARY, DATE, TIMESTAMP, UNKNOWN); a dialect has full
/// control over which pairs are allowed and at what cost.
///
/// Cost magnitudes. Overload resolution sums per-argument coercion costs
/// (Coercion::overallCost) to compare candidate signatures. For sums to
/// be meaningful, every CoercionEntry.cost in a single TypeCoercer
/// instance must be in the same small magnitude -- today's defaults use
/// costs 1-9, one per source-type series. A dialect that mixes costs of
/// vastly different magnitudes (e.g., 1-10 for some entries, 100-1000
/// for others) will produce surprising tie-breaking. There is no
/// hardcoded surcharge added at lookup time: the dialect's rule cost is
/// returned verbatim, including for INT -> DECIMAL widening -- the rule's
/// stored cost applies to every compatible target (p, s).
///
/// DECIMAL handling is two-step: the dialect's rule resolves a
/// fixed-(p, s) decimal, then the type system's generic DECIMAL->DECIMAL
/// compatibility widens it to the caller's request. Rule keys collapse on
/// the name "DECIMAL" regardless of (p, s) -- one rule per
/// (sourceName, targetName) pair on each side.
///
/// Source DECIMAL: a dialect registers one rule per (DECIMAL, target).
/// The source must be DECIMAL(1, 0) (the canonical placeholder -- see
/// CoercionEntry doc); the rule fires for any actual DECIMAL(p, s) source
/// because the source's precision/scale is not part of the lookup key.
///
/// Target DECIMAL: a dialect registers one rule per (source, DECIMAL),
/// with the rule's stored target serving as the minimum-width decimal
/// that holds every value of the source (e.g., INTEGER -> DECIMAL(10, 0)).
/// At lookup, the type system extends this fixed target to the caller's
/// requested (p, s) via ShortDecimalType / LongDecimalType isCoercibleTo
/// -- the dialect does not control widening; it picks only the
/// minimum-width target.
///
/// DECIMAL -> DECIMAL is **not customizable** by dialects. Rule entries
/// with both source and target DECIMAL are not honored: the same-name
/// short-circuit at the top of coerceTypeBase returns cost 0 for any two
/// DECIMALs regardless of (p, s) before the rule lookup runs.
/// Precision/scale reconciliation between two DECIMALs is hardcoded in
/// the type system -- LongDecimalType::commonSuperType inside
/// leastCommonSuperType for plan-level operations (UNION, CASE result
/// type, etc.), and SignatureBinder's integer-parameter binding for
/// function signatures of the form DECIMAL(P, S). Dialects that need
/// non-standard DECIMAL->DECIMAL semantics must extend the type system,
/// not TypeCoercer.
///
/// Container types (ARRAY, MAP, ROW) and FUNCTION/OPAQUE are not
/// customizable directly -- coercibility for those is structural (names and
/// arities must match, children are recursed element-wise, see
/// coercible()), so a dialect controls their behavior only indirectly via
/// element-type rules.
///
/// Custom types (e.g. JSON, TIMESTAMP WITH TIME ZONE) are out of scope for
/// TypeCoercer; they're coerced via the global CastRulesRegistry registered
/// alongside registerCustomType().
class TypeCoercer {
 public:
  /// Construct from a complete rule set. Rules are frozen on construction.
  /// Dialect-specific factories (e.g. presto::typeCoercer()) build a
  /// TypeCoercer and expose it as a singleton rather than mutating an
  /// existing instance.
  explicit TypeCoercer(const std::vector<CoercionEntry>& rules);

  /// Velox's default coercer holding today's conservative built-in rules.
  /// Used by callers that have not selected a dialect-specific coercer.
  static const TypeCoercer& defaults();

  /// Checks if 'fromType' can be implicitly converted to 'toType' via a
  /// single rule lookup. Does NOT recurse into container children -- for
  /// container coercibility (e.g., ARRAY<INT> vs ARRAY<BIGINT>), use
  /// coercible() instead.
  ///
  /// Prefer this over the string overload when the target type is available,
  /// as it avoids reconstructing the type from a name (which can throw for
  /// parametric custom types like BigintEnum).
  ///
  /// For DECIMAL targets, the rule's stored type is the minimum-width
  /// decimal for the source. If 'toType' is a wider DECIMAL that the
  /// minimum can be coerced to (per ShortDecimalType / LongDecimalType
  /// isCoercibleTo), the returned Coercion carries 'toType' and the rule's
  /// cost. If 'toType' is too narrow, returns nullopt.
  ///
  /// @return "to" type and cost if conversion is possible.
  std::optional<Coercion> coerceTypeBase(
      const TypePtr& fromType,
      const TypePtr& toType) const;

  /// Checks if the base of 'fromType' can be implicitly converted to a type
  /// with the given name. Used by SignatureBinder which only has a type name.
  ///
  /// @return "to" type and cost if conversion is possible.
  std::optional<Coercion> coerceTypeBase(
      const TypePtr& fromType,
      const std::string& toTypeName) const;

  /// Checks if 'fromType' can be implicitly converted to 'toType', recursing
  /// into container children. For primitives this delegates to
  /// coerceTypeBase(); for containers (ARRAY, MAP, ROW, FUNCTION) the names
  /// and arities must match, then each child must be element-wise coercible.
  ///
  /// @return Total cost (sum of child costs for containers) if possible.
  /// std::nullopt otherwise.
  std::optional<int32_t> coercible(
      const TypePtr& fromType,
      const TypePtr& toType) const;

  /// Returns least common type for 'a' and 'b', i.e. a type that both 'a' and
  /// 'b' are coercible to. Returns nullptr if no such type exists.
  ///
  /// When `a` and `b` are ROW types with different field names, the resulting
  /// ROW has empty field names for any positions where the corresponding field
  /// names do not match.
  TypePtr leastCommonSuperType(const TypePtr& a, const TypePtr& b) const;

 private:
  struct PairHash {
    size_t operator()(const std::pair<std::string, std::string>& p) const {
      return folly::hash::hash_combine(
          std::hash<std::string>{}(p.first),
          std::hash<std::string>{}(p.second));
    }
  };

  // Flat map: (fromName, toName) -> Coercion{type, cost}.
  std::unordered_map<std::pair<std::string, std::string>, Coercion, PairHash>
      rules_;
};

} // namespace facebook::velox
