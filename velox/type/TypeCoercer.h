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
#include <algorithm>
#include <span>
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

  /// Resolved return type and default null behavior of a function signature.
  /// The return type is never null. A bound candidate always resolves one.
  struct CandidateMetadata {
    TypePtr returnType;
    bool nullOnNull{false};
  };

  /// Returns the index of the strictly-lowest-cost candidate, or std::nullopt
  /// if 'candidates' is empty or two or more tie for lowest cost.
  template <typename T>
  static std::optional<size_t> pickLowestCost(
      const std::vector<std::pair<std::vector<Coercion>, T>>& candidates) {
    const auto tied = minCostCandidates(candidates);
    return tied.size() == 1 ? std::optional<size_t>(tied[0]) : std::nullopt;
  }

  /// Resolves a lowest-cost tie between candidates when the tie is caused only
  /// by bare UNKNOWN (untyped null) arguments. Returns the winning candidate's
  /// index, or std::nullopt when the tie is genuinely ambiguous.
  ///
  /// Looks only at "unknown-only-coercion" candidates -- those whose coercions
  /// fall solely on UNKNOWN positions, coercing nothing but the nulls. They are
  /// more specific than candidates that also widen a real argument. Among them:
  ///   - exactly one      -> it wins;
  ///   - several, but interchangeable -> lowest index wins;
  ///   - anything else    -> ambiguous (std::nullopt).
  ///
  /// Interchangeable means they share the same return type and all return null
  /// on a null argument, so the pick cannot change the result.
  ///
  /// 'argTypes' are the call's actual argument types, used to locate the
  /// UNKNOWN positions. 'resolutionAt(i)' returns candidate i's
  /// CandidateMetadata.
  template <typename T, typename ResolutionAt>
  static std::optional<size_t> pickLowestCost(
      const std::vector<std::pair<std::vector<Coercion>, T>>& candidates,
      const std::vector<TypePtr>& argTypes,
      ResolutionAt&& resolutionAt) {
    const auto tied = minCostCandidates(candidates);
    if (tied.size() == 1) {
      return tied[0];
    }
    return tryResolveTie(
        candidates, tied, argTypes, std::forward<ResolutionAt>(resolutionAt));
  }

 private:
  // Returns true if 'coercions' has at least one coercion and every coercion
  // falls on an UNKNOWN position of 'argTypes' (an exact bind has none).
  static bool isUnknownOnlyCoercion(
      const std::vector<Coercion>& coercions,
      const std::vector<TypePtr>& argTypes);

  // Returns the indices of the candidates tied at the lowest summed cost.
  template <typename T>
  static std::vector<size_t> minCostCandidates(
      const std::vector<std::pair<std::vector<Coercion>, T>>& candidates) {
    std::vector<size_t> tied;
    int64_t minCost{0};
    for (auto i = 0; i < candidates.size(); ++i) {
      const auto cost = overallCost(candidates[i].first);
      if (tied.empty() || cost < minCost) {
        minCost = cost;
        tied.clear();
        tied.push_back(i);
      } else if (cost == minCost) {
        tied.push_back(i);
      }
    }
    return tied;
  }

  // Resolves only an UNKNOWN-induced tie: keeps the unknown-only-coercion
  // candidates among 'tied' and returns the winner, else std::nullopt.
  template <typename T, typename ResolutionAt>
  static std::optional<size_t> tryResolveTie(
      const std::vector<std::pair<std::vector<Coercion>, T>>& candidates,
      std::span<const size_t> tied,
      const std::vector<TypePtr>& argTypes,
      ResolutionAt&& resolutionAt) {
    std::vector<size_t> unknownOnly;
    for (auto index : tied) {
      if (isUnknownOnlyCoercion(candidates[index].first, argTypes)) {
        unknownOnly.push_back(index);
      }
    }
    if (unknownOnly.empty()) {
      return std::nullopt;
    }

    // Lowest index, so the pick is deterministic regardless of candidate order.
    const size_t selectedIndex =
        *std::min_element(unknownOnly.begin(), unknownOnly.end());
    if (unknownOnly.size() == 1) {
      return selectedIndex;
    }

    const auto selectedReturnType = resolutionAt(selectedIndex).returnType;
    VELOX_DCHECK_NOT_NULL(selectedReturnType);
    for (auto index : unknownOnly) {
      const auto metadata = resolutionAt(index);
      if (!metadata.nullOnNull || *metadata.returnType != *selectedReturnType) {
        return std::nullopt;
      }
    }
    return selectedIndex;
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
/// coerce()), so a dialect controls their behavior only indirectly via
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

  /// Returns the coercion from 'fromType' to 'toType', or std::nullopt.
  /// Scalars resolve via a single rule lookup; containers match on name and
  /// arity, then recurse element-wise. A bare UNKNOWN coerces to any resolved
  /// target, ranked above every UNKNOWN->scalar rule.
  std::optional<Coercion> coerce(const TypePtr& fromType, const TypePtr& toType)
      const;

  /// Returns least common type for 'a' and 'b', i.e. a type that both 'a' and
  /// 'b' are coercible to. Returns nullptr if no such type exists.
  ///
  /// When `a` and `b` are ROW types with different field names, the resulting
  /// ROW has empty field names for any positions where the corresponding field
  /// names do not match.
  TypePtr leastCommonSuperType(const TypePtr& a, const TypePtr& b) const;

 private:
  // Returns the coercion from scalar 'fromType' to 'toType' via a single rule
  // lookup, or std::nullopt.
  std::optional<Coercion> coerceTypeBase(
      const TypePtr& fromType,
      const TypePtr& toType) const;

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

  // Derived from the rule set at construction (max UNKNOWN->scalar cost + 1).
  int32_t unknownFallbackCost_{1};
};

} // namespace facebook::velox
