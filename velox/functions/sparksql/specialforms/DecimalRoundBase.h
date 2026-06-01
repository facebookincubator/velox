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
#include <string_view>

#include "velox/expression/ConstantExpr.h"
#include "velox/expression/VectorFunction.h"
#include "velox/type/Type.h"

namespace facebook::velox::functions::sparksql {

/// Design overview — decimal rounding in Spark
/// =============================================
///
/// Spark's decimal_round, decimal_ceil, and decimal_floor are special forms
/// (not simple functions) because their *result type* depends on the runtime
/// value of the scale argument, not just input types. The type resolution
/// happens at plan time via constructSpecialForm / makeSpecialForm in each
/// .cpp file.
///
/// All three operations share the same structure:
///   1. Compute scale factors (divide/multiply powers of 10) from input/result
///      precision and the target scale.
///   2. For each row: divide to truncate, optionally adjust (the rounding
///      direction), multiply back. Check for overflow.
///
/// The only difference between round/ceil/floor is step 2's adjustment logic.
/// This file captures the shared parts:
///
///   DecimalRoundBase     — static helpers: computeFactors, dispatchTypes,
///                          createFunction, buildExpr, extractConstantScaleArg,
///                          clampScale, getResultPrecisionScale.
///
///   ScaleFactors         — precomputed divide/multiply factors, overflow
///   bound.
///                          Computed once per expression, not per row.
///
///   DecimalRoundFunction<TResult, TInput, Policy>
///                        — the VectorFunction. Handles constant/flat dispatch
///                          and calls Policy::applyOne per element.
///
/// Each rounding form defines a tiny Policy struct in its .cpp file:
///   - RoundHalfUpPolicy  (DecimalRound.cpp)  — round half-up, never overflows.
///   - CeilFloorPolicy<>  (DecimalCeilFloor.cpp) — directional, may overflow.
///
/// A Policy must provide:
///   - static constexpr bool canOverflow
///   - Constructor(const ScaleFactors& factors)
///   - std::optional<TResult> applyOne(const TInput& value) const
class DecimalRoundBase {
 public:
  template <typename T>
  struct TypeTag {
    using type = T;
  };

  /// Precomputed scale factors shared by all rounding policies. Computed once
  /// from (scale, inputPrecision, inputScale, resultPrecision) and passed to
  /// the policy constructor — policies should not recompute these.
  struct ScaleFactors {
    int32_t scale; // Target scale (clamped to [-38, 38]).
    uint8_t inputPrecision;
    uint8_t inputScale;
    uint8_t resultPrecision;
    uint8_t resultScale;
    // divideFactor: 10^(inputScale - scale). Present when truncation is needed
    // (scale < inputScale). The rounding kernel divides by this to drop digits.
    std::optional<int128_t> divideFactor;
    // multiplyFactor: 10^(-scale). Present when scale < 0 (rounding to tens,
    // hundreds, etc). After truncation, multiply back to restore magnitude.
    std::optional<int128_t> multiplyFactor;
    // overflowBound: 10^resultPrecision. If |result| >= this, it overflows.
    int128_t overflowBound;
  };

  static int32_t clampScale(int32_t s) {
    constexpr int32_t kMax = LongDecimalType::kMaxPrecision;
    return std::max(-kMax, std::min(s, kMax));
  }

  /// Computes the divide and multiply factors for a given scale adjustment.
  /// The logic is shared across all three rounding directions.
  static ScaleFactors computeFactors(
      int32_t scale,
      uint8_t inputPrecision,
      uint8_t inputScale,
      uint8_t resultPrecision,
      uint8_t resultScale) {
    ScaleFactors f{};
    f.scale = clampScale(scale);
    f.inputPrecision = inputPrecision;
    f.inputScale = inputScale;
    f.resultPrecision = resultPrecision;
    f.resultScale = resultScale;
    f.overflowBound = DecimalUtil::kPowersOfTen[resultPrecision];

    if (f.scale < static_cast<int32_t>(inputScale)) {
      const int32_t divDigits = static_cast<int32_t>(inputScale) - f.scale;
      VELOX_DCHECK_GT(divDigits, 0);
      const int32_t cappedDivDigits = std::min(
          divDigits, static_cast<int32_t>(LongDecimalType::kMaxPrecision));
      f.divideFactor = DecimalUtil::kPowersOfTen[cappedDivDigits];
      if (f.scale < 0) {
        VELOX_DCHECK_LE(
            -f.scale, static_cast<int32_t>(LongDecimalType::kMaxPrecision));
        f.multiplyFactor = DecimalUtil::kPowersOfTen[-f.scale];
      }
    }
    return f;
  }

  static int32_t extractConstantScaleArg(
      const exec::ExprPtr& expr,
      std::string_view funcName) {
    VELOX_USER_CHECK_EQ(
        expr->type()->kind(),
        TypeKind::INTEGER,
        "The second argument of {} must be INTEGER, got: {}.",
        funcName,
        expr->type()->toString());
    auto constantExpr = std::dynamic_pointer_cast<exec::ConstantExpr>(expr);
    VELOX_USER_CHECK_NOT_NULL(
        constantExpr,
        "The second argument of {} must be a constant expression.",
        funcName);
    VELOX_CHECK(
        constantExpr->value()->isConstantEncoding(),
        "ConstantExpr must hold a constant-encoded vector.");
    auto* constantVector =
        constantExpr->value()->asUnchecked<ConstantVector<int32_t>>();
    VELOX_USER_CHECK(
        !constantVector->isNullAt(0),
        "The second argument of {} must not be NULL.",
        funcName);
    return constantVector->valueAt(0);
  }

  /// Dispatches on the physical types of input/result decimals. We use explicit
  /// branching rather than VELOX_DYNAMIC_DECIMAL_TYPE_DISPATCH because we need
  /// to handle all four combinations of short/long independently (the macro
  /// only dispatches on a single type at a time).
  template <typename Fn>
  static auto
  dispatchTypes(const TypePtr& inputType, const TypePtr& resultType, Fn&& fn) {
    if (inputType->isShortDecimal()) {
      if (resultType->isShortDecimal()) {
        return fn(TypeTag<int64_t>{}, TypeTag<int64_t>{});
      }
      return fn(TypeTag<int128_t>{}, TypeTag<int64_t>{});
    }
    if (resultType->isShortDecimal()) {
      return fn(TypeTag<int64_t>{}, TypeTag<int128_t>{});
    }
    return fn(TypeTag<int128_t>{}, TypeTag<int128_t>{});
  }

  /// Creates a VectorFunction for a given Policy, dispatching on decimal types.
  template <typename PolicyFactory>
  static std::shared_ptr<exec::VectorFunction> createFunction(
      const TypePtr& inputType,
      int32_t scale,
      const TypePtr& resultType,
      PolicyFactory&& factory) {
    const auto [inputPrecision, inputScale] =
        getDecimalPrecisionScale(*inputType);
    const auto [resultPrecision, resultScale] =
        getDecimalPrecisionScale(*resultType);
    const auto factors = computeFactors(
        scale, inputPrecision, inputScale, resultPrecision, resultScale);
    return dispatchTypes(
        inputType,
        resultType,
        [&](auto resultTag,
            auto inputTag) -> std::shared_ptr<exec::VectorFunction> {
          return factory(resultTag, inputTag, factors);
        });
  }

  /// Builds the final exec::Expr from a VectorFunction. Shared boilerplate
  /// for all decimal rounding constructSpecialForm implementations.
  static exec::ExprPtr buildExpr(
      const TypePtr& type,
      std::vector<exec::ExprPtr>&& args,
      std::shared_ptr<exec::VectorFunction> func,
      std::string_view funcName,
      bool trackCpuUsage) {
    return std::make_shared<exec::Expr>(
        type,
        std::move(args),
        std::move(func),
        exec::VectorFunctionMetadata{},
        std::string(funcName),
        trackCpuUsage);
  }
};

/// Generic VectorFunction for decimal rounding, parameterized on a Policy.
/// Policy must define:
///   - static constexpr bool canOverflow
///   - Constructor(const DecimalRoundBase::ScaleFactors&)
///   - std::optional<TResult> applyOne(const TInput&) const
///
/// Relies on the Expr framework's defaultNullBehavior (true by default):
/// 'rows' never includes positions where the input is null, so no manual
/// null propagation is needed.
template <typename TResult, typename TInput, typename Policy>
class DecimalRoundFunction : public exec::VectorFunction {
 public:
  explicit DecimalRoundFunction(const DecimalRoundBase::ScaleFactors& factors)
      : policy_(factors) {}

  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& resultType,
      exec::EvalCtx& context,
      VectorPtr& result) const override {
    context.ensureWritable(rows, resultType, result);
    auto* flat = result->asUnchecked<FlatVector<TResult>>();
    flat->clearNulls(rows);
    auto* rawResults = flat->mutableRawValues();

    if (args[0]->isConstantEncoding()) {
      auto value =
          args[0]->template asUnchecked<ConstantVector<TInput>>()->valueAt(0);
      auto rounded = policy_.applyOne(value);
      rows.applyToSelected([&](auto row) {
        if constexpr (Policy::canOverflow) {
          if (FOLLY_UNLIKELY(!rounded.has_value())) {
            flat->setNull(row, true);
            return;
          }
        }
        rawResults[row] = rounded.value();
      });
    } else {
      auto* rawValues =
          args[0]->template asUnchecked<FlatVector<TInput>>()->rawValues();
      rows.applyToSelected([&](auto row) {
        auto rounded = policy_.applyOne(rawValues[row]);
        if constexpr (Policy::canOverflow) {
          if (FOLLY_UNLIKELY(!rounded.has_value())) {
            flat->setNull(row, true);
            return;
          }
        }
        rawResults[row] = rounded.value();
      });
    }
  }

  bool supportsFlatNoNullsFastPath() const override {
    return !Policy::canOverflow;
  }

 private:
  Policy policy_;
};

} // namespace facebook::velox::functions::sparksql
