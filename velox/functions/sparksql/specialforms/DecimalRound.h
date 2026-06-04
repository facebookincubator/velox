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
#include "velox/expression/FunctionCallToSpecialForm.h"
#include "velox/expression/VectorFunction.h"
#include "velox/type/Type.h"

namespace facebook::velox::functions::sparksql {

// Forward declaration — defined below after DecimalRoundOps.
template <typename TResult, typename TInput, typename Policy>
class DecimalRoundFunction;

/// Shared infrastructure for Spark's decimal rounding special forms
/// (decimal_round, decimal_ceil, decimal_floor).
///
/// These are special forms because the result type depends on the runtime value
/// of the scale argument. All three share the same structure: compute scale
/// factors, then for each row divide/truncate/adjust/multiply. The only
/// difference is the adjustment logic (rounding direction).
///
/// DecimalRoundOps provides static helpers used by all three forms.
/// DecimalRoundFunction<Policy> is the shared VectorFunction template.
/// Each form defines a Policy struct (in the .cpp) with applyOne().
class DecimalRoundOps {
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

  static int32_t clampScale(int32_t scale) {
    constexpr int32_t kMax = LongDecimalType::kMaxPrecision;
    return std::max(-kMax, std::min(scale, kMax));
  }

  /// Computes the divide and multiply factors for a given scale adjustment.
  static ScaleFactors computeFactors(
      int32_t scale,
      uint8_t inputPrecision,
      uint8_t inputScale,
      uint8_t resultPrecision,
      uint8_t resultScale);

  /// Extracts a constant INTEGER scale argument from an expression.
  static int32_t extractConstantScaleArg(
      const exec::ExprPtr& expr,
      std::string_view funcName);

  /// Creates a DecimalRoundFunction<Policy> dispatching on decimal types.
  /// Policy is instantiated with the appropriate TResult and TInput types.
  template <template <typename, typename> class Policy>
  static std::shared_ptr<exec::VectorFunction> createFunction(
      const TypePtr& inputType,
      int32_t scale,
      const TypePtr& resultType) {
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
          using TResult = typename decltype(resultTag)::type;
          using TInput = typename decltype(inputTag)::type;
          return std::make_shared<
              DecimalRoundFunction<TResult, TInput, Policy<TResult, TInput>>>(
              factors);
        });
  }

 private:
  /// Dispatches on the physical types of input/result decimals.
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
};

/// Generic VectorFunction for decimal rounding, parameterized on a Policy.
/// Policy must define:
///   - static constexpr bool canOverflow
///   - Constructor(const DecimalRoundOps::ScaleFactors&)
///   - std::optional<TResult> applyOne(const TInput&) const
template <typename TResult, typename TInput, typename Policy>
class DecimalRoundFunction : public exec::VectorFunction {
 public:
  explicit DecimalRoundFunction(const DecimalRoundOps::ScaleFactors& factors)
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

/// Spark decimal_round special form. The result type depends on the value of
/// the constant scale argument, requiring special-form type resolution.
class DecimalRoundCallToSpecialForm : public exec::FunctionCallToSpecialForm {
 public:
  // Throws not supported exception.
  TypePtr resolveType(const std::vector<TypePtr>& argTypes) override;

  /// @brief Returns an expression for decimal_round special form. The
  /// expression is a regular expression based on a custom VectorFunction
  /// implementation.
  /// @param type Result type. Must be short or long decimal.
  /// @param args One or two inputs. First input must be decimal. Second
  /// optional input is the new scale to be rounded to, and must be constant
  /// INTEGER.
  exec::ExprPtr constructSpecialForm(
      const TypePtr& type,
      std::vector<exec::ExprPtr>&& args,
      bool trackCpuUsage,
      const core::QueryConfig& config) override;

  /// Returns the result precision and scale after rounding from the input
  /// precision and scale to a new scale. The calculation logic is consistent
  /// with Spark version after 3.3.
  static std::pair<uint8_t, uint8_t>
  getResultPrecisionScale(uint8_t precision, uint8_t scale, int32_t roundScale);

  static constexpr const char* kRoundDecimal = "decimal_round";
  // Ceil/floor constants live here because all three forms share
  // getResultPrecisionScale and are registered together.
  static constexpr const char* kCeilDecimal = "decimal_ceil";
  static constexpr const char* kFloorDecimal = "decimal_floor";
};

/// Registers decimal_round, decimal_ceil, and decimal_floor special forms.
void registerDecimalRoundingForms();

} // namespace facebook::velox::functions::sparksql
