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

#include "velox/functions/sparksql/specialforms/DecimalCeilFloor.h"

#include "velox/functions/sparksql/specialforms/DecimalScaleDispatch.h"

namespace facebook::velox::functions::sparksql {
namespace {

// VectorFunction implementation for Spark's RoundCeil / RoundFloor over a
// decimal first argument with a constant integer target scale. The result
// type is precomputed by getResultPrecisionScale() and supplied as
// resultType. Values that overflow the result precision are returned as
// NULL, matching Spark's catch-and-null behavior in RoundBase.
template <typename TResult, typename TInput, bool ceiling>
class DecimalCeilFloorFunction : public exec::VectorFunction {
 public:
  DecimalCeilFloorFunction(
      int32_t roundScale,
      uint8_t inputPrecision,
      uint8_t inputScale,
      uint8_t resultPrecision,
      uint8_t resultScale)
      : roundScale_{detail::clampDecimalScale(roundScale)},
        inputScale_{inputScale},
        resultPrecision_{resultPrecision} {
    // Validate that result type matches expected Spark semantics.
    const auto [expectedP, expectedS] =
        DecimalCeilFloorCallToSpecialFormBase::getResultPrecisionScale(
            inputPrecision, inputScale, roundScale);
    VELOX_USER_CHECK_EQ(
        expectedP,
        resultPrecision,
        "The result precision of {} is inconsistent with Spark expected.",
        ceiling ? "decimal_ceil" : "decimal_floor");
    VELOX_USER_CHECK_EQ(
        expectedS,
        resultScale,
        "The result scale of {} is inconsistent with Spark expected.",
        ceiling ? "decimal_ceil" : "decimal_floor");
    // Precompute scaling factors for the three regimes:
    //   roundScale_ >= inputScale_  : no-op (resultScale == inputScale, the
    //                                  unscaled value is already correct).
    //                                  This matches Spark's
    //                                  `newScale = min(s, _scale)` rule, which
    //                                  caps the result scale at the input scale
    //                                  rather than padding with zeros.
    //   0 <= roundScale_ < inputScale_ : divide by 10^(s-n) (divideFactor).
    //   roundScale_ < 0             : divide by 10^(s-n) then multiply by
    //                                  10^(-n) (both factors).
    if (roundScale_ < static_cast<int32_t>(inputScale_)) {
      const int32_t divDigits = static_cast<int32_t>(inputScale_) - roundScale_;
      VELOX_USER_CHECK_GT(divDigits, 0);
      // For high-scale inputs combined with negative round scales, the
      // logical divisor (10^divDigits) can exceed 10^38. Any valid decimal
      // unscaled value is bounded by 10^38, so dividing by 10^38 already
      // yields quotient 0 with the full input as remainder, which is
      // semantically equivalent to dividing by any larger power of ten.
      // Capping keeps us within DecimalUtil::kPowersOfTen bounds while
      // preserving Spark's value+null semantics.
      const int32_t cappedDivDigits = std::min(
          divDigits, static_cast<int32_t>(LongDecimalType::kMaxPrecision));
      divideFactor_ = DecimalUtil::kPowersOfTen[cappedDivDigits];
      if (roundScale_ < 0) {
        VELOX_DCHECK_LE(
            -roundScale_, static_cast<int32_t>(LongDecimalType::kMaxPrecision));
        multiplyFactor_ = DecimalUtil::kPowersOfTen[-roundScale_];
      }
    }
    // Pre-compute the absolute precision bound for the result type. We use
    // it to detect overflow and emit NULL (matching Spark semantics).
    overflowBound_ = DecimalUtil::kPowersOfTen[resultPrecision_];
  }

  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& resultType,
      exec::EvalCtx& context,
      VectorPtr& result) const override {
    VELOX_USER_CHECK(
        args[0]->isConstantEncoding() || args[0]->isFlatEncoding(),
        "Decimal ceil/floor first arg must be flat or constant, got: {}.",
        args[0]->encoding());
    context.ensureWritable(rows, resultType, result);
    result->clearNulls(rows);
    auto* flat = result->asUnchecked<FlatVector<TResult>>();
    auto* rawResults = flat->mutableRawValues();
    if (args[0]->isConstantEncoding()) {
      auto* constArg = args[0]->asUnchecked<ConstantVector<TInput>>();
      if (constArg->isNullAt(0)) {
        rows.applyToSelected([&](auto row) { flat->setNull(row, true); });
        return;
      }
      bool isNull;
      const TResult value = applyOne(constArg->valueAt(0), isNull);
      rows.applyToSelected([&](auto row) {
        if (FOLLY_UNLIKELY(isNull)) {
          flat->setNull(row, true);
        } else {
          rawResults[row] = value;
        }
      });
    } else {
      auto* flatArg = args[0]->asUnchecked<FlatVector<TInput>>();
      const auto* rawValues = flatArg->rawValues();
      const uint64_t* nulls = flatArg->rawNulls();
      if (nulls) {
        rows.applyToSelected([&](auto row) {
          if (bits::isBitNull(nulls, row)) {
            flat->setNull(row, true);
            return;
          }
          bool isNull;
          const TResult value = applyOne(rawValues[row], isNull);
          if (FOLLY_UNLIKELY(isNull)) {
            flat->setNull(row, true);
          } else {
            rawResults[row] = value;
          }
        });
      } else {
        rows.applyToSelected([&](auto row) {
          bool isNull;
          const TResult value = applyOne(rawValues[row], isNull);
          if (FOLLY_UNLIKELY(isNull)) {
            flat->setNull(row, true);
          } else {
            rawResults[row] = value;
          }
        });
      }
    }
  }

  bool supportsFlatNoNullsFastPath() const override {
    return false;
  }

 private:
  inline TResult applyOne(const TInput& input, bool& isNull) const {
    isNull = false;
    int128_t out;
    if (!divideFactor_.has_value()) {
      // No-op path: roundScale_ >= inputScale_, so the unscaled value is
      // already correct (resultScale == inputScale per Spark's
      // newScale = min(s, _scale) rule). The overflow bound below still
      // guards against malformed plans.
      out = static_cast<int128_t>(input);
    } else {
      const int128_t divisor = divideFactor_.value();
      const int128_t in = static_cast<int128_t>(input);
      const int128_t quotient = in / divisor;
      const int128_t remainder = in % divisor;
      int128_t rounded;
      if constexpr (ceiling) {
        rounded = quotient + (remainder > 0 ? 1 : 0);
      } else {
        rounded = quotient + (remainder < 0 ? -1 : 0);
      }
      if (multiplyFactor_.has_value() && !mulWithBoundCheck(rounded)) {
        isNull = true;
        return TResult{};
      }
      out = rounded;
    }
    if (out >= overflowBound_ || out <= -overflowBound_) {
      isNull = true;
      return TResult{};
    }
    return static_cast<TResult>(out);
  }

  // Multiplies `value` in-place by multiplyFactor_. Returns false if the
  // product would exceed the result-precision bound (overflowBound_). The
  // pre-check is performed against overflowBound_ to keep the int128
  // arithmetic well within range.
  inline bool mulWithBoundCheck(int128_t& value) const {
    const int128_t factor = multiplyFactor_.value();
    VELOX_DCHECK_GT(
        factor, 0, "multiplyFactor_ is always a positive power of 10.");
    const int128_t maxAbs = overflowBound_ / factor;
    if (value >= maxAbs || value <= -maxAbs) {
      return false;
    }
    value *= factor;
    return true;
  }

  const int32_t roundScale_;
  const uint8_t inputScale_;
  const uint8_t resultPrecision_;
  std::optional<int128_t> divideFactor_;
  std::optional<int128_t> multiplyFactor_;
  int128_t overflowBound_;
};

template <bool ceiling>
std::shared_ptr<exec::VectorFunction> createDecimalCeilFloor(
    const TypePtr& inputType,
    int32_t scale,
    const TypePtr& resultType) {
  const auto [inputPrecision, inputScale] =
      getDecimalPrecisionScale(*inputType);
  const auto [resultPrecision, resultScale] =
      getDecimalPrecisionScale(*resultType);
  return detail::dispatchDecimalTypes(
      inputType, resultType, [&](auto resultTag, auto inputTag) {
        using TResult = typename decltype(resultTag)::type;
        using TInput = typename decltype(inputTag)::type;
        return std::make_shared<
            DecimalCeilFloorFunction<TResult, TInput, ceiling>>(
            scale, inputPrecision, inputScale, resultPrecision, resultScale);
      });
}

} // namespace

TypePtr DecimalCeilFloorCallToSpecialFormBase::resolveType(
    const std::vector<TypePtr>& /*argTypes*/) {
  VELOX_FAIL(
      "Decimal ceil/floor with scale special form does not support type resolution.");
}

exec::ExprPtr DecimalCeilFloorCallToSpecialFormBase::makeSpecialForm(
    const TypePtr& type,
    std::vector<exec::ExprPtr>&& args,
    bool trackCpuUsage,
    const core::QueryConfig& /*config*/,
    bool ceiling,
    std::string_view funcName) {
  VELOX_USER_CHECK(
      type->isDecimal(), "The result type of {} must be decimal.", funcName);
  VELOX_USER_CHECK_EQ(
      args.size(),
      2,
      "{} expects two arguments (decimal value and target scale).",
      funcName);
  VELOX_USER_CHECK(
      args[0]->type()->isDecimal(),
      "The first argument of {} must be decimal.",
      funcName);

  const int32_t scale = detail::extractConstantScaleArg(args[1], funcName);
  auto func = ceiling
      ? createDecimalCeilFloor<true>(args[0]->type(), scale, type)
      : createDecimalCeilFloor<false>(args[0]->type(), scale, type);
  return std::make_shared<exec::Expr>(
      type,
      std::move(args),
      std::move(func),
      exec::VectorFunctionMetadata{},
      std::string(funcName),
      trackCpuUsage);
}

exec::ExprPtr DecimalCeilCallToSpecialForm::constructSpecialForm(
    const TypePtr& type,
    std::vector<exec::ExprPtr>&& args,
    bool trackCpuUsage,
    const core::QueryConfig& config) {
  return makeSpecialForm(
      type, std::move(args), trackCpuUsage, config, true, kCeilDecimal);
}

exec::ExprPtr DecimalFloorCallToSpecialForm::constructSpecialForm(
    const TypePtr& type,
    std::vector<exec::ExprPtr>&& args,
    bool trackCpuUsage,
    const core::QueryConfig& config) {
  return makeSpecialForm(
      type, std::move(args), trackCpuUsage, config, false, kFloorDecimal);
}

} // namespace facebook::velox::functions::sparksql
