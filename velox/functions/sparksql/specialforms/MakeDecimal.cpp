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

#include "velox/functions/sparksql/specialforms/MakeDecimal.h"
#include "velox/expression/ConstantExpr.h"

namespace facebook::velox::functions::sparksql {
namespace {
template <typename T, bool nullOnOverflow>
class MakeDecimalFunction final : public exec::VectorFunction {
 public:
  explicit MakeDecimalFunction(uint8_t resultPrecision)
      : resultPrecision_(resultPrecision) {}

  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args, // Not using const ref so we can reuse args
      const TypePtr& outputType,
      exec::EvalCtx& context,
      VectorPtr& resultRef) const final {
    VELOX_USER_CHECK_EQ(args.size(), 2);
    VELOX_USER_CHECK(
        args[0]->isConstantEncoding() || args[0]->isFlatEncoding(),
        "Single-arg deterministic functions receive their only argument as flat or constant vector.");
    context.ensureWritable(rows, outputType, resultRef);
    auto result = resultRef->asUnchecked<FlatVector<T>>()->mutableRawValues();
    int128_t bound = DecimalUtil::kPowersOfTen[resultPrecision_];
    if (args[0]->isConstantEncoding()) {
      // Fast path for constant vectors.
      auto constant =
          args[0]->asUnchecked<ConstantVector<int64_t>>()->valueAt(0);
      if constexpr (std::is_same_v<T, int64_t>) {
        rows.applyToSelected([&](int row) {
          // The unscaled value cannot fit in the requested precision.
          if (constant <= -bound || constant >= bound) {
            resultRef->setNull(row, true);
          } else {
            VELOX_USER_FAIL(
                "Unscaled value {} too large for precision {}.",
                constant,
                resultPrecision_);
          }
        });
      } else {
        rows.applyToSelected(
            [&](int row) { result[row] = (int128_t)constant; });
      }
    } else {
      // Fast path for flat.
      auto flatValues =
          args[0]->asUnchecked<FlatVector<int64_t>>()->mutableRawValues();
      if constexpr (std::is_same_v<T, int64_t>) {
        rows.applyToSelected([&](int row) {
          auto unscaled = flatValues[row];
          if (unscaled <= -bound || unscaled >= bound) {
            // The unscaled value cannot fit in the requested precision.
            if constexpr (nullOnOverflow) {
              resultRef->setNull(row, true);
            } else {
              VELOX_USER_FAIL(
                  "Unscaled value {} too large for precision {}.",
                  unscaled,
                  resultPrecision_);
            }
          } else {
            result[row] = unscaled;
          }
        });
      } else {
        rows.applyToSelected(
            [&](int row) { result[row] = (int128_t)flatValues[row]; });
      }
    }
  }

 private:
  const uint8_t resultPrecision_;
};

std::shared_ptr<exec::VectorFunction> makeMakeDecimalByUnscaledValue(
    const TypePtr& type,
    bool nullOnOverflow) {
  if (type->isShortDecimal()) {
    if (nullOnOverflow) {
      return std::make_shared<MakeDecimalFunction<int64_t, true>>(
          type->asShortDecimal().precision());
    } else {
      return std::make_shared<MakeDecimalFunction<int64_t, false>>(
          type->asShortDecimal().precision());
    }
  } else {
    if (nullOnOverflow) {
      return std::make_shared<MakeDecimalFunction<int128_t, true>>(
          type->asLongDecimal().precision());
    } else {
      return std::make_shared<MakeDecimalFunction<int128_t, false>>(
          type->asLongDecimal().precision());
    }
  }
}
} // namespace

TypePtr MakeDecimalCallToSpecialForm::resolveType(
    const std::vector<TypePtr>& argTypes) {
  VELOX_FAIL("Make decimal function does not support type resolution.");
}

exec::ExprPtr MakeDecimalCallToSpecialForm::constructSpecialForm(
    const TypePtr& type,
    std::vector<exec::ExprPtr>&& compiledChildren,
    bool trackCpuUsage,
    const core::QueryConfig& config) {
  VELOX_USER_CHECK_GE(
      compiledChildren.size(), 2, "Make_decimal expects two arguments.");
  auto constantExpr =
      std::dynamic_pointer_cast<exec::ConstantExpr>(compiledChildren[1]);
  VELOX_USER_CHECK_NOT_NULL(constantExpr);
  VELOX_USER_CHECK(constantExpr->value()->isConstantEncoding());
  bool nullOnOverflow =
      constantExpr->value()->asUnchecked<ConstantVector<bool>>()->valueAt(0);

  return std::make_shared<exec::Expr>(
      type,
      std::move(compiledChildren),
      makeMakeDecimalByUnscaledValue(type, nullOnOverflow),
      kMakeDecimal,
      trackCpuUsage);
}
} // namespace facebook::velox::functions::sparksql
