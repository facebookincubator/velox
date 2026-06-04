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

#include "velox/functions/sparksql/specialforms/DecimalRoundOps.h"

namespace facebook::velox::functions::sparksql {

DecimalRoundOps::ScaleFactors DecimalRoundOps::computeFactors(
    int32_t scale,
    uint8_t inputPrecision,
    uint8_t inputScale,
    uint8_t resultPrecision,
    uint8_t resultScale) {
  ScaleFactors factors{};
  factors.scale = clampScale(scale);
  factors.inputPrecision = inputPrecision;
  factors.inputScale = inputScale;
  factors.resultPrecision = resultPrecision;
  factors.resultScale = resultScale;
  factors.overflowBound = DecimalUtil::kPowersOfTen[resultPrecision];

  if (factors.scale < static_cast<int32_t>(inputScale)) {
    const int32_t divDigits = static_cast<int32_t>(inputScale) - factors.scale;
    VELOX_DCHECK_GT(divDigits, 0);
    const int32_t cappedDivDigits = std::min(
        divDigits, static_cast<int32_t>(LongDecimalType::kMaxPrecision));
    factors.divideFactor = DecimalUtil::kPowersOfTen[cappedDivDigits];
    if (factors.scale < 0) {
      VELOX_DCHECK_LE(
          -factors.scale, static_cast<int32_t>(LongDecimalType::kMaxPrecision));
      factors.multiplyFactor = DecimalUtil::kPowersOfTen[-factors.scale];
    }
  }
  return factors;
}

int32_t DecimalRoundOps::extractConstantScaleArg(
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

} // namespace facebook::velox::functions::sparksql
