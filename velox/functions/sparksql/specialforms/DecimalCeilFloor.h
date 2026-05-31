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

#include "velox/expression/FunctionCallToSpecialForm.h"
#include "velox/functions/sparksql/specialforms/DecimalRound.h"

namespace facebook::velox::functions::sparksql {

/// Base class for Spark RoundCeil / RoundFloor 2-argument decimal forms.
/// Spark's `ceiling(x, scale)` / `floor(x, scale)` route to RoundCeil /
/// RoundFloor, which take a decimal value and a foldable INTEGER target scale,
/// and return a decimal value rounded toward +∞ (ceil) or -∞ (floor).
///
/// The result type is fully determined by the input decimal type and the
/// constant target scale, which is why these are modeled as special forms
/// (the constant value cannot be expressed via SignatureVariable constraints).
///
/// The result type rules are identical to DecimalRound (Spark's
/// RoundBase.dataType) — see DecimalRoundCallToSpecialForm::getResultPrecisionScale.
class DecimalCeilFloorCallToSpecialFormBase
    : public exec::FunctionCallToSpecialForm {
 public:
  TypePtr resolveType(const std::vector<TypePtr>& argTypes) override;

  /// Delegates to DecimalRoundCallToSpecialForm::getResultPrecisionScale —
  /// the result type rules are shared across Round, RoundCeil, and RoundFloor.
  static std::pair<uint8_t, uint8_t>
  getResultPrecisionScale(uint8_t precision, uint8_t scale, int32_t roundScale) {
    return DecimalRoundCallToSpecialForm::getResultPrecisionScale(
        precision, scale, roundScale);
  }

 protected:
  exec::ExprPtr makeSpecialForm(
      const TypePtr& type,
      std::vector<exec::ExprPtr>&& args,
      bool trackCpuUsage,
      const core::QueryConfig& config,
      bool ceiling,
      std::string_view funcName);
};

/// Spark decimal_ceil special form: rounds a decimal value toward +∞ to the
/// specified target scale. Scale is silently clamped to [-38, 38]. Overflow
/// beyond the result precision returns NULL.
class DecimalCeilCallToSpecialForm
    : public DecimalCeilFloorCallToSpecialFormBase {
 public:
  exec::ExprPtr constructSpecialForm(
      const TypePtr& type,
      std::vector<exec::ExprPtr>&& args,
      bool trackCpuUsage,
      const core::QueryConfig& config) override;

  static constexpr const char* kCeilDecimal = "decimal_ceil";
};

/// Spark decimal_floor special form: rounds a decimal value toward -∞ to the
/// specified target scale. Scale is silently clamped to [-38, 38]. Overflow
/// beyond the result precision returns NULL.
class DecimalFloorCallToSpecialForm
    : public DecimalCeilFloorCallToSpecialFormBase {
 public:
  exec::ExprPtr constructSpecialForm(
      const TypePtr& type,
      std::vector<exec::ExprPtr>&& args,
      bool trackCpuUsage,
      const core::QueryConfig& config) override;

  static constexpr const char* kFloorDecimal = "decimal_floor";
};

} // namespace facebook::velox::functions::sparksql
