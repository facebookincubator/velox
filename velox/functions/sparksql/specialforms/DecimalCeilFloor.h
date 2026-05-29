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

namespace facebook::velox::functions::sparksql {

/// Base class for Spark RoundCeil / RoundFloor 2-argument decimal forms.
/// Spark's `ceiling(x, scale)` / `floor(x, scale)` route to RoundCeil /
/// RoundFloor, which take a decimal value and a foldable INTEGER target scale,
/// and return a decimal value rounded toward +∞ (ceil) or -∞ (floor).
///
/// The result type is fully determined by the input decimal type and the
/// constant target scale, which is why these are modeled as special forms
/// (the constant value cannot be expressed via SignatureVariable constraints).
class DecimalCeilFloorCallToSpecialFormBase
    : public exec::FunctionCallToSpecialForm {
 public:
  TypePtr resolveType(const std::vector<TypePtr>& argTypes) override;

  /// Returns the result precision and scale after applying ceil/floor to a
  /// decimal of (precision, scale) at the requested target `roundScale`.
  /// Logic mirrors Spark's RoundBase.dataType for DecimalType, which is shared
  /// by Round, RoundCeil and RoundFloor.
  static std::pair<uint8_t, uint8_t>
  getResultPrecisionScale(uint8_t precision, uint8_t scale, int32_t roundScale);

 protected:
  // Produces the final special-form expression for the given rounding mode.
  // `funcName` is used for diagnostics and for the registered special-form
  // name (e.g. "decimal_ceil" or "decimal_floor"). `ceiling` selects rounding
  // direction: true → toward +∞ (ceil), false → toward -∞ (floor).
  exec::ExprPtr makeSpecialForm(
      const TypePtr& type,
      std::vector<exec::ExprPtr>&& args,
      bool trackCpuUsage,
      const core::QueryConfig& config,
      bool ceiling,
      const std::string& funcName);
};

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
