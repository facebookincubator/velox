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

/// Implements pmod_with_mode(dividend, divisor, failOnError,
/// checkZeroBeforeLeft). Both options must be non-null constant booleans.
/// Evaluates the divisor before the dividend, skipping the dividend for null
/// divisors and legacy zero divisors. checkZeroBeforeLeft also skips it for
/// ANSI zero divisors, matching Spark codegen for non-nullable operands.
class PmodCallToSpecialForm : public exec::FunctionCallToSpecialForm {
 public:
  /// Accepts two identical primitive numeric types followed by two booleans.
  TypePtr resolveType(const std::vector<TypePtr>& argTypes) override;

  /// Captures the options independently of the current query configuration.
  exec::ExprPtr constructSpecialForm(
      const TypePtr& type,
      std::vector<exec::ExprPtr>&& args,
      bool trackCpuUsage,
      const core::QueryConfig& config) override;

  static constexpr const char* kPmodWithMode = "pmod_with_mode";
};

} // namespace facebook::velox::functions::sparksql
