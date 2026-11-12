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

namespace facebook::velox::functions::prestosql {

/// Enforces the type rules for SQL `value AT TIME ZONE zone`.
///
/// States Presto's rule for the value type. Requires a VARCHAR time zone,
/// allowing UNKNOWN to be coerced to VARCHAR. Takes the result type and
/// evaluation from the 'at_timezone' function registered with the same prefix.
///
/// | Value type               | Accepted                                    |
/// |--------------------------|---------------------------------------------|
/// | TIMESTAMP                | Yes                                         |
/// | TIMESTAMP WITH TIME ZONE | Yes                                         |
/// | TIME WITH TIME ZONE      | Yes                                         |
/// | TIME                     | Only with coercions, as TIME WITH TIME ZONE |
/// | Anything else, e.g. DATE | No                                          |
///
/// Presto function registration installs it under the same 'at_timezone' name
/// as the scalar implementations:
///
///   exec::registerFunctionCallToSpecialForm(
///       prefix + "at_timezone",
///       std::make_unique<AtTimeZoneCallToSpecialForm>(
///           prefix + "at_timezone"));
class AtTimeZoneCallToSpecialForm : public exec::FunctionCallToSpecialForm {
 public:
  /// @param functionName Name of the registered 'at_timezone' function,
  /// including its prefix.
  explicit AtTimeZoneCallToSpecialForm(std::string functionName);

  /// Resolves final argument types without requesting coercions.
  TypePtr resolveType(const std::vector<TypePtr>& argTypes) override;

  /// Requests only TIME to TIME WITH TIME ZONE and UNKNOWN zone to VARCHAR.
  TypePtr resolveTypeWithCoercions(
      const std::vector<TypePtr>& argTypes,
      std::vector<TypePtr>& coercions,
      const TypeCoercer& coercer) override;

  /// Builds the executable expression from a value and a time zone.
  exec::ExprPtr constructSpecialForm(
      const TypePtr& type,
      std::vector<exec::ExprPtr>&& compiledChildren,
      bool trackCpuUsage,
      const core::QueryConfig& config) override;

 private:
  // Name of the 'at_timezone' function, including its prefix, that supplies
  // the result type and evaluates the form.
  const std::string functionName_;
};

} // namespace facebook::velox::functions::prestosql
