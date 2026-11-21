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

/// Name of the AT TIME ZONE function, without the registration prefix.
inline constexpr const char* kAtTimeZone = "at_timezone";

/// Implements SQL `value AT TIME ZONE zone`.
///
/// Applies Presto's rule for the value type in both modes. Requires a VARCHAR
/// time zone, allowing UNKNOWN to be coerced to VARCHAR.
///
/// | Value type               | Accepted                                    |
/// |--------------------------|---------------------------------------------|
/// | TIMESTAMP                | Yes                                         |
/// | TIMESTAMP WITH TIME ZONE | Yes                                         |
/// | TIME WITH TIME ZONE      | Yes                                         |
/// | TIME                     | Only with coercions, as TIME WITH TIME ZONE |
/// | Anything else, e.g. DATE | No                                          |
///
/// 'ignoreTimeZoneLabel' selects the result type for TIMESTAMP and TIMESTAMP
/// WITH TIME ZONE values: TIMESTAMP when true, TIMESTAMP WITH TIME ZONE when
/// false. TIME WITH TIME ZONE values return TIME WITH TIME ZONE in both modes.
///
/// The form creates its implementations itself. It is the only registration
/// under its name, so lookups that skip special forms, such as
/// getFunctionSignatures, do not see AT TIME ZONE.
class AtTimeZoneCallToSpecialForm : public exec::FunctionCallToSpecialForm {
 public:
  /// @param name Registered name, including its prefix.
  /// @param ignoreTimeZoneLabel Selects the result type, as described above.
  /// @param owner Owner recorded in the expression's metadata. Held as a view,
  /// so it must outlive the form.
  AtTimeZoneCallToSpecialForm(
      std::string name,
      bool ignoreTimeZoneLabel,
      std::string_view owner);

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
  // Names the expression in stats and errors.
  const std::string name_;
  // Selects the result type and implementation for timestamp values.
  const bool ignoreTimeZoneLabel_;
  // Recorded in the expression's metadata for logging and listeners.
  const std::string_view owner_;
};

} // namespace facebook::velox::functions::prestosql
