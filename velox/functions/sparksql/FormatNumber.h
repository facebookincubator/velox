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

#include <type_traits>

#include "velox/expression/StringWriter.h"
#include "velox/functions/Macros.h"

namespace facebook::velox::functions::sparksql {

namespace detail {

// Formats a number with US locale (comma thousands separator, dot decimal
// separator) and d decimal places. Writes the result directly into 'out'.
// Uses the default format pattern "#,###,###,###,###,###,##0" with d
// fractional digits appended, matching Java DecimalFormat with Locale.US.
void formatInteger(
    int64_t value,
    int32_t decimalPlaces,
    exec::StringWriter& out);

void formatFloatingPoint(
    double value,
    int32_t decimalPlaces,
    exec::StringWriter& out);

} // namespace detail

/// format_number(x, d) -> varchar
///
/// Formats number x with d decimal places using US locale formatting (comma
/// as thousands separator, dot as decimal separator), matching Java
/// DecimalFormat with the default pattern "#,###,###,###,###,###,##0" and
/// Locale.US. Returns null if d < 0.
///
/// Currently only supports the integer second argument (number of decimal
/// places). Spark also supports a string format argument, which is not yet
/// implemented.
// TODO: Support string format argument (user-specified format pattern).
///
/// Unlike CAST(x AS VARCHAR), this adds thousands separators and fixed decimal
/// places with HALF_EVEN (banker's) rounding.
template <typename TExec>
struct FormatNumberFunction {
  VELOX_DEFINE_FUNCTION_TYPES(TExec);

  template <typename T>
  bool call(out_type<Varchar>& result, const T& value, int32_t decimalPlaces) {
    if (decimalPlaces < 0) {
      return false;
    }
    if constexpr (std::is_floating_point_v<T>) {
      detail::formatFloatingPoint(
          static_cast<double>(value), decimalPlaces, result);
    } else {
      detail::formatInteger(static_cast<int64_t>(value), decimalPlaces, result);
    }
    return true;
  }
};

} // namespace facebook::velox::functions::sparksql
