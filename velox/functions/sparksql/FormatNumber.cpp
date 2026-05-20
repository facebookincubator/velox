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

#include "velox/functions/sparksql/FormatNumber.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <locale>

#include <fmt/format.h>

namespace facebook::velox::functions::sparksql::detail {

namespace {

// Java DecimalFormat internally caps fraction digits at 340.
constexpr int32_t kMaxFractionDigits = 340;

// Custom numpunct that provides US-style thousands grouping without
// depending on system locale availability.
struct UsNumpunct : std::numpunct<char> {
  char do_thousands_sep() const override {
    return ',';
  }
  std::string do_grouping() const override {
    return "\3";
  }
  char do_decimal_point() const override {
    return '.';
  }
};

// Locale with US-style thousands grouping for fmt.
const std::locale& usLocale() {
  static const std::locale loc(std::locale::classic(), new UsNumpunct());
  return loc;
}

} // namespace

void formatInteger(
    int64_t value,
    int32_t decimalPlaces,
    exec::StringWriter& out) {
  int32_t cappedPlaces = std::min(decimalPlaces, kMaxFractionDigits);

  // Use fmt::memory_buffer (stack-allocated for typical sizes) to avoid
  // per-row heap allocation.
  fmt::memory_buffer buf;
  fmt::format_to(std::back_inserter(buf), usLocale(), "{:Ld}", value);

  if (cappedPlaces > 0) {
    buf.push_back('.');
    for (int32_t i = 0; i < cappedPlaces; ++i) {
      buf.push_back('0');
    }
  }

  out.append(std::string_view(buf.data(), buf.size()));
}

void formatFloatingPoint(
    double value,
    int32_t decimalPlaces,
    exec::StringWriter& out) {
  if (std::isnan(value)) {
    out.append("NaN");
    return;
  }
  if (std::isinf(value)) {
    // Java DecimalFormat with US locale uses the infinity symbol (U+221E).
    if (value < 0) {
      out.append("-\xE2\x88\x9E");
    } else {
      out.append("\xE2\x88\x9E");
    }
    return;
  }

  int32_t cappedPlaces = std::min(decimalPlaces, kMaxFractionDigits);

  // Use fmt::memory_buffer (stack-allocated for typical sizes) to avoid
  // per-row heap allocation.
  fmt::memory_buffer buf;
  fmt::format_to(
      std::back_inserter(buf), usLocale(), "{:.{}Lf}", value, cappedPlaces);
  out.append(std::string_view(buf.data(), buf.size()));
}

} // namespace facebook::velox::functions::sparksql::detail
