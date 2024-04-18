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

#include "velox/functions/sparksql/specialforms/SparkCastHooks.h"
#include "velox/functions/lib/string/StringImpl.h"
#include "velox/type/TimestampConversion.h"

namespace facebook::velox::functions::sparksql {

SparkCastHooks::SparkCastHooks(const core::QueryConfig& config)
    : CastHooks(), legacyCast_(config.isLegacyCast()) {}

Timestamp SparkCastHooks::castStringToTimestamp(const StringView& view) const {
  return util::fromTimestampString(view.data(), view.size());
}

int32_t SparkCastHooks::castStringToDate(const StringView& dateString) const {
  // Allows all patterns supported by Spark:
  // `[+-]yyyy*`
  // `[+-]yyyy*-[m]m`
  // `[+-]yyyy*-[m]m-[d]d`
  // `[+-]yyyy*-[m]m-[d]d *`
  // `[+-]yyyy*-[m]m-[d]dT*`
  // The asterisk `*` in `yyyy*` stands for any numbers.
  // For the last two patterns, the trailing `*` can represent none or any
  // sequence of characters, e.g:
  //   "1970-01-01 123"
  //   "1970-01-01 (BC)"
  return util::castFromDateString(
      removeWhiteSpaces(dateString), util::ParseMode::kNonStandardCast);
}

bool SparkCastHooks::legacy() const {
  return legacyCast_;
}

StringView SparkCastHooks::removeWhiteSpaces(const StringView& view) const {
  StringView output;
  stringImpl::trimUnicodeWhiteSpace<true, true, StringView, StringView>(
      output, view);
  return output;
}

const TimestampToStringOptions& SparkCastHooks::timestampToStringOptions()
    const {
  static constexpr TimestampToStringOptions options = {
      .precision = TimestampToStringOptions::Precision::kMicroseconds,
      .leadingPositiveSign = true,
      .skipTrailingZeros = true,
      .zeroPaddingYear = true,
      .dateTimeSeparator = ' ',
  };
  return options;
}

bool SparkCastHooks::truncate() const {
  return true;
}

std::string SparkCastHooks::castMapToString(
    const MapVector* mapVector,
    const int row) const {
  auto keys = mapVector->mapKeys();
  auto values = mapVector->mapValues();
  auto rawOffsets = mapVector->rawOffsets();
  auto rawSizes = mapVector->rawSizes();

  // spark.sql.legacy.castComplexTypesToString.enabled
  // When true, maps and structs are wrapped by [] in casting to strings, and
  // NULL elements of structs/maps/arrays will be omitted while converting to
  // strings. Otherwise, if this is false, which is the default, maps and
  // structs are wrapped by {}, and NULL elements will be converted to "null".
  std::string_view leftBracket = legacyCast_ ? "[" : "{";
  std::string_view rightBracket = legacyCast_ ? "]" : "}";
  std::string_view nullString = legacyCast_ ? "" : "null";

  std::stringstream out;
  std::string_view delimiter = ", ";
  out << leftBracket;
  if (rawSizes[row] == 0) {
    out << rightBracket;
    return out.str();
  }
  for (vector_size_t i = 0; i < rawSizes[row]; i++) {
    if (i > 0) {
      out << delimiter;
    }
    out << keys->toString(rawOffsets[row] + i) << " ->";
    vector_size_t index = rawOffsets[row] + i;
    if (values->isNullAt(index)) {
      if (!nullString.empty()) {
        out << " " << nullString;
      }
    } else {
      out << " " << values->toString(index);
    }
  }
  out << rightBracket;
  return out.str();
}
} // namespace facebook::velox::functions::sparksql
