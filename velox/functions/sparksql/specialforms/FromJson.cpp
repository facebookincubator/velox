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

#include "velox/functions/sparksql/specialforms/FromJson.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

#include <fmt/format.h>
#include <folly/Conv.h>
#include <folly/Unicode.h>

#include "velox/expression/EvalCtx.h"
#include "velox/expression/SpecialForm.h"
#include "velox/expression/VectorReaders.h"
#include "velox/expression/VectorWriters.h"
#include "velox/functions/lib/JsonUtil.h"
#include "velox/functions/lib/Utf8Utils.h"
#include "velox/functions/lib/string/StringCore.h"
#include "velox/functions/lib/string/StringImpl.h"
#include "velox/functions/prestosql/json/JsonStringUtil.h"
#include "velox/functions/prestosql/json/SIMDJsonUtil.h"
#include "velox/functions/sparksql/specialforms/JavaFloatingDecimal.h"
#include "velox/type/DecimalUtil.h"
#include "velox/type/TimestampConversion.h"

using namespace facebook::velox::exec;

namespace facebook::velox::functions::sparksql {
namespace {

bool isJsonDigit(char character) {
  return character >= '0' && character <= '9';
}

// Validates strict JSON number syntax before using simdjson's looser parser.
bool isValidJsonNumber(std::string_view input) {
  size_t position{0};
  if (input.empty()) {
    return false;
  }
  if (input[position] == '-') {
    if (++position == input.size()) {
      return false;
    }
  }

  if (input[position] == '0') {
    ++position;
    if (position < input.size() && isJsonDigit(input[position])) {
      return false;
    }
  } else if (input[position] >= '1' && input[position] <= '9') {
    while (position < input.size() && isJsonDigit(input[position])) {
      ++position;
    }
  } else {
    return false;
  }

  if (position < input.size() && input[position] == '.') {
    ++position;
    const auto fractionStart = position;
    while (position < input.size() && isJsonDigit(input[position])) {
      ++position;
    }
    if (position == fractionStart) {
      return false;
    }
  }

  if (position < input.size() &&
      (input[position] == 'e' || input[position] == 'E')) {
    ++position;
    if (position < input.size() &&
        (input[position] == '+' || input[position] == '-')) {
      ++position;
    }
    const auto exponentStart = position;
    while (position < input.size() && isJsonDigit(input[position])) {
      ++position;
    }
    if (position == exponentStart) {
      return false;
    }
  }

  return position == input.size();
}

// Spark accepts single-quoted JSON strings by default. Detect them cheaply so
// standard JSON remains allocation-free.
bool normalizeSingleQuotedJson(std::string_view input, std::string& output) {
  if (input.find('\'') == std::string_view::npos) {
    return false;
  }

  enum class Quote { kNone, kSingle, kDouble };

  Quote quote{Quote::kNone};
  bool converted{false};
  output.clear();
  output.reserve(input.size());
  for (size_t i = 0; i < input.size(); ++i) {
    const auto character = input[i];
    if (quote == Quote::kNone) {
      if (character == '\'') {
        output.push_back('"');
        quote = Quote::kSingle;
        converted = true;
      } else {
        output.push_back(character);
        if (character == '"') {
          quote = Quote::kDouble;
        }
      }
      continue;
    }

    if (quote == Quote::kDouble) {
      if (character == '\\' && i + 1 < input.size()) {
        const auto escaped = input[++i];
        if (escaped == '\'') {
          output.push_back('\'');
          converted = true;
        } else {
          output.push_back('\\');
          output.push_back(escaped);
        }
      } else {
        output.push_back(character);
        if (character == '"') {
          quote = Quote::kNone;
        }
      }
      continue;
    }

    if (character == '\\' && i + 1 < input.size()) {
      const auto escaped = input[++i];
      if (escaped == '\'') {
        output.push_back('\'');
      } else {
        output.push_back('\\');
        output.push_back(escaped);
      }
    } else if (character == '\'') {
      output.push_back('"');
      quote = Quote::kNone;
    } else if (character == '"') {
      output.append("\\\"");
    } else {
      output.push_back(character);
    }
  }
  return converted && quote == Quote::kNone;
}

bool isJsonWhitespace(char character) {
  return character == ' ' || character == '\t' || character == '\n' ||
      character == '\r';
}

// Spark's streaming parser consumes one root value and does not inspect the
// remaining input.
size_t firstJsonValueSize(std::string_view input) {
  size_t position{0};
  while (position < input.size() && isJsonWhitespace(input[position])) {
    ++position;
  }
  if (position == input.size()) {
    return input.size();
  }

  if (input.substr(position, 4) == "null") {
    return position + 4;
  }
  if (input[position] != '{' && input[position] != '[') {
    return input.size();
  }

  std::vector<char> closingDelimiters;
  char quote{0};
  bool escaped{false};
  for (size_t i = position; i < input.size(); ++i) {
    const auto character = input[i];
    if (quote != 0) {
      if (escaped) {
        escaped = false;
      } else if (character == '\\') {
        escaped = true;
      } else if (character == quote) {
        quote = 0;
      }
      continue;
    }

    if (character == '"' || character == '\'') {
      quote = character;
    } else if (character == '{') {
      closingDelimiters.push_back('}');
    } else if (character == '[') {
      closingDelimiters.push_back(']');
    } else if (character == '}' || character == ']') {
      if (closingDelimiters.empty() || closingDelimiters.back() != character) {
        return input.size();
      }
      closingDelimiters.pop_back();
      if (closingDelimiters.empty()) {
        const auto valueSize = i + 1;
        const auto trailingInput = input.substr(valueSize);
        if (std::all_of(
                trailingInput.begin(), trailingInput.end(), isJsonWhitespace)) {
          return input.size();
        }
        return valueSize;
      }
    }
  }
  return input.size();
}

int hexDigitValue(char character) {
  if (character >= '0' && character <= '9') {
    return character - '0';
  }
  if (character >= 'a' && character <= 'f') {
    return character - 'a' + 10;
  }
  if (character >= 'A' && character <= 'F') {
    return character - 'A' + 10;
  }
  return -1;
}

bool parseUnicodeEscape(
    std::string_view input,
    size_t position,
    uint16_t& codeUnit) {
  if (position + 6 > input.size() || input[position] != '\\' ||
      input[position + 1] != 'u') {
    return false;
  }

  codeUnit = 0;
  for (size_t i = position + 2; i < position + 6; ++i) {
    const auto digit = hexDigitValue(input[i]);
    if (digit < 0) {
      return false;
    }
    codeUnit = static_cast<uint16_t>((codeUnit << 4) | digit);
  }
  return true;
}

int unicodeDecimalDigit(char32_t codePoint) {
  constexpr char32_t kDecimalDigitStarts[] = {
      0x0030, 0x0660, 0x06F0, 0x07C0, 0x0966, 0x09E6, 0x0A66, 0x0AE6,
      0x0B66, 0x0BE6, 0x0C66, 0x0CE6, 0x0D66, 0x0DE6, 0x0E50, 0x0ED0,
      0x0F20, 0x1040, 0x1090, 0x17E0, 0x1810, 0x1946, 0x19D0, 0x1A80,
      0x1A90, 0x1B50, 0x1BB0, 0x1C40, 0x1C50, 0xA620, 0xA8D0, 0xA900,
      0xA9D0, 0xA9F0, 0xAA50, 0xABF0, 0xFF10,
  };
  const auto* upper = std::upper_bound(
      std::begin(kDecimalDigitStarts),
      std::end(kDecimalDigitStarts),
      codePoint);
  if (upper == std::begin(kDecimalDigitStarts)) {
    return -1;
  }
  const auto zero = *std::prev(upper);
  return codePoint < zero + 10 ? static_cast<int>(codePoint - zero) : -1;
}

bool normalizeUnicodeDecimalDigits(
    std::string_view input,
    std::string& output) {
  if (stringCore::isAscii(input.data(), input.size())) {
    return false;
  }

  output.clear();
  output.reserve(input.size());
  bool converted{false};
  const auto* position = reinterpret_cast<const unsigned char*>(input.data());
  const auto* const end = position + input.size();
  while (position < end) {
    const auto* characterStart = position;
    const auto codePoint = folly::utf8ToCodePoint(position, end, true);
    const auto digit = unicodeDecimalDigit(codePoint);
    if (digit >= 0) {
      output.push_back(static_cast<char>('0' + digit));
      converted = true;
    } else {
      output.append(
          reinterpret_cast<const char*>(characterStart),
          position - characterStart);
    }
  }
  return converted;
}

using UnpairedSurrogate = std::pair<size_t, uint16_t>;

// Locates unpaired UTF-16 escapes by decoded character position so Spark can
// preserve Jackson's serialized representation.
std::vector<UnpairedSurrogate> findUnpairedSurrogates(
    std::string_view escapedInput) {
  if (escapedInput.size() >= 2 && escapedInput.front() == '"' &&
      escapedInput.back() == '"') {
    escapedInput.remove_prefix(1);
    escapedInput.remove_suffix(1);
  }

  std::vector<UnpairedSurrogate> surrogates;
  size_t characterIndex{0};
  for (size_t i = 0; i < escapedInput.size(); ++characterIndex) {
    if (escapedInput[i] != '\\') {
      const auto characterLength = firstByteCharLength(escapedInput.data() + i);
      i += std::min(
          escapedInput.size() - i,
          static_cast<size_t>(std::max(characterLength, 1)));
      continue;
    }

    uint16_t codeUnit;
    if (!parseUnicodeEscape(escapedInput, i, codeUnit)) {
      i += std::min<size_t>(2, escapedInput.size() - i);
      continue;
    }

    const bool isHighSurrogate = codeUnit >= 0xD800 && codeUnit <= 0xDBFF;
    const bool isLowSurrogate = codeUnit >= 0xDC00 && codeUnit <= 0xDFFF;
    if (isHighSurrogate) {
      uint16_t lowCodeUnit;
      if (parseUnicodeEscape(escapedInput, i + 6, lowCodeUnit) &&
          lowCodeUnit >= 0xDC00 && lowCodeUnit <= 0xDFFF) {
        i += 12;
        continue;
      }
    }
    if (isHighSurrogate || isLowSurrogate) {
      surrogates.emplace_back(characterIndex, codeUnit);
    }
    i += 6;
  }
  return surrogates;
}

// Java's UTF-8 encoder replaces an unpaired UTF-16 surrogate with '?'.
bool replaceUnpairedSurrogateCharacters(
    std::string_view escapedInput,
    std::string_view decodedInput,
    std::string& output) {
  const auto surrogates = findUnpairedSurrogates(escapedInput);
  if (surrogates.empty()) {
    return false;
  }

  output.clear();
  output.reserve(decodedInput.size());
  size_t surrogatePosition{0};
  size_t characterIndex{0};
  for (size_t i = 0; i < decodedInput.size(); ++characterIndex) {
    const auto characterLength = firstByteCharLength(decodedInput.data() + i);
    const auto characterSize = std::min(
        decodedInput.size() - i,
        static_cast<size_t>(std::max(characterLength, 1)));
    if (surrogatePosition < surrogates.size() &&
        surrogates[surrogatePosition].first == characterIndex) {
      output.push_back('?');
      ++surrogatePosition;
    } else {
      output.append(decodedInput.substr(i, characterSize));
    }
    i += characterSize;
  }
  return true;
}

// Appends Velox's canonical JSON escaping without the surrounding quotes.
void appendNormalizedJsonString(std::string_view input, std::string& output) {
  const auto oldSize = output.size();
  const auto normalizedSize =
      normalizedSizeForJsonCast(input.data(), input.size());
  output.resize(oldSize + normalizedSize);
  normalizeForJsonCast(input.data(), input.size(), output.data() + oldSize);
}

// Preserves Jackson's treatment of unpaired UTF-16 escapes while delegating
// ordinary JSON escaping to Velox's shared implementation.
void appendSparkJsonString(
    std::string_view escapedInput,
    std::string_view decodedInput,
    std::string& output) {
  const auto surrogates = findUnpairedSurrogates(escapedInput);
  output.push_back('"');
  size_t surrogatePosition{0};
  size_t characterIndex{0};
  size_t segmentStart{0};
  for (size_t i = 0; i < decodedInput.size(); ++characterIndex) {
    const auto characterLength = firstByteCharLength(decodedInput.data() + i);
    const auto characterSize = std::min(
        decodedInput.size() - i,
        static_cast<size_t>(std::max(characterLength, 1)));
    if (surrogatePosition < surrogates.size() &&
        surrogates[surrogatePosition].first == characterIndex) {
      appendNormalizedJsonString(
          decodedInput.substr(segmentStart, i - segmentStart), output);
      const auto escapeStart = output.size();
      output.resize(escapeStart + 6);
      auto* escapePosition = output.data() + escapeStart;
      writeUtf16Escape(surrogates[surrogatePosition].second, escapePosition);
      segmentStart = i + characterSize;
      ++surrogatePosition;
    }
    i += characterSize;
  }
  appendNormalizedJsonString(decodedInput.substr(segmentStart), output);
  output.push_back('"');
}

constexpr simdjson::error_code kMalformedAtomError{simdjson::T_ATOM_ERROR};
constexpr simdjson::error_code kObjectTruncationError{simdjson::F_ATOM_ERROR};

bool isSparkSpecialNumber(std::string_view token) {
  return token == "NaN" || token == "+INF" || token == "+Infinity" ||
      token == "Infinity" || token == "-INF" || token == "-Infinity";
}

// Writes Spark's quoted spellings for non-finite floating-point values.
bool appendSparkSpecialNumber(std::string_view token, std::string& output) {
  if (token == "NaN") {
    output.append(R"("NaN")");
    return true;
  }
  if (token == "+INF" || token == "+Infinity" || token == "Infinity") {
    output.append(R"("Infinity")");
    return true;
  }
  if (token == "-INF" || token == "-Infinity") {
    output.append(R"("-Infinity")");
    return true;
  }
  return false;
}

// Recursively serializes a parsed value using Spark's compact JSON spelling.
template <typename JsonInput>
simdjson::error_code appendSparkJsonValue(
    JsonInput value,
    std::string& output) {
  SIMDJSON_ASSIGN_OR_RAISE(const auto type, value.type());
  switch (type) {
    case simdjson::ondemand::json_type::string: {
      const auto escapedString = trimToken(value.raw_json_token());
      const auto stringResult = value.get_string(/*allow_replacement=*/true);
      if (stringResult.error() != simdjson::SUCCESS) {
        return kMalformedAtomError;
      }
      appendSparkJsonString(escapedString, stringResult.value_unsafe(), output);
      return simdjson::SUCCESS;
    }
    case simdjson::ondemand::json_type::number: {
      const auto token = trimToken(value.raw_json());
      if (appendSparkSpecialNumber(token, output)) {
        return simdjson::SUCCESS;
      }
      if (!isValidJsonNumber(token)) {
        return kMalformedAtomError;
      }

      if (token.find_first_of(".eE") == std::string_view::npos) {
        output.append(token == "-0" ? "0" : token);
        return simdjson::SUCCESS;
      }

      double number{0.0};
      const auto numberResult = value.get_double();
      if (numberResult.error() == simdjson::SUCCESS) {
        number = numberResult.value_unsafe();
      } else {
        const auto castResult =
            util::Converter<TypeKind::DOUBLE>::tryCast(token);
        if (castResult.hasError()) {
          return simdjson::NUMBER_ERROR;
        }
        number = castResult.value();
      }
      if (std::isfinite(number)) {
        appendJavaDouble(number, output);
      } else if (std::isnan(number)) {
        output.append(R"("NaN")");
      } else {
        output.append(
            std::signbit(number) ? R"("-Infinity")" : R"("Infinity")");
      }
      return simdjson::SUCCESS;
    }
    case simdjson::ondemand::json_type::boolean: {
      const auto booleanResult = value.get_bool();
      if (booleanResult.error() != simdjson::SUCCESS) {
        return kMalformedAtomError;
      }
      output.append(booleanResult.value_unsafe() ? "true" : "false");
      return simdjson::SUCCESS;
    }
    case simdjson::ondemand::json_type::array: {
      SIMDJSON_ASSIGN_OR_RAISE(auto array, value.get_array());
      output.push_back('[');
      bool first{true};
      for (const auto& elementResult : array) {
        SIMDJSON_ASSIGN_OR_RAISE(auto element, elementResult);
        if (!first) {
          output.push_back(',');
        }
        first = false;
        const auto result = appendSparkJsonValue(element, output);
        if (result == kMalformedAtomError) {
          return simdjson::INCORRECT_TYPE;
        }
        SIMDJSON_TRY(result);
      }
      output.push_back(']');
      return simdjson::SUCCESS;
    }
    case simdjson::ondemand::json_type::object: {
      SIMDJSON_ASSIGN_OR_RAISE(auto object, value.get_object());
      output.push_back('{');
      bool first{true};
      for (const auto& fieldResult : object) {
        SIMDJSON_ASSIGN_OR_RAISE(auto field, fieldResult);
        const auto escapedKey = field.escaped_key();
        SIMDJSON_ASSIGN_OR_RAISE(auto key, field.unescaped_key(true));
        if (!first) {
          output.push_back(',');
        }
        first = false;
        appendSparkJsonString(escapedKey, key, output);
        output.push_back(':');
        const auto result = appendSparkJsonValue(field.value(), output);
        if (result == kMalformedAtomError || result == kObjectTruncationError ||
            result == simdjson::INCORRECT_TYPE) {
          return kObjectTruncationError;
        }
        SIMDJSON_TRY(result);
      }
      output.push_back('}');
      return simdjson::SUCCESS;
    }
    case simdjson::ondemand::json_type::null:
      if (trimToken(value.raw_json()) != "null") {
        return kMalformedAtomError;
      }
      output.append("null");
      return simdjson::SUCCESS;
    case simdjson::ondemand::json_type::unknown: {
      const auto token = trimToken(value.raw_json());
      return appendSparkSpecialNumber(token, output) ? simdjson::SUCCESS
                                                     : kMalformedAtomError;
    }
  }
  VELOX_UNREACHABLE();
}

template <typename JsonInput>
simdjson::error_code validateJsonValue(JsonInput value) {
  SIMDJSON_ASSIGN_OR_RAISE(const auto type, value.type());
  switch (type) {
    case simdjson::ondemand::json_type::string:
      return value.get_string(/*allow_replacement=*/true).error() ==
              simdjson::SUCCESS
          ? simdjson::SUCCESS
          : kMalformedAtomError;
    case simdjson::ondemand::json_type::number: {
      const auto token = trimToken(value.raw_json());
      std::string ignored;
      return appendSparkSpecialNumber(token, ignored) ||
              isValidJsonNumber(token)
          ? simdjson::SUCCESS
          : kMalformedAtomError;
    }
    case simdjson::ondemand::json_type::boolean:
      return value.get_bool().error() == simdjson::SUCCESS
          ? simdjson::SUCCESS
          : kMalformedAtomError;
    case simdjson::ondemand::json_type::array: {
      SIMDJSON_ASSIGN_OR_RAISE(auto array, value.get_array());
      for (const auto& elementResult : array) {
        SIMDJSON_ASSIGN_OR_RAISE(auto element, elementResult);
        const auto result = validateJsonValue(element);
        if (result == kMalformedAtomError) {
          return simdjson::INCORRECT_TYPE;
        }
        SIMDJSON_TRY(result);
      }
      return simdjson::SUCCESS;
    }
    case simdjson::ondemand::json_type::object: {
      SIMDJSON_ASSIGN_OR_RAISE(auto object, value.get_object());
      for (const auto& fieldResult : object) {
        SIMDJSON_ASSIGN_OR_RAISE(auto field, fieldResult);
        SIMDJSON_TRY(field.unescaped_key(/*allow_replacement=*/true).error());
        const auto result = validateJsonValue(field.value());
        if (result == kMalformedAtomError || result == kObjectTruncationError ||
            result == simdjson::INCORRECT_TYPE) {
          return kObjectTruncationError;
        }
        SIMDJSON_TRY(result);
      }
      return simdjson::SUCCESS;
    }
    case simdjson::ondemand::json_type::null:
      return trimToken(value.raw_json()) == "null" ? simdjson::SUCCESS
                                                   : kMalformedAtomError;
    case simdjson::ondemand::json_type::unknown: {
      const auto token = trimToken(value.raw_json());
      return isSparkSpecialNumber(token) ? simdjson::SUCCESS
                                         : kMalformedAtomError;
    }
  }
  VELOX_UNREACHABLE();
}

bool isConversionError(simdjson::error_code error) {
  return error == simdjson::INCORRECT_TYPE ||
      error == simdjson::NUMBER_OUT_OF_RANGE;
}

// Preserves the generic container casts used by from_json across a batch.
void copyParsedJsonValue(
    exec::GenericWriter& writer,
    const exec::GenericView& view) {
  const auto kind = view.kind();
  if (kind == TypeKind::ARRAY) {
    auto& output = writer.castTo<Array<Any>>();
    auto input = view.castTo<Array<Any>>();
    for (auto element : input) {
      if (element.has_value()) {
        copyParsedJsonValue(output.add_item(), element.value());
      } else {
        output.add_null();
      }
    }
    return;
  }

  if (kind == TypeKind::MAP) {
    auto& output = writer.castTo<Map<Any, Any>>();
    auto input = view.castTo<Map<Any, Any>>();
    for (auto [key, value] : input) {
      if (value.has_value()) {
        auto [keyWriter, valueWriter] = output.add_item();
        copyParsedJsonValue(keyWriter, key);
        copyParsedJsonValue(valueWriter, value.value());
      } else {
        auto& keyWriter = output.add_null();
        copyParsedJsonValue(keyWriter, key);
      }
    }
    return;
  }

  if (kind == TypeKind::ROW) {
    auto& output = writer.castTo<DynamicRow>();
    auto input = view.castTo<DynamicRow>();
    for (column_index_t i = 0; i < input.size(); ++i) {
      const auto field = input.at(i);
      if (field.has_value()) {
        copyParsedJsonValue(output.get_writer_at(i), field.value());
      } else {
        output.set_null_at(i);
      }
    }
    return;
  }

  writer.copy_from(view);
}

int64_t floorDivide(int64_t dividend, int64_t divisor) {
  VELOX_DCHECK_GT(divisor, 0);
  return dividend >= 0 ? dividend / divisor
                       : -((-dividend + divisor - 1) / divisor);
}

// Mirrors Spark's RebaseDateTime.rebaseJulianToGregorianDays for legacy
// Julian-calendar day values, including dates before the precomputed table.
int32_t rebaseJulianToGregorianDays(int32_t days) {
  constexpr std::array<int32_t, 14> kSwitchDays{
      -719'164,
      -682'945,
      -646'420,
      -609'895,
      -536'845,
      -500'320,
      -463'795,
      -390'745,
      -354'220,
      -317'695,
      -244'645,
      -208'120,
      -171'595,
      -141'427,
  };
  constexpr std::array<int32_t, 14> kDifferences{
      2, 1, 0, -1, -2, -3, -4, -5, -6, -7, -8, -9, -10, 0};
  if (days >= kSwitchDays.front()) {
    const auto upper =
        std::upper_bound(kSwitchDays.begin(), kSwitchDays.end(), days);
    const auto index = std::distance(kSwitchDays.begin(), upper) - 1;
    return days + kDifferences[index];
  }

  const int64_t julianDayNumber = 2'440'588LL + days;
  const int64_t shiftedJulianDays = julianDayNumber + 32'082;
  const int64_t fourYearCycle = floorDivide(4 * shiftedJulianDays + 3, 1'461);
  const int64_t dayOfFourYearCycle =
      shiftedJulianDays - floorDivide(1'461 * fourYearCycle, 4);
  const int64_t monthIndex = floorDivide(5 * dayOfFourYearCycle + 2, 153);
  const int64_t day =
      dayOfFourYearCycle - floorDivide(153 * monthIndex + 2, 5) + 1;
  const int64_t month = monthIndex + 3 - 12 * floorDivide(monthIndex, 10);
  const int64_t year = fourYearCycle - 4'800 + floorDivide(monthIndex, 10);

  const auto firstDay = util::daysSinceEpochFromDate(
      static_cast<int32_t>(year), static_cast<int32_t>(month), 1);
  VELOX_DCHECK(!firstDay.hasError());
  const auto rebased = firstDay.value() + day - 1;
  VELOX_DCHECK_GE(rebased, std::numeric_limits<int32_t>::min());
  VELOX_DCHECK_LE(rebased, std::numeric_limits<int32_t>::max());
  return static_cast<int32_t>(rebased);
}

bool isSignedDecimalInteger(std::string_view input) {
  if (input.empty()) {
    return false;
  }
  if (input.front() == '-' || input.front() == '+') {
    input.remove_prefix(1);
  }
  return !input.empty() && std::all_of(input.begin(), input.end(), isJsonDigit);
}

// Struct to store schema information for a JSON row, used for efficient field
// lookup and null handling.
struct JsonRowSchemaInfo {
  // Unique key for this schema info, computed from the nesting level and field
  // index.
  uint64_t key;

  // Shared pointer to a vector indicating which fields are missing in the
  // current JSON object.
  std::shared_ptr<std::vector<bool>> isFieldMissing;

  // Maps field names to their column indices for fast lookup.
  folly::F14FastMap<std::string, column_index_t> fieldIndices;

  // Maps field names to their node index in the schema tree for fast lookup.
  folly::F14FastMap<std::string, column_index_t> nodeIndices;

  JsonRowSchemaInfo(
      uint64_t key,
      std::shared_ptr<std::vector<bool>>&& isFieldMissing,
      folly::F14FastMap<std::string, column_index_t>&& fieldIndices,
      folly::F14FastMap<std::string, column_index_t>&& nodeIndices)
      : key(key),
        isFieldMissing(std::move(isFieldMissing)),
        fieldIndices(std::move(fieldIndices)),
        nodeIndices(std::move(nodeIndices)) {}
};

// Struct for extracting JSON data and writing it with type-specific handling.
template <typename Input>
struct ExtractJsonTypeImpl {
  template <TypeKind kind>
  static simdjson::error_code apply(
      Input input,
      exec::GenericWriter& writer,
      memory::MemoryPool* pool,
      bool isRoot,
      const folly::F14FastMap<int64_t, JsonRowSchemaInfo>& jsonRowSchemaInfo,
      column_index_t nodeIndex) {
    SIMDJSON_ASSIGN_OR_RAISE(const auto type, input.type());
    if (type == simdjson::ondemand::json_type::boolean) {
      const auto booleanResult = input.get_bool();
      if (booleanResult.error() != simdjson::SUCCESS) {
        return kMalformedAtomError;
      }
      if constexpr (kind == TypeKind::BOOLEAN) {
        writer.castTo<bool>() = booleanResult.value_unsafe();
        return simdjson::SUCCESS;
      }
      if constexpr (kind == TypeKind::VARCHAR) {
        writer.castTo<Varchar>().append(
            booleanResult.value_unsafe() ? "true" : "false");
        return simdjson::SUCCESS;
      }
      return simdjson::INCORRECT_TYPE;
    }
    if (type == simdjson::ondemand::json_type::string) {
      bool acceptsString{false};
      if constexpr (
          kind == TypeKind::VARCHAR || kind == TypeKind::REAL ||
          kind == TypeKind::DOUBLE || kind == TypeKind::HUGEINT) {
        acceptsString = true;
      } else if constexpr (kind == TypeKind::INTEGER) {
        acceptsString = writer.type() == DATE();
      } else if constexpr (kind == TypeKind::BIGINT) {
        acceptsString = writer.type()->isShortDecimal();
      }
      if (!acceptsString) {
        if (input.get_string(/*allow_replacement=*/true).error() !=
            simdjson::SUCCESS) {
          return simdjson::STRING_ERROR;
        }
        return simdjson::INCORRECT_TYPE;
      }
    }
    if (type == simdjson::ondemand::json_type::null &&
        trimToken(input.raw_json()) != "null") {
      return kMalformedAtomError;
    }
    if (type == simdjson::ondemand::json_type::number) {
      const auto token = trimToken(input.raw_json());
      if (!isValidJsonNumber(token) && !isSparkSpecialNumber(token)) {
        return kMalformedAtomError;
      }
    }
    if (type == simdjson::ondemand::json_type::unknown &&
        !isSparkSpecialNumber(trimToken(input.raw_json()))) {
      return kMalformedAtomError;
    }
    return KindDispatcher<kind>::apply(
        input, writer, pool, isRoot, jsonRowSchemaInfo, nodeIndex);
  }

 private:
  // Dummy is needed because full/explicit specialization is not allowed inside
  // class.
  template <TypeKind kind, typename Dummy = void>
  struct KindDispatcher {
    static simdjson::error_code apply(
        Input,
        exec::GenericWriter&,
        memory::MemoryPool* /*pool*/,
        bool /*isRoot*/,
        const folly::
            F14FastMap<int64_t, JsonRowSchemaInfo>& /*jsonRowSchemaInfo*/,
        column_index_t /*nodeIndex*/) {
      VELOX_NYI("Parse json to {} is not supported.", TypeTraits<kind>::name);
      return simdjson::error_code::UNEXPECTED_ERROR;
    }
  };

  template <typename Dummy>
  struct KindDispatcher<TypeKind::VARCHAR, Dummy> {
    static simdjson::error_code apply(
        Input value,
        exec::GenericWriter& writer,
        memory::MemoryPool* /*pool*/,
        bool /*isRoot*/,
        const folly::
            F14FastMap<int64_t, JsonRowSchemaInfo>& /*jsonRowSchemaInfo*/,
        column_index_t /*nodeIndex*/) {
      SIMDJSON_ASSIGN_OR_RAISE(auto type, value.type());
      if (type == simdjson::ondemand::json_type::string) {
        const auto escapedString = trimToken(value.raw_json_token());
        SIMDJSON_ASSIGN_OR_RAISE(
            const auto stringValue,
            value.get_string(/*allow_replacement=*/true));
        std::string repairedString;
        if (replaceUnpairedSurrogateCharacters(
                escapedString, stringValue, repairedString)) {
          writer.castTo<Varchar>().append(repairedString);
        } else {
          writer.castTo<Varchar>().append(stringValue);
        }
        return simdjson::SUCCESS;
      }

      std::string stringValue;
      SIMDJSON_TRY(appendSparkJsonValue(value, stringValue));
      writer.castTo<Varchar>().append(stringValue);
      return simdjson::SUCCESS;
    }
  };

  template <typename Dummy>
  struct KindDispatcher<TypeKind::BOOLEAN, Dummy> {
    static simdjson::error_code apply(
        Input value,
        exec::GenericWriter& writer,
        memory::MemoryPool* /*pool*/,
        bool /*isRoot*/,
        const folly::
            F14FastMap<int64_t, JsonRowSchemaInfo>& /*jsonRowSchemaInfo*/,
        column_index_t /*nodeIndex*/) {
      SIMDJSON_ASSIGN_OR_RAISE(auto type, value.type());
      if (type == simdjson::ondemand::json_type::boolean) {
        auto& w = writer.castTo<bool>();
        SIMDJSON_ASSIGN_OR_RAISE(w, value.get_bool());
        return simdjson::SUCCESS;
      }
      return simdjson::INCORRECT_TYPE;
    }
  };

  template <typename Dummy>
  struct KindDispatcher<TypeKind::TINYINT, Dummy> {
    static simdjson::error_code apply(
        Input value,
        exec::GenericWriter& writer,
        memory::MemoryPool* /*pool*/,
        bool /*isRoot*/,
        const folly::
            F14FastMap<int64_t, JsonRowSchemaInfo>& /*jsonRowSchemaInfo*/,
        column_index_t /*nodeIndex*/) {
      return castJsonToInt<int8_t>(value, writer);
    }
  };

  template <typename Dummy>
  struct KindDispatcher<TypeKind::SMALLINT, Dummy> {
    static simdjson::error_code apply(
        Input value,
        exec::GenericWriter& writer,
        memory::MemoryPool* /*pool*/,
        bool /*isRoot*/,
        const folly::
            F14FastMap<int64_t, JsonRowSchemaInfo>& /*jsonRowSchemaInfo*/,
        column_index_t /*nodeIndex*/) {
      return castJsonToInt<int16_t>(value, writer);
    }
  };

  template <typename Dummy>
  struct KindDispatcher<TypeKind::INTEGER, Dummy> {
    static simdjson::error_code apply(
        Input value,
        exec::GenericWriter& writer,
        memory::MemoryPool* /*pool*/,
        bool /*isRoot*/,
        const folly::
            F14FastMap<int64_t, JsonRowSchemaInfo>& /*jsonRowSchemaInfo*/,
        column_index_t /*nodeIndex*/) {
      if (writer.type() == DATE()) {
        return castJsonToDate(value, writer);
      }
      return castJsonToInt<int32_t>(value, writer);
    }
  };

  template <typename Dummy>
  struct KindDispatcher<TypeKind::BIGINT, Dummy> {
    static simdjson::error_code apply(
        Input value,
        exec::GenericWriter& writer,
        memory::MemoryPool* /*pool*/,
        bool /*isRoot*/,
        const folly::
            F14FastMap<int64_t, JsonRowSchemaInfo>& /*jsonRowSchemaInfo*/,
        column_index_t /*nodeIndex*/) {
      if (writer.type()->isShortDecimal()) {
        return castJsonToDecimal<int64_t>(value, writer);
      }
      return castJsonToInt<int64_t>(value, writer);
    }
  };

  template <typename Dummy>
  struct KindDispatcher<TypeKind::HUGEINT, Dummy> {
    static simdjson::error_code apply(
        Input value,
        exec::GenericWriter& writer,
        memory::MemoryPool* /*pool*/,
        bool /*isRoot*/,
        const folly::
            F14FastMap<int64_t, JsonRowSchemaInfo>& /*jsonRowSchemaInfo*/,
        column_index_t /*nodeIndex*/) {
      VELOX_CHECK(writer.type()->isLongDecimal());
      return castJsonToDecimal<int128_t>(value, writer);
    }
  };

  template <typename Dummy>
  struct KindDispatcher<TypeKind::REAL, Dummy> {
    static simdjson::error_code apply(
        Input value,
        exec::GenericWriter& writer,
        memory::MemoryPool* /*pool*/,
        bool /*isRoot*/,
        const folly::
            F14FastMap<int64_t, JsonRowSchemaInfo>& /*jsonRowSchemaInfo*/,
        column_index_t /*nodeIndex*/) {
      return castJsonToFloatingPoint<float>(value, writer);
    }
  };

  template <typename Dummy>
  struct KindDispatcher<TypeKind::DOUBLE, Dummy> {
    static simdjson::error_code apply(
        Input value,
        exec::GenericWriter& writer,
        memory::MemoryPool* /*pool*/,
        bool /*isRoot*/,
        const folly::
            F14FastMap<int64_t, JsonRowSchemaInfo>& /*jsonRowSchemaInfo*/,
        column_index_t /*nodeIndex*/) {
      return castJsonToFloatingPoint<double>(value, writer);
    }
  };

  template <typename Dummy>
  struct KindDispatcher<TypeKind::ARRAY, Dummy> {
    static simdjson::error_code apply(
        Input value,
        exec::GenericWriter& writer,
        memory::MemoryPool* pool,
        bool isRoot,
        const folly::F14FastMap<int64_t, JsonRowSchemaInfo>& jsonRowSchemaInfo,
        column_index_t nodeIndex) {
      auto& writerTyped = writer.castTo<Array<Any>>();
      const auto& elementType = writer.type()->childAt(0);
      SIMDJSON_ASSIGN_OR_RAISE(auto type, value.type());
      if (type == simdjson::ondemand::json_type::array) {
        SIMDJSON_ASSIGN_OR_RAISE(auto array, value.get_array());
        bool hasConversionError{false};
        for (const auto& elementResult : array) {
          SIMDJSON_ASSIGN_OR_RAISE(auto element, elementResult);
          if (hasConversionError) {
            const auto result = validateJsonValue(element);
            if (result == kObjectTruncationError) {
              return result;
            }
            if (result == kMalformedAtomError ||
                result == simdjson::INCORRECT_TYPE) {
              return kMalformedAtomError;
            }
            if (result != simdjson::SUCCESS) {
              return result;
            }
            continue;
          }

          SIMDJSON_ASSIGN_OR_RAISE(const auto jsonType, element.type());
          if (jsonType == simdjson::ondemand::json_type::null) {
            if (trimToken(element.raw_json()) != "null") {
              hasConversionError = true;
              continue;
            }
            writerTyped.add_null();
          } else {
            const auto result = VELOX_DYNAMIC_TYPE_DISPATCH(
                ExtractJsonTypeImpl<simdjson::ondemand::value>::apply,
                elementType->kind(),
                element,
                writerTyped.add_item(),
                pool,
                false,
                jsonRowSchemaInfo,
                nodeIndex + 1);
            if (result == kMalformedAtomError) {
              if (jsonType == simdjson::ondemand::json_type::array ||
                  jsonType == simdjson::ondemand::json_type::object) {
                return result;
              }
              hasConversionError = true;
              continue;
            }
            const bool keepPartialRootStruct = isRoot &&
                elementType->kind() == TypeKind::ROW &&
                jsonType == simdjson::ondemand::json_type::object &&
                isConversionError(result);
            if (isConversionError(result) && !keepPartialRootStruct) {
              hasConversionError = true;
              continue;
            }
            if (result != simdjson::SUCCESS && !keepPartialRootStruct) {
              return result;
            }
          }
        }
        if (hasConversionError) {
          return simdjson::INCORRECT_TYPE;
        }
      } else if (
          type == simdjson::ondemand::json_type::object &&
          elementType->kind() == TypeKind::ROW && isRoot) {
        const auto result = VELOX_DYNAMIC_TYPE_DISPATCH(
            ExtractJsonTypeImpl<simdjson::ondemand::value>::apply,
            elementType->kind(),
            value,
            writerTyped.add_item(),
            pool,
            false,
            jsonRowSchemaInfo,
            nodeIndex + 1);
        if (result != simdjson::SUCCESS) {
          return result;
        }
      } else {
        return simdjson::INCORRECT_TYPE;
      }
      return simdjson::SUCCESS;
    }
  };

  template <typename Dummy>
  struct KindDispatcher<TypeKind::MAP, Dummy> {
    static simdjson::error_code apply(
        Input value,
        exec::GenericWriter& writer,
        memory::MemoryPool* pool,
        bool isRoot,
        const folly::F14FastMap<int64_t, JsonRowSchemaInfo>& jsonRowSchemaInfo,
        column_index_t nodeIndex) {
      auto& writerTyped = writer.castTo<Map<Any, Any>>();
      const auto& valueType = writer.type()->childAt(1);
      SIMDJSON_ASSIGN_OR_RAISE(auto object, value.get_object());
      bool hasConversionError{false};
      for (const auto& fieldResult : object) {
        SIMDJSON_ASSIGN_OR_RAISE(auto field, fieldResult);
        const auto escapedKey = field.escaped_key();
        SIMDJSON_ASSIGN_OR_RAISE(auto key, field.unescaped_key(true));
        std::string repairedKey;
        const std::string_view sparkKey =
            replaceUnpairedSurrogateCharacters(escapedKey, key, repairedKey)
            ? std::string_view(repairedKey)
            : key;
        auto fieldValue = field.value();
        if (hasConversionError) {
          const auto result = validateJsonValue(fieldValue);
          if (result == simdjson::SUCCESS) {
            continue;
          }
          if (result == kMalformedAtomError ||
              result == kObjectTruncationError ||
              result == simdjson::INCORRECT_TYPE) {
            return kObjectTruncationError;
          }
          return result;
        }
        SIMDJSON_ASSIGN_OR_RAISE(const auto jsonType, fieldValue.type());
        if (jsonType == simdjson::ondemand::json_type::null) {
          if (trimToken(fieldValue.raw_json()) != "null") {
            return kObjectTruncationError;
          }
          writerTyped.add_null().castTo<Varchar>().append(sparkKey);
        } else {
          auto writers = writerTyped.add_item();
          std::get<0>(writers).castTo<Varchar>().append(sparkKey);
          const auto result = VELOX_DYNAMIC_TYPE_DISPATCH(
              ExtractJsonTypeImpl<simdjson::ondemand::value>::apply,
              valueType->kind(),
              fieldValue,
              std::get<1>(writers),
              pool,
              false,
              jsonRowSchemaInfo,
              nodeIndex + 1);
          if (result == kMalformedAtomError ||
              result == kObjectTruncationError ||
              result == simdjson::STRING_ERROR) {
            return kObjectTruncationError;
          }
          const bool keepPartialRootStruct = isRoot &&
              valueType->kind() == TypeKind::ROW &&
              jsonType == simdjson::ondemand::json_type::object &&
              isConversionError(result);
          if (isConversionError(result) && !keepPartialRootStruct) {
            hasConversionError = true;
            continue;
          }
          if (result != simdjson::SUCCESS && !keepPartialRootStruct) {
            return result;
          }
        }
      }
      return hasConversionError ? simdjson::INCORRECT_TYPE : simdjson::SUCCESS;
    }
  };

  template <typename Dummy>
  struct KindDispatcher<TypeKind::ROW, Dummy> {
    static simdjson::error_code apply(
        Input value,
        exec::GenericWriter& writer,
        memory::MemoryPool* pool,
        bool isRoot,
        const folly::F14FastMap<int64_t, JsonRowSchemaInfo>& jsonRowSchemaInfo,
        column_index_t nodeIndex) {
      const auto& rowType = writer.type()->asRow();
      auto& writerTyped = writer.castTo<DynamicRow>();
      const auto setAllFieldsNull = [&]() {
        for (column_index_t i = 0; i < rowType.size(); ++i) {
          writerTyped.set_null_at(i);
        }
      };

      const auto typeResult = value.type();
      if (typeResult.error() != ::simdjson::SUCCESS) {
        if (!isRoot) {
          return typeResult.error();
        }
        setAllFieldsNull();
        return simdjson::SUCCESS;
      }

      const auto jsonType = typeResult.value_unsafe();
      if (jsonType == simdjson::ondemand::json_type::object) {
        SIMDJSON_ASSIGN_OR_RAISE(auto object, value.get_object());
        const auto& schemaInfo = jsonRowSchemaInfo.at(nodeIndex);
        const auto& isFieldMissing = schemaInfo.isFieldMissing;
        const auto& fieldIndices = schemaInfo.fieldIndices;
        const auto& nodeIndices = schemaInfo.nodeIndices;
        std::fill(isFieldMissing->begin(), isFieldMissing->end(), true);
        bool hasConversionError{false};
        for (const auto& fieldResult : object) {
          SIMDJSON_ASSIGN_OR_RAISE(auto field, fieldResult);
          const auto escapedKey = field.escaped_key();
          SIMDJSON_ASSIGN_OR_RAISE(auto key, field.unescaped_key(true));
          const auto fieldIterator = findUnpairedSurrogates(escapedKey).empty()
              ? fieldIndices.find(key)
              : fieldIndices.end();
          if (fieldIterator == fieldIndices.end()) {
            const auto result = validateJsonValue(field.value());
            if ((result == kMalformedAtomError ||
                 result == kObjectTruncationError ||
                 result == simdjson::INCORRECT_TYPE) &&
                !isRoot) {
              return kObjectTruncationError;
            }
            SIMDJSON_TRY(result);
            continue;
          }

          const auto index = fieldIterator->second;
          const bool isDuplicate = !isFieldMissing->at(index);
          isFieldMissing->at(index) = false;
          auto fieldValue = field.value();
          SIMDJSON_ASSIGN_OR_RAISE(const auto fieldType, fieldValue.type());
          const auto truncateCurrentField = [&]() {
            if (!isDuplicate) {
              writerTyped.set_null_at(index);
            }
            for (column_index_t i = 0; i < rowType.size(); ++i) {
              if (isFieldMissing->at(i)) {
                writerTyped.set_null_at(i);
              }
            }
            return isRoot ? simdjson::SUCCESS : kObjectTruncationError;
          };
          if (fieldType == simdjson::ondemand::json_type::null) {
            if (trimToken(fieldValue.raw_json()) != "null") {
              if (isRoot) {
                return kMalformedAtomError;
              }
              return truncateCurrentField();
            }
            writerTyped.set_null_at(index);
            continue;
          }

          VectorPtr scratchVector;
          exec::VectorWriter<Any> scratchWriter;
          exec::GenericWriter* fieldWriter;
          if (isDuplicate) {
            scratchVector = BaseVector::create(rowType.childAt(index), 1, pool);
            scratchWriter.init(*scratchVector);
            scratchWriter.setOffset(0);
            fieldWriter = &scratchWriter.current();
          } else {
            fieldWriter = &writerTyped.get_writer_at(index);
          }
          const auto result = VELOX_DYNAMIC_TYPE_DISPATCH(
              ExtractJsonTypeImpl<simdjson::ondemand::value>::apply,
              rowType.childAt(index)->kind(),
              fieldValue,
              *fieldWriter,
              pool,
              false,
              jsonRowSchemaInfo,
              nodeIndices.at(key));
          if (isDuplicate) {
            if (result == simdjson::SUCCESS) {
              scratchWriter.commit(true);
              scratchWriter.finish();
              DecodedVector decoded(*scratchVector);
              exec::VectorReader<Any> scratchReader(&decoded);
              writerTyped.set_null_at(index);
              copyParsedJsonValue(
                  writerTyped.get_writer_at(index), scratchReader[0]);
            } else {
              scratchWriter.commit(false);
              scratchWriter.finish();
            }
          }
          if (result == kMalformedAtomError ||
              result == simdjson::STRING_ERROR) {
            if (isRoot) {
              return result;
            }
            return truncateCurrentField();
          }
          if (result == kObjectTruncationError) {
            return truncateCurrentField();
          }
          if (result != simdjson::SUCCESS) {
            if (!isConversionError(result)) {
              return result;
            }
            if (!isDuplicate) {
              writerTyped.set_null_at(index);
            }
            hasConversionError = true;
          }
        }

        for (column_index_t i = 0; i < rowType.size(); ++i) {
          if (isFieldMissing->at(i)) {
            writerTyped.set_null_at(i);
          }
        }
        if (!isRoot && hasConversionError) {
          return simdjson::INCORRECT_TYPE;
        }
      } else if (isRoot) {
        setAllFieldsNull();
      } else {
        setAllFieldsNull();
        return simdjson::INCORRECT_TYPE;
      }
      return simdjson::SUCCESS;
    }
  };

  static simdjson::error_code castJsonToDate(
      Input value,
      exec::GenericWriter& writer) {
    SIMDJSON_ASSIGN_OR_RAISE(auto type, value.type());
    if (type != simdjson::ondemand::json_type::string) {
      return simdjson::INCORRECT_TYPE;
    }
    std::string_view input;
    SIMDJSON_ASSIGN_OR_RAISE(input, value.get_string());
    std::string normalizedDigits;
    std::string_view dayInput = input;
    const bool hasNormalizedDigits =
        normalizeUnicodeDecimalDigits(input, normalizedDigits);
    if (hasNormalizedDigits) {
      dayInput = normalizedDigits;
    }

    const auto parseDate = [](std::string_view date) {
      auto result =
          util::fromDateString(StringView(date), util::ParseMode::kSparkCast);
      if (!result.hasError() ||
          stringImpl::stringPosition</*isAscii=*/true>(date, kGMT, 1) <= 0) {
        return result;
      }

      std::vector<char> cleanedDate(date.size());
      const auto size = stringCore::replace</*ignoreEmptyReplaced=*/true>(
          cleanedDate.data(), date, kGMT, std::string_view(), false);
      return util::fromDateString(
          StringView(cleanedDate.data(), static_cast<int32_t>(size)),
          util::ParseMode::kSparkCast);
    };

    if (isSignedDecimalInteger(dayInput)) {
      auto digits = dayInput;
      if (digits.front() == '-' || digits.front() == '+') {
        digits.remove_prefix(1);
      }
      if (!hasNormalizedDigits && digits.size() >= 4) {
        const auto dateResult = parseDate(input);
        if (!dateResult.hasError()) {
          writer.castTo<int32_t>() = dateResult.value();
          return simdjson::SUCCESS;
        }
      }

      const auto dayResult = folly::tryTo<int32_t>(dayInput);
      if (dayResult.hasError()) {
        return simdjson::INCORRECT_TYPE;
      }
      writer.castTo<int32_t>() = rebaseJulianToGregorianDays(dayResult.value());
      return simdjson::SUCCESS;
    }

    auto dateResult = parseDate(input);
    if (dateResult.hasError()) {
      auto trimmedInput = input;
      while (!trimmedInput.empty() && isJsonWhitespace(trimmedInput.back())) {
        trimmedInput.remove_suffix(1);
      }
      if (trimmedInput.size() != input.size()) {
        dateResult = parseDate(trimmedInput);
      }
    }
    if (dateResult.hasError()) {
      return simdjson::INCORRECT_TYPE;
    }
    writer.castTo<int32_t>() = dateResult.value();
    return simdjson::SUCCESS;
  }

  template <typename T>
  static simdjson::error_code castJsonToInt(
      Input value,
      exec::GenericWriter& writer) {
    SIMDJSON_ASSIGN_OR_RAISE(auto type, value.type());
    switch (type) {
      case simdjson::ondemand::json_type::number: {
        const auto numberResult = value.get_number();
        if (numberResult.error() != simdjson::SUCCESS) {
          const auto token = trimToken(value.raw_json());
          if (isSparkSpecialNumber(token)) {
            return simdjson::INCORRECT_TYPE;
          }
          return isValidJsonNumber(token) ? simdjson::NUMBER_OUT_OF_RANGE
                                          : numberResult.error();
        }
        const auto num = numberResult.value_unsafe();
        switch (num.get_number_type()) {
          case simdjson::ondemand::number_type::signed_integer:
            return convertIfInRange<T>(num.get_int64(), writer);
          case simdjson::ondemand::number_type::unsigned_integer:
            return simdjson::NUMBER_OUT_OF_RANGE;
          default:
            return simdjson::INCORRECT_TYPE;
        }
      }
      default:
        return simdjson::INCORRECT_TYPE;
    }
    return simdjson::SUCCESS;
  }

  template <typename T>
  static simdjson::error_code parseSpecialFloatingStrings(
      const std::string_view& s,
      exec::GenericWriter& writer) {
    constexpr T kNaN = std::numeric_limits<T>::quiet_NaN();
    constexpr T kInf = std::numeric_limits<T>::infinity();
    // Strip surrounding quotes if any.
    std::string_view stripped = s;
    if (s.size() >= 2 && s.front() == '"' && s.back() == '"') {
      stripped = s.substr(1, s.size() - 2);
    }
    if (stripped == "NaN") {
      writer.castTo<T>() = kNaN;
    } else if (
        stripped == "+INF" || stripped == "+Infinity" ||
        stripped == "Infinity") {
      writer.castTo<T>() = kInf;
    } else if (stripped == "-INF" || stripped == "-Infinity") {
      writer.castTo<T>() = -kInf;
    } else {
      return simdjson::INCORRECT_TYPE;
    }
    return simdjson::SUCCESS;
  }

  // Casts a JSON value to a float point, handling both numeric special cases
  // for NaN and Infinity.
  template <typename T>
  static simdjson::error_code castJsonToFloatingPoint(
      Input value,
      exec::GenericWriter& writer) {
    SIMDJSON_ASSIGN_OR_RAISE(const auto type, value.type());
    std::string_view token;
    if (type == simdjson::ondemand::json_type::string) {
      SIMDJSON_ASSIGN_OR_RAISE(token, value.get_string());
    } else if (
        type == simdjson::ondemand::json_type::number ||
        type == simdjson::ondemand::json_type::unknown) {
      token = trimToken(value.raw_json());
    } else {
      return simdjson::INCORRECT_TYPE;
    }

    // Spark support such special floating point with/without quotes:
    // NaN, +INF, +Infinity, Infinity, -INF, -Infinity.
    if (parseSpecialFloatingStrings<T>(token, writer) == simdjson::SUCCESS) {
      return simdjson::SUCCESS;
    }
    if (type == simdjson::ondemand::json_type::string) {
      return simdjson::INCORRECT_TYPE;
    }
    if (type != simdjson::ondemand::json_type::number ||
        !isValidJsonNumber(token)) {
      return simdjson::NUMBER_ERROR;
    }

    if constexpr (std::is_same_v<T, float>) {
      const auto castResult = util::Converter<TypeKind::REAL>::tryCast(token);
      if (!castResult.hasError()) {
        writer.castTo<T>() = castResult.value();
        return simdjson::SUCCESS;
      }
      return simdjson::NUMBER_ERROR;
    }

    const auto result = value.get_double();
    if (result.error() == simdjson::SUCCESS) {
      writer.castTo<T>() = result.value_unsafe();
      return simdjson::SUCCESS;
    }

    // simdjson parses floating point numbers in the range
    // [std::numeric_limits<double>::lowest(),
    // std::numeric_limits<double>::max()], i.e., from approximately
    // -1.7976e308 to 1.7975e308. Values outside this range
    // (<= -1e308 or >= 1e308) are rejected and simdjson returns
    // NUMBER_ERROR. However, our expected behavior is to convert such
    // extreme values to -INF or +INF, so we add extra logic here to
    // handle NUMBER_ERROR and perform the conversion.
    auto castResult = util::Converter<TypeKind::DOUBLE>::tryCast(token);
    if (!castResult.hasError()) {
      writer.castTo<T>() = castResult.value();
      return simdjson::SUCCESS;
    }
    return simdjson::NUMBER_ERROR;
  }

  template <typename To, typename From>
  static simdjson::error_code convertIfInRange(
      From x,
      exec::GenericWriter& writer) {
    static_assert(std::is_signed_v<From> && std::is_signed_v<To>);
    // Match Jackson's getByteValue, which accepts -128 through 255 and wraps
    // 128 through 255.
    if constexpr (std::is_same_v<To, int8_t>) {
      constexpr From kMin = std::numeric_limits<int8_t>::lowest();
      constexpr From kMax = std::numeric_limits<uint8_t>::max();
      if (!(kMin <= x && x <= kMax)) {
        return simdjson::NUMBER_OUT_OF_RANGE;
      }
      const auto wrapped = x > std::numeric_limits<int8_t>::max()
          ? x - (std::numeric_limits<uint8_t>::max() + 1)
          : x;
      writer.castTo<int8_t>() = static_cast<int8_t>(wrapped);
      return simdjson::SUCCESS;
    }
    if constexpr (sizeof(To) < sizeof(From)) {
      constexpr From kMin = std::numeric_limits<To>::lowest();
      constexpr From kMax = std::numeric_limits<To>::max();
      if (!(kMin <= x && x <= kMax)) {
        return simdjson::NUMBER_OUT_OF_RANGE;
      }
    }
    writer.castTo<To>() = x;
    return simdjson::SUCCESS;
  }

  template <typename T>
  static simdjson::error_code castJsonToDecimal(
      Input value,
      exec::GenericWriter& writer) {
    SIMDJSON_ASSIGN_OR_RAISE(auto type, value.type());
    std::string_view s;
    std::string normalizedDigits;
    std::string normalizedString;
    switch (type) {
      case simdjson::ondemand::json_type::string: {
        SIMDJSON_ASSIGN_OR_RAISE(s, value.get_string());
        if (normalizeUnicodeDecimalDigits(s, normalizedDigits)) {
          s = normalizedDigits;
        }
        if (s.find(',') != std::string_view::npos) {
          normalizedString = s;
          normalizedString.erase(
              std::remove(
                  normalizedString.begin(), normalizedString.end(), ','),
              normalizedString.end());
          s = normalizedString;
        }
        break;
      }
      case simdjson::ondemand::json_type::number: {
        s = trimToken(value.raw_json());
        if (isSparkSpecialNumber(s)) {
          return simdjson::INCORRECT_TYPE;
        }
        if (!isValidJsonNumber(s)) {
          return simdjson::NUMBER_ERROR;
        }
        break;
      }
      default:
        return simdjson::INCORRECT_TYPE;
    }
    const auto toPrecisionScale = getDecimalPrecisionScale(*writer.type());
    T decimalValue;
    const auto status = velox::DecimalUtil::castFromString<T>(
        StringView(s),
        toPrecisionScale.first,
        toPrecisionScale.second,
        decimalValue);
    if (!status.ok()) {
      return simdjson::INCORRECT_TYPE;
    }
    writer.castTo<T>() = decimalValue;
    return simdjson::SUCCESS;
  }

  constexpr static std::string_view kGMT{"GMT"};
};

/// @brief Parses a JSON string into the specified data type. Supports ROW,
/// ARRAY, and MAP as root types. Key Behavior:
/// - Failure Handling: Returns `NULL` for invalid arrays and maps. A nonempty
/// invalid ROW input returns a non-null row whose fields are all null.
/// - Boolean: Only `true` and `false` are valid; others return `NULL`.
/// - Integral Types: Accepts only integers; floats or strings return `NULL`.
/// - Float/Double: All numbers are valid; strings like `"NaN"`, `"+INF"`,
/// `"+Infinity"`, `"Infinity"`, `"-INF"`, `"-Infinity"` are accepted, others
/// return `NULL`.
/// - Array: Accepts JSON objects only if the array is the root type with ROW
/// child type.
/// - Map: Keys must be `VARCHAR` type.
/// - Row: Partial parsing is supported, but JSON arrays cannot be parsed into a
/// ROW type.
template <TypeKind kind>
class FromJsonFunction final : public exec::VectorFunction {
 public:
  explicit FromJsonFunction(const TypePtr& type) {
    column_index_t index = 0;
    constructRowSchemaInfoMap(type, index);
  }

  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args, // Not using const ref so we can reuse args
      const TypePtr& outputType,
      exec::EvalCtx& context,
      VectorPtr& result) const final {
    VELOX_USER_CHECK(
        args[0]->isConstantEncoding() || args[0]->isFlatEncoding(),
        "Single-arg deterministic functions receive their only argument as flat or constant vector.");
    context.ensureWritable(rows, outputType, result);
    result->clearNulls(rows);
    if (args[0]->isConstantEncoding()) {
      parseJsonConstant(args[0], context, rows, *result);
    } else {
      parseJsonFlat(args[0], context, rows, *result);
    }
  }

 private:
  // Applies Spark's UTF-8, single-quote, UTF-16, and document-framing
  // defaults before handing the first JSON value to simdjson.
  simdjson::simdjson_result<simdjson::ondemand::document> parseJson(
      const simdjson::padded_string_view& paddedInput,
      std::string_view input) const {
    std::string_view sparkInput = input;
    bool transformed{false};
    if (!stringCore::isAscii(input.data(), input.size()) &&
        !simdjson::validate_utf8(input.data(), input.size())) {
      repairedInput_.clear();
      replaceInvalidUTF8Characters(
          repairedInput_, input.data(), static_cast<int32_t>(input.size()));
      sparkInput = repairedInput_;
      transformed = true;
    }

    constexpr std::string_view kUtf8ByteOrderMark{"\xef\xbb\xbf", 3};
    if (sparkInput.starts_with(kUtf8ByteOrderMark)) {
      return simdjson::UTF8_ERROR;
    }

    const auto valueSize = firstJsonValueSize(sparkInput);
    if (valueSize != sparkInput.size()) {
      sparkInput = sparkInput.substr(0, valueSize);
      transformed = true;
    }

    if (normalizeSingleQuotedJson(sparkInput, normalizedInput_)) {
      sparkInput = normalizedInput_;
      transformed = true;
    }
    if (!transformed) {
      return simdjsonParse(paddedInput);
    }

    preparedInput_.assign(sparkInput);
    const auto preparedSize = preparedInput_.size();
    preparedInput_.resize(preparedSize + simdjson::SIMDJSON_PADDING);
    simdjson::padded_string_view preparedInput(
        preparedInput_.data(), preparedSize, preparedInput_.size());
    return simdjsonParse(preparedInput);
  }

  static bool isEmptyJsonInput(std::string_view input) {
    return std::all_of(input.begin(), input.end(), isJsonWhitespace);
  }

  static void commitAllNullRow(exec::VectorWriter<Any>& writer) {
    writer.commitNull();
    auto& genericWriter = writer.current();
    auto& rowWriter = genericWriter.castTo<DynamicRow>();
    const auto& rowType = genericWriter.type()->asRow();
    for (column_index_t i = 0; i < rowType.size(); ++i) {
      rowWriter.set_null_at(i);
    }
    writer.commit(true);
  }

  static void commitFailedInput(
      std::string_view input,
      exec::VectorWriter<Any>& writer) {
    if constexpr (kind == TypeKind::ROW) {
      if (!isEmptyJsonInput(input)) {
        commitAllNullRow(writer);
        return;
      }
    }
    writer.commitNull();
  }

  void parseJsonConstant(
      VectorPtr& input,
      exec::EvalCtx& context,
      const SelectivityVector& rows,
      BaseVector& result) const {
    // Result is guaranteed to be a flat writable vector.
    auto* flatResult = result.as<typename KindToFlatVector<kind>::type>();
    exec::VectorWriter<Any> writer;
    writer.init(*flatResult);
    const auto constInput = input->asUnchecked<ConstantVector<StringView>>();
    if (constInput->isNullAt(0)) {
      context.applyToSelectedNoThrow(rows, [&](vector_size_t row) {
        writer.setOffset(row);
        writer.commitNull();
      });
    } else {
      const auto constant = constInput->valueAt(0);
      paddedInput_.resize(constant.size() + simdjson::SIMDJSON_PADDING);
      memcpy(paddedInput_.data(), constant.data(), constant.size());
      simdjson::padded_string_view paddedInput(
          paddedInput_.data(), constant.size(), paddedInput_.size());

      simdjson::ondemand::document jsonDoc;
      auto error =
          parseJson(
              paddedInput, std::string_view(constant.data(), constant.size()))
              .get(jsonDoc);

      const auto constantInput =
          std::string_view(constant.data(), constant.size());
      context.applyToSelectedNoThrow(rows, [&](vector_size_t row) {
        writer.setOffset(row);
        if (error != simdjson::SUCCESS) {
          commitFailedInput(constantInput, writer);
          return;
        }

        jsonDoc.rewind();
        if (extractJsonToWriter(jsonDoc, writer, rowSchemaInfoMap_) !=
            simdjson::SUCCESS) {
          commitFailedInput(constantInput, writer);
        }
      });
    }

    writer.finish();
  }

  void parseJsonFlat(
      VectorPtr& input,
      exec::EvalCtx& context,
      const SelectivityVector& rows,
      BaseVector& result) const {
    auto* flatResult = result.as<typename KindToFlatVector<kind>::type>();
    exec::VectorWriter<Any> writer;
    writer.init(*flatResult);
    auto* inputVector = input->asUnchecked<FlatVector<StringView>>();
    size_t maxSize = 0;
    rows.applyToSelected([&](vector_size_t row) {
      if (inputVector->isNullAt(row)) {
        return;
      }
      const auto& input = inputVector->valueAt(row);
      maxSize = std::max(maxSize, input.size());
    });
    paddedInput_.resize(maxSize + simdjson::SIMDJSON_PADDING);
    context.applyToSelectedNoThrow(rows, [&](vector_size_t row) {
      writer.setOffset(row);
      if (inputVector->isNullAt(row)) {
        writer.commitNull();
        return;
      }
      const auto& input = inputVector->valueAt(row);
      memcpy(paddedInput_.data(), input.data(), input.size());
      simdjson::padded_string_view paddedInput(
          paddedInput_.data(), input.size(), paddedInput_.size());
      simdjson::ondemand::document doc;
      const auto inputView = std::string_view(input.data(), input.size());
      auto error = parseJson(paddedInput, inputView).get(doc);
      if (error != simdjson::SUCCESS) {
        commitFailedInput(inputView, writer);
        return;
      }
      if (extractJsonToWriter(doc, writer, rowSchemaInfoMap_) !=
          simdjson::SUCCESS) {
        commitFailedInput(inputView, writer);
      }
    });
    writer.finish();
  }

  // Constructs a map from the schema tree node index to JsonRowSchemaInfo.
  column_index_t constructRowSchemaInfoMap(
      const TypePtr& type,
      column_index_t& index) {
    auto nodeKey = index++;
    switch (type->kind()) {
      case TypeKind::ARRAY: {
        constructRowSchemaInfoMap(type->childAt(0), index);
        break;
      }
      case TypeKind::MAP: {
        constructRowSchemaInfoMap(type->childAt(1), index);
        break;
      }
      case TypeKind::ROW: {
        const auto& rowType = asRowType(type);
        auto isFieldMissing = std::make_shared<std::vector<bool>>();
        isFieldMissing->resize(rowType->size(), true);
        folly::F14FastMap<std::string, column_index_t> fieldIndices;
        folly::F14FastMap<std::string, column_index_t> nodeIndices;
        const auto size = rowType->size();
        for (auto i = 0; i < size; ++i) {
          const auto& key = rowType->nameOf(i);
          fieldIndices[key] = i;
          nodeIndices[key] = constructRowSchemaInfoMap(type->childAt(i), index);
        }
        rowSchemaInfoMap_.insert_or_assign(
            nodeKey,
            JsonRowSchemaInfo(
                nodeKey,
                std::move(isFieldMissing),
                std::move(fieldIndices),
                std::move(nodeIndices)));
        break;
      }
      default:
        break;
    }
    return nodeKey;
  }

  // Extracts data from json doc and writes it to writer.
  // @param rowSchemaInfoMap A map from schema tree node index to
  // JsonRowSchemaInfo.
  static simdjson::error_code extractJsonToWriter(
      simdjson::ondemand::document& doc,
      exec::VectorWriter<Any>& writer,
      const folly::F14FastMap<int64_t, JsonRowSchemaInfo>& rowSchemaInfoMap) {
    SIMDJSON_ASSIGN_OR_RAISE(const auto isNull, doc.is_null());
    if (isNull) {
      if constexpr (kind == TypeKind::ROW) {
        commitAllNullRow(writer);
      } else {
        writer.commitNull();
      }
    } else {
      SIMDJSON_TRY(
          ExtractJsonTypeImpl<simdjson::ondemand::document&>::apply<kind>(
              doc,
              writer.current(),
              writer.vector().pool(),
              true,
              rowSchemaInfoMap,
              0));
      writer.commit(true);
    }
    return simdjson::SUCCESS;
  }

  // The buffer with extra bytes for parser::parse(),
  mutable std::string paddedInput_;
  // Reusable buffer for normalizing Spark's default single-quoted JSON.
  mutable std::string normalizedInput_;
  // Reusable buffer for applying Spark's invalid UTF-8 replacement behavior.
  mutable std::string repairedInput_;
  // Reusable padded buffer for inputs that require Spark compatibility changes.
  mutable std::string preparedInput_;
  // Map from row schema tree node index to schema information for JSON rows.
  folly::F14FastMap<int64_t, JsonRowSchemaInfo> rowSchemaInfoMap_;
};

/// Determines whether a given type is supported.
/// @param isRootType A flag indicating whether the type is the root type in
/// the evaluation context. Only ROW, ARRAY, and MAP are allowed as root types;
/// this flag helps differentiate such cases.
bool isSupportedType(const TypePtr& type, bool isRootType) {
  switch (type->kind()) {
    case TypeKind::ARRAY: {
      return isSupportedType(type->childAt(0), false);
    }
    case TypeKind::ROW: {
      for (const auto& child : asRowType(type)->children()) {
        if (!isSupportedType(child, false)) {
          return false;
        }
      }
      return true;
    }
    case TypeKind::MAP: {
      return (
          type->childAt(0)->kind() == TypeKind::VARCHAR &&
          isSupportedType(type->childAt(1), false));
    }
    case TypeKind::HUGEINT:
    case TypeKind::BIGINT:
    case TypeKind::INTEGER:
    case TypeKind::BOOLEAN:
    case TypeKind::SMALLINT:
    case TypeKind::TINYINT:
    case TypeKind::DOUBLE:
    case TypeKind::REAL:
    case TypeKind::VARCHAR: {
      return !isRootType;
    }
    default:
      return false;
  }
}

} // namespace

TypePtr FromJsonCallToSpecialForm::resolveType(
    const std::vector<TypePtr>& /*argTypes*/) {
  VELOX_FAIL("from_json function does not support type resolution.");
}

exec::ExprPtr FromJsonCallToSpecialForm::constructSpecialForm(
    const TypePtr& type,
    std::vector<exec::ExprPtr>&& args,
    bool trackCpuUsage,
    const core::QueryConfig& /*config*/) {
  VELOX_USER_CHECK_EQ(args.size(), 1, "from_json expects one argument.");
  VELOX_USER_CHECK_EQ(
      args[0]->type()->kind(),
      TypeKind::VARCHAR,
      "The first argument of from_json should be of varchar type.");
  VELOX_USER_CHECK(
      isSupportedType(type, true), "Unsupported type {}.", type->toString());

  std::shared_ptr<exec::VectorFunction> func;
  if (type->kind() == TypeKind::ARRAY) {
    func = std::make_shared<FromJsonFunction<TypeKind::ARRAY>>(type);
  } else if (type->kind() == TypeKind::MAP) {
    func = std::make_shared<FromJsonFunction<TypeKind::MAP>>(type);
  } else {
    func = std::make_shared<FromJsonFunction<TypeKind::ROW>>(type);
  }

  return std::make_shared<exec::Expr>(
      type,
      std::move(args),
      func,
      exec::VectorFunctionMetadata{},
      kFromJson,
      trackCpuUsage);
}
} // namespace facebook::velox::functions::sparksql
