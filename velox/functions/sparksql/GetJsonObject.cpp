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

#include "velox/functions/sparksql/GetJsonObject.h"

#include <cctype>
#include <cstddef>
#include <cstring>
#include <string>
#include <string_view>

#include "folly/Unicode.h"

#include "velox/functions/lib/Utf8Utils.h"
#include "velox/functions/prestosql/json/JsonStringUtil.h"

namespace facebook::velox::functions::sparksql::detail {

namespace {

// Counts the supplementary-plane characters (4-byte UTF-8 sequences) in `raw`.
size_t countSupplementaryCharacters(std::string_view raw) {
  size_t count = 0;
  const auto* p = reinterpret_cast<const unsigned char*>(raw.data());
  const auto* const end = p + raw.size();
  while (p < end) {
    const int length = validateAndGetNextUtf8Length(p, end);
    if (length == 4) {
      ++count;
    }
    p += length > 0 ? length : 1;
  }
  return count;
}

// Writes `raw` to `out`, applying the escaping described on
// appendWithSupplementaryEscapes. `out` must have room for the result (see
// countSupplementaryCharacters for sizing); returns the past-the-end write
// position.
char* escapeSupplementaryCharacters(std::string_view raw, char* out) {
  const auto* const base = reinterpret_cast<const unsigned char*>(raw.data());
  const auto* const end = base + raw.size();
  const auto* p = base;
  const auto* runStart = base;
  while (p < end) {
    const int length = validateAndGetNextUtf8Length(p, end);
    if (length == 4) {
      // Flush the verbatim run accumulated before this sequence in one copy.
      if (p > runStart) {
        const auto runSize = static_cast<size_t>(p - runStart);
        std::memcpy(out, runStart, runSize);
        out += runSize;
      }
      // folly::utf8ToCodePoint advances `p` past the 4-byte sequence.
      encodeUtf16Hex(folly::utf8ToCodePoint(p, end, true), out);
      runStart = p;
    } else {
      p += length > 0 ? length : 1;
    }
  }
  // Flush the trailing verbatim run.
  if (p > runStart) {
    const auto runSize = static_cast<size_t>(p - runStart);
    std::memcpy(out, runStart, runSize);
    out += runSize;
  }
  return out;
}

} // namespace

bool isValidJsonValue(simdjson::ondemand::value value) {
  simdjson::ondemand::json_type type;
  if (value.type().get(type)) {
    return false;
  }
  switch (type) {
    case simdjson::ondemand::json_type::object: {
      simdjson::ondemand::object object;
      if (value.get_object().get(object)) {
        return false;
      }
      for (auto field : object) {
        simdjson::ondemand::value child;
        if (field.value().get(child) || !isValidJsonValue(child)) {
          return false;
        }
      }
      return true;
    }
    case simdjson::ondemand::json_type::array: {
      simdjson::ondemand::array array;
      if (value.get_array().get(array)) {
        return false;
      }
      for (auto element : array) {
        simdjson::ondemand::value child;
        if (element.get(child) || !isValidJsonValue(child)) {
          return false;
        }
      }
      return true;
    }
    case simdjson::ondemand::json_type::string:
      return true;
    case simdjson::ondemand::json_type::number:
      return isValidJsonNumber(trimToken(value.raw_json_token()));
    case simdjson::ondemand::json_type::boolean: {
      const auto token = trimToken(value.raw_json_token());
      return token == "true" || token == "false";
    }
    case simdjson::ondemand::json_type::null:
      return trimToken(value.raw_json_token()) == "null";
    default:
      return false;
  }
}

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

bool escapeControlCharacters(std::string_view input, std::string& output) {
  output.clear();
  std::string brackets;
  char quote{0};
  bool rootClosed{false};
  for (size_t i = 0; i < input.size(); ++i) {
    const char character = input[i];
    if (quote == 0) {
      if (character == '"' || character == '\'') {
        quote = character;
      } else if (!rootClosed && (character == '{' || character == '[')) {
        brackets.push_back(character == '{' ? '}' : ']');
      } else if (!brackets.empty() && (character == '}' || character == ']')) {
        if (brackets.back() != character) {
          return false;
        }
        brackets.pop_back();
        rootClosed = brackets.empty();
      }
    } else if (character == '\\' && i + 1 < input.size()) {
      if (!output.empty()) {
        output.append(input.data() + i, 2);
      }
      ++i;
      continue;
    } else if (character == quote) {
      quote = 0;
    } else if (static_cast<unsigned char>(character) < 0x20) {
      if (output.empty()) {
        output.reserve(input.size() + 5);
        output.append(input.data(), i);
      }
      char escape[6];
      char* end{escape};
      encodeAscii(static_cast<int8_t>(character), end);
      output.append(escape, end);
      continue;
    }
    if (!output.empty()) {
      output.push_back(character);
    }
  }
  return true;
}

std::string_view rootScalar(std::string_view json) {
  const size_t begin = json.find_first_not_of(" \t\n\r");
  if (begin == std::string_view::npos || json[begin] == '{' ||
      json[begin] == '[') {
    return {};
  }
  size_t end{begin + 1};
  if (json[begin] == '"') {
    while (end < json.size() && json[end] != '"') {
      end += json[end] == '\\' ? 2 : 1;
    }
    return end < json.size() ? json.substr(begin, end + 1 - begin)
                             : std::string_view{};
  }
  while (end < json.size() &&
         (std::isalnum(static_cast<unsigned char>(json[end])) ||
          json[end] == '.' || json[end] == '+' || json[end] == '-')) {
    ++end;
  }
  const bool isNumber = std::isdigit(static_cast<unsigned char>(json[begin])) ||
      json[begin] == '-';
  if (isNumber && end < json.size() &&
      std::string_view(" \t\n\r").find(json[end]) == std::string_view::npos) {
    return {};
  }
  return json.substr(begin, end - begin);
}

bool compactJson(std::string_view raw, std::string& out) {
  out.clear();
  bool inString{false};
  for (size_t i = 0; i < raw.size(); ++i) {
    const char c = raw[i];
    if (!inString) {
      if (c != ' ' && c != '\t' && c != '\n' && c != '\r') {
        out.push_back(c);
        inString = c == '"';
      }
      continue;
    }
    if (c == '"') {
      out.push_back(c);
      inString = false;
    } else if (c != '\\') {
      out.push_back(c);
    } else if (i + 1 == raw.size()) {
      return false;
    } else if (raw[i + 1] == '/') {
      out.push_back('/');
      ++i;
    } else if (raw[i + 1] != 'u') {
      out.append(raw.data() + i, 2);
      ++i;
    } else {
      uint16_t codeUnit;
      if (!parseUnicodeEscape(raw, i, codeUnit)) {
        return false;
      }
      char escape[6];
      char* end = escape;
      if (codeUnit < 0x80) {
        encodeAscii(static_cast<int8_t>(codeUnit), end);
        out.append(escape, end);
      } else if (codeUnit >= 0xD800 && codeUnit <= 0xDFFF) {
        writeUtf16Escape(codeUnit, end);
        out.append(escape, end);
      } else {
        out.append(folly::codePointToUtf8(codeUnit));
      }
      i += 5;
    }
  }
  return true;
}

void appendWithSupplementaryEscapes(
    std::string_view raw,
    exec::StringWriter& out) {
  const size_t supplementaryCount = countSupplementaryCharacters(raw);
  if (supplementaryCount == 0) {
    // Fast path: nothing to escape, the raw slice already matches Spark.
    out.append(raw);
    return;
  }
  // Each 4-byte sequence (1 char) becomes a 12-byte surrogate-pair escape
  // ('\uXXXX\uXXXX'), i.e. 8 bytes larger than the raw 4. Escape directly into
  // `out`'s buffer -- reserve the exact worst case, write in place, then resize
  // down to what was actually written, avoiding an intermediate string + copy.
  const size_t oldSize = out.size();
  const size_t maxSize = oldSize + raw.size() + 8 * supplementaryCount;
  out.reserve(maxSize);
  char* const begin = out.data() + oldSize;
  const char* const written = escapeSupplementaryCharacters(raw, begin);
  out.resize(oldSize + static_cast<size_t>(written - begin));
}

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

} // namespace facebook::velox::functions::sparksql::detail
