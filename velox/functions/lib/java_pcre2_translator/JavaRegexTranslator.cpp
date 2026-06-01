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
//
// Originally authored by Oleksii PELYKH for pcre4j; ported from
// org.pcre4j.regex.translate.JavaRegexTranslator (Java) under
// Apache-2.0 by the same author for inclusion in Velox.
//
#include "velox/functions/lib/java_pcre2_translator/JavaRegexTranslator.h"

#include "velox/functions/lib/java_pcre2_translator/ClassBodyParser.h"
#include "velox/functions/lib/java_pcre2_translator/ClassRenderer.h"
#include "velox/functions/lib/java_pcre2_translator/PropertyMap.h"

#include <unicode/uchar.h>
#include <unicode/utypes.h>

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace facebook::velox::functions::java_pcre2_translator {
namespace {

bool isValidQuantifierBody(std::string_view body) {
  if (body.empty()) {
    return false;
  }
  std::size_t k = 0;
  while (k < body.size() && body[k] >= '0' && body[k] <= '9') {
    ++k;
  }
  if (k == 0) {
    return false;
  }
  if (k == body.size()) {
    return true;
  }
  if (body[k] != ',') {
    return false;
  }
  ++k;
  while (k < body.size() && body[k] >= '0' && body[k] <= '9') {
    ++k;
  }
  return k == body.size();
}

bool isHexDigit(char ch) {
  return (ch >= '0' && ch <= '9') || (ch >= 'a' && ch <= 'f') ||
      (ch >= 'A' && ch <= 'F');
}

std::uint32_t hexValue(char ch) {
  if (ch >= '0' && ch <= '9') {
    return ch - '0';
  }
  if (ch >= 'a' && ch <= 'f') {
    return ch - 'a' + 10;
  }
  return ch - 'A' + 10;
}

std::uint32_t parseFourHex(std::string_view s, std::size_t from) {
  std::uint32_t value = 0;
  for (std::size_t i = 0; i < 4; ++i) {
    value = (value << 4) | hexValue(s[from + i]);
  }
  return value;
}

bool isOctalDigit(char ch) {
  return ch >= '0' && ch <= '7';
}

std::string toLowerHex(std::uint32_t cp) {
  constexpr char kHex[] = "0123456789abcdef";
  if (cp == 0) {
    return "0";
  }
  std::string out;
  while (cp != 0) {
    out.push_back(kHex[cp & 0xF]);
    cp >>= 4;
  }
  std::reverse(out.begin(), out.end());
  return out;
}

bool hasOddTrailingBackslashes(const std::string& sb) {
  std::size_t count = 0;
  for (std::size_t j = sb.size(); j > 0 && sb[j - 1] == '\\'; --j) {
    ++count;
  }
  return (count & 1U) == 1U;
}

std::size_t findPropertyTokenEnd(std::string_view s, std::size_t start) {
  if (start + 3 >= s.size()) {
    return start;
  }
  if (s[start + 2] != '{') {
    return start;
  }
  const auto closeIdx = s.find('}', start + 3);
  if (closeIdx == std::string_view::npos) {
    return start;
  }
  return closeIdx + 1;
}

bool isCasedLetterCategory(std::string_view resolved) {
  return resolved == "Lu" || resolved == "Ll" || resolved == "Lt" ||
      resolved == "Lowercase" || resolved == "Uppercase" ||
      resolved == "Titlecase" || resolved == "[a-z]" || resolved == "[A-Z]";
}

std::size_t tryAppendPropertyToken(
    std::string_view s,
    std::size_t start,
    char pOrP,
    std::string& out,
    bool caseless) {
  const std::size_t tokenEnd = findPropertyTokenEnd(s, start);
  if (tokenEnd <= start) {
    return start;
  }
  const std::size_t braceOpen = s.find('{', start + 2);
  const std::string name(s.substr(braceOpen + 1, tokenEnd - braceOpen - 2));
  auto replacement = PropertyMap::apply(name);
  if (replacement) {
    if (auto normalized = PropertyMap::apply(*replacement)) {
      replacement = std::move(normalized);
    }
  }
  const std::string_view effective = replacement
      ? std::string_view(*replacement)
      : std::string_view(name);

  if (caseless && isCasedLetterCategory(effective)) {
    if (pOrP == 'P') {
      out += "[^\\p{Lu}\\p{Ll}\\p{Lt}]";
    } else {
      out += "[\\p{Lu}\\p{Ll}\\p{Lt}]";
    }
    return tokenEnd;
  }

  if (!replacement) {
    out.append(s.substr(start, tokenEnd - start));
  } else if (*replacement == PropertyMap::kNeverMatch) {
    if (pOrP == 'P') {
      out += "[\\x{0}-\\x{10FFFF}]";
    } else {
      out += "(?!)";
    }
  } else if (replacement->rfind("[^", 0) == 0) {
    if (pOrP == 'P') {
      out.push_back('[');
      out.append(replacement->substr(2));
    } else {
      out += *replacement;
    }
  } else if (!replacement->empty() && replacement->front() == '[') {
    if (pOrP == 'P') {
      out += "[^";
      out.append(replacement->substr(1));
    } else {
      out += *replacement;
    }
  } else if (replacement->rfind("\\P{", 0) == 0) {
    if (pOrP == 'P') {
      out += "\\p{";
      out.append(replacement->substr(3));
    } else {
      out += *replacement;
    }
  } else {
    out.push_back('\\');
    out.push_back(pOrP);
    out.push_back('{');
    out += *replacement;
    out.push_back('}');
  }
  return tokenEnd;
}

std::string rewritePropertiesOnly(
    std::string_view s,
    std::size_t from,
    std::size_t to) {
  std::string sb;
  sb.reserve(to - from + 8);
  std::size_t i = from;
  bool inQuote = false;
  while (i < to) {
    const char c = s[i];
    if (c == '\\' && i + 1 < to) {
      const char next = s[i + 1];
      if (!inQuote && next == 'Q') {
        sb += "\\Q";
        i += 2;
        inQuote = true;
        continue;
      }
      if (inQuote && next == 'E') {
        sb += "\\E";
        i += 2;
        inQuote = false;
        continue;
      }
      if (!inQuote && (next == 'p' || next == 'P') &&
          !hasOddTrailingBackslashes(sb)) {
        const auto tokenEnd = tryAppendPropertyToken(s, i, next, sb, false);
        if (tokenEnd > i) {
          i = tokenEnd;
          continue;
        }
      }
      sb.push_back(c);
      ++i;
      continue;
    }
    sb.push_back(c);
    ++i;
  }
  return sb;
}

bool isJavaModeFlag(char c) {
  return c == 'i' || c == 'd' || c == 'm' || c == 's' || c == 'u' ||
      c == 'c' || c == 'x' || c == 'U';
}

std::string filterModeFlags(
    std::string_view s,
    std::size_t from,
    std::size_t to) {
  std::string out;
  out.reserve(to - from);
  for (std::size_t k = from; k < to; ++k) {
    const char f = s[k];
    if (f != 'u' && f != 'U' && f != 'd' && f != 'c') {
      out.push_back(f);
    }
  }
  return out;
}

struct ModeTranslation {
  std::size_t end{std::string_view::npos};
  char term{0};
  bool hasDash{false};
  bool onI{false};
  bool offI{false};
  bool onU{false};
  bool offU{false};
  bool onX{false};
  bool offX{false};
};

bool containsFlag(std::string_view s, std::size_t from, std::size_t to, char flag) {
  for (std::size_t i = from; i < to; ++i) {
    if (s[i] == flag) {
      return true;
    }
  }
  return false;
}

ModeTranslation tryTranslateModeModifier(
    std::string_view s,
    std::size_t start,
    std::size_t len,
    std::string& out) {
  std::size_t j = start + 2;

  const std::size_t onStart = j;
  while (j < len && isJavaModeFlag(s[j])) {
    ++j;
  }
  const std::size_t onEnd = j;

  std::size_t offStart = std::string_view::npos;
  std::size_t offEnd = std::string_view::npos;
  if (j < len && s[j] == '-') {
    ++j;
    offStart = j;
    while (j < len && isJavaModeFlag(s[j])) {
      ++j;
    }
    offEnd = j;
  }

  if (j >= len) {
    return {};
  }
  const char term = s[j];
  if (term != ')' && term != ':') {
    return {};
  }

  const std::string filteredOn = filterModeFlags(s, onStart, onEnd);
  const std::string filteredOff = offStart != std::string_view::npos
      ? filterModeFlags(s, offStart, offEnd)
      : "";
  const bool hasOn = !filteredOn.empty();
  const bool hasOff = !filteredOff.empty();
  const bool hasDash = offStart != std::string_view::npos;

  if (term == ')') {
    if (hasOn || hasOff) {
      out += "(?";
      out += filteredOn;
      if (hasDash) {
        out.push_back('-');
        out += filteredOff;
      }
      out.push_back(')');
    }
  } else {
    if (!hasOn && !hasOff) {
      out += "(?:";
    } else {
      out += "(?";
      out += filteredOn;
      if (hasDash) {
        out.push_back('-');
        out += filteredOff;
      }
      out.push_back(':');
    }
  }

  ModeTranslation result;
  result.end = j + 1;
  result.term = term;
  result.hasDash = hasDash;
  result.onI = containsFlag(s, onStart, onEnd, 'i');
  result.offI = offStart != std::string_view::npos &&
      containsFlag(s, offStart, offEnd, 'i');
  result.onU = containsFlag(s, onStart, onEnd, 'U');
  result.offU = offStart != std::string_view::npos &&
      containsFlag(s, offStart, offEnd, 'U');
  result.onX = containsFlag(s, onStart, onEnd, 'x');
  result.offX = offStart != std::string_view::npos &&
      containsFlag(s, offStart, offEnd, 'x');
  return result;
}

int countCapturingGroups(std::string_view pattern) {
  int count = 0;
  bool inClass = false;
  bool inQuote = false;
  int classDepth = 0;
  for (std::size_t i = 0; i < pattern.size(); ++i) {
    const char c = pattern[i];
    if (c == '\\' && i + 1 < pattern.size()) {
      const char next = pattern[i + 1];
      if (!inQuote && next == 'Q') {
        inQuote = true;
        ++i;
        continue;
      }
      if (inQuote && next == 'E') {
        inQuote = false;
        ++i;
        continue;
      }
      ++i;
      continue;
    }
    if (inQuote) {
      continue;
    }
    if (c == '[') {
      if (!inClass) {
        inClass = true;
        classDepth = 1;
      } else {
        ++classDepth;
      }
      continue;
    }
    if (c == ']' && inClass) {
      --classDepth;
      if (classDepth == 0) {
        inClass = false;
      }
      continue;
    }
    if (inClass) {
      continue;
    }
    if (c == '(') {
      if (i + 1 >= pattern.size() || pattern[i + 1] != '?') {
        ++count;
      } else if (i + 3 < pattern.size() && pattern[i + 2] == '<' &&
          pattern[i + 3] != '=' && pattern[i + 3] != '!') {
        ++count;
      } else if (i + 3 < pattern.size() && pattern[i + 2] == 'P' &&
          pattern[i + 3] == '<') {
        ++count;
      }
    }
  }
  return count;
}

bool containsAscii(std::string_view s, std::string_view needle) {
  return s.find(needle) != std::string_view::npos;
}

std::string expandCasedPropertiesInClass(std::string_view classText) {
  const bool hasProp =
      containsAscii(classText, "\\p{") || containsAscii(classText, "\\P{");
  const bool hasAsciiCasedRange =
      containsAscii(classText, "a-z") || containsAscii(classText, "A-Z");
  if (!hasProp && !hasAsciiCasedRange) {
    return std::string(classText);
  }

  std::string sb;
  sb.reserve(classText.size() + 32);
  bool appendedCasedUnion = false;
  for (std::size_t i = 0; i < classText.size(); ++i) {
    const char c = classText[i];
    if (c == '\\' && i + 3 < classText.size() &&
        (classText[i + 1] == 'p' || classText[i + 1] == 'P') &&
        classText[i + 2] == '{') {
      const auto close = classText.find('}', i + 3);
      if (close != std::string_view::npos) {
        const auto body = classText.substr(i + 3, close - i - 3);
        if (isCasedLetterCategory(body)) {
          if (classText[i + 1] == 'P') {
            sb.append(classText.substr(i, close + 1 - i));
          } else {
            sb += "\\p{Lu}\\p{Ll}\\p{Lt}";
            appendedCasedUnion = true;
          }
          i = close;
          continue;
        }
      }
    }
    sb.push_back(c);
  }
  if (hasAsciiCasedRange && !appendedCasedUnion && sb.size() > 1 &&
      sb.back() == ']') {
    sb.insert(sb.size() - 1, "\\p{Lu}\\p{Ll}\\p{Lt}");
  }
  return sb;
}

bool decodeUtf8CodePoint(std::string_view s, std::size_t& i, std::uint32_t& cp) {
  const unsigned char b0 = static_cast<unsigned char>(s[i]);
  if (b0 < 0x80) {
    cp = b0;
    ++i;
    return true;
  }
  int need = 0;
  cp = 0;
  if ((b0 & 0xE0) == 0xC0) {
    need = 1;
    cp = b0 & 0x1F;
  } else if ((b0 & 0xF0) == 0xE0) {
    need = 2;
    cp = b0 & 0x0F;
  } else if ((b0 & 0xF8) == 0xF0) {
    need = 3;
    cp = b0 & 0x07;
  } else {
    ++i;
    return false;
  }
  if (i + need >= s.size()) {
    ++i;
    return false;
  }
  for (int n = 1; n <= need; ++n) {
    const unsigned char bx = static_cast<unsigned char>(s[i + n]);
    if ((bx & 0xC0) != 0x80) {
      ++i;
      return false;
    }
    cp = (cp << 6) | (bx & 0x3F);
  }
  i += need + 1;
  return true;
}

bool containsRawSurrogate(std::string_view s, std::size_t from, std::size_t to) {
  const std::size_t limit = std::min(to, s.size());
  for (std::size_t k = from; k < limit;) {
    std::uint32_t cp = 0;
    const std::size_t before = k;
    if (!decodeUtf8CodePoint(s.substr(0, limit), k, cp)) {
      if (k <= before) {
        ++k;
      }
      continue;
    }
    if (cp >= 0xD800 && cp <= 0xDFFF) {
      return true;
    }
  }
  return false;
}

bool containsRawSurrogate(std::string_view s) {
  return containsRawSurrogate(s, 0, s.size());
}

bool containsSurrogateHexToken(std::string_view s) {
  for (std::size_t i = 0; i + 3 < s.size(); ++i) {
    if (s[i] != '\\' || s[i + 1] != 'x' || s[i + 2] != '{') {
      continue;
    }
    std::size_t k = i + 3;
    std::uint32_t cp = 0;
    bool any = false;
    while (k < s.size() && s[k] != '}') {
      if (!isHexDigit(s[k])) {
        any = false;
        break;
      }
      cp = (cp << 4) | hexValue(s[k]);
      any = true;
      ++k;
    }
    if (any && k < s.size() && s[k] == '}' && cp >= 0xD800 &&
        cp <= 0xDFFF) {
      return true;
    }
  }
  return false;
}

bool needsRawByteModeForPcre2(std::string_view translatedPattern) {
  return containsSurrogateHexToken(translatedPattern) ||
      containsRawSurrogate(translatedPattern);
}

bool hasOddBackslashesBefore(std::string_view s, std::size_t pos) {
  std::size_t count = 0;
  while (pos > 0 && s[pos - 1] == '\\') {
    --pos;
    ++count;
  }
  return (count & 1U) == 1U;
}

std::size_t trySkipClass(std::string_view s, std::size_t start) {
  std::size_t pos = start;
  try {
    ClassBodyParser::parseClass(s, pos);
    return pos;
  } catch (const std::invalid_argument&) {
    return start;
  }
}

bool validModeModifierHasUnsupportedRe2Flag(
    std::string_view s,
    std::size_t start,
    char& flag) {
  std::size_t j = start + 2;
  bool hasUnsupported = false;
  char unsupported = 0;

  while (j < s.size() && isJavaModeFlag(s[j])) {
    if (s[j] == 'U' || s[j] == 'd' || s[j] == 'c') {
      hasUnsupported = true;
      unsupported = s[j];
    }
    ++j;
  }

  if (j < s.size() && s[j] == '-') {
    ++j;
    while (j < s.size() && isJavaModeFlag(s[j])) {
      ++j;
    }
  }

  if (j >= s.size() || (s[j] != ')' && s[j] != ':')) {
    return false;
  }

  flag = unsupported;
  return hasUnsupported;
}

void appendRe2ModeModifier(
    std::string_view s,
    std::size_t start,
    std::size_t end,
    char term,
    bool hasDash,
    std::string& out) {
  const auto appendFlags = [&](std::size_t from, std::size_t to) {
    for (std::size_t k = from; k < to; ++k) {
      if (s[k] != 'x' && s[k] != 'u' && s[k] != 'U' && s[k] != 'd' &&
          s[k] != 'c') {
        out.push_back(s[k]);
      }
    }
  };

  std::size_t j = start + 2;
  const std::size_t onStart = j;
  while (j < end && isJavaModeFlag(s[j])) {
    ++j;
  }
  const std::size_t onEnd = j;

  std::size_t offStart = std::string_view::npos;
  std::size_t offEnd = std::string_view::npos;
  if (hasDash) {
    ++j;
    offStart = j;
    while (j < end && isJavaModeFlag(s[j])) {
      ++j;
    }
    offEnd = j;
  }

  const auto hasKeptFlag = [&](std::size_t from, std::size_t to) {
    for (std::size_t k = from; k < to; ++k) {
      if (s[k] != 'x' && s[k] != 'u' && s[k] != 'U' && s[k] != 'd' &&
          s[k] != 'c') {
        return true;
      }
    }
    return false;
  };

  const bool hasOn = hasKeptFlag(onStart, onEnd);
  const bool hasOff = offStart != std::string_view::npos &&
      hasKeptFlag(offStart, offEnd);

  if (term == ')') {
    if (hasOn || hasOff) {
      out += "(?";
      appendFlags(onStart, onEnd);
      if (hasOff) {
        out.push_back('-');
        appendFlags(offStart, offEnd);
      }
      out.push_back(')');
    }
    return;
  }

  if (!hasOn && !hasOff) {
    out += "(?:";
  } else {
    out += "(?";
    appendFlags(onStart, onEnd);
    if (hasOff) {
      out.push_back('-');
      appendFlags(offStart, offEnd);
    }
    out.push_back(':');
  }
}

void rejectUnsupportedRe2Features(std::string_view javaPattern) {
  bool inQuotation = false;
  bool commentsMode = false;
  std::vector<bool> commentsStack;

  for (std::size_t i = 0; i < javaPattern.size();) {
    const char c = javaPattern[i];

    if (c == '\\' && i + 1 < javaPattern.size()) {
      const char next = javaPattern[i + 1];
      if (!inQuotation && next == 'Q') {
        inQuotation = true;
        i += 2;
        continue;
      }
      if (inQuotation && next == 'E') {
        inQuotation = false;
        i += 2;
        continue;
      }
      if (inQuotation) {
        ++i;
        continue;
      }
      if (next >= '1' && next <= '9') {
        throw EvaluationFailedException(
            "RE2 does not support Java backreferences (\\1-\\9)");
      }
      if (next == 'k' && i + 2 < javaPattern.size() &&
          javaPattern[i + 2] == '<') {
        throw EvaluationFailedException(
            "RE2 does not support Java named backreferences (\\k<name>)");
      }
      i += 2;
      continue;
    }

    if (inQuotation) {
      ++i;
      continue;
    }

    if (commentsMode && c == '#' && !hasOddBackslashesBefore(javaPattern, i)) {
      while (i < javaPattern.size() && javaPattern[i] != '\n') {
        ++i;
      }
      continue;
    }

    if (c == '[' && !hasOddBackslashesBefore(javaPattern, i)) {
      const auto classEnd = trySkipClass(javaPattern, i);
      if (classEnd > i) {
        i = classEnd;
        continue;
      }
    }

    if (c == '(' && i + 1 < javaPattern.size() && javaPattern[i + 1] == '?' &&
        !hasOddBackslashesBefore(javaPattern, i)) {
      if (i + 2 < javaPattern.size()) {
        const char op = javaPattern[i + 2];
        if (op == '=' || op == '!') {
          throw EvaluationFailedException(
              "RE2 does not support Java lookaround assertions");
        }
        if (op == '>') {
          throw EvaluationFailedException(
              "RE2 does not support Java atomic groups (?>...)");
        }
        if (op == '<' && i + 3 < javaPattern.size() &&
            (javaPattern[i + 3] == '=' || javaPattern[i + 3] == '!')) {
          throw EvaluationFailedException(
              "RE2 does not support Java lookaround assertions");
        }
      }

      char unsupportedFlag = 0;
      if (validModeModifierHasUnsupportedRe2Flag(
              javaPattern, i, unsupportedFlag)) {
        if (unsupportedFlag == 'U') {
          throw EvaluationFailedException(
              "RE2 does not support Java UNICODE_CHARACTER_CLASS flag (?U)");
        }
        if (unsupportedFlag == 'c') {
          throw EvaluationFailedException(
              "RE2 does not support Java CANON_EQ flag (?c)");
        }
        throw EvaluationFailedException(
            "RE2 does not support Java UNIX_LINES flag (?d)");
      }

      std::string ignored;
      const auto modeResult =
          tryTranslateModeModifier(javaPattern, i, javaPattern.size(), ignored);
      if (modeResult.end != std::string_view::npos) {
        if (modeResult.term == ':') {
          commentsStack.push_back(commentsMode);
        }
        if (modeResult.onX) {
          commentsMode = true;
        }
        if (modeResult.hasDash && modeResult.offX) {
          commentsMode = false;
        }
        i = modeResult.end;
        continue;
      }
    }

    if (c == '(' && !hasOddBackslashesBefore(javaPattern, i)) {
      commentsStack.push_back(commentsMode);
    } else if (c == ')' && !hasOddBackslashesBefore(javaPattern, i) &&
        !commentsStack.empty()) {
      commentsMode = commentsStack.back();
      commentsStack.pop_back();
    }

    if ((c == '*' || c == '?' || c == '+') && i + 1 < javaPattern.size() &&
        javaPattern[i + 1] == '+' && !hasOddBackslashesBefore(javaPattern, i)) {
      throw EvaluationFailedException(
          "RE2 does not support Java possessive quantifiers");
    }

    if (c == '{' && !hasOddBackslashesBefore(javaPattern, i)) {
      const auto close = javaPattern.find('}', i + 1);
      if (close != std::string_view::npos &&
          isValidQuantifierBody(javaPattern.substr(i + 1, close - i - 1)) &&
          close + 1 < javaPattern.size() && javaPattern[close + 1] == '+') {
        throw EvaluationFailedException(
            "RE2 does not support Java possessive quantifiers");
      }
    }

    ++i;
  }
}

void appendCommentsModeClassForRe2(
    std::string_view pattern,
    std::size_t classStart,
    std::size_t classEnd,
    std::string& out) {
  out.push_back('[');
  bool inQuotation = false;

  for (std::size_t i = classStart + 1; i + 1 < classEnd;) {
    const char c = pattern[i];
    if (c == '\\' && i + 1 < classEnd) {
      const char next = pattern[i + 1];
      if (!inQuotation && next == 'Q') {
        inQuotation = true;
      } else if (inQuotation && next == 'E') {
        inQuotation = false;
      }
      out.push_back(c);
      out.push_back(next);
      i += 2;
      continue;
    }

    if (!inQuotation && c == '#') {
      while (i + 1 < classEnd && pattern[i] != '\n') {
        ++i;
      }
      if (i + 1 >= classEnd) {
        throw EvaluationFailedException(
            "Java COMMENTS mode comment in character class is not terminated");
      }
      continue;
    }

    if (!inQuotation && std::isspace(static_cast<unsigned char>(c))) {
      ++i;
      continue;
    }

    out.push_back(c);
    ++i;
  }

  out.push_back(']');
}

std::string translateCommentsModeForRe2(std::string_view pattern) {
  std::string out;
  out.reserve(pattern.size());
  bool inQuotation = false;
  bool commentsMode = false;
  std::vector<bool> commentsStack;

  for (std::size_t i = 0; i < pattern.size();) {
    const char c = pattern[i];

    if (c == '\\' && i + 1 < pattern.size()) {
      const char next = pattern[i + 1];
      if (!inQuotation && next == 'Q') {
        out += "\\Q";
        inQuotation = true;
        i += 2;
        continue;
      }
      if (inQuotation && next == 'E') {
        out += "\\E";
        inQuotation = false;
        i += 2;
        continue;
      }
      if (commentsMode && std::isspace(static_cast<unsigned char>(next))) {
        out.push_back(next);
        i += 2;
        continue;
      }
      out.push_back(c);
      out.push_back(next);
      i += 2;
      continue;
    }

    if (inQuotation) {
      out.push_back(c);
      ++i;
      continue;
    }

    if (c == '[' && !hasOddTrailingBackslashes(out)) {
      const auto classEnd = trySkipClass(pattern, i);
      if (classEnd > i) {
        if (commentsMode) {
          appendCommentsModeClassForRe2(pattern, i, classEnd, out);
        } else {
          out.append(pattern.substr(i, classEnd - i));
        }
        i = classEnd;
        continue;
      }
    }

    if (commentsMode && c == '(' && i + 2 < pattern.size() &&
        pattern[i + 1] == '?' && !hasOddTrailingBackslashes(out) &&
        (std::isspace(static_cast<unsigned char>(pattern[i + 2])) ||
         pattern[i + 2] == '#')) {
      throw EvaluationFailedException(
          "Java COMMENTS mode does not ignore whitespace in inline group prefixes");
    }

    if (commentsMode && c == '#') {
      while (i < pattern.size() && pattern[i] != '\n') {
        ++i;
      }
      if (i < pattern.size()) {
        ++i;
      }
      continue;
    }

    if (commentsMode && std::isspace(static_cast<unsigned char>(c))) {
      ++i;
      continue;
    }

    if (c == '(' && i + 1 < pattern.size() && pattern[i + 1] == '?' &&
        !hasOddTrailingBackslashes(out)) {
      std::string ignored;
      const auto modeResult =
          tryTranslateModeModifier(pattern, i, pattern.size(), ignored);
      if (modeResult.end != std::string_view::npos) {
        appendRe2ModeModifier(
            pattern,
            i,
            modeResult.end,
            modeResult.term,
            modeResult.hasDash,
            out);
        if (modeResult.term == ':') {
          commentsStack.push_back(commentsMode);
        }
        if (modeResult.onX) {
          commentsMode = true;
        }
        if (modeResult.hasDash && modeResult.offX) {
          commentsMode = false;
        }
        i = modeResult.end;
        continue;
      }
    }

    if (c == '(' && !hasOddTrailingBackslashes(out)) {
      commentsStack.push_back(commentsMode);
    } else if (c == ')' && !hasOddTrailingBackslashes(out) &&
        !commentsStack.empty()) {
      commentsMode = commentsStack.back();
      commentsStack.pop_back();
    }

    out.push_back(c);
    ++i;
  }

  return out;
}

std::string translatePcre2OctalEscapesForRe2(std::string_view pattern) {
  std::string out;
  out.reserve(pattern.size());
  for (std::size_t i = 0; i < pattern.size();) {
    if (i + 3 < pattern.size() && pattern[i] == '\\' && pattern[i + 1] == 'o' &&
        pattern[i + 2] == '{') {
      const auto close = pattern.find('}', i + 3);
      if (close != std::string_view::npos) {
        std::uint32_t value = 0;
        bool valid = close > i + 3;
        for (std::size_t k = i + 3; k < close; ++k) {
          if (!isOctalDigit(pattern[k])) {
            valid = false;
            break;
          }
          value = (value << 3) + (pattern[k] - '0');
        }
        if (valid) {
          out += "\\x{";
          out += toLowerHex(value);
          out.push_back('}');
          i = close + 1;
          continue;
        }
      }
    }
    out.push_back(pattern[i++]);
  }
  return out;
}

std::string rewriteJavaNamedGroupsForRe2(std::string_view pattern) {
  std::string out;
  out.reserve(pattern.size() + 8);
  bool inQuotation = false;

  for (std::size_t i = 0; i < pattern.size();) {
    const char c = pattern[i];
    if (c == '\\' && i + 1 < pattern.size()) {
      const char next = pattern[i + 1];
      if (!inQuotation && next == 'Q') {
        inQuotation = true;
      } else if (inQuotation && next == 'E') {
        inQuotation = false;
      }
      out.push_back(c);
      out.push_back(next);
      i += 2;
      continue;
    }

    if (!inQuotation && c == '[' && !hasOddBackslashesBefore(pattern, i)) {
      const auto classEnd = trySkipClass(pattern, i);
      if (classEnd > i) {
        out.append(pattern.substr(i, classEnd - i));
        i = classEnd;
        continue;
      }
    }

    if (!inQuotation && c == '(' && i + 3 < pattern.size() &&
        pattern[i + 1] == '?' && pattern[i + 2] == '<' &&
        pattern[i + 3] != '=' && pattern[i + 3] != '!' &&
        !hasOddBackslashesBefore(pattern, i)) {
      out += "(?P<";
      i += 3;
      continue;
    }

    out.push_back(c);
    ++i;
  }
  return out;
}

} // namespace

std::string toPcre2Pattern(
    std::string_view javaPattern,
    bool& needsRawByteMode) {
  needsRawByteMode = false;
  if (javaPattern.empty()) {
    return std::string(javaPattern);
  }

  const std::size_t len = javaPattern.size();
  std::string out;
  out.reserve(len + 32);

  std::size_t i = 0;
  bool inQuotation = false;
  bool caseless = false;
  bool unicodeCharacterClass = false;
  bool commentsMode = false;
  struct GroupFrame {
    bool previousCaseless;
    bool previousUnicodeCharacterClass;
    bool previousCommentsMode;
  };
  std::vector<GroupFrame> groupStack;

  while (i < len) {
    const char c = javaPattern[i];

    if (c == '\\' && i + 1 < len) {
      const char next = javaPattern[i + 1];

      if (!inQuotation && next == 'Q') {
        out += "\\Q";
        i += 2;
        inQuotation = true;
        continue;
      }

      if (inQuotation && next == 'E') {
        out += "\\E";
        i += 2;
        inQuotation = false;
        continue;
      }

      if (inQuotation) {
        out.push_back(c);
        ++i;
        continue;
      }

      if (next == 'p' || next == 'P') {
        if (!hasOddTrailingBackslashes(out)) {
          const auto tokenEnd = tryAppendPropertyToken(javaPattern, i, next, out, caseless);
          if (tokenEnd > i) {
            i = tokenEnd;
            continue;
          }
        }
      }

      if (next == 'u' && i + 6 <= len) {
        std::size_t k = i + 2;
        const std::size_t hexEnd = k + 4;
        while (k < hexEnd && isHexDigit(javaPattern[k])) {
          ++k;
        }
        if (k - (i + 2) == 4) {
          const std::uint32_t cp = parseFourHex(javaPattern, i + 2);
          if (cp >= 0xD800 && cp <= 0xDBFF) {
            if (i + 12 <= len && javaPattern[i + 6] == '\\' &&
                javaPattern[i + 7] == 'u') {
              bool hasLowSurrogate = true;
              for (std::size_t p = i + 8; p < i + 12; ++p) {
                hasLowSurrogate = hasLowSurrogate && isHexDigit(javaPattern[p]);
              }
              if (hasLowSurrogate) {
                const std::uint32_t low = parseFourHex(javaPattern, i + 8);
                if (low >= 0xDC00 && low <= 0xDFFF) {
                  const std::uint32_t scalar =
                      0x10000 + ((cp - 0xD800) << 10) + (low - 0xDC00);
                  out += "\\x{";
                  out += toLowerHex(scalar);
                  out.push_back('}');
                  i += 12;
                  continue;
                }
              }
            }
            throw EvaluationFailedException(
                "Lone high-surrogate Unicode escape cannot be safely translated");
          }
          if (cp >= 0xDC00 && cp <= 0xDFFF) {
            throw EvaluationFailedException(
                "Lone low-surrogate Unicode escape cannot be safely translated");
          }
          out += "\\x{";
          out.append(javaPattern.substr(i + 2, 4));
          out.push_back('}');
          i = k;
          continue;
        }
      }

      if (next == 'N' && i + 2 < len && javaPattern[i + 2] == '{') {
        const auto close = javaPattern.find('}', i + 3);
        if (close != std::string_view::npos) {
          const std::string name(javaPattern.substr(i + 3, close - i - 3));
          UErrorCode status = U_ZERO_ERROR;
          const UChar32 cp =
              u_charFromName(U_EXTENDED_CHAR_NAME, name.c_str(), &status);
          if (U_SUCCESS(status)) {
            out += "\\x{";
            out += toLowerHex(static_cast<std::uint32_t>(cp));
            out.push_back('}');
          } else {
            out.append(javaPattern.substr(i, close + 1 - i));
          }
          i = close + 1;
          continue;
        }
      }

      if (next == 'x' && i + 2 < len && javaPattern[i + 2] == '{') {
        const auto close = javaPattern.find('}', i + 3);
        if (close != std::string_view::npos) {
          out.append(javaPattern.substr(i, close + 1 - i));
          i = close + 1;
          continue;
        }
      }

      if (next == '0' && i + 2 < len && isOctalDigit(javaPattern[i + 2])) {
        std::size_t k = i + 2;
        const std::size_t last = std::min(k + 3, len);
        while (k < last && isOctalDigit(javaPattern[k])) {
          ++k;
        }
        if (k - (i + 2) == 3 && javaPattern[i + 2] > '3') {
          --k;
        }
        int value = 0;
        for (std::size_t p = i + 2; p < k; ++p) {
          value = value * 8 + (javaPattern[p] - '0');
        }
        out += "\\o{";
        char buf[16];
        std::snprintf(buf, sizeof(buf), "%o", value);
        out += buf;
        out.push_back('}');
        i = k;
        continue;
      }

      if (next >= '1' && next <= '9') {
        std::size_t k = i + 2;
        while (k < len && std::isdigit(static_cast<unsigned char>(javaPattern[k]))) {
          ++k;
        }
        const int groupCount = countCapturingGroups(javaPattern);
        std::size_t useDigits = k - (i + 1);
        auto parseDigits = [&](std::size_t digits) {
          int value = 0;
          for (std::size_t p = i + 1; p < i + 1 + digits; ++p) {
            const int digit = javaPattern[p] - '0';
            if (value > (groupCount + 1 - digit) / 10) {
              return groupCount + 1;
            }
            value = value * 10 + digit;
          }
          return value;
        };
        int backrefN = parseDigits(useDigits);
        while (useDigits > 1 && backrefN > groupCount) {
          --useDigits;
          backrefN = parseDigits(useDigits);
        }
        if (backrefN > groupCount) {
          out += "(*F)";
        } else {
          out += "\\g{";
          out += std::to_string(backrefN);
          out.push_back('}');
        }
        for (std::size_t p = i + 1 + useDigits; p < k; ++p) {
          out.push_back(javaPattern[p]);
        }
        i = k;
        continue;
      }

      out.push_back(c);
      ++i;
      continue;
    }

    if (inQuotation) {
      out.push_back(c);
      ++i;
      continue;
    }

    if (commentsMode && c == '#' && !hasOddTrailingBackslashes(out)) {
      while (i < len) {
        const char commentChar = javaPattern[i++];
        out.push_back(commentChar);
        if (commentChar == '\n') {
          break;
        }
      }
      continue;
    }

    if (c == '[' && !hasOddTrailingBackslashes(out)) {
      const std::size_t classStart = i;
      std::size_t pos = i;
      try {
        const ClassNode classNode = ClassBodyParser::parseClass(javaPattern, pos);
        const std::size_t classEnd = pos;
        if (containsRawSurrogate(javaPattern, classStart, classEnd)) {
          out += rewritePropertiesOnly(javaPattern, classStart, classEnd);
          i = classEnd;
          continue;
        }
        const auto classText = javaPattern.substr(classStart, classEnd - classStart);
        if (unicodeCharacterClass && classText.find("&&") != std::string_view::npos &&
            (classText.find("\\d") != std::string_view::npos ||
             classText.find("\\D") != std::string_view::npos ||
             classText.find("\\w") != std::string_view::npos ||
             classText.find("\\W") != std::string_view::npos ||
             classText.find("\\s") != std::string_view::npos ||
             classText.find("\\S") != std::string_view::npos)) {
          throw EvaluationFailedException(
              "UNICODE_CHARACTER_CLASS intersection cannot be safely translated");
        }
        const auto renderResult = ClassRenderer::renderWithSignal(classNode);
        const std::string& rendered = renderResult.text;
        const std::string renderedWithMappedProperties =
            rewritePropertiesOnly(rendered, 0, rendered.size());
        const std::string maybeFolded = caseless
            ? expandCasedPropertiesInClass(renderedWithMappedProperties)
            : renderedWithMappedProperties;
        if (renderResult.intersectionUnresolved) {
          out += rewritePropertiesOnly(javaPattern, classStart, classEnd);
        } else {
          out += maybeFolded;
        }
        i = classEnd;
        continue;
      } catch (const std::invalid_argument& e) {
        if (e.what() != nullptr &&
            std::string_view(e.what()).rfind("Bad intersection syntax", 0) == 0) {
          throw EvaluationFailedException("Bad intersection syntax");
        }
        out.push_back(c);
        ++i;
        continue;
      }
    }

    if (c == '(' && i + 1 < len && javaPattern[i + 1] == '?' &&
        !hasOddTrailingBackslashes(out)) {
      const auto modeResult = tryTranslateModeModifier(javaPattern, i, len, out);
      if (modeResult.end != std::string_view::npos) {
        if (modeResult.term == ':') {
          groupStack.push_back(
              {caseless, unicodeCharacterClass, commentsMode});
        }
        if (modeResult.onI) {
          caseless = true;
        }
        if (modeResult.hasDash && modeResult.offI) {
          caseless = false;
        }
        if (modeResult.onU) {
          unicodeCharacterClass = true;
        }
        if (modeResult.hasDash && modeResult.offU) {
          unicodeCharacterClass = false;
        }
        if (modeResult.onX) {
          commentsMode = true;
        }
        if (modeResult.hasDash && modeResult.offX) {
          commentsMode = false;
        }
        i = modeResult.end;
        continue;
      }
    }

    if (c == '(' && !hasOddTrailingBackslashes(out)) {
      groupStack.push_back({caseless, unicodeCharacterClass, commentsMode});
    }

    if (c == '{' && !hasOddTrailingBackslashes(out)) {
      const auto close = javaPattern.find('}', i + 1);
      if (close == std::string_view::npos) {
        throw EvaluationFailedException("Unclosed counted closure");
      }
      const auto body = javaPattern.substr(i + 1, close - i - 1);
      if (!isValidQuantifierBody(body)) {
        throw EvaluationFailedException("Illegal repetition");
      }
    }

    const bool closesGroup = c == ')' && !hasOddTrailingBackslashes(out);
    out.push_back(c);
    if (closesGroup) {
      if (!groupStack.empty()) {
        const auto frame = groupStack.back();
        groupStack.pop_back();
        caseless = frame.previousCaseless;
        unicodeCharacterClass = frame.previousUnicodeCharacterClass;
        commentsMode = frame.previousCommentsMode;
      }
    }
    ++i;
  }

  needsRawByteMode = needsRawByteModeForPcre2(out);
  return out;
}

std::string toPcre2Pattern(std::string_view javaPattern) {
  bool needsRawByteMode = false;
  return toPcre2Pattern(javaPattern, needsRawByteMode);
}

std::string toRe2Pattern(std::string_view javaPattern) {
  rejectUnsupportedRe2Features(javaPattern);
  return rewriteJavaNamedGroupsForRe2(
      translatePcre2OctalEscapesForRe2(
          toPcre2Pattern(translateCommentsModeForRe2(javaPattern))));
}

} // namespace facebook::velox::functions::java_pcre2_translator
