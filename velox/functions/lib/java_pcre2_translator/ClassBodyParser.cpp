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
// org.pcre4j.regex.translate.ClassBodyParser (Java) under Apache-2.0 by the
// same author for inclusion in Velox.
//
#include "velox/functions/lib/java_pcre2_translator/ClassBodyParser.h"

#include "velox/functions/lib/java_pcre2_translator/PropertyMap.h"

#include <unicode/uchar.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace facebook::velox::functions::java_pcre2_translator {
namespace {

bool startsWith(std::string_view s, std::string_view prefix) {
  return s.size() >= prefix.size() && s.substr(0, prefix.size()) == prefix;
}

bool isBlockPropertyName(std::string_view s) {
  return startsWith(s, "In") || startsWith(s, "blk=") ||
      startsWith(s, "block=");
}

void expect(std::string_view s, std::size_t& pos, char expected) {
  if (pos >= s.size() || s[pos] != expected) {
    throw std::invalid_argument(
        "Expected '" + std::string(1, expected) + "' at index " +
        std::to_string(pos));
  }
  ++pos;
}

int hexDigit(char c) {
  if (c >= '0' && c <= '9') {
    return c - '0';
  }
  if (c >= 'a' && c <= 'f') {
    return c - 'a' + 10;
  }
  if (c >= 'A' && c <= 'F') {
    return c - 'A' + 10;
  }
  throw std::invalid_argument("Invalid hex digit: " + std::string(1, c));
}

std::int32_t codePointAt(std::string_view s, std::size_t pos, std::size_t& width) {
  const auto b0 = static_cast<unsigned char>(s[pos]);
  if (b0 < 0x80) {
    width = 1;
    return b0;
  }
  if ((b0 & 0xE0) == 0xC0 && pos + 1 < s.size()) {
    width = 2;
    return ((b0 & 0x1F) << 6) | (static_cast<unsigned char>(s[pos + 1]) & 0x3F);
  }
  if ((b0 & 0xF0) == 0xE0 && pos + 2 < s.size()) {
    width = 3;
    return ((b0 & 0x0F) << 12) |
        ((static_cast<unsigned char>(s[pos + 1]) & 0x3F) << 6) |
        (static_cast<unsigned char>(s[pos + 2]) & 0x3F);
  }
  if ((b0 & 0xF8) == 0xF0 && pos + 3 < s.size()) {
    width = 4;
    return ((b0 & 0x07) << 18) |
        ((static_cast<unsigned char>(s[pos + 1]) & 0x3F) << 12) |
        ((static_cast<unsigned char>(s[pos + 2]) & 0x3F) << 6) |
        (static_cast<unsigned char>(s[pos + 3]) & 0x3F);
  }
  width = 1;
  return b0;
}

ClassNode makeUnion(const std::vector<ClassNode>& items) {
  if (items.empty()) {
    return ClassNode(Union(std::vector<ClassNode>{}));
  }
  if (items.size() == 1) {
    return items.front();
  }
  return ClassNode(Union(items));
}

ClassNode parseIntersection(std::string_view s, std::size_t& pos);
ClassNode parseUnion(std::string_view s, std::size_t& pos);
ClassNode parseItem(std::string_view s, std::size_t& pos);
ClassNode parseAtom(std::string_view s, std::size_t& pos);
ClassNode parseEscape(std::string_view s, std::size_t& pos);

ClassNode parseIntersection(std::string_view s, std::size_t& pos) {
  ClassNode first = parseUnion(s, pos);
  if (pos + 1 < s.size() && s[pos] == '&' && s[pos + 1] == '&') {
    std::vector<ClassNode> operands;
    operands.push_back(first);
    while (pos + 1 < s.size() && s[pos] == '&' && s[pos + 1] == '&') {
      pos += 2;
      if (pos >= s.size() || s[pos] == ']') {
        throw std::invalid_argument("Bad intersection syntax near index " + std::to_string(pos));
      }
      operands.push_back(parseUnion(s, pos));
    }
    return ClassNode(Intersection(operands));
  }
  return first;
}

ClassNode parseUnion(std::string_view s, std::size_t& pos) {
  std::vector<ClassNode> items;
  while (pos < s.size()) {
    const char ch = s[pos];
    if (ch == ']') {
      break;
    }
    if (ch == '&' && pos + 1 < s.size() && s[pos + 1] == '&') {
      break;
    }
    items.push_back(parseItem(s, pos));
  }
  return makeUnion(items);
}

ClassNode parseItem(std::string_view s, std::size_t& pos) {
  ClassNode atom = parseAtom(s, pos);

  if (const auto* litLo = atom.getIf<Literal>(); litLo != nullptr &&
      pos < s.size() && s[pos] == '-' && pos + 1 < s.size() && s[pos + 1] != ']') {
    ++pos;
    ClassNode atomHi = parseAtom(s, pos);
    if (const auto* litHi = atomHi.getIf<Literal>()) {
      return ClassNode(Range(litLo->cp, litHi->cp));
    }
    return ClassNode(Union(std::vector<ClassNode>{atom, ClassNode(Literal('-')), atomHi}));
  }

  if (atom.is<PropertyLeaf>() && pos < s.size() && s[pos] == '-' &&
      pos + 1 < s.size() && s[pos + 1] != ']') {
    ++pos;
    ClassNode next = parseAtom(s, pos);
    return ClassNode(Union(std::vector<ClassNode>{atom, ClassNode(Literal('-')), next}));
  }
  return atom;
}

ClassNode parseAtom(std::string_view s, std::size_t& pos) {
  if (pos >= s.size()) {
    throw std::invalid_argument("Unexpected end of pattern inside character class");
  }
  if (s[pos] == '[') {
    return ClassBodyParser::parseClass(s, pos);
  }
  if (s[pos] == '\\') {
    return parseEscape(s, pos);
  }
  std::size_t width = 0;
  const auto cp = codePointAt(s, pos, width);
  pos += width;
  return ClassNode(Literal(cp));
}

ClassNode parsePropertyEscape(std::string_view s, std::size_t& pos, char esc) {
  const bool neg = esc == 'P';
  if (pos < s.size() && s[pos] == '{') {
    ++pos;
    const std::size_t start = pos;
    while (pos < s.size() && s[pos] != '}') {
      ++pos;
    }
    const std::string propName(s.substr(start, pos - start));
    if (pos < s.size()) {
      ++pos;
    }
    const auto rewritten = PropertyMap::apply(propName);
    std::string token;
    if (!rewritten.has_value()) {
      token = std::string("\\") + esc + "{" + propName + "}";
    } else if (startsWith(*rewritten, "[") && rewritten->back() == ']' && !neg) {
      std::size_t rewritePos = 0;
      auto node = ClassBodyParser::parseClass(*rewritten, rewritePos);
      if (rewritePos != rewritten->size()) {
        throw std::invalid_argument("Unexpected trailing content in property rewrite");
      }
      return node;
    } else if (startsWith(*rewritten, "[")) {
      std::size_t rewritePos = 0;
      auto node = ClassBodyParser::parseClass(*rewritten, rewritePos);
      if (rewritePos != rewritten->size()) {
        throw std::invalid_argument("Unexpected trailing content in property rewrite");
      }
      return ClassNode(Negated(node));
    } else if (startsWith(*rewritten, "\\P{")) {
      token = neg ? ("\\p{" + rewritten->substr(3)) : *rewritten;
    } else {
      std::string propertyName = *rewritten;
      if (isBlockPropertyName(propName) && !startsWith(propertyName, "In")) {
        propertyName = "In" + propertyName;
      }
      token = std::string("\\") + esc + "{" + propertyName + "}";
    }
    return ClassNode(PropertyLeaf(token, neg));
  }
  return ClassNode(PropertyLeaf(std::string("\\") + esc, neg));
}

ClassNode parseEscape(std::string_view s, std::size_t& pos) {
  expect(s, pos, '\\');
  if (pos >= s.size()) {
    throw std::invalid_argument("Trailing backslash inside character class");
  }
  const char esc = s[pos++];
  switch (esc) {
    case 'n':
      return ClassNode(Literal('\n'));
    case 't':
      return ClassNode(Literal('\t'));
    case 'r':
      return ClassNode(Literal('\r'));
    case 'f':
      return ClassNode(Literal('\f'));
    case 'a':
      return ClassNode(Literal(0x07));
    case 'e':
      return ClassNode(Literal(0x1B));
    case '0': {
      int val = 0;
      int count = 0;
      while (pos < s.size() && count < 3) {
        const char d = s[pos];
        if (d < '0' || d > '7') {
          break;
        }
        const int next = val * 8 + (d - '0');
        if (next > 0xFF) {
          break;
        }
        val = next;
        ++pos;
        ++count;
      }
      return ClassNode(Literal(val));
    }
    case 'c': {
      if (pos >= s.size()) {
        throw std::invalid_argument("Incomplete \\c escape");
      }
      const auto ctrl = static_cast<std::int32_t>(s[pos]) & 0x1F;
      ++pos;
      return ClassNode(Literal(ctrl));
    }
    case 'x': {
      if (pos < s.size() && s[pos] == '{') {
        ++pos;
        std::uint32_t val = 0;
        bool any = false;
        while (pos < s.size() && s[pos] != '}') {
          val = val * 16 + hexDigit(s[pos++]);
          if (val > 0x10FFFF) {
            throw std::invalid_argument("\\x{...} code point out of Unicode range");
          }
          any = true;
        }
        if (pos >= s.size() || s[pos] != '}') {
          throw std::invalid_argument("Unterminated \\x{...} escape");
        }
        if (!any) {
          throw std::invalid_argument("Empty \\x{} escape");
        }
        ++pos;
        return ClassNode(Literal(static_cast<std::int32_t>(val)));
      }
      if (pos + 1 >= s.size()) {
        throw std::invalid_argument("Incomplete \\x escape (need 2 hex digits)");
      }
      const int hi = hexDigit(s[pos++]);
      const int lo = hexDigit(s[pos++]);
      return ClassNode(Literal(hi * 16 + lo));
    }
    case 'u': {
      if (pos + 3 >= s.size()) {
        throw std::invalid_argument("Incomplete \\u escape (need 4 hex digits)");
      }
      int val = 0;
      for (int i = 0; i < 4; ++i) {
        val = val * 16 + hexDigit(s[pos++]);
      }
      return ClassNode(Literal(val));
    }
    case 'Q': {
      std::vector<ClassNode> literals;
      while (pos < s.size()) {
        if (s[pos] == '\\' && pos + 1 < s.size() && s[pos + 1] == 'E') {
          pos += 2;
          break;
        }
        std::size_t width = 0;
        const auto cp = codePointAt(s, pos, width);
        literals.emplace_back(Literal(cp));
        pos += width;
      }
      return makeUnion(literals);
    }
    case 'd':
      return ClassNode(PropertyLeaf("\\d", false));
    case 'D':
      return ClassNode(PropertyLeaf("\\D", true));
    case 'w':
      return ClassNode(PropertyLeaf("\\w", false));
    case 'W':
      return ClassNode(PropertyLeaf("\\W", true));
    case 's':
      return ClassNode(PropertyLeaf("\\s", false));
    case 'S':
      return ClassNode(PropertyLeaf("\\S", true));
    case 'h':
      return ClassNode(PropertyLeaf("\\h", false));
    case 'H':
      return ClassNode(PropertyLeaf("\\H", true));
    case 'v':
      return ClassNode(PropertyLeaf("\\v", false));
    case 'V':
      return ClassNode(PropertyLeaf("\\V", true));
    case 'p':
    case 'P':
      return parsePropertyEscape(s, pos, esc);
    case 'N': {
      if (pos < s.size() && s[pos] == '{') {
        const std::size_t start = pos;
        while (pos < s.size() && s[pos] != '}') {
          ++pos;
        }
        if (pos < s.size()) {
          ++pos;
        }
        const std::string braced(s.substr(start, pos - start));
        if (braced.size() >= 2 && braced.front() == '{' && braced.back() == '}') {
          const std::string name = braced.substr(1, braced.size() - 2);
          UErrorCode status = U_ZERO_ERROR;
          const UChar32 cp = u_charFromName(U_EXTENDED_CHAR_NAME, name.c_str(), &status);
          if (U_SUCCESS(status)) {
            return ClassNode(Literal(cp));
          }
        }
        return ClassNode(PropertyLeaf("\\N" + braced, false));
      }
      return ClassNode(Literal('N'));
    }
    default:
      return ClassNode(Literal(esc));
  }
}

} // namespace

ClassNode ClassBodyParser::parseClass(std::string_view s, std::size_t& pos) {
  expect(s, pos, '[');
  return parseClassBody(s, pos);
}

ClassNode ClassBodyParser::parseClassBody(std::string_view s, std::size_t& pos) {
  const bool negated = pos < s.size() && s[pos] == '^';
  if (negated) {
    ++pos;
  }

  ClassNode body = parseIntersection(s, pos);
  if (pos >= s.size() || s[pos] != ']') {
    throw std::invalid_argument("Unterminated character class");
  }
  ++pos;
  if (negated) {
    return ClassNode(Negated(body));
  }
  return body;
}

} // namespace facebook::velox::functions::java_pcre2_translator
