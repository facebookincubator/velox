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
#include "velox/functions/lib/java_regex/JavaRegexTranslator.h"

#include "velox/functions/lib/java_regex/JavaUnicodeBlocks.h"

#include <cctype>
#include <cstdint>
#include <optional>

namespace facebook::velox::functions {

namespace {

// Java capturing-group names are [a-zA-Z][a-zA-Z0-9]*. RE2 additionally allows
// '_', so accept the union: rewriting a name RE2 would have accepted is never a
// regression, and anything else is left verbatim for RE2 to reject.
bool isGroupName(const std::string& p, size_t begin, size_t end) {
  if (begin >= end) {
    return false;
  }
  for (size_t i = begin; i < end; ++i) {
    const unsigned char c = p[i];
    const bool alpha = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
    const bool digit = c >= '0' && c <= '9';
    if (!(alpha || c == '_' || (digit && i > begin))) {
      return false;
    }
  }
  return true;
}

} // namespace

std::string normalizeJavaRegexForRe2(const std::string& pattern) {
  const size_t n = pattern.size();
  std::string out;
  out.reserve(n + 8);

  auto isHexDigit = [](unsigned char c) {
    return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') ||
        (c >= 'A' && c <= 'F');
  };

  bool inClass = false; // True while scanning inside a [...] character class.
  size_t i = 0;
  while (i < n) {
    const char c = pattern[i];
    if (c == '\\' && i + 1 < n) {
      const char next = pattern[i + 1];
      // \Q...\E literal quoting: copy verbatim, no interpretation inside.
      if (next == 'Q') {
        const size_t end = pattern.find("\\E", i + 2);
        const size_t stop = (end == std::string::npos) ? n : end + 2;
        out.append(pattern, i, stop - i);
        i = stop;
        continue;
      }
      // \uXXXX -> \x{XXXX} (requires exactly four hex digits). A UTF-16
      // high-surrogate (\uD800-\uDBFF) immediately followed by a low-surrogate
      // (\uDC00-\uDFFF) is folded into the single supplementary code point it
      // encodes, which RE2 matches natively; Java treats the pair as one char.
      if (next == 'u' && i + 5 < n && isHexDigit(pattern[i + 2]) &&
          isHexDigit(pattern[i + 3]) && isHexDigit(pattern[i + 4]) &&
          isHexDigit(pattern[i + 5])) {
        auto hex4 = [&](size_t p) -> unsigned {
          unsigned v = 0;
          for (size_t k = 0; k < 4; ++k) {
            const unsigned char h = pattern[p + k];
            v = (v << 4) + (h <= '9' ? h - '0' : (h | 0x20) - 'a' + 10);
          }
          return v;
        };
        const unsigned hi = hex4(i + 2);
        if (hi >= 0xD800 && hi <= 0xDBFF && i + 11 < n &&
            pattern[i + 6] == '\\' && pattern[i + 7] == 'u' &&
            isHexDigit(pattern[i + 8]) && isHexDigit(pattern[i + 9]) &&
            isHexDigit(pattern[i + 10]) && isHexDigit(pattern[i + 11]) &&
            hex4(i + 8) >= 0xDC00 && hex4(i + 8) <= 0xDFFF) {
          const unsigned cp =
              0x10000 + ((hi - 0xD800) << 10) + (hex4(i + 8) - 0xDC00);
          out += "\\x{";
          char digits[8];
          int d = 0;
          for (unsigned v = cp; v != 0; v >>= 4) {
            const unsigned nib = v & 0xF;
            digits[d++] = nib < 10 ? ('0' + nib) : ('A' + nib - 10);
          }
          while (d > 0) {
            out += digits[--d];
          }
          out += '}';
          i += 12;
          continue;
        }
        out += "\\x{";
        out.append(pattern, i + 2, 4);
        out += '}';
        i += 6;
        continue;
      }
      // Any other escape: copy the escaped pair verbatim (e.g. \\, \[, \d, \.).
      out += c;
      out += next;
      i += 2;
      continue;
    }

    // POSIX class such as [:alpha:] appears as "[[:...:]]"; copy it verbatim so
    // the inner brackets do not corrupt character-class tracking.
    if (inClass && c == '[' && i + 1 < n && pattern[i + 1] == ':') {
      const size_t close = pattern.find(":]", i + 2);
      if (close != std::string::npos) {
        out.append(pattern, i, close + 2 - i);
        i = close + 2;
        continue;
      }
    }

    if (c == '[') {
      inClass = true;
    } else if (c == ']') {
      inClass = false;
    }
    out += c;
    ++i;
  }
  return out;
}

std::string expandJavaUnicodePropertiesForRe2(const std::string& pattern) {
  const size_t n = pattern.size();
  std::string out;
  out.reserve(n + 16);

  // Normalizes a property name for block/alias lookup: lower-case, and strip
  // spaces, underscores and hyphens (Java block matching is loose).
  auto normalize = [](const std::string& s) {
    std::string r;
    r.reserve(s.size());
    for (const char ch : s) {
      if (ch == ' ' || ch == '_' || ch == '-') {
        continue;
      }
      r += static_cast<char>((ch >= 'A' && ch <= 'Z') ? (ch - 'A' + 'a') : ch);
    }
    return r;
  };

  // Emits a code-point range, respecting character-class context. Returns an
  // empty string when the range cannot be expressed (negation inside a class).
  auto emitRange = [](uint32_t lo, uint32_t hi, bool negate, bool inClass) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "\\x{%X}-\\x{%X}", lo, hi);
    if (inClass) {
      return negate ? std::string() : std::string(buf);
    }
    return (negate ? std::string("[^") : std::string("[")) + buf + "]";
  };

  // Translates a single \p{...}/\P{...} name. Returns "" to signal "leave the
  // original token verbatim" (RE2 handles native forms; unknown forms fall
  // back).
  auto translate =
      [&](std::string name, bool negate, bool inClass) -> std::string {
    if (!name.empty() && name[0] == '^') {
      negate = !negate;
      name.erase(0, 1);
    }
    const std::string norm = normalize(name);
    // Block forms.
    std::string blockKey;
    if (norm.rfind("block=", 0) == 0) {
      blockKey = norm.substr(6);
    } else if (norm.rfind("blk=", 0) == 0) {
      blockKey = norm.substr(4);
    } else if (norm.rfind("in", 0) == 0 && norm.size() > 2) {
      blockKey = norm.substr(2);
    }
    if (!blockKey.empty()) {
      const auto range = lookupJavaUnicodeBlock(blockKey);
      // If the block is unknown, leave verbatim: the name might actually be a
      // script (e.g. \p{Inherited}) that RE2 supports.
      return range ? emitRange(range->first, range->second, negate, inClass)
                   : std::string();
    }
    // Alias forms (with or without a leading 'Is').
    const std::string base =
        (norm.rfind("is", 0) == 0 && norm.size() > 2) ? norm.substr(2) : norm;
    if (base == "ascii") {
      return emitRange(0x0, 0x7F, negate, inClass);
    }
    if (base == "l1") {
      return emitRange(0x0, 0xFF, negate, inClass);
    }
    if (base == "lc") {
      if (negate) {
        return std::string(); // negated cased-letter set: leave for fallback.
      }
      const std::string body = "\\p{Lu}\\p{Ll}\\p{Lt}";
      return inClass ? body : "[" + body + "]";
    }
    // Script/category with a Java 'Is' prefix: drop it and let RE2 validate the
    // native name (\p{IsGreek} -> \p{Greek}, \p{IsLu} -> \p{Lu}).
    if (name.size() > 2 && name[0] == 'I' && name[1] == 's') {
      const std::string nativeName = name.substr(2);
      return (negate ? "\\P{" : "\\p{") + nativeName + "}";
    }
    return std::string(); // RE2-native or unknown: leave verbatim.
  };

  bool inClass = false;
  size_t i = 0;
  while (i < n) {
    const char c = pattern[i];
    if (c == '\\' && i + 1 < n) {
      const char next = pattern[i + 1];
      // \Q...\E literal quoting: copy verbatim.
      if (next == 'Q') {
        const size_t end = pattern.find("\\E", i + 2);
        const size_t stop = (end == std::string::npos) ? n : end + 2;
        out.append(pattern, i, stop - i);
        i = stop;
        continue;
      }
      // \p{...} / \P{...} Unicode property.
      if ((next == 'p' || next == 'P') && i + 2 < n && pattern[i + 2] == '{') {
        const size_t close = pattern.find('}', i + 3);
        if (close != std::string::npos) {
          const std::string name = pattern.substr(i + 3, close - (i + 3));
          const std::string frag = translate(name, next == 'P', inClass);
          if (!frag.empty()) {
            out += frag;
          } else {
            out.append(pattern, i, close + 1 - i); // leave verbatim
          }
          i = close + 1;
          continue;
        }
      }
      // Any other escape: copy the pair verbatim.
      out += c;
      out += next;
      i += 2;
      continue;
    }
    // POSIX class such as [:alpha:]: copy verbatim so inner brackets do not
    // corrupt class tracking.
    if (inClass && c == '[' && i + 1 < n && pattern[i + 1] == ':') {
      const size_t cl = pattern.find(":]", i + 2);
      if (cl != std::string::npos) {
        out.append(pattern, i, cl + 2 - i);
        i = cl + 2;
        continue;
      }
    }
    if (c == '[') {
      inClass = true;
    } else if (c == ']') {
      inClass = false;
    }
    out += c;
    ++i;
  }
  return out;
}

std::string rewriteJavaNamedGroups(const std::string& pattern) {
  const size_t n = pattern.size();
  std::string out;
  out.reserve(n + 8);

  bool inClass = false;
  size_t i = 0;
  while (i < n) {
    if (pattern[i] == '\\' && i + 1 < n) {
      // \Q...\E literal quoting: copy verbatim. A '(?<' inside a quoted run is
      // literal text, not a group opener, so rewriting it would change what the
      // pattern matches.
      if (pattern[i + 1] == 'Q') {
        const size_t end = pattern.find("\\E", i + 2);
        const size_t stop = (end == std::string::npos) ? n : end + 2;
        out.append(pattern, i, stop - i);
        i = stop;
        continue;
      }
      // Copy escaped pairs verbatim so an escaped '(' never starts a group.
      out.append(pattern, i, 2);
      i += 2;
      continue;
    }

    if (!inClass && pattern.compare(i, 3, "(?<") == 0) {
      // '(?<=' and '(?<!' are lookbehind, not named groups. RE2 cannot express
      // lookbehind at all, so leave them untouched and let RE2 reject them
      // rather than corrupting the pattern into a bogus group name.
      const char after = i + 3 < n ? pattern[i + 3] : '\0';
      if (after != '=' && after != '!') {
        const size_t close = pattern.find('>', i + 3);
        if (close != std::string::npos && isGroupName(pattern, i + 3, close)) {
          out += "(?P<";
          out.append(pattern, i + 3, close - (i + 3));
          out += '>';
          i = close + 1;
          continue;
        }
      }
    }

    // '(' is a literal inside a character class, so group syntax does not
    // apply there.
    if (pattern[i] == '[') {
      inClass = true;
    } else if (pattern[i] == ']') {
      inClass = false;
    }
    out += pattern[i];
    ++i;
  }
  return out;
}

std::string translateJavaRegexToRe2(const std::string& pattern) {
  return normalizeJavaRegexForRe2(
      expandJavaUnicodePropertiesForRe2(rewriteJavaNamedGroups(pattern)));
}

} // namespace facebook::velox::functions
