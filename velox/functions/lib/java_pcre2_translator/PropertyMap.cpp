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
// org.pcre4j.regex.translate.PropertyMap (Java) under Apache-2.0 by the
// same author for inclusion in Velox.
//
#include "velox/functions/lib/java_pcre2_translator/PropertyMap.h"

#include "velox/functions/lib/java_pcre2_translator/JdkPropertyExpander.h"

#include <cctype>
#include <unordered_map>

namespace facebook::velox::functions::java_pcre2_translator {

namespace {

const std::unordered_map<std::string, std::string>& table() {
  static const std::unordered_map<std::string, std::string> kTable{
      // --- Short alias: L1 (JDK's Latin-1 shorthand) ---
      {"L1", "[\\x{00}-\\x{FF}]"},

      // --- \p{javaXxx} Java-specific properties ---
      {"javaTitleCase", "Lt"},
      {"javaDigit", "Nd"},
      {"javaLetter", "L"},
      {"javaLetterOrDigit", "[\\p{L}\\p{Nd}]"},
      {"javaAlphabetic", "Alphabetic"},
      {"javaIdeographic", "Ideographic"},
      {"javaMirrored", "Bidi_Mirrored"},
      {"javaDefined", "\\P{Cn}"},
      {"javaISOControl", "[\\x00-\\x1F\\x{7F}-\\x{9F}]"},
      {"javaJavaIdentifierStart", "[\\p{L}\\p{Nl}_$]"},
      {"javaJavaIdentifierPart", "[\\p{L}\\p{Nl}\\p{Mn}\\p{Mc}\\p{Nd}\\p{Pc}_$]"},
      {"javaUnicodeIdentifierStart", "[\\p{L}\\p{Nl}]"},
      {"javaUnicodeIdentifierPart",
       "[\\p{L}\\p{Nl}\\p{Mn}\\p{Mc}\\p{Nd}\\p{Pc}]"},
      {"javaIdentifierIgnorable",
       "[\\x{00}-\\x{08}\\x{0E}-\\x{1B}\\x{7F}-\\x{9F}\\p{Cf}]"},
      // Per Character.isWhitespace() Javadoc:
      {"javaWhitespace",
       "[\\t\\n\\x0B\\f\\r \\x{1C}-\\x{1F}\\x{1680}"
       "\\x{2000}-\\x{200A}\\x{2028}\\x{2029}\\x{205F}\\x{3000}]"},

      // --- POSIX-style class names accepted by Java's \p{Xxx} (default,
      // non-UNICODE) ---
      {"Lower", "[a-z]"},
      {"Upper", "[A-Z]"},
      {"Alpha", "[a-zA-Z]"},
      {"Digit", "[0-9]"},
      {"Alnum", "[a-zA-Z0-9]"},
      {"Punct", "[!-/:-@\\[-`{-~]"},
      {"Graph", "[!-~]"},
      {"Print", "[ -~]"},
      {"Blank", "[ \\t]"},
      {"Cntrl", "[\\x00-\\x1F\\x{7F}]"},
      {"XDigit", "[0-9a-fA-F]"},
      {"Space", "[ \\t\\n\\x0B\\f\\r]"},

      // --- Java property names not recognised as PCRE2 long names ---
      {"Control", "Cc"},
      {"Format", "Cf"},
      {"TitleCase", "Lt"},
      {"UpperCase", "Lu"},
      {"LowerCase", "Ll"},
      {"Letter", "L"},
      {"Mark", "M"},
      {"Number", "N"},
      {"Punctuation", "P"},
      {"Symbol", "S"},
      {"Separator", "Z"},
      {"Other", "C"},
      {"Assigned", "\\P{Cn}"},
      {"Unassigned", "Cn"},
  };
  return kTable;
}

std::string toLower(std::string_view s) {
  std::string out(s);
  for (char& c : out) {
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  }
  return out;
}

std::optional<std::string> resolveOrPass(std::string_view value) {
  auto it = table().find(std::string(value));
  if (it != table().end()) {
    return it->second;
  }
  return std::string(value);
}

std::string camelCaseToUnderscores(std::string_view s);

std::string upperBlockKey(std::string_view value) {
  std::string out(value);
  for (char& c : out) {
    if (c == ' ') {
      c = '_';
    } else {
      c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    }
  }
  return out;
}

std::string normalizedBlockKey(std::string_view value) {
  std::string out;
  out.reserve(value.size());
  for (char c : value) {
    const auto uc = static_cast<unsigned char>(c);
    if (c == '_' || c == '-' || std::isspace(uc)) {
      continue;
    }
    out.push_back(static_cast<char>(std::toupper(uc)));
  }
  return out;
}

std::string resolveBlock(std::string_view blockName) {
  const std::string upper = upperBlockKey(blockName);
  if (upper == "HIGH_SURROGATES") {
    return "[\\x{D800}-\\x{DB7F}]";
  }
  if (upper == "HIGH_PRIVATE_USE_SURROGATES") {
    return "[\\x{DB80}-\\x{DBFF}]";
  }
  if (upper == "LOW_SURROGATES") {
    return "[\\x{DC00}-\\x{DFFF}]";
  }
  const std::string normalized = normalizedBlockKey(blockName);
  if (normalized == "HIGHSURROGATES") {
    return "[\\x{D800}-\\x{DB7F}]";
  }
  if (normalized == "HIGHPRIVATEUSESURROGATES") {
    return "[\\x{DB80}-\\x{DBFF}]";
  }
  if (normalized == "LOWSURROGATES") {
    return "[\\x{DC00}-\\x{DFFF}]";
  }
  if (auto materialized = JdkPropertyExpander::materializeUnicodeBlock(blockName)) {
    return *materialized;
  }
  return camelCaseToUnderscores(blockName);
}

// Inserts an `_` between every lowercase→uppercase boundary in a CamelCase
// string.  E.g. `BasicLatin` → `Basic_Latin`.  Returns `s` unchanged when
// the input already contains an underscore.
std::string camelCaseToUnderscores(std::string_view s) {
  if (s.find('_') != std::string_view::npos) {
    return std::string(s);
  }
  std::string out;
  out.reserve(s.size() + 8);
  for (std::size_t i = 0; i < s.size(); ++i) {
    const char c = s[i];
    if (i > 0 && std::isupper(static_cast<unsigned char>(c)) &&
        std::islower(static_cast<unsigned char>(s[i - 1]))) {
      out.push_back('_');
    }
    out.push_back(c);
  }
  return out;
}

} // namespace

std::optional<std::string> PropertyMap::apply(std::string_view name) {
  // 0. Strip Java/Unicode qualifier prefixes: gc=Lu, sc=Greek, blk=Latin, …
  const auto eq = name.find('=');
  if (eq != std::string_view::npos && eq > 0) {
    const std::string key = toLower(name.substr(0, eq));
    const std::string_view value = name.substr(eq + 1);
    if (key == "gc" || key == "general_category") {
      return resolveOrPass(value);
    }
    if (key == "sc" || key == "script") {
      return resolveOrPass(value);
    }
    if (key == "blk" || key == "block") {
      return resolveBlock(value);
    }
    return std::nullopt;
  }

  if (name == "javaLowerCase" || name == "javaUpperCase" ||
      name == "javaSpaceChar") {
    return JdkPropertyExpander::materializeJavaProperty(name);
  }

  // 1. Exact table match.
  const auto& t = table();
  auto it = t.find(std::string(name));
  if (it != t.end()) {
    return it->second;
  }

  // 2. \p{IsXxx} → strip "Is" prefix; prefer known JDK alias mapping over
  //    passthrough.
  if (name.size() > 2 && name[0] == 'I' && name[1] == 's') {
    const std::string stripped(name.substr(2));
    auto mit = t.find(stripped);
    if (mit != t.end()) {
      return mit->second;
    }
    return stripped;
  }

  // 3. \p{InXxx} → strip "In" prefix; insert underscores at CamelCase
  //    boundaries so PCRE2's block-name lookup succeeds.  Note that
  //    ALL_CAPS_WITH_UNDERSCORES block names were already handled in step 1.
  if (name.size() > 2 && name[0] == 'I' && name[1] == 'n') {
    return resolveBlock(name.substr(2));
  }

  // 4. No rewrite.
  return std::nullopt;
}

} // namespace facebook::velox::functions::java_pcre2_translator
