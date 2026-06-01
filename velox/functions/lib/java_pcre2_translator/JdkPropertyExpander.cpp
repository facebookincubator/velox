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
// org.pcre4j.regex.translate.JdkPropertyExpander (Java) under Apache-2.0 by
// the same author for inclusion in Velox.
//
#include "velox/functions/lib/java_pcre2_translator/JdkPropertyExpander.h"

#include <unicode/uchar.h>
#include <unicode/uscript.h>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace facebook::velox::functions::java_pcre2_translator {
namespace {

class SpanBuilder {
 public:
  void add(std::int32_t cp) {
    if (spanStart_ < 0) {
      spanStart_ = cp;
      spanEnd_ = cp;
    } else if (cp == spanEnd_ + 1) {
      spanEnd_ = cp;
    } else {
      pairs_.push_back(spanStart_);
      pairs_.push_back(spanEnd_);
      spanStart_ = cp;
      spanEnd_ = cp;
    }
  }

  RangeSet build() {
    if (spanStart_ >= 0) {
      pairs_.push_back(spanStart_);
      pairs_.push_back(spanEnd_);
      spanStart_ = -1;
    }
    return RangeSet::fromSortedPairs(std::move(pairs_));
  }

 private:
  std::vector<std::int32_t> pairs_;
  std::int32_t spanStart_{-1};
  std::int32_t spanEnd_{-1};
};

std::string upperAscii(std::string_view s) {
  std::string out(s);
  for (char& c : out) {
    c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
  }
  return out;
}

void addAlias(
    std::unordered_map<std::string, SpanBuilder>& builders,
    const char* alias,
    std::int32_t cp) {
  if (alias != nullptr && alias[0] != '\0') {
    builders[upperAscii(alias)].add(cp);
  }
}

const char* categoryName(int32_t type) {
  switch (type) {
    case U_UPPERCASE_LETTER:
      return "LU";
    case U_LOWERCASE_LETTER:
      return "LL";
    case U_TITLECASE_LETTER:
      return "LT";
    case U_MODIFIER_LETTER:
      return "LM";
    case U_OTHER_LETTER:
      return "LO";
    case U_NON_SPACING_MARK:
      return "MN";
    case U_ENCLOSING_MARK:
      return "ME";
    case U_COMBINING_SPACING_MARK:
      return "MC";
    case U_DECIMAL_DIGIT_NUMBER:
      return "ND";
    case U_LETTER_NUMBER:
      return "NL";
    case U_OTHER_NUMBER:
      return "NO";
    case U_SPACE_SEPARATOR:
      return "ZS";
    case U_LINE_SEPARATOR:
      return "ZL";
    case U_PARAGRAPH_SEPARATOR:
      return "ZP";
    case U_CONTROL_CHAR:
      return "CC";
    case U_FORMAT_CHAR:
      return "CF";
    case U_SURROGATE:
      return "CS";
    case U_PRIVATE_USE_CHAR:
      return "CO";
    case U_UNASSIGNED:
      return "CN";
    case U_DASH_PUNCTUATION:
      return "PD";
    case U_START_PUNCTUATION:
      return "PS";
    case U_END_PUNCTUATION:
      return "PE";
    case U_CONNECTOR_PUNCTUATION:
      return "PC";
    case U_OTHER_PUNCTUATION:
      return "PO";
    case U_MATH_SYMBOL:
      return "SM";
    case U_CURRENCY_SYMBOL:
      return "SC";
    case U_MODIFIER_SYMBOL:
      return "SK";
    case U_OTHER_SYMBOL:
      return "SO";
    case U_INITIAL_PUNCTUATION:
      return "PI";
    case U_FINAL_PUNCTUATION:
      return "PF";
    default:
      return nullptr;
  }
}

RangeSet unionOf(
    const std::unordered_map<std::string, RangeSet>& map,
    std::initializer_list<const char*> keys) {
  RangeSet result = RangeSet::empty();
  for (const char* key : keys) {
    auto it = map.find(key);
    if (it != map.end()) {
      result = result.unionWith(it->second);
    }
  }
  return result;
}

std::unordered_map<std::string, RangeSet> buildPositiveMap() {
  std::unordered_map<std::string, SpanBuilder> catBuilders;
  for (const char* cat : {"LU", "LL", "LT", "LM", "LO", "MN", "ME", "MC",
                          "ND", "NL", "NO", "PC", "PD", "PS", "PE", "PI",
                          "PF", "PO", "SM", "SC", "SK", "SO", "ZS", "ZL",
                          "ZP", "CC", "CF", "CS", "CO", "CN"}) {
    catBuilders.emplace(cat, SpanBuilder{});
  }

  std::unordered_map<std::string, SpanBuilder> scriptBuilders;
  std::unordered_map<std::string, SpanBuilder> blockBuilders;
  std::unordered_map<std::string, SpanBuilder> binaryBuilders;

  // Strategy choice: use Velox's existing ICU dependency instead of adding a new
  // dependency or generating source tables. ICU's u_charType/uscript_getScript
  // provide the same kind of full-code-point scan as Java Character APIs.
  for (std::int32_t cp = 0; cp <= RangeSet::kMaxCp; ++cp) {
    if (const char* cat = categoryName(u_charType(static_cast<UChar32>(cp)))) {
      catBuilders[cat].add(cp);
    }

    UErrorCode status = U_ZERO_ERROR;
    const UScriptCode script = uscript_getScript(static_cast<UChar32>(cp), &status);
    if (U_SUCCESS(status)) {
      const char* name = uscript_getName(script);
      if (name != nullptr) {
        scriptBuilders[upperAscii(name)].add(cp);
      }
      addAlias(
          scriptBuilders,
          u_getPropertyValueName(UCHAR_SCRIPT, script, U_SHORT_PROPERTY_NAME),
          cp);
    }

    const auto block = ublock_getCode(static_cast<UChar32>(cp));
    addAlias(
        blockBuilders,
        u_getPropertyValueName(UCHAR_BLOCK, block, U_LONG_PROPERTY_NAME),
        cp);
    addAlias(
        blockBuilders,
        u_getPropertyValueName(UCHAR_BLOCK, block, U_SHORT_PROPERTY_NAME),
        cp);

    if (u_hasBinaryProperty(static_cast<UChar32>(cp), UCHAR_ALPHABETIC)) {
      binaryBuilders["ALPHABETIC"].add(cp);
    }
    if (u_hasBinaryProperty(static_cast<UChar32>(cp), UCHAR_IDEOGRAPHIC)) {
      binaryBuilders["IDEOGRAPHIC"].add(cp);
    }
    if (u_hasBinaryProperty(static_cast<UChar32>(cp), UCHAR_BIDI_MIRRORED)) {
      binaryBuilders["BIDI_MIRRORED"].add(cp);
    }
  }

  std::unordered_map<std::string, RangeSet> map;
  for (auto& [cat, builder] : catBuilders) {
    map.emplace(cat, builder.build());
  }

  map.emplace("L", unionOf(map, {"LU", "LL", "LT", "LM", "LO"}));
  map.emplace("LC", unionOf(map, {"LU", "LL", "LT"}));
  map.emplace("M", unionOf(map, {"MN", "ME", "MC"}));
  map.emplace("N", unionOf(map, {"ND", "NL", "NO"}));
  map.emplace("P", unionOf(map, {"PC", "PD", "PS", "PE", "PI", "PF", "PO"}));
  map.emplace("S", unionOf(map, {"SM", "SC", "SK", "SO"}));
  map.emplace("Z", unionOf(map, {"ZS", "ZL", "ZP"}));
  map.emplace("C", unionOf(map, {"CC", "CF", "CS", "CO", "CN"}));

  for (auto& [script, builder] : scriptBuilders) {
    map.emplace(script, builder.build());
  }
  for (auto& [block, builder] : blockBuilders) {
    auto range = builder.build();
    map.emplace("IN" + block, range);
    map.emplace(block, std::move(range));
  }
  for (auto& [binaryProperty, builder] : binaryBuilders) {
    map.emplace(binaryProperty, builder.build());
  }
  map.emplace("ASCII", RangeSet::range(0, 0x7F));
  return map;
}

const std::unordered_map<std::string, RangeSet>& positiveMap() {
  static const auto kMap = buildPositiveMap();
  return kMap;
}

std::optional<RangeSet> compute(std::string_view token) {
  bool negate = false;
  std::string name;
  if (token.size() >= 4 && token.substr(0, 3) == "\\p{" && token.back() == '}') {
    name = upperAscii(token.substr(3, token.size() - 4));
  } else if (token.size() >= 4 && token.substr(0, 3) == "\\P{" && token.back() == '}') {
    negate = true;
    name = upperAscii(token.substr(3, token.size() - 4));
  } else {
    return std::nullopt;
  }

  auto lookupName = name;
  if (name.rfind("BLK=", 0) == 0) {
    lookupName = "IN" + name.substr(4);
  } else if (name.rfind("BLOCK=", 0) == 0) {
    lookupName = "IN" + name.substr(6);
  }

  auto it = positiveMap().find(lookupName);
  if (it == positiveMap().end()) {
    return std::nullopt;
  }
  return negate ? std::optional<RangeSet>(it->second.complement())
                : std::optional<RangeSet>(it->second);
}

std::mutex cacheMutex;
std::unordered_map<std::string, std::optional<RangeSet>> cache;

} // namespace

std::optional<RangeSet> JdkPropertyExpander::expand(std::string_view pcre2Token) {
  const std::string key(pcre2Token);
  std::lock_guard<std::mutex> l(cacheMutex);
  auto it = cache.find(key);
  if (it != cache.end()) {
    return it->second;
  }
  auto result = compute(key);
  cache.emplace(key, result);
  return result;
}

} // namespace facebook::velox::functions::java_pcre2_translator
