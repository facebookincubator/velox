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

#include <map>
#include <string>
#include <string_view>

#include "velox/external/regex_compat/RegexTypes.h"

// Opaque PCRE2 8-bit types so this header doesn't drag in <pcre2.h>.
struct pcre2_real_code_8;
typedef struct pcre2_real_code_8 pcre2_code_8;

namespace facebook::velox::regex_compat {

/// PCRE2 (8-bit) backend in the regex-compat test suite.  Public method names
/// and signatures mirror `re2::RE2`'s subset that Velox uses.
///
/// **Pattern / replacement input is Java `java.util.regex` syntax.**
/// PCRE2 natively understands the Java pattern syntax for the common cases
/// (`(?<name>)` named groups, `\d`/`\w`/`\b` etc.) plus a superset of features
/// (lookaround, backreferences, atomic groups, etc.) — so no Java→PCRE2
/// pattern translation is performed by this class.  For replacement strings,
/// PCRE2's `pcre2_substitute_8` with `PCRE2_SUBSTITUTE_EXTENDED` natively
/// understands `$N` and `${name}` Java-style references.
///
/// Java syntax that PCRE2 cannot express (Java-specific property tokens like
/// `\p{InGreek}`, character-class intersection `[a-c&&b-d]`, the meaning swap
/// of `(?U)` flag, etc.) is NOT translated here — those cases are intentionally
/// left to surface as test failures, documenting the need for a future
/// Java→PCRE2 translator (cf. pcre4j PR #606).
class Pcre2Regex {
 public:
  explicit Pcre2Regex(std::string_view javaPattern, Options opt = {});
  ~Pcre2Regex();

  Pcre2Regex(const Pcre2Regex&) = delete;
  Pcre2Regex& operator=(const Pcre2Regex&) = delete;

  bool ok() const;
  const std::string& error() const;
  int NumberOfCapturingGroups() const;
  const std::map<std::string, int>& NamedCapturingGroups() const;

  bool Match(
      std::string_view input,
      std::size_t startpos,
      std::size_t endpos,
      Anchor anchor,
      std::string_view* submatch,
      int nsubmatch) const;

  static bool FullMatch(std::string_view input, const Pcre2Regex& re);
  static bool PartialMatch(std::string_view input, const Pcre2Regex& re);

  /// Java `$N` / `${name}` replacement syntax, handled natively by PCRE2 via
  /// `PCRE2_SUBSTITUTE_EXTENDED`.  Returns the number of replacements done.
  static int GlobalReplace(
      std::string* str,
      const Pcre2Regex& re,
      std::string_view javaReplacement);

 private:
  pcre2_code_8* code_ = nullptr;
  std::string error_;
  int captureCount_ = 0;
  std::map<std::string, int> named_;
};

} // namespace facebook::velox::regex_compat
