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
#include <memory>
#include <string>
#include <string_view>

#include "velox/external/regex_compat/RegexTypes.h"

namespace re2 {
class RE2;
}

namespace facebook::velox::regex_compat {

/// `re2::RE2` backend in the regex-compat test suite.  Public method names
/// and signatures mirror the subset of `re2::RE2` that
/// `velox/functions/lib/Re2Functions.cpp` actually consumes — this keeps the
/// test-suite typed-test surface aligned with Velox's existing RE2 usage.
///
/// **Pattern / replacement input** is Java `java.util.regex` syntax.
/// Internally, the constructor and `GlobalReplace` call into Velox's existing
/// `prepareRegexpReplacePattern` / `prepareRegexpReplaceReplacement`
/// (`Re2Functions.h:402,422`) to translate Java syntax to RE2 syntax.  Java
/// features that RE2 cannot express (lookaround / backrefs / possessive /
/// atomic group) cause `ok() == false` with an error message coming directly
/// from `re2::RE2::error()`; no separate pre-flight scanner is run.
class Re2Regex {
 public:
  explicit Re2Regex(std::string_view javaPattern, Options opt = {});
  ~Re2Regex();

  Re2Regex(const Re2Regex&) = delete;
  Re2Regex& operator=(const Re2Regex&) = delete;

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

  // Static convenience helpers matching `re2::RE2`.
  static bool FullMatch(std::string_view input, const Re2Regex& re);
  static bool PartialMatch(std::string_view input, const Re2Regex& re);

  /// Globally replace all matches in `*str`.  `javaReplacement` uses Java
  /// `$N` / `${name}` syntax; this method internally translates via Velox
  /// `prepareRegexpReplaceReplacement` before invoking `re2::RE2::GlobalReplace`.
  /// Returns the number of replacements performed.
  static int GlobalReplace(
      std::string* str,
      const Re2Regex& re,
      std::string_view javaReplacement);

  // Internal access for the GlobalReplace implementation.
  const re2::RE2& raw() const;

 private:
  std::unique_ptr<re2::RE2> re_;
  std::string error_;
  std::map<std::string, int> named_;
};

} // namespace facebook::velox::regex_compat
