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

#include <re2/re2.h>
#include <string>

#include "folly/CPortability.h"
#include "velox/common/base/Exceptions.h"
#include "velox/functions/lib/Re2Functions.h"

namespace facebook::velox::functions {

/// This function preprocesses an input pattern string to follow RE2 syntax for
/// Re2RegexpReplacePresto. Specifically, Presto using RE2J supports named
/// capturing groups as (?<name>regex) or (?P<name>regex), but RE2 only supports
/// (?P<name>regex), so we convert the former format to the latter.
FOLLY_ALWAYS_INLINE std::string preparePrestoRegexpReplacePattern(
    const StringView& pattern) {
  std::string newPattern = pattern.getString();

  RE2::GlobalReplace(
      &newPattern,
      RE2(re2::StringPiece{"[(][?]<([^>]*)>"}, RE2::Quiet),
      "(?P<\\1>");

  return newPattern;
}

/// This function preprocesses an input replacement string to follow RE2 syntax
/// for Re2RegexpReplacePresto. Specifically, Presto using RE2J supports
/// referencing capturing groups with $g or ${name} in replacement, but RE2 only
/// supports referencing numbered capturing groups with \g. So we replace
/// references to named groups with references to the corresponding numbered
/// groups. In addition, Presto using RE2J expects the literal $ character to be
/// escaped as \$, but RE2 does not allow escaping $ in replacement, so we
/// unescape \$ in this function.
FOLLY_ALWAYS_INLINE std::string preparePrestoRegexpReplaceReplacement(
    const RE2& re,
    const StringView& replacement) {
  if (replacement.size() == 0) {
    return std::string{};
  }

  auto newReplacement = replacement.getString();

  RE2 extractor("\\${([^}]*)}", RE2::Quiet);
  if (UNLIKELY(!extractor.ok())) {
    VELOX_FAIL("Invalid regular expression:{}.", extractor.error());
  }

  // If newReplacement contains a reference to a
  // named capturing group ${name}, replace the name with its index.
  re2::StringPiece groupName[2];
  while (extractor.Match(
      newReplacement,
      0,
      newReplacement.size(),
      RE2::UNANCHORED,
      groupName,
      2)) {
    auto groupIter = re.NamedCapturingGroups().find(groupName[1].as_string());
    if (groupIter == re.NamedCapturingGroups().end()) {
      VELOX_USER_FAIL(
          "Invalid replacement sequence: unknown group {{ {} }}.",
          groupName[1].as_string());
    }

    RE2::GlobalReplace(
        &newReplacement,
        RE2(re2::StringPiece{fmt::format(
                "\\${{{}}}", groupName[1].as_string())},
            RE2::Quiet),
        re2::StringPiece{fmt::format("${}", groupIter->second)});
  }

  // Convert references to numbered capturing groups from $g to \g.
  RE2::GlobalReplace(
      &newReplacement,
      RE2(re2::StringPiece{"\\$(\\d+)"}, RE2::Quiet),
      re2::StringPiece{"\\\\\\1"});

  // Un-escape dollar-sign '$'.
  RE2::GlobalReplace(
      &newReplacement,
      RE2(re2::StringPiece{"\\\\\\$"}, RE2::Quiet),
      re2::StringPiece{"$"});

  return newReplacement;
}

template <typename T>
using Re2RegexpReplacePresto = Re2RegexpReplace<
    T,
    preparePrestoRegexpReplacePattern,
    preparePrestoRegexpReplaceReplacement>;

} // namespace facebook::velox::functions
