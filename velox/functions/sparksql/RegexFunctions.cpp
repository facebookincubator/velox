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
#include <folly/container/F14Map.h>
#include "velox/functions/lib/Re2Functions.h"
#include "velox/functions/lib/string/StringImpl.h"

namespace facebook::velox::functions::sparksql {
namespace {

using ::re2::RE2;

template <typename T>
re2::StringPiece toStringPiece(const T& string) {
  return re2::StringPiece(string.data(), string.size());
}

void checkForBadPattern(const RE2& re) {
  if (UNLIKELY(!re.ok())) {
    VELOX_USER_FAIL("invalid regular expression:{}", re.error());
  }
}

void ensureRegexIsConstant(
    const char* functionName,
    const VectorPtr& patternVector) {
  if (!patternVector || !patternVector->isConstantEncoding()) {
    VELOX_USER_FAIL("{} requires a constant pattern.", functionName);
  }
}

FOLLY_ALWAYS_INLINE std::string prepareRegexpPattern(
    const StringView& pattern) {
  static const RE2 kRegex("[(][?]<([^>]*)>");

  // (?<name>regex) => (?P<name>regex)
  std::string newPattern = pattern.getString();
  RE2::GlobalReplace(&newPattern, kRegex, R"((?P<\1>)");

  return newPattern;
}

FOLLY_ALWAYS_INLINE std::string prepareReplacement(
    const RE2& re,
    const StringView& replacement) {
  if (replacement.size() == 0) {
    return std::string{};
  }

  auto newReplacement = replacement.getString();

  // If newReplacement contains a reference to a
  // named capturing group ${name}, replace the name with its index.
  static const RE2 kExtractRegex(R"(\${([^}]*)})");
  VELOX_DCHECK(
      kExtractRegex.ok(),
      "Invalid regular expression {}: {}.",
      R"(\${([^}]*)})",
      kExtractRegex.error());
  re2::StringPiece groupName[2];
  while (kExtractRegex.Match(
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
        fmt::format(R"(\${{{}}})", groupName[1].as_string()),
        fmt::format("${}", groupIter->second));
  }

  // Convert references to numbered capturing groups from $g to \g.
  static const RE2 kConvertRegex(R"(\$(\d+))");
  VELOX_DCHECK(
      kConvertRegex.ok(),
      "Invalid regular expression {}: {}.",
      R"(\$(\d+))",
      kConvertRegex.error());
  RE2::GlobalReplace(&newReplacement, kConvertRegex, R"(\\\1)");

  // Un-escape dollar-sign '$'.
  static const RE2 kUnescapeRegex(R"(\\\$)");
  VELOX_DCHECK(
      kUnescapeRegex.ok(),
      "Invalid regular expression {}: {}.",
      R"(\\\$)",
      kUnescapeRegex.error());
  RE2::GlobalReplace(&newReplacement, kUnescapeRegex, "$");

  // TODO(zhaokuo03): need to replace \\ in replacement

  return newReplacement;
}

// REGEXP_REPLACE(string, pattern, overwrite) → string
// REGEXP_REPLACE(string, pattern, overwrite, position) → string
//
// If a string has a substring that matches the given pattern, replace
// the match in the string wither overwrite and return the string. If
// optional paramter position is provided, only make replacements
// after that positon in the string (1 indexed).
//
// If position <= 0, throw error.
// If position > length string, return string.
template <typename T>
struct RegexpReplaceFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  static constexpr bool is_default_ascii_behavior = true;

  FOLLY_ALWAYS_INLINE void initialize(
      const std::vector<TypePtr>& inputTypes,
      const core::QueryConfig& config,
      const arg_type<Varchar>* str,
      const arg_type<Varchar>* pattern,
      const arg_type<Varchar>* replacement) {
    initialize(inputTypes, config, str, pattern, replacement, nullptr);
  }

  FOLLY_ALWAYS_INLINE void initialize(
      const std::vector<TypePtr>& /*inputTypes*/,
      const core::QueryConfig& config,
      const arg_type<Varchar>* /*string*/,
      const arg_type<Varchar>* pattern,
      const arg_type<Varchar>* replacement,
      const arg_type<int64_t>* /*position*/) {
    if (pattern != nullptr) {
      const auto processedPattern = prepareRegexpPattern(*pattern);
      re_.emplace(processedPattern, RE2::Quiet);
      VELOX_USER_CHECK(
          re_->ok(),
          "Invalid regular expression {}: {}.",
          processedPattern,
          re_->error());
    }

    if (replacement != nullptr) {
      // Constant 'replacement' with non-constant 'pattern' needs to be
      // processed separately for each row.
      if (pattern != nullptr) {
        ensureProcessedReplacement(re_.value(), *replacement);
        constantReplacement_ = true;
      }
    }
  }

  void call(
      out_type<Varchar>& result,
      const arg_type<Varchar>& stringInput,
      const arg_type<Varchar>& pattern,
      const arg_type<Varchar>& replace) {
    call(result, stringInput, pattern, replace, 1);
  }

  void call(
      out_type<Varchar>& result,
      const arg_type<Varchar>& stringInput,
      const arg_type<Varchar>& pattern,
      const arg_type<Varchar>& replace,
      const arg_type<int64_t>& position) {
    if (performChecks(result, stringInput, pattern, replace, position - 1)) {
      return;
    }
    size_t start = functions::stringImpl::cappedByteLength<false>(
        stringInput, position - 1);
    if (start > stringInput.size() + 1) {
      result = stringInput;
      return;
    }
    performReplace(result, stringInput, pattern, replace, start);
  }

  void callAscii(
      out_type<Varchar>& result,
      const arg_type<Varchar>& stringInput,
      const arg_type<Varchar>& pattern,
      const arg_type<Varchar>& replace) {
    callAscii(result, stringInput, pattern, replace, 1);
  }

  void callAscii(
      out_type<Varchar>& result,
      const arg_type<Varchar>& stringInput,
      const arg_type<Varchar>& pattern,
      const arg_type<Varchar>& replace,
      const arg_type<int64_t>& position) {
    if (performChecks(result, stringInput, pattern, replace, position - 1)) {
      return;
    }
    performReplace(result, stringInput, pattern, replace, position - 1);
  }

 private:
  bool performChecks(
      out_type<Varchar>& result,
      const arg_type<Varchar>& stringInput,
      const arg_type<Varchar>& pattern,
      const arg_type<Varchar>& replace,
      const arg_type<int64_t>& position) {
    VELOX_USER_CHECK_GE(
        position + 1, 1, "regexp_replace requires a position >= 1");
    if (position > stringInput.size()) {
      result = stringInput;
      return true;
    }

    if (stringInput.size() == 0 && pattern.size() == 0 && position == 1) {
      result = replace;
      return true;
    }
    return false;
  }

  void performReplace(
      out_type<Varchar>& result,
      const arg_type<Varchar>& stringInput,
      const arg_type<Varchar>& pattern,
      const arg_type<Varchar>& replace,
      const arg_type<int64_t>& position) {
    auto& re = ensurePattern(pattern);
    const auto& processedReplacement = ensureProcessedReplacement(re, replace);

    std::string prefix(stringInput.data(), position);
    std::string targetString(
        stringInput.data() + position, stringInput.size() - position);

    RE2::GlobalReplace(&targetString, re, processedReplacement);
    result = prefix + targetString;
  }

  RE2& ensurePattern(const arg_type<Varchar>& pattern) {
    if (!re_.has_value()) {
      auto processedPattern = prepareRegexpPattern(pattern);
      return *cache_.findOrCompile(StringView(processedPattern));
    } else {
      return re_.value();
    }
  }

  const std::string& ensureProcessedReplacement(
      RE2& re,
      const arg_type<Varchar>& replacement) {
    if (!constantReplacement_) {
      processedReplacement_ = prepareReplacement(re, replacement);
    }

    return processedReplacement_;
  }

  // Used when pattern is constant.
  std::optional<RE2> re_;

  // True if replacement is constant.
  bool constantReplacement_{false};

  // Constant replacement if 'constantReplacement_' is true, or 'current'
  // replacement.
  std::string processedReplacement_;

  // Used when pattern is not constant.
  detail::ReCache cache_;
};

} // namespace

// These functions delegate to the RE2-based implementations in
// common/RegexFunctions.h, but check to ensure that syntax that has different
// semantics between Spark (which uses java.util.regex) and RE2 throws an
// error.
std::shared_ptr<exec::VectorFunction> makeRLike(
    const std::string& name,
    const std::vector<exec::VectorFunctionArg>& inputArgs,
    const core::QueryConfig& config) {
  // Return any errors from re2Search() first.
  auto result = makeRe2Search(name, inputArgs, config);
  ensureRegexIsConstant("RLIKE", inputArgs[1].constantValue);
  return result;
}

std::shared_ptr<exec::VectorFunction> makeRegexExtract(
    const std::string& name,
    const std::vector<exec::VectorFunctionArg>& inputArgs,
    const core::QueryConfig& config) {
  auto result = makeRe2Extract(name, inputArgs, config, /*emptyNoMatch=*/true);
  ensureRegexIsConstant("REGEXP_EXTRACT", inputArgs[1].constantValue);
  return result;
}

void registerRegexpReplace(const std::string& prefix) {
  registerFunction<RegexpReplaceFunction, Varchar, Varchar, Varchar, Varchar>(
      {prefix + "regexp_replace"});
  registerFunction<
      RegexpReplaceFunction,
      Varchar,
      Varchar,
      Varchar,
      Varchar,
      int32_t>({prefix + "regexp_replace"});
}

} // namespace facebook::velox::functions::sparksql
