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

#include <memory>
#include <string>
#include <string_view>

#include <re2/re2.h>

#include "velox/functions/Udf.h"
#include "velox/functions/lib/Re2Functions.h"
#include "velox/functions/sparksql/UriParser.h"

namespace facebook::velox::functions::sparksql {

/// parse_url(url, part) → varchar
/// parse_url(url, 'QUERY', key) → varchar
/// Extracts a part of a URL, reproducing the semantics of Spark's ParseUrl
/// expression: the URL is parsed with java.net.URI rules (see detail::
/// parseUrl) and every part is returned in its raw, still percent-encoded
/// form. The three-argument form extracts a query parameter with the same
/// (&|^)key=([^&]*) regex Spark compiles.
template <typename T>
struct ParseURLFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  // ASCII input always produces ASCII result.
  static constexpr bool is_default_ascii_behavior = true;

  static constexpr std::string_view kRegexPrefix = "(&|^)";
  static constexpr std::string_view kRegexSuffix = "=([^&]*)";

  ParseURLFunction() : cache_(0) {}

  FOLLY_ALWAYS_INLINE
  void initialize(
      const std::vector<TypePtr>& /* inputTypes */,
      const core::QueryConfig& config,
      const arg_type<Varchar>* urlStr,
      const arg_type<Varchar>* part,
      const arg_type<Varchar>* key) {
    cache_.setMaxCompiledRegexes(config.exprMaxCompiledRegexes());
    // A constant key compiles its regex once here; a non-constant key is
    // looked up in the regex cache in call().
    if (key) {
      constPattern_ = std::make_unique<re2::RE2>(buildQueryPattern(key->str()));
      VELOX_USER_CHECK(constPattern_->ok(), "invalid key: {}", key->str());
    }
  }

  FOLLY_ALWAYS_INLINE
  bool call(
      out_type<Varchar>& output,
      const arg_type<Varchar>& urlStr,
      const arg_type<Varchar>& part) {
    detail::ParsedUrl parsed;
    if (!detail::parseUrl(
            std::string_view(urlStr.data(), urlStr.size()), parsed)) {
      return false;
    }
    return extractPart(output, parsed, part);
  }

  FOLLY_ALWAYS_INLINE
  bool call(
      out_type<Varchar>& output,
      const arg_type<Varchar>& urlStr,
      const arg_type<Varchar>& part,
      const arg_type<Varchar>& key) {
    if (part != "QUERY") {
      return false;
    }
    detail::ParsedUrl parsed;
    if (!detail::parseUrl(
            std::string_view(urlStr.data(), urlStr.size()), parsed) ||
        !parsed.query.has_value()) {
      return false;
    }
    const re2::RE2* pattern = constPattern_.get();
    if (pattern == nullptr) {
      // A non-constant key is looked up in the regex cache so each distinct
      // key compiles at most once instead of once per row. An invalid key or
      // a full cache fails the query, like the regexp functions.
      const std::string queryPattern = buildQueryPattern(key.str());
      pattern = cache_.findOrCompile(StringView(queryPattern));
    }
    re2::StringPiece value;
    if (!RE2::PartialMatch(
            re2::StringPiece(parsed.query->data(), parsed.query->size()),
            *pattern,
            nullptr,
            &value)) {
      return false;
    }
    // A non-participating capture group yields a null result, matching
    // java.util.regex's Matcher.group(2) returning null.
    if (value.data() == nullptr) {
      return false;
    }
    output = std::string(value.data(), value.size());
    return true;
  }

 private:
  static void assignOutput(
      out_type<Varchar>& output,
      std::string_view value) {
    output = StringView(value.data(), static_cast<int32_t>(value.size()));
  }

  /// Builds the query-parameter extraction regex for key: the same
  /// (&|^)key=([^&]*) pattern Spark compiles.
  static std::string buildQueryPattern(std::string_view key) {
    return fmt::format("{}{}{}", kRegexPrefix, key, kRegexSuffix);
  }

  /// Returns the requested part, or false for null. A view field with
  /// data() == nullptr means the component is absent.
  static bool extractPart(
      out_type<Varchar>& output,
      const detail::ParsedUrl& parsed,
      const arg_type<Varchar>& part) {
    if (part == "PROTOCOL") {
      if (parsed.protocol.data() == nullptr) {
        return false;
      }
      assignOutput(output, parsed.protocol);
    } else if (part == "HOST") {
      if (parsed.host.data() == nullptr) {
        return false;
      }
      assignOutput(output, parsed.host);
    } else if (part == "PATH") {
      if (parsed.path.data() == nullptr) {
        return false;
      }
      assignOutput(output, parsed.path);
    } else if (part == "QUERY") {
      if (!parsed.query.has_value()) {
        return false;
      }
      assignOutput(output, *parsed.query);
    } else if (part == "REF") {
      if (!parsed.ref.has_value()) {
        return false;
      }
      assignOutput(output, *parsed.ref);
    } else if (part == "FILE") {
      if (parsed.path.data() == nullptr) {
        return false;
      }
      std::string outputStr(parsed.path);
      if (parsed.query.has_value()) {
        outputStr += '?';
        outputStr += *parsed.query;
      }
      output = outputStr;
    } else if (part == "AUTHORITY") {
      if (!parsed.authority.has_value()) {
        return false;
      }
      assignOutput(output, *parsed.authority);
    } else if (part == "USERINFO") {
      if (!parsed.userInfo.has_value()) {
        return false;
      }
      assignOutput(output, *parsed.userInfo);
    } else {
      return false;
    }
    return true;
  }

  // The regex for a constant query key, compiled once in initialize().
  std::unique_ptr<re2::RE2> constPattern_;

  // Cache of compiled regexes for non-constant query keys, bounded by
  // 'expression.max_compiled_regexes'.
  mutable facebook::velox::functions::detail::ReCache cache_;
};

} // namespace facebook::velox::functions::sparksql
