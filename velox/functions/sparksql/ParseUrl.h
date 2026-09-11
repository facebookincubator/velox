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
#include <optional>
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

  // Every part except a FILE that concatenates the path and query is a
  // direct slice of the URL argument, so the result reuses its string
  // buffer instead of copying.
  static constexpr int32_t reuse_strings_from_arg = 0;

  // ASCII input always produces ASCII result.
  static constexpr bool is_default_ascii_behavior = true;

  static constexpr std::string_view kRegexPrefix = "(&|^)";
  static constexpr std::string_view kRegexSuffix = "=([^&]*)";

  ParseURLFunction() : cache_(0) {}

  // The extractable parts, keyed once per part string instead of compared
  // on every row.
  enum class Part {
    kProtocol,
    kHost,
    kPath,
    kQuery,
    kRef,
    kFile,
    kAuthority,
    kUserInfo,
    kUnknown
  };

  static Part parsePart(std::string_view part) {
    if (part == "PROTOCOL") {
      return Part::kProtocol;
    } else if (part == "HOST") {
      return Part::kHost;
    } else if (part == "PATH") {
      return Part::kPath;
    } else if (part == "QUERY") {
      return Part::kQuery;
    } else if (part == "REF") {
      return Part::kRef;
    } else if (part == "FILE") {
      return Part::kFile;
    } else if (part == "AUTHORITY") {
      return Part::kAuthority;
    } else if (part == "USERINFO") {
      return Part::kUserInfo;
    }
    return Part::kUnknown;
  }

  FOLLY_ALWAYS_INLINE
  void initialize(
      const std::vector<TypePtr>& /*inputTypes*/,
      const core::QueryConfig& config,
      const arg_type<Varchar>* urlStr,
      const arg_type<Varchar>* part,
      const arg_type<Varchar>* key) {
    cache_.setMaxCompiledRegexes(config.exprMaxCompiledRegexes());
    // A constant URL is parsed once here; the parsed views point into the
    // constant vector's string buffer, which outlives every call.
    if (urlStr) {
      constUrl_ = detail::ParsedUrl{};
      if (!detail::parseUrl(
              std::string_view(urlStr->data(), urlStr->size()), *constUrl_)) {
        constUrl_.reset();
        constUrlInvalid_ = true;
      }
    }
    // A constant part is keyed once instead of on every row.
    if (part) {
      constPart_ = parsePart(std::string_view(part->data(), part->size()));
    }
    // A constant key's pattern is only remembered here; the regex itself is
    // compiled lazily on the first call that extracts a query parameter, so
    // an invalid key does not fail rows that never use it.
    if (key) {
      constQueryPattern_ = buildQueryPattern(key->str());
    }
  }

  // Returns the parsed URL, using the cached parse for a constant URL.
  // Returns nullptr for an invalid URL.
  detail::ParsedUrl* parseUrlArg(const arg_type<Varchar>& urlStr) {
    if (constUrl_.has_value()) {
      return &*constUrl_;
    }
    if (constUrlInvalid_) {
      return nullptr;
    }
    parsedScratch_ = detail::ParsedUrl{};
    if (!detail::parseUrl(
            std::string_view(urlStr.data(), urlStr.size()), parsedScratch_)) {
      return nullptr;
    }
    return &parsedScratch_;
  }

  // Returns the part key, using the cached key for a constant part.
  Part partArg(const arg_type<Varchar>& part) const {
    return constPart_.has_value()
        ? *constPart_
        : parsePart(std::string_view(part.data(), part.size()));
  }

  FOLLY_ALWAYS_INLINE
  bool call(
      out_type<Varchar>& output,
      const arg_type<Varchar>& urlStr,
      const arg_type<Varchar>& part) {
    const auto* parsed = parseUrlArg(urlStr);
    if (parsed == nullptr) {
      return false;
    }
    return extractPart(output, *parsed, partArg(part));
  }

  FOLLY_ALWAYS_INLINE
  bool call(
      out_type<Varchar>& output,
      const arg_type<Varchar>& urlStr,
      const arg_type<Varchar>& part,
      const arg_type<Varchar>& key) {
    if (partArg(part) != Part::kQuery) {
      return false;
    }
    const auto* parsed = parseUrlArg(urlStr);
    if (parsed == nullptr || !parsed->query.has_value()) {
      return false;
    }
    const re2::RE2* pattern = nullptr;
    if (constQueryPattern_.has_value()) {
      // A constant key compiles its regex once, on the first call that uses
      // it; an invalid key fails the query here.
      if (constPattern_ == nullptr) {
        constPattern_ = std::make_unique<re2::RE2>(*constQueryPattern_);
        VELOX_USER_CHECK(
            constPattern_->ok(), "invalid key: {}", *constQueryPattern_);
      }
      pattern = constPattern_.get();
    } else {
      // A non-constant key is looked up in the regex cache so each distinct
      // key compiles at most once instead of once per row. An invalid key or
      // a full cache fails the query, like the regexp functions.
      const std::string queryPattern = buildQueryPattern(key.str());
      pattern = cache_.findOrCompile(StringView(queryPattern));
    }
    re2::StringPiece value;
    if (!RE2::PartialMatch(
            re2::StringPiece(parsed->query->data(), parsed->query->size()),
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
    // The capture is a slice of the query, which is a slice of the URL
    // argument, so it can be stored without copying.
    output.setNoCopy(StringView(value.data(), value.size()));
    return true;
  }

 private:
  // Stores a part that is a direct slice of the URL argument without
  // copying; the result vector reuses the argument's string buffer.
  static void assignOutput(out_type<Varchar>& output, std::string_view value) {
    output.setNoCopy(StringView(value.data(), value.size()));
  }

  // Builds the query-parameter extraction regex for key: the same
  // (&|^)key=([^&]*) pattern Spark compiles.
  static std::string buildQueryPattern(std::string_view key) {
    return fmt::format("{}{}{}", kRegexPrefix, key, kRegexSuffix);
  }

  // Returns the requested part, or false for null. A view field with
  // data() == nullptr means the component is absent.
  static bool extractPart(
      out_type<Varchar>& output,
      const detail::ParsedUrl& parsed,
      Part part) {
    switch (part) {
      case Part::kProtocol:
        if (parsed.protocol.data() == nullptr) {
          return false;
        }
        assignOutput(output, parsed.protocol);
        return true;
      case Part::kHost:
        if (parsed.host.data() == nullptr) {
          return false;
        }
        assignOutput(output, parsed.host);
        return true;
      case Part::kPath:
        if (parsed.path.data() == nullptr) {
          return false;
        }
        assignOutput(output, parsed.path);
        return true;
      case Part::kQuery:
        if (!parsed.query.has_value()) {
          return false;
        }
        assignOutput(output, *parsed.query);
        return true;
      case Part::kRef:
        if (!parsed.ref.has_value()) {
          return false;
        }
        assignOutput(output, *parsed.ref);
        return true;
      case Part::kFile: {
        if (parsed.path.data() == nullptr) {
          return false;
        }
        // FILE synthesizes a new string that is not a slice of the URL
        // argument, so it must be copied into the result.
        std::string outputStr(parsed.path);
        if (parsed.query.has_value()) {
          outputStr += '?';
          outputStr += *parsed.query;
        }
        output = outputStr;
        return true;
      }
      case Part::kAuthority:
        if (!parsed.authority.has_value()) {
          return false;
        }
        assignOutput(output, *parsed.authority);
        return true;
      case Part::kUserInfo:
        if (!parsed.userInfo.has_value()) {
          return false;
        }
        assignOutput(output, *parsed.userInfo);
        return true;
      case Part::kUnknown:
        return false;
    }
    return false;
  }

  // The parse of a constant URL, or an empty optional after initialize()
  // when the constant URL is invalid. Unset when the URL is not constant.
  std::optional<detail::ParsedUrl> constUrl_;
  bool constUrlInvalid_ = false;

  // Scratch space for parsing a non-constant URL, reused across calls.
  detail::ParsedUrl parsedScratch_;

  // A constant part, keyed once in initialize() instead of per row.
  std::optional<Part> constPart_;

  // The pattern of a constant query key, remembered in initialize(). The
  // compiled regex is built lazily on first use in call().
  std::optional<std::string> constQueryPattern_;
  std::unique_ptr<re2::RE2> constPattern_;

  // Cache of compiled regexes for non-constant query keys, bounded by
  // 'expression.max_compiled_regexes'.
  facebook::velox::functions::detail::ReCache cache_;
};

} // namespace facebook::velox::functions::sparksql
