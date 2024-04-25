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

#include <boost/regex.hpp>
#include <folly/CPortability.h>
#include <cctype>
#include "velox/functions/Macros.h"

namespace facebook::velox::functions::sparksql {
namespace detail {

// Checks if the URI contains ascii whitespaces or
// unescaped '%' chars.
bool isValidURI(StringView input) {
  const char* p = input.data();
  const char* end = p + input.size();
  char buf[3];
  buf[2] = '\0';
  char* endptr;
  for (; p < end; ++p) {
    if (stringImpl::isAsciiWhiteSpace(*p)) {
      return false;
    }

    if (*p == '%') {
      if (p + 2 < end) {
        buf[0] = p[1];
        buf[1] = p[2];
        strtol(buf, &endptr, 16);
        p += 2;
        if (endptr != buf + 2) {
          return false;
        }
      } else {
        return false;
      }
    }
  }
  return true;
}

static const boost::regex kUriRegex(
    "(([a-zA-Z][a-zA-Z0-9+.-]*):)?" // scheme:
    "([^?#]*)" // authority and path
    "(?:\\?([^#]*))?" // ?query
    "(?:#(.*))?"); // #fragment
template <typename T>
T submatch(const boost::cmatch& match, int indx);

template <>
FOLLY_ALWAYS_INLINE std::string_view submatch(
    const boost::cmatch& match,
    int idx) {
  const auto& sub = match[idx];
  return std::string_view(sub.first, sub.length());
}

template <>
FOLLY_ALWAYS_INLINE StringView submatch(const boost::cmatch& match, int idx) {
  const auto& sub = match[idx];
  return StringView(sub.first, sub.length());
}

template <typename TInString>
bool parse(const TInString& rawUrl, boost::cmatch& match) {
  if (!isValidURI(rawUrl)) {
    return false;
  }
  return boost::regex_match(
      rawUrl.data(), rawUrl.data() + rawUrl.size(), match, kUriRegex);
}
} // namespace detail

// parse_url(url, partToExtract) → string
// parse_url(url, partToExtract, key) → string
//
// Extracts a part of a URL. The partToExtract argument can be one of
// 'PROTOCOL', 'HOST', 'PATH', 'REF', 'AUTHORITY', 'FILE', 'USERINFO',
// 'QUERY', 'FRAGMENT'.
//
// If the URL is invalid, return null.
template <typename T>
struct ParseUrlFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  // Results refer to strings in the first argument.
  static constexpr int32_t reuse_strings_from_arg = 0;

  // ASCII input always produces ASCII result.
  static constexpr bool is_default_ascii_behavior = true;

  bool call(
      out_type<Varchar>& result,
      const arg_type<Varchar>& url,
      const arg_type<Varchar>& partToExtract) {
    boost::cmatch match;
    if (!detail::parse(url, match)) {
      return false;
    }
    if (partToExtract == "PROTOCOL") {
      return simpleMatch(match, kProto, result);
    } else if (partToExtract == "QUERY") {
      return simpleMatch(match, kQuery, result);
    } else if (partToExtract == "REF") {
      return simpleMatch(match, kRef, result);
    }

    boost::cmatch authAndPathMatch;
    boost::cmatch authorityMatch;
    bool hasAuthority = false;
    if (!matchAuthorityAndPath(
            match, authAndPathMatch, authorityMatch, hasAuthority)) {
      return false;
    } else if (partToExtract == "HOST") {
      if (!hasAuthority) {
        return false;
      }
      return simpleMatch(authorityMatch, kHost, result);
    } else if (partToExtract == "PATH") {
      return matchPath(match, authAndPathMatch, hasAuthority, result);
    } else if (partToExtract == "FILE") {
      return matchFile(match, authAndPathMatch, hasAuthority, result);
    } else if (partToExtract == "USERINFO") {
      return matchUserinfo(match, authorityMatch, hasAuthority, result);
    } else if (partToExtract == "AUTHORITY") {
      return matchAuthority(authorityMatch, hasAuthority, result);
    }

    return false;
  }

  bool call(
      out_type<Varchar>& result,
      const arg_type<Varchar>& url,
      const arg_type<Varchar>& partToExtract,
      const arg_type<Varchar>& key) {
    // Only "QUERY" support the third parameter.
    if (partToExtract != "QUERY") {
      return false;
    }
    if (key.empty()) {
      return false;
    }

    boost::cmatch match;
    if (!detail::parse(url, match)) {
      return false;
    }

    // Parse query string.
    static const boost::regex kQueryParamRegex(
        "(^|&)" // start of query or start of parameter "&"
        "([^=&]*)=?" // parameter name and "=" if value is expected
        "([^=&]*)" // parameter value
        "(?=(&|$))" // forward reference, next should be end of query or
                    // start of next parameter
    );

    auto query = detail::submatch<StringView>(match, kQuery);
    const boost::cregex_iterator begin(
        query.data(), query.data() + query.size(), kQueryParamRegex);
    boost::cregex_iterator end;

    for (auto it = begin; it != end; ++it) {
      if (it->length(2) != 0) { // key shouldn't be empty.
        auto k = detail::submatch<StringView>((*it), 2);
        if (key.compare(k) == 0) {
          auto value = detail::submatch<StringView>((*it), 3);
          if (value != "") {
            result.setNoCopy(value);
            return true;
          } else {
            return false;
          }
        }
      }
    }

    return false;
  }

 private:
  bool matchAuthorityAndPath(
      const boost::cmatch& urlMatch,
      boost::cmatch& authAndPathMatch,
      boost::cmatch& authorityMatch,
      bool& hasAuthority) {
    static const boost::regex kAuthorityAndPathRegex("//([^/]*)(/.*)?");
    auto authorityAndPath = detail::submatch<StringView>(urlMatch, kAuthPath);
    if (!boost::regex_match(
            authorityAndPath.begin(),
            authorityAndPath.end(),
            authAndPathMatch,
            kAuthorityAndPathRegex)) {
      // Does not start with //, doesn't have authority.
      hasAuthority = false;
      return true;
    }

    static const boost::regex kAuthorityRegex(
        "(?:([^@:]*)(?::([^@]*))?@)?" // username, password.
        "(\\[[^\\]]*\\]|[^\\[:]*)" // host (IP-literal (e.g. '['+IPv6+']',
        // dotted-IPv4, or named host).
        "(?::(\\d*))?"); // port.

    const auto authority = authAndPathMatch[1];
    if (!boost::regex_match(
            authority.first,
            authority.second,
            authorityMatch,
            kAuthorityRegex)) {
      return false; // Invalid URI Authority.
    }

    hasAuthority = true;
    return true;
  }

  FOLLY_ALWAYS_INLINE bool simpleMatch(
      const boost::cmatch& urlMatch,
      const int index,
      out_type<Varchar>& result) {
    StringView sub = detail::submatch<StringView>(urlMatch, index);
    if (sub.empty()) {
      return false;
    }
    result.setNoCopy(sub);
    return true;
  }

  bool matchPath(
      const boost::cmatch& match,
      const boost::cmatch& authAndPathMatch,
      const bool hasAuthority,
      out_type<Varchar>& result) {
    StringView path = detail::submatch<StringView>(match, kAuthPath);
    if (hasAuthority) {
      path = detail::submatch<StringView>(authAndPathMatch, 2);
    }
    result.setNoCopy(path);
    return true;
  }

  bool matchFile(
      const boost::cmatch& match,
      const boost::cmatch& authAndPathMatch,
      const bool hasAuthority,
      out_type<Varchar>& result) {
    // Path[?Query].
    std::string_view path =
        detail::submatch<std::string_view>(match, kAuthPath);
    if (hasAuthority) {
      path = detail::submatch<std::string_view>(authAndPathMatch, 2);
    }
    std::string_view query = detail::submatch<std::string_view>(match, kQuery);
    if (!query.empty()) {
      result.setNoCopy(
          StringView(path.data(), (query.data() + query.size()) - path.data()));
    } else {
      result.setNoCopy(StringView(path.data(), path.size()));
    }
    return true;
  }

  bool matchUserinfo(
      const boost::cmatch& match,
      const boost::cmatch& authorityMatch,
      const bool hasAuthority,
      out_type<Varchar>& result) {
    // Username[:Password].
    if (!hasAuthority) {
      return false;
    }
    std::string_view username =
        detail::submatch<std::string_view>(authorityMatch, kUser);
    std::string_view password =
        detail::submatch<std::string_view>(authorityMatch, kPass);
    if (!password.empty()) {
      result.setNoCopy(
          StringView(username.data(), password.end() - username.begin()));
      return true;
    } else if (!username.empty()) {
      result.setNoCopy(StringView(username.data(), username.size()));
      return true;
    } else {
      return false;
    }
  }

  bool matchAuthority(
      const boost::cmatch& authorityMatch,
      const bool hasAuthority,
      out_type<Varchar>& result) {
    // [Userinfo@]Host[:Port].
    if (!hasAuthority) {
      return false;
    }
    std::string_view host =
        detail::submatch<std::string_view>(authorityMatch, kHost);
    std::string_view first = host;
    std::string_view last = host;

    std::string_view username =
        detail::submatch<std::string_view>(authorityMatch, kUser);
    std::string_view port =
        detail::submatch<std::string_view>(authorityMatch, kPort);
    if (!username.empty()) {
      first = username;
    }
    if (!port.empty()) {
      last = port;
    }
    result.setNoCopy(StringView(first.data(), last.end() - first.begin()));
    return true;
  }

  static constexpr int kAuthPath = 3;
  static constexpr int kQuery = 4;
  static constexpr int kHost = 3;
  static constexpr int kProto = 2;
  static constexpr int kRef = 5;
  // submatch indexes for authorityMatch
  static constexpr int kPathHasAuth = 2;
  static constexpr int kPathNoAuth = 3;
  static constexpr int kUser = 1;
  static constexpr int kPass = 2;
  static constexpr int kPort = 4;
};

} // namespace facebook::velox::functions::sparksql
