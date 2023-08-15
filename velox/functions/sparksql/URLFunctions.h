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
#include <cctype>
#include "velox/functions/Macros.h"

namespace facebook::velox::functions::sparksql {
namespace {
FOLLY_ALWAYS_INLINE StringView submatch(const boost::cmatch& match, int idx) {
  const auto& sub = match[idx];
  return StringView(sub.first, sub.length());
}

template <typename TInString>
bool parse(const TInString& rawUrl, boost::cmatch& match) {
  static const boost::regex kUriRegex(
      "(([a-zA-Z][a-zA-Z0-9+.-]*):)?" // scheme:
      "([^?#]*)" // authority and path
      "(?:\\?([^#]*))?" // ?query
      "(?:#(.*))?"); // #fragment

  return boost::regex_match(
      rawUrl.data(), rawUrl.data() + rawUrl.size(), match, kUriRegex);
}
} // namespace

template <typename T>
struct ParseUrlFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  // Results refer to strings in the first argument.
  static constexpr int32_t reuse_strings_from_arg = 0;

  // ASCII input always produces ASCII result.
  static constexpr bool is_default_ascii_behavior = true;

  static constexpr int kProto = 2;
  static constexpr int kAuthPath = 3;
  static constexpr int kQuery = 4;
  static constexpr int kRef = 5;

  FOLLY_ALWAYS_INLINE bool call(
      out_type<Varchar>& result,
      const arg_type<Varchar>& url,
      const arg_type<Varchar>& partToExtract) {
    boost::cmatch match;
    if (!parse(url, match)) {
      return false;
    }

    if (partToExtract == "PROTOCOL") {
      auto proto = submatch(match, kProto);
      if (proto.empty()) {
        return false;
      }
      result.setNoCopy(proto);
      return true;
    }

    if (partToExtract == "QUERY") {
      auto query = submatch(match, kQuery);
      if (query.empty()) {
        return false;
      }
      result.setNoCopy(query);
      return true;
    }

    if (partToExtract == "REF") {
      auto ref = submatch(match, kRef);
      if (ref.empty()) {
        return false;
      }
      result.setNoCopy(ref);
      return true;
    }

    boost::cmatch authAndPathMatch;
    boost::cmatch authorityMatch;
    bool hasAuthority = false;
    if (!matchAuthorityAndPath(
            match, authAndPathMatch, authorityMatch, hasAuthority)) {
      return false;
    }

    if (partToExtract == "HOST") {
      if (!hasAuthority) {
        return false;
      }
      result.setNoCopy(submatch(authorityMatch, 3));
      return true;
    }

    if (partToExtract == "PATH") {
      auto path = submatch(match, kAuthPath);
      if (hasAuthority) {
        path = submatch(authAndPathMatch, 2);
      }
      result.setNoCopy(path);
      return true;
    }

    // Path[?Query].
    if (partToExtract == "FILE") {
      auto path = submatch(match, kAuthPath);
      if (hasAuthority) {
        path = submatch(authAndPathMatch, 2);
      }
      auto query = submatch(match, kQuery);
      result = path;
      if (!query.empty()) {
        result += "?";
        result += query;
      }
      return true;
    }

    // Username[:Password].
    if (partToExtract == "USERINFO") {
      if (!hasAuthority) {
        return false;
      }
      auto username = submatch(authorityMatch, 1);
      auto password = submatch(authorityMatch, 2);
      result = username;
      if (!password.empty()) {
        result += ":";
        result += password;
      }
      return result.size();
    }

    // [Userinfo@]Host[:Port].
    if (partToExtract == "AUTHORITY") {
      if (!hasAuthority) {
        return false;
      }
      auto username = submatch(authorityMatch, 1);
      auto password = submatch(authorityMatch, 2);
      if (!username.empty() || !password.empty()) {
        result = username;
        if (!password.empty()) {
          result += ":";
          result += password;
        }
        result += "@";
      }
      result += submatch(authorityMatch, 3);
      auto port = submatch(authorityMatch, 4);
      if (!port.empty()) {
        result += ":";
        result += port;
      }
      return result.size();
    }

    return false;
  }

  FOLLY_ALWAYS_INLINE bool call(
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
    if (!parse(url, match)) {
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

    auto query = submatch(match, kQuery);
    const boost::cregex_iterator begin(
        query.data(), query.data() + query.size(), kQueryParamRegex);
    boost::cregex_iterator end;

    for (auto it = begin; it != end; ++it) {
      if (it->length(2) != 0) { // key shouldn't be empty.
        auto k = submatch((*it), 2);
        if (key.compare(k) == 0) {
          auto value = submatch((*it), 3);
          result.setNoCopy(value);
          return true;
        }
      }
    }

    return false;
  }

 private:
  FOLLY_ALWAYS_INLINE bool matchAuthorityAndPath(
      const boost::cmatch& urlMatch,
      boost::cmatch& authAndPathMatch,
      boost::cmatch& authorityMatch,
      bool& hasAuthority) {
    static const boost::regex kAuthorityAndPathRegex("//([^/]*)(/.*)?");
    auto authorityAndPath = submatch(urlMatch, kAuthPath);
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
};

} // namespace facebook::velox::functions::sparksql
