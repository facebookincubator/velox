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
namespace {

static const boost::regex kUriRegex(
    "(([a-zA-Z][a-zA-Z0-9+.-]*):)?" // scheme:
    "([^?#]*)" // authority and path
    "(?:\\?([^#]*))?" // ?query
    "(?:#(.*))?"); // #fragment

FOLLY_ALWAYS_INLINE StringView submatch(const boost::cmatch& match, int idx) {
  const auto& sub = match[idx];
  return StringView(sub.first, sub.length());
}

template <typename TInString>
bool parse(const TInString& rawUrl, boost::cmatch& match) {
  return boost::regex_match(
      rawUrl.data(), rawUrl.data() + rawUrl.size(), match, kUriRegex);
}

enum submatchEnum {
  PROTOCOL = 2,
  QUERY = 4,
  REF = 5,
  HOST = 0,
  PATH = 1,
  FILE = 3,
  USERINFO = 7,
  AUTHORITY = 8,
  UNKNOWN = 9
};

struct submatchMap : std::map<std::string, submatchEnum> {
  submatchMap() {
    (*this)["PROTOCOL"] = PROTOCOL;
    (*this)["QUERY"] = QUERY;
    (*this)["REF"] = REF;
    (*this)["HOST"] = HOST;
    (*this)["PATH"] = PATH;
    (*this)["FILE"] = FILE;
    (*this)["USERINFO"] = USERINFO;
    (*this)["AUTHORITY"] = AUTHORITY;
  }
};

static const std::unordered_set<submatchEnum> requiresAuthority = {
    submatchEnum::HOST,
    submatchEnum::USERINFO,
    submatchEnum::AUTHORITY};

} // namespace

struct MatchPair {
  int submatchIndex;
  const boost::cmatch& match;
  MatchPair(int index, const boost::cmatch& m)
      : submatchIndex(index), match(m) {}
};

template <typename T>
struct ParseUrlFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  // Results refer to strings in the first argument.
  static constexpr int32_t reuse_strings_from_arg = 0;

  // ASCII input always produces ASCII result.
  static constexpr bool is_default_ascii_behavior = true;

  static constexpr int kAuthPath = 3;
  static constexpr int kQuery = 4;
  static constexpr int kHost = 3;
  // submatch indexes for authorityMatch
  static constexpr int kPathHasAuth = 2;
  static constexpr int kPathNoAuth = 3;
  static constexpr int kUser = 1;
  static constexpr int kPass = 2;
  static constexpr int kPort = 4;

  FOLLY_ALWAYS_INLINE bool call(
      out_type<Varchar>& result,
      const arg_type<Varchar>& url,
      const arg_type<Varchar>& partToExtract) {
    boost::cmatch match;
    if (!parse(url, match)) {
      return false;
    }

    submatchEnum partToExtractEnum = getSubmatchEnum(partToExtract);
    switch (partToExtractEnum) {
      case UNKNOWN:
        return false;
      case PROTOCOL:
      case QUERY:
      case REF:
        return multiMatch(result, {MatchPair(partToExtractEnum, match)});
      default:
        break;
    }
    // Cases above do not require authroity matching and handling
    boost::cmatch authAndPathMatch;
    boost::cmatch authorityMatch;
    bool hasAuthority = false;
    if (!matchAuthorityAndPath(
            match, authAndPathMatch, authorityMatch, hasAuthority)) {
      return false;
    }
    if (!hasAuthority &&
        requiresAuthority.find(partToExtractEnum) != requiresAuthority.end()) {
      return false;
    }
    boost::cmatch* matchToUse = hasAuthority ? &authAndPathMatch : &match;
    int pathMatch = hasAuthority ? kPathHasAuth : kPathNoAuth;
    switch (partToExtractEnum) {
      case HOST:
        return multiMatch(result, {MatchPair(kHost, authorityMatch)});
      case PATH:
        return multiMatch(result, {MatchPair(pathMatch, *matchToUse)}, true);
        return true;
      // Path[?Query].
      case FILE: {
        if (match[kQuery].matched) {
          return multiMatch(
              result,
              {MatchPair(pathMatch, *matchToUse), MatchPair(kQuery, match)});
        }
        multiMatch(result, {MatchPair(pathMatch, *matchToUse)});
        return true;
      }
      // Username[:Password].
      case USERINFO: {
        if (authorityMatch[kPass].matched) {
          return multiMatch(
              result,
              {MatchPair(kUser, authorityMatch),
               MatchPair(kPass, authorityMatch)});
        }
        return multiMatch(result, {MatchPair{kUser, authorityMatch}});
      }
      // [Userinfo@]Host[:Port].
      case AUTHORITY: {
        MatchPair start = MatchPair(
            authorityMatch[kUser].matched ? kUser : kHost, authorityMatch);
        MatchPair end = MatchPair(
            authorityMatch[kPort].matched ? kPort : kHost, authorityMatch);
        return multiMatch(result, {start, end});
      }
      default:
        return false;
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
  bool multiMatch(
      out_type<Varchar>& result,
      std::vector<MatchPair> matches,
      bool returnEmtpyString = false) {
    if (matches.empty() ||
        matches[0].match[matches[0].submatchIndex].matched == false) {
      if (returnEmtpyString) {
        result.setNoCopy(StringView("", 0));
        return true;
      }
      return false;
    }
    size_t i = 0;
    const boost::sub_match<const char*>* firstMatch = nullptr;
    while (i < matches.size()) {
      if (matches[i].match[matches[i].submatchIndex].matched) {
        firstMatch = &(matches[i].match[matches[i].submatchIndex]);
        break;
      }
      i++;
    }
    if (firstMatch == nullptr) {
      return false;
    }
    const char* start = firstMatch->first;
    size_t totalLength = firstMatch->length();

    if (matches.size() > 1) {
      while (i < matches.size()) {
        const auto& currentMatch = matches[i].match[matches[i].submatchIndex];
        if (!currentMatch.matched) {
          return false;
        }
        totalLength = (currentMatch.first - start) + currentMatch.length();
        i++;
      }
    }
    result.setNoCopy(StringView(start, totalLength));
    return true;
  }

  submatchEnum getSubmatchEnum(const arg_type<Varchar>& partToExtract) {
    static const submatchMap submatchMap;
    auto it = submatchMap.find(partToExtract);
    if (it != submatchMap.end()) {
      return it->second;
    }
    return submatchEnum::UNKNOWN;
  }

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