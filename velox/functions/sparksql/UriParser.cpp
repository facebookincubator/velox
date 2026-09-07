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

#include "velox/functions/sparksql/UriParser.h"

#include <limits>

namespace facebook::velox::functions::sparksql {
namespace detail {

void splitRef(std::string_view& url, std::optional<std::string_view>& ref) {
  if (!url.empty() && url.front() == '#') {
    ref = url.substr(1);
    url.remove_prefix(url.size());
  }
}

bool splitQuery(
    std::string_view& remainder,
    std::optional<std::string_view>& query) {
  if (remainder.empty() || remainder.front() != '?') {
    return true;
  }
  remainder.remove_prefix(1);
  const auto queryEnd = remainder.find('#');
  const auto queryPart = remainder.substr(
      0, queryEnd == std::string_view::npos ? remainder.size() : queryEnd);
  if (!checkComponent<isUric>(queryPart, true)) {
    return false;
  }
  query = queryPart;
  remainder.remove_prefix(queryPart.size());
  return true;
}

bool splitPath(std::string_view& remainder, ParsedUrl& parsed) {
  const auto pathEnd = remainder.find_first_of("?#");
  const auto path = remainder.substr(
      0, pathEnd == std::string_view::npos ? remainder.size() : pathEnd);
  if (!checkComponent<isPathChar>(path, true)) {
    return false;
  }
  parsed.path = path;
  remainder.remove_prefix(path.size());
  return true;
}

bool checkPort(std::string_view port) {
  int64_t value = 0;
  for (const char c : port) {
    value = value * 10 + (c - '0');
    if (value > std::numeric_limits<int32_t>::max()) {
      return false;
    }
  }
  return true;
}

bool scanIpv4Address(
    std::string_view input,
    bool numericOnly,
    std::string_view& out) {
  const auto candidate = leadingRun<isIpv4Char>(input);
  if (candidate.empty() || (numericOnly && candidate.size() != input.size())) {
    return false;
  }
  auto rest = candidate;
  for (int byteIndex = 0; byteIndex < 4; ++byteIndex) {
    const auto digits = leadingRun<isDigit>(rest);
    // More than three digits cannot be a byte; parsing them into an int
    // would wrap around, so reject the whole address up front.
    if (digits.empty() || digits.size() > 3) {
      return false;
    }
    int value = 0;
    for (const char c : digits) {
      value = value * 10 + (c - '0');
    }
    if (value > 255) {
      return false;
    }
    rest.remove_prefix(digits.size());
    if (byteIndex < 3) {
      if (rest.empty() || rest.front() != '.') {
        return false;
      }
      rest.remove_prefix(1);
    }
  }
  if (!rest.empty()) {
    return false;
  }
  out = candidate;
  return true;
}

bool parseHostname(std::string_view& input, ParsedUrl& parsed) {
  const auto host = input;
  std::string_view lastLabel;
  while (true) {
    const auto label = leadingRun<isAlnum>(input);
    if (label.empty()) {
      break;
    }
    lastLabel = label;
    input.remove_prefix(label.size());
    const auto labelTail = leadingRun<isAlnumOrDash>(input);
    if (!labelTail.empty() && labelTail.back() == '-') {
      return false;
    }
    input.remove_prefix(labelTail.size());
    if (input.empty() || input.front() != '.') {
      break;
    }
    input.remove_prefix(1);
    if (input.empty()) {
      break;
    }
  }
  if (!input.empty() && input.front() != ':') {
    return false;
  }
  if (lastLabel.empty()) {
    return false;
  }
  // lastLabel.data() == host.data() means the hostname has exactly one
  // label, so the "last label must start with a letter" rule does not
  // apply.
  if (lastLabel.data() != host.data() && !isAlpha(lastLabel.front())) {
    return false;
  }
  parsed.host = host.substr(0, host.size() - input.size());
  return true;
}

bool scanHexGroups(
    std::string_view& input,
    int& byteCount,
    bool& startsWithGroup) {
  auto group = leadingHexDigits(input);
  const auto ipv4Ahead = !group.empty() && input.size() > group.size() &&
      input[group.size()] == '.';
  if (group.empty() || ipv4Ahead) {
    startsWithGroup = false;
    return true;
  }
  if (group.size() > 4) {
    return false;
  }
  startsWithGroup = true;
  byteCount += 2;
  input.remove_prefix(group.size());
  while (!input.empty() && input.front() == ':' &&
         !(input.size() > 1 && input[1] == ':')) {
    const auto beforeColon = input;
    input.remove_prefix(1);
    group = leadingHexDigits(input);
    if (group.empty()) {
      return false;
    }
    if (input.size() > group.size() && input[group.size()] == '.') {
      // The embedded IPv4 address starts after the ':' we just consumed;
      // give the ':' back so the caller sees it.
      input = beforeColon;
      return true;
    }
    if (group.size() > 4) {
      return false;
    }
    byteCount += 2;
    input.remove_prefix(group.size());
  }
  return true;
}

bool skipIpv4Address(std::string_view& input) {
  std::string_view address;
  if (!scanIpv4Address(input, true, address)) {
    return false;
  }
  input.remove_prefix(address.size());
  return true;
}

bool scanHexTail(std::string_view& input, int& byteCount) {
  if (input.empty()) {
    return true;
  }
  bool startsWithGroup;
  if (!scanHexGroups(input, byteCount, startsWithGroup)) {
    return false;
  }
  if (startsWithGroup) {
    if (!input.empty() && input.front() == ':') {
      input.remove_prefix(1);
      if (!skipIpv4Address(input)) {
        return false;
      }
      byteCount += 4;
    }
    return true;
  }
  if (!skipIpv4Address(input)) {
    return false;
  }
  byteCount += 4;
  return true;
}

bool parseIpv6Reference(std::string_view literal) {
  int byteCount = 0;
  bool compressed = false;
  bool startsWithGroup;
  auto input = literal;
  if (!scanHexGroups(input, byteCount, startsWithGroup)) {
    return false;
  }
  if (startsWithGroup) {
    if (input.substr(0, 2) == "::") {
      compressed = true;
      input.remove_prefix(2);
      if (!scanHexTail(input, byteCount)) {
        return false;
      }
    } else if (!input.empty() && input.front() == ':') {
      input.remove_prefix(1);
      if (!skipIpv4Address(input)) {
        return false;
      }
      byteCount += 4;
    }
  } else if (input.substr(0, 2) == "::") {
    compressed = true;
    input.remove_prefix(2);
    if (!scanHexTail(input, byteCount)) {
      return false;
    }
  }
  return input.empty() && byteCount <= 16 &&
      ((!compressed && byteCount == 16) || (compressed && byteCount < 16));
}

bool parseIpv6Host(std::string_view& hostAndPort, ParsedUrl& parsed) {
  const auto closingBracket = hostAndPort.find(']');
  if (closingBracket == std::string_view::npos) {
    return false;
  }
  const auto literal = hostAndPort.substr(1, closingBracket - 1);
  const auto scopeStart = literal.find('%');
  if (scopeStart != std::string_view::npos) {
    const auto scope = literal.substr(scopeStart + 1);
    if (scope.empty() || !checkComponent<isAlnum>(scope, false)) {
      return false;
    }
    if (!parseIpv6Reference(literal.substr(0, scopeStart))) {
      return false;
    }
  } else if (!parseIpv6Reference(literal)) {
    return false;
  }
  parsed.host = hostAndPort.substr(0, closingBracket + 1);
  hostAndPort.remove_prefix(closingBracket + 1);
  return true;
}

bool parseServer(std::string_view authority, ParsedUrl& parsed) {
  auto hostAndPort = authority;
  // The userinfo ends at the first '@'.
  const auto userInfoEnd = authority.find('@');
  if (userInfoEnd != std::string_view::npos) {
    const auto userInfo = authority.substr(0, userInfoEnd);
    if (!checkComponent<isUserInfoChar>(userInfo, true)) {
      return false;
    }
    parsed.userInfo = userInfo;
    hostAndPort = authority.substr(userInfoEnd + 1);
  }
  if (!hostAndPort.empty() && hostAndPort.front() == '[') {
    if (!parseIpv6Host(hostAndPort, parsed)) {
      return false;
    }
  } else {
    std::string_view address;
    if (scanIpv4Address(hostAndPort, false, address)) {
      const auto afterAddress = hostAndPort.substr(address.size());
      if (afterAddress.empty() || afterAddress.front() == ':') {
        parsed.host = address;
        hostAndPort.remove_prefix(address.size());
      }
    }
    if (parsed.host.empty()) {
      if (!parseHostname(hostAndPort, parsed)) {
        return false;
      }
    }
  }
  // All three host forms rejoin here for the port check.
  if (hostAndPort.empty()) {
    return true;
  }
  if (hostAndPort.front() != ':') {
    return false;
  }
  hostAndPort.remove_prefix(1);
  const auto portEnd = hostAndPort.find('/');
  const auto port = hostAndPort.substr(
      0, portEnd == std::string_view::npos ? hostAndPort.size() : portEnd);
  if (!checkComponent<isDigit>(port, false) || !checkPort(port)) {
    return false;
  }
  hostAndPort.remove_prefix(port.size());
  return hostAndPort.empty();
}

bool parseAuthority(std::string_view authority, ParsedUrl& parsed) {
  // An authority scans with the '%'-permitting server set unless it
  // starts with ']' - counter-intuitively, an authority with no ']' at
  // all still gets the permissive set; only a leading ']' falls back to
  // the strict one.
  // Multi-byte characters that are neither spaces nor ISO controls scan
  // as server characters (and registry characters). In the permissive
  // branch '%' is a set member and never reaches escape validation, so
  // a malformed pair there is only caught by the reg-name scan below.
  const auto serverChars = authority.front() == ']'
      ? scanComponent<isServerChar>(authority, true)
      : scanComponent<isServerPercentChar>(authority, true);
  if (serverChars == ScanResult::kMalformedEscape) {
    return false;
  }
  const auto regChars = scanComponent<isRegNameChar>(authority, true);
  if (regChars == ScanResult::kMalformedEscape) {
    return false;
  }
  if (serverChars == ScanResult::kValid && parseServer(authority, parsed)) {
    parsed.authority = authority;
    return true;
  }
  // Registry-based fallback: drop whatever a failed server attempt
  // stored, but keep the raw authority.
  parsed.host = std::string_view{};
  parsed.userInfo = std::nullopt;
  if (regChars == ScanResult::kValid) {
    parsed.authority = authority;
    return true;
  }
  return false;
}

bool parseHierarchical(std::string_view& remainder, ParsedUrl& parsed) {
  if (remainder.substr(0, 2) == "//") {
    remainder.remove_prefix(2);
    // The authority ends at the first '/', '?' or '#'.
    const auto authorityEnd = remainder.find_first_of("/?#");
    const auto authority = remainder.substr(
        0,
        authorityEnd == std::string_view::npos ? remainder.size()
                                               : authorityEnd);
    if (authority.empty()) {
      if (authorityEnd == std::string_view::npos) {
        // '//' with nothing after it fails to parse entirely.
        return false;
      }
      // An empty authority before a path, query, or fragment is valid.
    } else if (!parseAuthority(authority, parsed)) {
      return false;
    }
    remainder.remove_prefix(authority.size());
  }
  return splitPath(remainder, parsed) && splitQuery(remainder, parsed.query);
}

bool parseUrl(std::string_view url, ParsedUrl& parsed) {
  auto remainder = url;

  // The scheme is everything before the first ':' that appears before
  // any '/', '?' or '#'.
  const auto schemeEnd = remainder.find_first_of(":/?#");
  if (schemeEnd != std::string_view::npos && remainder[schemeEnd] == ':') {
    if (schemeEnd == 0) {
      return false;
    }
    const auto scheme = remainder.substr(0, schemeEnd);
    if (!isAlpha(scheme.front()) ||
        !checkComponent<isSchemeChar>(scheme.substr(1), false)) {
      return false;
    }
    parsed.protocol = scheme;
    remainder.remove_prefix(schemeEnd + 1);
    if (remainder.empty() || remainder.front() != '/') {
      // Opaque URI: everything before '#' is the scheme-specific part;
      // there is no path, query, or authority at all.
      const auto opaqueEnd = remainder.find('#');
      const auto opaquePart = remainder.substr(
          0,
          opaqueEnd == std::string_view::npos ? remainder.size() : opaqueEnd);
      if (opaquePart.empty() || !checkComponent<isUric>(opaquePart, true)) {
        return false;
      }
      remainder.remove_prefix(opaquePart.size());
      splitRef(remainder, parsed.ref);
      if (parsed.ref.has_value() &&
          !checkComponent<isUric>(*parsed.ref, true)) {
        return false;
      }
      return remainder.empty();
    }
  }

  if (!parseHierarchical(remainder, parsed)) {
    return false;
  }
  splitRef(remainder, parsed.ref);
  if (parsed.ref.has_value() && !checkComponent<isUric>(*parsed.ref, true)) {
    return false;
  }
  return remainder.empty();
}

} // namespace detail
} // namespace facebook::velox::functions::sparksql
