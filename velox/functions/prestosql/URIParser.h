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
#include <optional>
#include "velox/type/StringView.h"

namespace facebook::velox::functions {
/// A struct containing the parts of the URI that were extracted during parsing.
/// If the field was not found, it is empty.
///
/// For fields that can contain percent-encoded characters, the `...HasEncoded`
/// flag indicates whether the field contains any percent-encoded characters.
struct URI {
  StringView scheme;
  StringView path;
  bool pathHasEncoded = false;
  StringView query;
  bool queryHasEncoded = false;
  StringView fragment;
  bool fragmentHasEncoded = false;
  StringView host;
  bool hostHasEncoded = false;
  StringView port;
};

/// Parse a URI string into a URI struct according to RFC 3986.
bool parseUri(const StringView& uriStr, URI& uri);

/// If the string starting at str is a valid IPv6 address, returns true and pos
/// is updated to the first character after the IP address. Otherwise returns
/// false and pos is unchanged.
bool tryConsumeIPV6Address(const char* str, const size_t len, int32_t& pos);

/// Find and extract the value for the parameter with key `param` from the query
/// portion of a URI `query`. `query` should already be decoded if necessary.
template <typename TString>
std::optional<StringView> extractParameter(
    const StringView& query,
    const TString& param) {
  if (!query.empty()) {
    // Parse query string using RE2.
    static const RE2 kQueryParamRegex(
        "(^|&)" // start of query or start of parameter "&"
        "([^=&]*)=?" // parameter name and "=" if value is expected
        "([^&]*)" // parameter value (allows "=" to appear)
    );

    re2::StringPiece input(query.data(), query.size());
    re2::StringPiece matches[4]; // Group 0 (full match) + 3 capturing groups
    size_t pos = 0;

    while (pos < input.size() &&
           kQueryParamRegex.Match(
               input, pos, input.size(), RE2::UNANCHORED, matches, 4)) {
      // Check if key (group 2) is not empty and matched
      if (matches[2].size() > 0) {
        StringView key(matches[2].data(), matches[2].size());
        if (param.compare(key) == 0) {
          // Return the value (group 3)
          return std::optional<StringView>(
              StringView(matches[3].data(), matches[3].size()));
        }
      }

      // Move past this match to continue searching
      pos = matches[0].end() - input.data();
      if (pos == matches[0].data() - input.data()) {
        // Avoid infinite loop on zero-width matches
        ++pos;
      }
    }
  }
  return std::nullopt;
}

} // namespace facebook::velox::functions
