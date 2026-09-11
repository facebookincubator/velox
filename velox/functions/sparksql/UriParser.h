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

#include <optional>
#include <string_view>

/// Parses URLs with the java.net.URI (JDK 8) grammar that Spark's
/// parse_url is defined against. See detail::parseUrl for the supported
/// grammar and its lenient deviations from RFC 3986.

namespace facebook::velox::functions::sparksql {
namespace detail {

/// The parts of a URL that Spark's parse_url can extract. Every field is
/// a view into the parsed input, so parsing allocates nothing. PROTOCOL,
/// HOST and PATH use the convention that data() == nullptr means the
/// component is absent (and thus null in Spark); an empty view means the
/// component is present but empty. The remaining components distinguish
/// the two with std::optional because both states are observable.
struct ParsedUrl {
  std::string_view protocol;
  std::string_view host;
  std::string_view path;
  std::optional<std::string_view> query;
  std::optional<std::string_view> ref;
  std::optional<std::string_view> authority;
  std::optional<std::string_view> userInfo;
};

/// Parses a complete URL into its components. The grammar follows the
/// java.net.URI (JDK 8) rules Spark's parse_url is defined against,
/// including these lenient deviations from RFC 3986:
/// - non-ASCII characters are accepted outside the authority;
/// - '[' and ']' are allowed in queries;
/// - scheme-less relative references and the empty string are valid;
/// - a '#' inside the fragment is rejected;
/// - authorities that are not server-based fall back to registry form.
/// Returns false when the URL is invalid. Error details are dropped:
/// Spark's ParseUrl maps every failure to null.
bool parseUrl(std::string_view url, ParsedUrl& parsed);

} // namespace detail
} // namespace facebook::velox::functions::sparksql
