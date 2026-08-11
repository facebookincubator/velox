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
//
// Originally authored by Oleksii PELYKH for pcre4j; ported from
// org.pcre4j.regex.translate.PropertyMap (Java) under Apache-2.0 by the
// same author for inclusion in Velox.
//
#pragma once

#include <optional>
#include <string>
#include <string_view>

namespace facebook::velox::functions::java_pcre2_translator {

/// Maps Java regex property names (as used in `\p{...}`) to PCRE2
/// equivalents.
///
/// Return convention for `apply(name)`:
///   * A bare name like `"Greek"` → caller emits `\p{Greek}` / `\P{Greek}`.
///   * A string starting with `'['` → caller substitutes the entire
///     `\p{name}` token with this string (used for expanded ranges and
///     multi-class expressions).
///   * `std::nullopt` → no rewrite; leave the token as-is.
class PropertyMap {
 public:
  static constexpr std::string_view kNeverMatch{"\x01NEVER_MATCH\x01"};

  /// Resolves a Java regex property name to a PCRE2 equivalent.  Returns
  /// `std::nullopt` when no rewrite is needed (the caller should pass the
  /// token through unchanged).
  static std::optional<std::string> apply(std::string_view name);

 private:
  PropertyMap() = delete;
};

} // namespace facebook::velox::functions::java_pcre2_translator
