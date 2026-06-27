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
// org.pcre4j.regex.translate.JavaRegexTranslator (Java) under
// Apache-2.0 by the same author for inclusion in Velox.
//
// This header is the public surface of the `java_pcre2_translator`
// library.  It declares free functions that rewrite a `java.util.regex`
// pattern string into an equivalent pattern accepted by either PCRE2 or
// RE2.
//
#pragma once

#include <string>
#include <string_view>

#include "velox/functions/lib/java_pcre2_translator/EvaluationFailedException.h"

namespace facebook::velox::functions::java_pcre2_translator {

/// Rewrites a `java.util.regex.Pattern` source string into an equivalent
/// pattern accepted by PCRE2.  Implements the 3-phase pipeline described
/// in pcre4j PR #606:
///
///   1. Expand top-level `\p{...}` / `\P{...}` property tokens via the
///      Java property → Unicode block alias map.
///   2. Re-parse each character-class body, flatten nested unions, resolve
///      `&&` intersections via range-set algebra, and escape `-` after
///      multi-char escapes to disambiguate from the range operator.
///   3. Rewrite Java inline flag groups whose semantics diverge in PCRE2
///      (notably `(?U)` which means UNICODE_CHARACTER_CLASS in Java but
///      "ungreedy" in PCRE2).
///
/// Throws `EvaluationFailedException` when the input cannot be safely
/// expressed in PCRE2 syntax (e.g. a property name with no PCRE2
/// equivalent).  Callers are expected to surface the message verbatim.
///
std::string toPcre2Pattern(std::string_view javaPattern);

/// Rewrites a Java pattern and reports whether the resulting PCRE2 compile
/// must omit PCRE2_UTF to allow lone surrogate code units.
std::string toPcre2Pattern(
    std::string_view javaPattern,
    bool& needsRawByteMode);

/// Rewrites a Java pattern for PCRE2 and pre-expands cased literal code points
/// for Java's CASE_INSENSITIVE | UNICODE_CASE semantics.  This is intentionally
/// limited to literals outside character classes and outside \Q...\E quotes.
std::string toPcre2PatternWithUnicodeCase(
    std::string_view javaPattern,
    bool& needsRawByteMode);

std::string toPcre2PatternWithUnicodeCase(std::string_view javaPattern);

/// Rewrites a `java.util.regex.Pattern` source string into an equivalent
/// pattern accepted by RE2.
///
/// This shares the PCRE2 property and character-class translation pipeline,
/// rewrites Java named groups `(?<name>...)` to RE2 `(?P<name>...)`, and
/// rejects Java features that RE2 cannot represent without changing
/// semantics.
std::string toRe2Pattern(std::string_view javaPattern);

std::string toRe2PatternWithUnicodeCase(std::string_view javaPattern);

} // namespace facebook::velox::functions::java_pcre2_translator
