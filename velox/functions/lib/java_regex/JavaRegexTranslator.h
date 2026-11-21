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

#include <string>

/// Java (java.util.regex) -> RE2 pattern translator.
///
/// RE2 implements a strict subset of java.util.regex syntax. Engines that must
/// reproduce Java regex semantics -- notably the Spark function set -- either
/// reject patterns RE2 cannot compile, or silently differ from Java where the
/// two engines spell the same concept differently.
///
/// This library closes the second category: the Java/RE2 differences that RE2
/// *can* represent natively, just with different syntax. It performs a pure
/// source-to-source rewrite of the pattern string. It adds no new runtime
/// dependency (in particular, no ICU and no second regex engine) and it does
/// not change which engine evaluates the pattern.
///
/// Design invariant: anything that cannot be proven RE2-equivalent is left
/// verbatim, so RE2's own compiler rejects it and the caller takes its normal
/// unsupported-pattern path. The translator therefore fails closed -- it can
/// widen the set of accepted patterns, never narrow it, and never silently
/// changes the meaning of a pattern RE2 already accepted.
///
/// Deliberately out of scope (left untouched, RE2 rejects them):
///   * lookaround, backreferences, atomic and possessive groups -- not
///     expressible in a finite automaton;
///   * character-class intersection ('[a-z&&[^aeiou]]') -- requires class
///     subtraction, which RE2 does not support;
///   * lookbehind ('(?<=', '(?<!'), which shares the '(?<' prefix with named
///     groups but is left verbatim rather than rewritten;
///   * properties requiring Unicode character data beyond block ranges, such
///     as '\p{javaLowerCase}' -- would require an ICU dependency;
///   * '(?u)' Unicode-aware case folding.
///
/// Not implemented on purpose: '(?m)' is *not* injected. RE2's default '^'/'$'
/// already match Java's default (non-MULTILINE) behaviour, so injecting it
/// would be a silent semantic regression rather than a fix.
namespace facebook::velox::functions {

/// Rewrites a java.util.regex pattern into an equivalent RE2 pattern for the
/// subset of Java/RE2 differences that RE2 can represent natively. This
/// improves Spark parity without changing the regex engine and without any
/// Unicode-data (ICU) dependency. Constructs RE2 cannot represent
/// (lookaround, backreferences, atomic/possessive groups) are intentionally
/// left untouched: RE2's compiler rejects them and callers fall back.
///
/// Rewrites performed (escape- and character-class-aware):
///   \uXXXX              -> \x{XXXX}         RE2 has no '\u' escape.
///   \uD83D\uDE00        -> \x{1F600}        A UTF-16 high/low surrogate pair
///                                           is folded into the single code
///                                           point it encodes (Java treats the
///                                           pair as one character).
/// '\Q...\E' literal quoting and POSIX classes ('[:alpha:]') are copied
/// verbatim so their contents do not perturb the scanner.
std::string normalizeJavaRegexForRe2(const std::string& pattern);

/// Expands java.util.regex Unicode *property* escapes into RE2-expressible
/// equivalents, ICU-free. RE2 already supports Unicode scripts ('\p{Greek}')
/// and general categories ('\p{L}', '\p{Lu}') natively, so the only real gaps
/// are Java's alternate spellings and Unicode *blocks* (which RE2 lacks).
/// Rewrites performed (escape- and character-class-aware):
///   \p{InX}, \p{block=X}, \p{blk=X}  -> [\x{lo}-\x{hi}]  (static block table)
///   \p{IsScript}, \p{IsCategory}     -> \p{Script}/\p{Category} (drop 'Is')
///   \p{ASCII}, \p{IsASCII}           -> [\x{0}-\x{7F}]
///   \p{L1}                           -> [\x{0}-\x{FF}]
///   \p{LC}, \p{IsLC}                 -> [\p{Lu}\p{Ll}\p{Lt}]
/// A leading '^' inside the braces (\p{^InX}) toggles negation. Properties that
/// cannot be safely expressed (e.g. a negated form inside a character class, or
/// unknown names such as \p{javaLowerCase} that need ICU) are left untouched so
/// RE2 rejects them and the caller falls back to Spark. RE2-native forms
/// (\p{L}, \p{Greek}) are deliberately left as-is. Genuinely ICU-dependent
/// features (class intersection '&&', '(?u)' case folding) are out of scope.
std::string expandJavaUnicodePropertiesForRe2(const std::string& pattern);

/// Rewrites Java named capturing groups into RE2 spelling:
///   (?<name>...)  ->  (?P<name>...)
/// Both engines support named groups, they simply spell them differently.
///
/// This is a scanner, not a text substitution: it skips escaped characters,
/// character classes and '\Q...\E' quoted runs, so a '(?<' that is literal
/// text is left alone. That is a behaviour fix relative to the legacy
/// prepareRegexpReplacePattern(), whose
/// RE2::GlobalReplace("[(][?]<([^>]*)>") rewrites those literals and silently
/// changes what the pattern matches:
///   "\Q(?<g>\E" -> "\Q(?P<g>\E"   (matched "(?<g>", now matches "(?P<g>")
///   "[(?<g>]"   -> "[(?P<g>]"     (class members change)
///   "\(?<g>"    -> "\(?P<g>"      (escaped paren is literal)
/// RE2 accepts the corrupted spellings, so the legacy path produced wrong
/// results rather than an error.
std::string rewriteJavaNamedGroups(const std::string& pattern);

/// Convenience composition applying the full Java -> RE2 translation, in the
/// order the stages must run: named groups first, then Unicode property
/// expansion (which can emit '\x{...}' ranges), then general syntax
/// normalization.
///
/// This is the entry point callers should use. The translation is
/// widening-only: a pattern RE2 already accepts is returned byte-identical, so
/// no query that offloads today can change its results. Everything rewritten
/// here is a construct RE2 rejects, which today forces a fallback.
///
/// Not in scope: differences in what an already-compilable pattern *matches*,
/// such as Java's '\s' including U+000B (vertical tab) where RE2's does not.
/// Correcting those is a semantic change that must be applied to every regex
/// function at once, or the same escape would mean different things in
/// different functions in one query.
std::string translateJavaRegexToRe2(const std::string& pattern);

} // namespace facebook::velox::functions
