# Velox regex compatibility test suite

A C++ test harness that compares three regex engines — Velox's existing
**RE2**, **PCRE2** (8-bit, JIT), and an embedded JVM running
**`java.util.regex`** — against the same inputs, expressed in Java regex
syntax.

The goal is to quantify how each engine handles Java-style patterns and
replacements so the Velox project can make data-driven decisions about
whether to introduce PCRE2 alongside RE2 in production, and (separately)
whether to invest in a Java → PCRE2 translator analogous to
[pcre4j PR #606](https://github.com/alexey-pelykh/pcre4j/pull/606).

This module is **opt-in** and **off by default**.  It does not affect
stock Velox builds in any way unless you enable the CMake options below.

## Enabling

```bash
cmake -S . -B build -GNinja \
  -DVELOX_ENABLE_REGEX_COMPAT_TESTS=ON \         # opt-in master switch (pulls in PCRE2)
  -DVELOX_ENABLE_REGEX_COMPAT_JAVA_BACKEND=ON    # opt-in JNI backend (requires JDK)
cmake --build build --target velox_regex_compat_test
build/velox/external/regex_compat/tests/velox_regex_compat_test
```

`VELOX_ENABLE_REGEX_COMPAT_JAVA_BACKEND` defaults to `ON`.  If
`find_package(JNI)` fails (no JDK installed) the option is silently
flipped to `OFF` and the suite still builds — only the Java backend is
disabled.

## Architecture

Three parallel, non-virtual concrete classes:

| Backend     | Implementation                                              |
| ----------- | ----------------------------------------------------------- |
| `Re2Regex`  | wraps `re2::RE2`; uses `java_pcre2_translator::toRe2Pattern` for Java pattern syntax and Velox's inline `prepareRegexpReplaceReplacement` from `Re2Functions.h` for Java replacement syntax |
| `Pcre2Regex`| wraps `pcre2_code_8`; uses `java_pcre2_translator::toPcre2Pattern`; `GlobalReplace` uses `PCRE2_SUBSTITUTE_EXTENDED` for `$N` / `${name}` |
| `JavaRegex` | drives `java.util.regex.Pattern` / `Matcher` through an embedded JVM (`JNI_CreateJavaVM`) using only standard JDK classes — no Gluten / Hadoop jars needed |

Their public methods deliberately mirror the subset of `re2::RE2` actually
used in `velox/functions/lib/Re2Functions.cpp`:

- `bool Match(input, startpos, endpos, anchor, submatch[], nsubmatch)`
- `int NumberOfCapturingGroups()`
- `const std::map<std::string, int>& NamedCapturingGroups()`
- `bool ok() / const std::string& error()`
- static `FullMatch / PartialMatch / GlobalReplace`
- `Anchor { kUnanchored, kAnchorStart, kAnchorBoth }`
- `Options { caseSensitive, dotNl, oneLine, logErrors, maxMem }`

The shared shape (plus identical method signatures) lets one
`TYPED_TEST_SUITE_P` declaration produce one test per backend at compile
time — see `tests/BackendTypedTest.cpp` and the three ported pcre4j
test files.

The stateful Java `Matcher` API (`find()` cursor, `group(i)`,
`replaceFirst`, …) lives in `tests/JavaMatcherAdapter.h` — a
template that reconstructs the state machine on top of the backend's
stateless `Match()`.  It is **test-only**; production backends do not
carry this state.

## What's tested

`velox_regex_compat_test` ships with **189 GTest cases** across 15
suites:

```
Re2RegexTest                11 cases — RE2-specific edge cases
Pcre2RegexTest              12 cases — PCRE2-specific, incl. lookahead + backref
JavaRegexTest               13 cases — Java-specific, incl. \p{InGreek}
BackendTest                 13 × 3   — core API typed across all backends
PatternPortedTest           13 × 3   — ported from pcre4j PatternTests.java
MatchingPortedTest          14 × 3   — ported from pcre4j MatcherMatchingTests.java
ReplacementPortedTest       11 × 3   — ported from pcre4j MatcherReplacementTests.java
```

A single typed test exercises both engine differences (e.g. PCRE2 supports
lookahead while RE2 doesn't) and cross-engine parity (e.g. all three
backends accept Java `(?<name>...)` named groups).

## Known cross-engine differences

| Java feature                       | Re2Regex                              | Pcre2Regex                  | JavaRegex |
| ---------------------------------- | ------------------------------------- | --------------------------- | --------- |
| `(?<name>...)` named groups        | translated via `toRe2Pattern`         | native                  | native    |
| `$N` / `${name}` in replacement    | translated via `prepareRegexpReplaceReplacement` | `PCRE2_SUBSTITUTE_EXTENDED` native | native |
| Lookaround `(?=...)`, `(?!...)`    | not supported (`ok() == false`)       | native                      | native    |
| Backreferences `\1`                | not supported                         | native                      | native    |
| Atomic groups `(?>...)`, possessive `*+` | not supported                   | native                      | native    |
| Java `\p{InGreek}` / `\p{javaXxx}` | translated where safe                  | translated where safe   | native    |
| Character-class intersection `[a-c&&b-d]` | translated where safe          | translated where safe   | native    |
| `(?U)` Java UNICODE_CHARACTER_CLASS | rejected to avoid RE2 ungreedy semantics | translated where safe | native    |
| Multiline `^`/`$`                  | injected `(?m)` prefix when `oneLine=false` | option-mapped              | option-mapped |
| `a{` incomplete quantifier         | accepted as literal                   | accepted as literal         | rejected (`PatternSyntaxException`) |

The translator rows are intentionally conservative: features are translated
only where the target engine can preserve Java semantics, and otherwise the
backend reports `ok() == false` with a translator error.

## Provenance

- `Re2Regex`, `Pcre2Regex`, `JavaRegex`, `JvmFixture` — original code,
  Apache-2.0.
- Ported test cases in `tests/Pattern…PortedTest.cpp` and
  `tests/Matcher…PortedTest.cpp` are 1:1 translations of the
  corresponding `org.pcre4j.regex.tests.*` Java tests from
  [pcre4j](https://github.com/alexey-pelykh/pcre4j).  The upstream Java
  code is GPL/LGPL; the C++ port re-implements them in Apache-2.0 form
  for the Velox project.

## What's **not** in this module (scope notes)

- **No production code change.**  This module sits under
  `velox/external/regex_compat/` precisely because it is a comparison
  experiment, not a Velox engine swap.  If/when a production decision
  is made the backend classes can be lifted to `velox/functions/lib/`
  but that is a separate task.
- **No production regex engine replacement.**  The Java regex translator is
  wired into this comparison suite's RE2 and PCRE2 backends to measure
  compatibility, not to change Velox production regex behavior.
- **No QueryConfig runtime switch.**  Whether Velox should expose
  RE2/PCRE2/Java as a runtime-selectable engine is a downstream
  decision; the backend classes here all happen to be method-compatible,
  but they are not unified behind a virtual base or `std::variant`
  facade.
