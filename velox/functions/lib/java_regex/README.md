# Java regex → RE2 translator

A dependency-free, source-to-source translator that rewrites `java.util.regex`
patterns into equivalent RE2 patterns.

RE2 implements a strict subset of `java.util.regex`. Front-ends that must
reproduce Java semantics — notably Spark — therefore reject patterns RE2 cannot
compile, and in a small number of cases evaluate a pattern that *both* engines
accept but interpret differently.

This library closes the second category and part of the first: the Java/RE2
differences that RE2 **can** represent natively, and simply spells differently.
It does not introduce a second regex engine, and it adds no new dependency (in
particular, no ICU).

## Design invariant: fail closed

Anything that cannot be proven RE2-equivalent is copied **verbatim**, so RE2's
own compiler rejects it and the caller takes its existing unsupported-pattern
path.

The translator can therefore only ever *widen* the set of accepted patterns. It
never narrows it, and it never changes the meaning of a pattern RE2 already
accepted: a pattern RE2 compiles today comes back byte-identical. This is
asserted, not just documented, by
`JavaRegexCoverageTest.translationIsWideningOnly`.

## What it rewrites

| Java | RE2 output | Why |
|---|---|---|
| `\uXXXX` | `\x{XXXX}` | RE2 has no `\u` escape |
| `\uD83D\uDE00` | `\x{1F600}` | Java treats a UTF-16 surrogate pair as one character |
| `(?<name>…)` | `(?P<name>…)` | Same feature, different spelling |
| `\p{InX}`, `\p{block=X}`, `\p{blk=X}` | `[\x{lo}-\x{hi}]` | RE2 has no Unicode *blocks*; expanded from a static table |
| `\p{IsScript}`, `\p{IsCategory}` | `\p{Script}`, `\p{Category}` | Java alternate spelling |
| `\p{ASCII}`, `\p{L1}` | `[\x{0}-\x{7F}]`, `[\x{0}-\x{FF}]` | Not RE2 built-ins |
| `\p{LC}` | `[\p{Lu}\p{Ll}\p{Lt}]` | Not an RE2 built-in |

All rewrites are escape-aware and character-class-aware: `\\u0041` (an escaped
backslash followed by a literal `u`) is left alone, and `\Q…\E` literal quoting
and POSIX classes (`[:alpha:]`) are copied verbatim so their contents cannot
perturb the scanner.

### Why no semantic fixes

There are Java/RE2 differences in what an *already-compilable* pattern matches —
most notably Java's `\s` matching U+000B (vertical tab) where RE2's does not.
Those are deliberately out of scope here.

The reason is integration, not difficulty. A semantic fix has to be applied to
every regex function at once, or the same escape means different things in
different functions within one query: `regexp_replace` would treat U+000B as
whitespace while `rlike` and `split` would not. Widening rewrites have no such
problem, because a function that has not adopted the translator merely falls
back more often, which is always correct.

So this change is scoped to widening only, and semantic parity is left to a
follow-up that can land across the whole function set together.

## What it deliberately does not handle

Left untouched so RE2 rejects them and the caller falls back:

* lookaround, backreferences, atomic and possessive groups — not expressible in
  a finite automaton, so no translation exists;
* character-class intersection (`[a-z&&[^aeiou]]`) — needs class subtraction,
  which RE2 does not support;
* properties needing Unicode character data beyond block ranges, e.g.
  `\p{javaLowerCase}` — would require an ICU dependency;
* `(?u)` Unicode-aware case folding.

`(?m)` is **not** injected, on purpose: RE2's default `^`/`$` already match
Java's default (non-`MULTILINE`) behaviour, so injecting it would be a silent
semantic regression rather than a fix.

Lookbehind (`(?<=`, `(?<!`) shares the `(?<` prefix with named groups but is not
one; it is left verbatim so RE2 rejects it, rather than being rewritten into a
malformed group name.

The named-group rewrite is a scanner, so it skips escapes, character classes
and `\Q...\E` quoted runs. This is a **bug fix relative to the legacy**
`prepareRegexpReplacePattern()`, which used
`RE2::GlobalReplace("[(][?]<([^>]*)>")` and therefore rewrote `(?<` sequences
that are literal text:

| Pattern | Legacy result | Correct (this library) |
| --- | --- | --- |
| `\Q(?<g>\E` | `\Q(?P<g>\E` | `\Q(?<g>\E` |
| `[(?<g>]` | `[(?P<g>]` | `[(?<g>]` |
| `\(?<g>` | `\(?P<g>` | `\(?<g>` |

RE2 compiles the corrupted spellings, so the legacy path silently matched the
wrong text instead of raising an error. Pinned by
`namedGroupRewriteFixesLegacyCorruption`.

Block-name resolution uses exact match with a normalized-prefix fallback rather
than Java's exact `UnicodeBlock` alias table; unknown names fall through to the
verbatim path.

## Testing strategy

`tests/JavaRegexTranslatorTest.cpp` covers three levels:

1. **Transformation tests** — exact input/output assertions per rewrite rule,
   including the escape-awareness and character-class edge cases, and negative
   cases asserting that out-of-scope constructs are returned unchanged.
2. **Coverage measurement** (`JavaRegexCoverageTest.offloadDeltaOverCorpus`) —
   compiles a feature-tagged corpus with real RE2 before and after translation
   and reports the per-feature offloadability delta. Controls in the corpus
   (already-supported patterns, and RE2-irreducible ones such as lookaround and
   backreferences) must not change, which pins the fail-closed invariant.
3. **Widening-only invariant** (`translationIsWideningOnly`) — asserts that
   every corpus pattern RE2 accepts today is returned byte-identical, so no
   currently-offloaded query can change its results.

These are pure unit tests: no JVM, no cluster, and no Velox runtime.

Over the in-tree corpus, translation raises the number of patterns RE2 can
compile from 12/40 to 35/40, with the `lookaround` and `backref` controls
staying at zero — the fail-closed invariant is asserted, not just documented.

The same measurement over the OpenJDK `java.util.regex` `TestCases.txt` corpus
(305 patterns) gives 69.2% -> 82.0%, +39 patterns, zero regressions. That file
is GPL-licensed and so is deliberately **not** vendored here; the in-tree corpus
is a representative subset that keeps the number reproducible from a checkout.

## Integration

`translateJavaRegexToRe2()` is the entry point and applies the stages in the
required order: named groups → property expansion → syntax normalization.

It is currently applied unconditionally in the Spark function set
(`velox/functions/sparksql/RegexFunctions.cpp`), which previously performed the
named-group rewrite only. The translator is Velox-agnostic — it takes and
returns `std::string` — so other front-ends can adopt it independently. Presto
paths are unchanged.

Because the translator is widening-only and fails closed, applying it
unconditionally is safe and needs no configuration flag: a pattern RE2 rejected
before is either rewritten into something RE2 accepts, or left byte-identical,
and a pattern RE2 already accepted is always left byte-identical. That is also
why it does not have to be adopted by every function at once — a function that
has not adopted it simply falls back more often.
