# Self-Review Checklist for Contributors

Run this checklist before requesting review. If you use Claude or a similar
AI tool, paste this file as context and ask it to review your diff against
these rules. Most items come from recurring review feedback — fixing them
before review saves everyone time.

## Before requesting review

- [ ] CI is green.
- [ ] PR title and description are clear and free of common content issues
      (dense prose, long inline lists, jargon, function-by-function
      walkthroughs, restating the diff). The `write-commit-message` skill at
      `.claude/skills/write-commit-message/` offers a draft + self-check
      workflow — one path, not a requirement.
- [ ] PR title describes what changed (not the symptom or the ticket).
- [ ] PR description matches what the code actually does. If the scope grew
      beyond the original description, update it. Do not include test
      results or pass/fail status — CI reports that.
- [ ] For refactoring PRs, the description explains the before/after
      component responsibilities — what existed before, what exists after,
      and how they assemble. A diff that "extracts X" without explaining
      the resulting design is not ready for review.
- [ ] Bug fixes are not mixed with refactoring. If you find a bug while
      refactoring, fix it in a separate PR.
- [ ] When fixing a regression, CC the original author and reviewers of
      the PR that introduced it.
- [ ] When fixing a bug caused by a pattern (e.g., using regex to parse
      escapes), audit other usages of the same pattern and note in the PR
      whether they are safe.

## Naming

- [ ] No abbreviated variable names. Use `array`, not `arr`. Use `value`,
      not `val`. Use `count`, not `cnt`. Use `byteIndex`, not `idx`.
      Only loop indices (`i`, `j`) are acceptable.
- [ ] No numbered or lettered variables (`bitmap1`, `bitmapA`, `rows2`).
      Use descriptive names that convey meaning, or inline the value to
      avoid naming altogether.
- [ ] Parameter names are consistent within a function and across related
      functions. Don't use `arg` in one place and `input` in another for
      the same concept.

## Code

- [ ] No non-trivial implementations in headers. If a method body has more
      than one statement, it belongs in the `.cpp` file.
- [ ] No setter injection when the value could be a constructor parameter.
- [ ] No comments referencing other implementations ("like Java Presto",
      "similar to Spark's X") unless the goal is to match that engine's
      semantics. Logic should stand on its own.
- [ ] No scaffolding-style comments in committed code: don't name sibling
      functions ("X handles Y; we handle Z"), don't describe rejected
      alternative paths ("without requiring a dynamic_cast"), don't
      contrast with what the code isn't doing. These belong in the PR
      description or a diff reply.
- [ ] When changing behavior or layout, update neighboring doc comments to
      match. Don't leave stale descriptions that describe an earlier
      version of the code.
- [ ] No default arguments when all callers can pass values explicitly.
      `= {}` / `= nullptr` on parameters that every caller supplies is
      noise; pass the value or split into two helpers.
- [ ] No unnecessary type aliases (`using ConnectorConfig = ...` when used once).
- [ ] No unnecessary `static_cast` when the type already matches.
- [ ] No undefined behavior in evaluation order. Watch for
      `{std::move(x), x->method()}` — evaluation order of arguments is
      unspecified.
- [ ] Existing utility methods are used instead of inline boilerplate.
      Search for existing helpers before writing your own.
- [ ] Constants shared across files are in a common header, not duplicated.

## Error handling

- [ ] `VELOX_CHECK_*` for internal invariants, `VELOX_USER_CHECK_*` for
      user-facing input validation. Don't mix them up. For functions,
      prefer the [non-throwing error path](https://velox-lib.io/blog/optimize-try-more#non-throwing-simple-functions)
      instead — see the [Function PR Guide](FUNCTION_PR_GUIDE.md).
- [ ] Error messages match the check. `VELOX_CHECK_GE(x, 0)` should not say
      "greater than 0" — it checks `>= 0`.
- [ ] Error messages are descriptive. `VELOX_CHECK_*` macros already append
      compared values — don't repeat them in the custom message.
- [ ] Validation is consistent: if a function requires exactly N bytes,
      don't silently skip invalid inputs before the check.

## Tests

- [ ] Bug fixes: write the test first, confirm it fails without the fix,
      then apply the fix. A test that passes with and without the fix proves
      nothing.
- [ ] New tests are placed next to related existing tests, grouped by
      topic — not appended at the end of the file.
- [ ] New tests follow the file's existing assertion style. Don't switch to
      `EXPECT_THAT(..., HasSubstr(...))` when neighboring tests pin the
      full string with `ASSERT_EQ`, or to individual `EXPECT_EQ`s when
      neighbors use gtest container matchers.
- [ ] New shape or variant cases fold into an existing `TEST_F` as
      additional assertions (use `{ SCOPED_TRACE("…") … }` scope blocks
      to disambiguate). Don't introduce `xMulti` / `xVariant` /
      `xExtended` `TEST_F`s for what should be cases inside the existing
      test.
- [ ] No new test case duplicates coverage already in an adjacent test in
      the same file. Fold overlapping cases into the existing test rather
      than adding a separate one.
- [ ] No copy-pasted test blocks. If two tests differ only in one parameter,
      use a loop or a local lambda.
- [ ] Use existing test helpers (`makeArrayFromJson`, `makeBitmapVector`,
      etc.) instead of verbose manual construction.
- [ ] No duplicated test helpers across files. Extract to a shared header.
- [ ] Test helpers are designed building blocks, not mechanical extractions
      of duplication. Name after what they assert (`assertMarkers`,
      `runSpillTest`); take the natural input, not whatever happened to
      vary across the first few callers.
- [ ] `TEST()` for empty fixtures, `TEST_F()` only when the fixture is used.
- [ ] Error message assertions match the full descriptive text, not just
      the auto-generated comparison output from `VELOX_CHECK_*` macros.
- [ ] Each test file has one test suite with a matching name.
- [ ] Test comments describe what behavior the test verifies, not the bug
      history or implementation details that led to writing the test. The
      bug story belongs in the commit message and PR description.

## Documentation

- [ ] New functions and features have doc entries (e.g., in
      `docs/functions/spark/aggregate.rst`).
- [ ] No unexplained jargon. Define acronyms and domain terms.
- [ ] Check if existing doc pages need updating when behavior changes.

## API changes

- [ ] New API surface is documented in the header.
- [ ] No redundant parameters (can types be derived from richer inputs?).
- [ ] Method names are consistent with stat/counter names they update.
- [ ] Consider debuggability: can you tell what's happening from
      stats/toString output?
- [ ] If the PR touches public APIs beyond the original scope, discuss
      with the reviewer first. Do not silently expand scope.

## Re-review checklist

Before requesting re-review after addressing feedback:

- [ ] Every review comment is either addressed in code or discussed in a
      reply. Do not silently skip items.
- [ ] Run the full diff review again — don't just fix the flagged lines.
      Addressing feedback often introduces new issues.
- [ ] Check that behavioral changes (e.g., removing configuration that
      existing code relied on, changing error handling from throwing to
      returning null) are intentional and mentioned.
- [ ] If a requested change is too large for this PR, say so and propose
      a plan (e.g., separate PR). Do not silently skip it.
