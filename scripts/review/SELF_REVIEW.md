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
- [ ] No copy-pasted test blocks. If two tests differ only in one parameter,
      use a loop or a local lambda.
- [ ] Use existing test helpers (`makeArrayFromJson`, `makeBitmapVector`,
      etc.) instead of verbose manual construction.
- [ ] No duplicated test helpers across files. Extract to a shared header.
- [ ] `TEST()` for empty fixtures, `TEST_F()` only when the fixture is used.
- [ ] Error message assertions match the full descriptive text, not just
      the auto-generated comparison output from `VELOX_CHECK_*` macros.
- [ ] Each test file has one test suite with a matching name.

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
