# PR Review Style Guide

## Procedure

1. Fetch PR using `scripts/review/fetch.py`. This is the only fetch needed —
   work from its output for all subsequent analysis. Do not make additional
   `gh api` or `gh pr diff` calls.
2. Draft review to `~/.claude/review-drafts/pr-XXXXX-r1-v1.md`.
3. Show draft and get approval before posting.
4. Post using `scripts/review/post.py`.

## First rule

Before writing anything, ask: "Do I understand what this PR does end-to-end,
and have I verified the claims?" If the answer is no, dig deeper before
drafting. The most common mistake is focusing on code details before
understanding the change.

## Tone

- **Opening:** "Thank you for the fix/contribution!" — then straight to the
  points. Don't elaborate on what the PR does or praise specific design choices.
- **Don't restate the PR description.** The author knows what they wrote.
- **Skip "Thank you" when the PR needs fundamental clarification.** Lead with
  the question instead.
- **Be extra respectful when asking contributors to engage upstream.** They're
  volunteering their time. Suggest filing an issue first — it shows respect for
  upstream maintainers' expertise.
- **Be encouraging with new contributors.** Acknowledge the value of the
  capability they're adding.

## Structure

- Reviews should be concise and actionable. The author wants to get their work
  done — don't waste their time with fluff.
- **Order: big picture first.** Documentation, design questions, then code,
  then tests. The most impactful feedback should come first.
- **Every point must be actionable.** No observations without asks. If it
  doesn't require the author to change something, don't include it.
- **Drop qualifiers when the fix is obvious.** Don't explain why something is
  wrong if the fix is self-evident.

## Rigor

- **Verify before claiming.** Don't assert facts about the codebase, other
  projects, or behavior without checking. When in doubt, read the code.
- **Verify author claims.** When an author says an API doesn't support
  something, or references another PR/diff/external behavior, check the source.
  Don't accept at face value.
- **Check terminology.** Use precise terms. Don't conflate catalog/connector,
  function/method, etc.

## Re-reviews

When the author says "addressed comments", re-review the full diff — don't
just check the boxes from the previous round. Docs, naming, design choices,
and new code added while addressing feedback all need fresh eyes.

## What to check

### Correctness

- **PR title and description.** Are they clear, succinct, accurate? Does the
  title describe what changed (not the symptom)? Does the description match
  what the code actually does?
- **CI status.** Is CI green? If red, is it related to the PR or pre-existing?
- **Design.** Question design choices, don't just document them. Flag
  surprising behavior (e.g., auto-selection), unnecessary complexity, or
  features that could be simpler.
- **API design.** Flag anti-patterns: setter injection when constructor params
  would work, bypassing existing methods with inline boilerplate, unnecessary
  type aliases. When a PR adds API surface for an external consumer, question
  the consumer's architecture — "why does the caller need this?" matters more
  than "what's the cleanest way to expose it?"
- **Registries.** New registries should follow existing patterns (e.g.,
  query-scoped registries design in #16993).

### Code quality

- **Velox coding conventions.** Ensure code follows
  [CODING_STYLE.md](../../CODING_STYLE.md) and the rules in
  [.claude/CLAUDE.md](../../.claude/CLAUDE.md).
- **Comments.** Flag verbose code comments that restate the code, duplicate
  docs elsewhere, or explain obvious behavior. Remove references to other
  implementations ("like Java Presto") — logic should stand on its own.
- **Naming.** Check variable names, file names, class names against coding
  style conventions. Do not abbreviate parameter names.
- **Enums.** `kPascalCase` enumerators, trailing commas, `VELOX_DEFINE_ENUM_NAME`.

### Testing

- **Tests.** Are they sufficient? Do they reproduce the bug? Is there an
  integration test, not just a unit test? Are expected values hand-computed
  (fragile) or derived from the test input? Are test helpers used to reduce
  duplication? Do test names follow conventions?
- **Test files.** Each test file should have one test suite with a matching
  name. Empty test fixtures should use `TEST()` instead of `TEST_F()`.

### Documentation

- Are new functions/features documented? Are docs updated for changed behavior?
  Are README changes accurate (not removing still-valid content)?
- Check if **existing** doc pages need updating — e.g., a change to plan output
  may require updating the print-plan-with-stats page, a dependency version
  bump may require updating the dependency table.
- **When unsure about conventions**, CC the maintainer rather than guessing.
