---
name: pr-review
description: Review Velox pull requests for code quality, memory safety, performance, and correctness. Use when reviewing PRs, when asked to review code changes, or when the user mentions "/pr-review".
---

# Velox PR Review Skill

Orchestrate Velox PR reviews. This skill is the single entry point for
generating reviews — both AI-driven (this skill running locally or in
CI) and AI-assisted (a maintainer iterating on a draft before posting).

The skill is intentionally thin on review *content*. Review style,
rigor, tone, and what to check live in human-readable guides under
`scripts/review/`. The skill loads those guides and walks the stages
below.

## Required reading

Before drafting any review:

- `CODING_STYLE.md` (repo root) — every modified line is checked
  against it. The `Common Mistakes` section is the authoritative
  checklist of the highest-volume real review hits.
- `scripts/review/REVIEW_GUIDE.md` — review style, tone, rigor,
  re-reviews, refactoring rules, and what to check. This guide is
  the source of truth for *how* to review.
- `scripts/review/FUNCTION_PR_GUIDE.md` — only when the diff touches
  `velox/functions/` (see Stage 2 for the load rule and subtree
  guard).

If anything in this skill ever contradicts `CODING_STYLE.md` or
`REVIEW_GUIDE.md`, prefer those guides — the skill orchestrates, the
guides decide.

## Stages

The review flow is a six-stage pipeline. Each stage is the same
across all invocation modes; only Stage 1 (fetch transport) and
Stages 4-6 (draft / approval gate / post transport) differ between
local CLI and CI.

### Stage 0: Mode detection

Three modes:

- **GitHub Actions tag mode** — invoked via `@claude /pr-review` on a
  PR comment. Detect by the presence of `<formatted_context>`,
  `<pr_or_issue_body>`, and `<comments>` tags in the prompt. The
  action pre-fetches PR metadata, body, comments, reviews, and
  changed-file list.
- **GitHub Actions `workflow_dispatch` mode** — invoked manually
  via the Actions UI. PR code is checked out at the merge ref; use
  `gh pr view`, `gh pr diff`, `gh pr view --json comments,reviews`
  to fetch data.
- **Local CLI mode** — invoked via `/pr-review <PR number or URL>`
  from a developer machine. Use `scripts/review/fetch.py` and
  `scripts/review/post.py` (not `gh` directly — the scripts handle
  pagination, error formatting, and the draft path convention).

### Stage 1: Fetch

**Local CLI mode:**

```bash
python3 scripts/review/fetch.py <PR_NUMBER>
# or
python3 scripts/review/fetch.py <github-pr-url>
```

This is the only fetch needed — work from its output for all
subsequent analysis. Do not make additional `gh api` calls.

**GitHub Actions tag mode:** PR metadata is already in the prompt
context. Get the diff and commit history via git:

```bash
git diff origin/<baseBranch>...HEAD
git diff --stat origin/<baseBranch>...HEAD
git log origin/<baseBranch>..HEAD --oneline
# If the base branch ref is not available locally:
git fetch origin <baseBranch> --depth=1
```

Do NOT use `gh` CLI in tag mode — only git commands are available.
All PR metadata, comments, and reviews are already injected.

**`workflow_dispatch` mode:**

```bash
gh pr view <PR_NUMBER> --json title,body,author,baseRefName,headRefName,files,additions,deletions,commits
gh pr diff <PR_NUMBER>
gh pr view <PR_NUMBER> --json comments,reviews
```

### Stage 2: Load content (one-time, conditional)

Always load `CODING_STYLE.md` and `REVIEW_GUIDE.md`.

**Conditional: `FUNCTION_PR_GUIDE.md`.** Load the guide when the diff
touches any file under `velox/functions/`. After loading, apply a
**subtree guard**: if every changed file under `velox/functions/` is
in one of the non-function subtrees below, skip *applying* the
checklist — the guide's questions (`.rst` doc entries, registration
prefixes, `SimpleFunction` API) won't fit:

- `velox/functions/*/tests/`
- `velox/functions/*/benchmarks/`
- `velox/functions/*/fuzzer/`
- `velox/functions/*/coverage/`
- `velox/functions/remote/server/` (RPC infrastructure)
- Build files (`CMakeLists.txt`)

If the diff touches both function code *and* one of these subtrees,
apply the checklist to the function code portion only.

**Conditional: prior `/pr-review` comments.** Parse the `Comments`
and `Reviews` sections of the fetch output to identify issues
already raised by a prior reviewer (human or Claude). Do not
re-flag those — re-flagging trains authors to ignore Claude
reviews. If a prior comment was addressed by a follow-up commit,
verify the fix in the diff rather than restating the original
concern.

**Important:** `scripts/review/fetch.py` currently exits hard on
metadata-fetch failure but returns empty silently on comment-fetch
or review-fetch failures. Before concluding "no prior reviews,"
check that the fetch output contains a `Comments` and/or `Reviews`
section header. If the headers are absent, treat it as "fetch did
not return comment data" — ask the user to re-run `fetch.py` or
check `gh auth status`. Don't assume silence means none exist.

If `/pr-review` was invoked in reply to a specific comment thread,
focus the review on that thread's concerns instead of re-reviewing
the whole PR.

### Stage 3: Analyze

Apply the rigor, structure, and tone rules from `REVIEW_GUIDE.md`.
That guide enumerates what to check (correctness, memory safety,
concurrency, performance, error handling, code quality, testing,
documentation) — the skill does not duplicate it here.

Also walk the PR title and body through the self-check in
`.claude/skills/write-commit-message/SKILL.md`. If **2 or more**
items fail (long lists embedded in sentences, function-by-function
walkthroughs, restating the diff, jargon nouns, long inline
code/error strings), include a single short note in the summary
comment asking the author to tighten the description:

```
The PR description would read more clearly with a rewrite. Specifically:
- <issue 1, one short sentence>
- <issue 2, one short sentence>

The `write-commit-message` skill at `.claude/skills/write-commit-message/`
can help (it has a self-check + drafting workflow), but any path is fine
as long as the result is clear.
```

One short paragraph in the summary — don't file an inline comment
per issue, and don't nag on 0-1 minor issues.

**Huge diffs.** For diffs over ~3000 changed lines, fall back to
file-by-file `Read` rather than holding the full diff in the prompt
at once. Note in the draft that coverage was per-file so the
maintainer knows.

### Stage 4: Draft

**Local CLI mode:** Write the draft to
`~/.claude/review-drafts/pr-NNNNN-rN-vN.md`. Auto-create the
directory if missing.

Filename rule:

- `rN` = (number of prior `/pr-review`-authored comments on this PR
  detected in the fetch output) + 1. Round 1 means no prior bot
  reviews; round 2 means one prior round, etc.
- `vN` = (number of existing drafts on disk matching
  `pr-NNNNN-rN-*.md` for the current `rN`) + 1. First draft of the
  round is `v1`; the next iteration after maintainer feedback is
  `v2`.

If two shells race on the same PR & round, last writer wins —
accepted risk, low likelihood.

**CI modes:** Build the comment body in memory. No draft file.

### Stage 5: Approval gate

**Local CLI mode:** Show the maintainer the draft path and wait.

- `"post"` → proceed to Stage 6.
- `"iterate"` (with feedback) → bump `vN`, redraft, return to
  Stage 5.
- Unclear feedback → ask **one** clarifying question. If the answer
  is still unclear, redraft using your best interpretation and
  surface the assumption explicitly in your response ("I
  interpreted X as Y; tell me if you wanted Z instead"). Never
  loop silently asking question after question.

**CI modes:** Skip — the bot posts directly.

The CI bot loses the human catch for hallucinated file paths and
stale comments-on-issues-already-raised. Stage 2's prior-comment
parsing is the bot's only defense, which is why the fetch-header
check in Stage 2 matters more in CI than locally.

### Stage 6: Post

**Local CLI mode:** Before invoking `post.py`, verify the draft file
exists and is non-empty. If missing or empty, surface a clear
message and offer to redraft — do NOT invoke `post.py` on a
missing/empty file (it would crash with an unfriendly
`FileNotFoundError`).

```bash
python3 scripts/review/post.py <PR> <event> <body-file>
# Events: APPROVE, REQUEST_CHANGES, COMMENT
```

Post inline comments separately via `mcp__github_inline_comment__create_inline_comment`
(see Inline Comments below). `post.py` posts the summary body only.

**GitHub Actions modes:** Post via `gh pr comment` for the summary
and `mcp__github_inline_comment__create_inline_comment` for inline
comments.

## Output Format

The summary comment is markdown formatted as follows. Within the
`Issues Found` section, order points big-picture first per
`REVIEW_GUIDE.md` (documentation, design questions, code, tests).

```markdown
### Summary
Brief overall assessment (1-2 sentences).

### Issues Found
List issues, categorized by severity:
 - 🔴 **Critical**: Must fix before merge.
 - 🟡 **Suggestion**: Should consider.
 - 🟢 **Nitpick**: Minor style issues.

Each issue should include:
- File and line reference.
- Description of the issue.
- Suggested fix if applicable.

### Positive Observations
Optional. Only if there's something specific worth calling out.
```

## Inline Comments

Use `mcp__github_inline_comment__create_inline_comment` to post
comments on specific lines in the PR diff. Inline comments should
be used whenever pointing at the exact line adds clarity beyond
the summary comment.

**Use inline comments for:**

- Concrete bugs or incorrect logic.
- Memory safety issues (use-after-free, dangling references, leaks).
- Off-by-one errors or boundary condition mistakes.
- Incorrect use of `VELOX_CHECK_*` vs. `VELOX_USER_CHECK_*`.

**Do NOT use inline comments for:**

- Style nitpicks or naming suggestions.
- General architectural feedback.
- Positive observations.
- Anything that applies broadly rather than to a specific line.

**No repetition.** Each observation appears in exactly one place —
either inline or in the summary, never both. Inline comments
supplement the summary; they do not duplicate it. Always post a
summary comment with the overall review.

## Files to reference

When reviewing, consult these project files for context:

- `CODING_STYLE.md` — complete coding style guide.
- `.claude/CLAUDE.md` — project-level rules and review scripts.
- `scripts/review/REVIEW_GUIDE.md` — review style and rigor.
- `scripts/review/FUNCTION_PR_GUIDE.md` — function-PR checklist.
- `scripts/review/SELF_REVIEW.md` — contributor pre-review checklist
  (useful context for what the author was expected to run before
  requesting review).
