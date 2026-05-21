---
name: pr-review
description: Review Velox pull requests for code quality, memory safety, performance, and correctness. Use when reviewing PRs, when asked to review code changes, or when the user mentions "/pr-review".
---

# Velox PR Review Skill

Review Velox pull requests focusing on what CI cannot check: code quality, memory
safety, concurrency, performance, and correctness. This is performance-critical C++
code for a database execution engine where bugs can cause data corruption, crashes,
or security vulnerabilities.

## Usage Modes

### GitHub Actions Mode

When invoked via `/pr-review [additional context]` on a GitHub PR, the action
pre-fetches PR metadata and injects it into the prompt. Detect this mode by the
presence of `<formatted_context>`, `<pr_or_issue_body>`, and `<comments>` tags in
the prompt.

The prompt already contains:
- PR metadata (title, author, branch names, additions/deletions, file count)
- PR body/description
- All comments and review comments (with file/line references)
- List of changed files with paths and change types

Use git commands to get the diff and commit history. The base branch name is in the prompt context (look for PR Branch: <head> -> <base> or the baseBranch field).

```bash
# Get the full diff against the base branch
git diff origin/<baseBranch>...HEAD

# Get diff stats
git diff --stat origin/<baseBranch>...HEAD

# Get commit history for this PR
git log origin/<baseBranch>..HEAD --oneline

# If the base branch ref is not available, fetch it first
git fetch origin <baseBranch> --depth=1
```

Do NOT use `gh` CLI commands in this mode -- only git commands are available.
All PR metadata, comments, and reviews are already in the prompt context;
only the diff and commit log need to be fetched via git.

If the reviewer provided additional context or instructions after the `/pr-review`
command, incorporate those into your review focus.

### Local CLI Mode

The user provides a PR number or URL:

```
/pr-review 12345
/pr-review https://github.com/facebookincubator/velox/pull/12345
```

Use `gh` CLI to fetch PR data:

```bash
gh pr view <PR_NUMBER> --json title,body,author,baseRefName,headRefName,files,additions,deletions,commits
gh pr diff <PR_NUMBER>
gh pr view <PR_NUMBER> --json comments,reviews
```

## Review Workflow

### Step 1: Read Project Guidelines

**Before reviewing, you MUST Read `CODING_STYLE.md` at the repo root in
full.** Do not skim, do not skip — every modified line in the diff must be
checked against it.

The "Common Mistakes" section is the authoritative checklist for the
highest-volume real review hits (`///` vs `//`, abbreviations, `*Utils`,
undocumented headers, header-body weight, `goto`, test-first for bug
fixes, naming conventions, assert forms, etc.).

This skill does not maintain a duplicate checklist — `CODING_STYLE.md`
is the single source of truth. If anything in this skill ever appears to
contradict `CODING_STYLE.md`, prefer `CODING_STYLE.md`.

### Step 2: Analyze Changes and Prior Review

Read through the diff systematically:
1. Identify the purpose of the change from title/description
2. Group changes by type (new code, tests, config, docs)
3. Note the scope of changes (files affected, lines changed)

The `<comments>` block in the prompt context contains all prior review
comments — including any from earlier `/pr-review` invocations on this PR.
Read them before reviewing:
- Do **not** re-flag issues already raised by a prior reviewer (human or
  Claude). Re-flagging trains authors to ignore Claude reviews.
- If a prior comment was addressed by a follow-up commit, verify the fix in
  the diff rather than restating the original concern.
- If `/pr-review` was invoked in reply to a specific comment thread, focus
  the review on that thread's concerns instead of re-reviewing the whole PR.

### Step 3: Deep Review

Trace the logic step by step. For each change, consider boundary conditions
(empty, null, max size, first/last iteration), failure modes (allocation
failures, exceptions, partial state), concurrency (race conditions, lock
ordering), and memory safety (ownership, lifetimes, dangling references). Be
strict — better to flag a potential issue than miss a real bug. The Review
Areas table below enumerates what to check; do not duplicate it as narrative.

## Review Areas

Analyze each of these areas thoroughly:

| Area | Focus |
|------|-------|
| Correctness & Edge Cases | Logic errors, off-by-one, null/empty handling, boundary conditions, integer overflow, floating point edge cases (NaN, Inf, negative zero) |
| Memory Safety | Use-after-free, double-free, leaks, dangling pointers/references, buffer overflows, ownership/lifetime issues, exception safety |
| Concurrency | Race conditions, data races, deadlocks, lock ordering, thread-safety of shared state |
| Performance | Unnecessary copies (move semantics?), inefficient algorithms, cache-unfriendly access, excessive allocations in hot paths |
| Error Handling | All error paths handled? Exceptions caught appropriately? Informative error messages? Correct use of VELOX_CHECK_* vs VELOX_USER_CHECK_*? |
| Code Quality | RAII, const-correctness, smart pointers, naming conventions, clear structure |
| Testing | Sufficient tests? Edge cases covered? Error paths tested? Using gtest matchers? **Bug-fix PRs**: does the diff add a test that would fail without the fix? Flag bug fixes that ship code-only. |

## Output Format

The output should be a markdown-formatted summary and should follow the following markdown format exactly:

```markdown
### Summary
Brief overall assessment (1-2 sentences)

### Issues Found
List any issue, categorized by severity:
 - 🔴 **Critical**: Must fix before merge
 - 🟡 **Suggestion**: Should consider
 - 🟢 **Nitpick**: Minor style issues

Each issue should also include:
- File and line reference
- Description of the issue
- Suggested fix if applicable

### Positive Observations
Note any particularly good patterns or improvements.
```

## Inline Comments

Use the `mcp__github_inline_comment__create_inline_comment` tool to post
comments directly on specific lines in the PR diff. Inline comments should
be used whenever pointing at the exact line adds clarity beyond the summary
comment.

**Use inline comments for:**
- Concrete bugs or incorrect logic
- Memory safety issues (use-after-free, dangling references, leaks)
- Off-by-one errors or boundary condition mistakes
- Incorrect use of VELOX_CHECK_* vs VELOX_USER_CHECK_*

**Do NOT use inline comments for:**
- Style nitpicks or naming suggestions
- General architectural feedback
- Positive observations
- Anything that applies broadly rather than to a specific line

**Always post a summary comment** with the overall review. Inline comments
supplement the summary — they do not replace it.

## Key Principles

1. **No repetition** - Each observation appears in exactly one place
2. **Focus on what CI cannot check** - Don't comment on formatting, linting, or type errors
3. **Be specific** - Reference file paths and line numbers
4. **Be actionable** - Provide concrete suggestions, not vague concerns
5. **Be proportionate** - Minor issues shouldn't block, but note them
6. **Assume competence** - The author knows C++; explain only non-obvious context
7. **Permission to be quiet** - If the PR has no meaningful issues, post a
   short LGTM (one or two sentences) and stop. Do not manufacture nitpicks
   to fill space — padding trains authors to ignore Claude reviews. A clean
   PR getting a clean and high-signal review is the correct outcome.

## Files to Reference

When reviewing, consult these project files for context:
- `CLAUDE.md` - Coding style and project guidelines
- `CODING_STYLE.md` - Complete coding style guide
