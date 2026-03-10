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

Read CLAUDE.md and CODING_STYLE.md for project-specific standards including:
- Naming conventions (PascalCase for types, camelCase for functions)
- Assert/CHECK usage (VELOX_CHECK_* vs VELOX_USER_CHECK_*)
- Variable conventions (uniform initialization, std::string_view)
- API design principles
- Test patterns (gtest matchers)

### Step 2: Analyze Changes

Read through the diff systematically:
1. Identify the purpose of the change from title/description
2. Group changes by type (new code, tests, config, docs)
3. Note the scope of changes (files affected, lines changed)

### Step 3: Deep Review

**Think deeply and carefully about this code.** Take your time to:
- Trace through the logic step by step
- Consider what happens at boundary conditions (empty inputs, null values, max sizes)
- Think about concurrency issues if multiple threads could access this code
- Consider memory safety: ownership, lifetimes, dangling references
- Look for off-by-one errors, integer overflow, and other subtle bugs
- Examine error handling paths - what happens when things fail?
- Consider how this code interacts with existing code in the codebase

**Be thorough and strict.** It's better to flag a potential issue that turns out
to be fine than to miss a real bug.

**Explore edge cases exhaustively:**
- What if the input is empty? Null? Maximum size?
- What if allocation fails? What if an exception is thrown?
- What happens on the first iteration? The last iteration?
- Are there race conditions if called concurrently?
- What assumptions does this code make? Are they documented and validated?

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
| Testing | Sufficient tests? Edge cases covered? Error paths tested? Using gtest matchers? |

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

## Key Principles

1. **No repetition** - Each observation appears in exactly one place
2. **Focus on what CI cannot check** - Don't comment on formatting, linting, or type errors
3. **Be specific** - Reference file paths and line numbers
4. **Be actionable** - Provide concrete suggestions, not vague concerns
5. **Be proportionate** - Minor issues shouldn't block, but note them
6. **Assume competence** - The author knows C++; explain only non-obvious context

## Files to Reference

When reviewing, consult these project files for context:
- `CLAUDE.md` - Coding style and project guidelines
- `CODING_STYLE.md` - Complete coding style guide
