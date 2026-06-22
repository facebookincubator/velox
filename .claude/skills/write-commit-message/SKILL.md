---
name: write-commit-message
description: Draft a commit message for a Velox commit. Use when the user asks to write, draft, or compose a commit message for a Velox change. Encodes the project's content rules so the draft is showable without a separate review pass.
---

# Write Commit Message

Drafts a commit message that follows the rules in `CODING_STYLE.md` and `CLAUDE.md` at the repo root. The rules there are authoritative — this skill is the workflow for applying them.

## Process

1. **Read the rules** — Open `CODING_STYLE.md` and `CLAUDE.md` and re-read the commit-message sections. Do not draft from memory.

2. **Gather facts from the diff itself** — Never draft from the conversation alone. The conversation has scaffolding; the diff is the truth.
   - Read the existing commit message (if amending).
   - Read the full diff, not just file names.
   - List the files touched.
   - If the change fixes a bug, identify the user-visible symptom (error string, wrong output, crash) from the diff or the conversation. If you cannot state the symptom concretely, ask the user before drafting.

3. **Draft** — Match length to the change. Each paragraph is one long line (no hard wraps). The right shape depends on what the diff actually does:

   - **Trivial** (typo fix, comment edit, one-line rename, dependency bump): title alone, or title + one sentence + one-line test plan. Padding a trivial change with three paragraphs is a fail.
   - **Small** (a focused bug fix, a single-file refactor, a small new helper): title + one paragraph (what + why with a concrete anchor) + one-line test plan.
   - **Standard** (most fixes and features): title + 2-4 short paragraphs as below.
   - **Large**: if you find yourself wanting >4 body paragraphs, the change should probably be split — or the extra material belongs in separate documentation (design doc, issue, wiki page) that the summary links to, not inlined in the commit message.

   **Title**: `[velox] type(scope): Description` or `[velox][PR] type(scope): Description` for GitHub-originated PRs — capital start, no trailing period, ≤67 chars. Type ∈ {feat, fix, refactor, test, docs}. Scope optional.

   Body paragraphs (include only those that carry weight for this change):
   - **What + why**: lead with user-visible behavior change. Include one concrete example query, error message, or before/after fact. A reader without internals knowledge should get the gist.
   - **Mechanism**: the core idea as ONE concept — a new field, a swapped algorithm, an added check, a rewrite step. Skip when the title + "what + why" already conveys it.
   - **Deferred**: name a deliberately-not-done case and how it surfaces (NYI message, follow-up issue). Skip if none.
   - **Test plan**: high-level coverage. Name the test file(s) and what scenarios they cover. Don't write "tests pass" or "CI green" — CI reports that; restating it is noise. State what was *covered*, not that it succeeded. For pure refactors with no new tests, omit the Test Plan section entirely — "existing tests cover this" / "covered by CI" is implied by "pure refactor" and adds no information.

   **Prose clarity** — write so a tired reader gets each sentence on first read.
   - Prefer short sentences. If a sentence has two clauses joined by "so", "because", "but", "even though", "although", or a comma + participle, consider splitting it. Contrastive joiners ("but X", "even though Y") are especially risky when both halves introduce a fact the reader does not already have — pack two new facts into one sentence and the reader stalls. State each rule in its own sentence, then connect them.
   - Avoid stacked abstractions like "left the outer scope advertising the column as X" or "the projection inherits the source's reverseLookup names". Replace with a concrete chain.
   - Avoid compiler/optimizer/execution-engine jargon ("outer reference", "outer scope", "binding context", "name resolution scope", "vector encoding") unless the rest of the paragraph already established it. If you must use it, define it inline with a tiny example.
   - Prefer plain verbs (`used`, `dropped`, `kept`) over jargon verbs (`advertise`, `surface`, `propagate`, `materialize`) unless the jargon is the precise term.
   - Avoid hyphenated compound-noun stacks ("user-written case", "lookup-based fallback", "context-aware resolver"). They require the reader to unpack a modifier chain before getting to the noun. Rewrite as a relative clause ("the case the user wrote") or a single concrete noun.
   - Prefer the word with one obvious meaning in this context. "Case" can mean legal case, match case, or upper/lower case — use "capitalization" when you mean letter case. Similarly: "operator" vs "function", "key" vs "column", "type" vs "kind" — pick the one a SQL reader and a C++ reader both interpret the same way.
   - Show a complete example query or input that demonstrates the failure, paired with the resulting error. Don't describe an input in prose ("a vector of nulls passed to X") when you can show one (`makeFlatVector<int64_t>({1, null, 2})`); the example carries the meaning without the reader holding context across sentences.
   - Break any long code, query, or error string out into a fenced block on its own line. "Long" means more than ~6 words or anything that wraps the surrounding paragraph awkwardly. This applies whether the long string is paired with another or stands alone — a single long error string embedded mid-sentence still overloads the reader. Patterns:

     Lead with the error, then explain:

     ```
     The query failed with:

         Function not registered: my_func(BIGINT)

     This happened because the function was registered only under the Presto namespace.
     ```

     Query + result:

     ```
     For example:

         SELECT my_func(c0) FROM (VALUES 1, 2) AS t(c0)

     fails with `Function not registered` because ...
     ```

     Short fragments (a single column name, a flag, a 2-3-word error name) stay inline.
   - When in doubt, read the paragraph aloud. If you pause mid-sentence to decode it, split or simplify it.

4. **Self-check before showing** — Walk every item; do not skip any.
   - [ ] Title matches `[velox] type(scope): Description` or `[velox][PR] type(scope): Description`, capital, no period, ≤67 chars.
   - [ ] Para 1 leads with behavior, not internal symbol names.
   - [ ] Para 1 has a concrete anchor (example, error message, before/after).
   - [ ] Mechanism is one concept, not a diff retrace with sibling-function names.
   - [ ] No reasoning journey (alternatives considered, sibling reused, layered fixes).
   - [ ] Code symbols in plain backticks; never `` \`escaped\` ``.
   - [ ] Each paragraph is ONE long line. No hard wrap at 72/80 columns. Bullet lists in the test plan are the exception.
   - [ ] No "tests pass" / "N tests pass" / "CI green".
   - [ ] Every factual claim verified against the diff, not recalled.
   - [ ] Reads in ~30 seconds.
   - [ ] Length matches the change. Trivial changes are not padded to standard length; standard changes are not condensed to one line.
   - [ ] Prose clarity: no sentence longer than ~30 words; no stacked abstractions ("X advertising Y", "scope of Z"). Each sentence is parseable on first read.

   If any item fails, fix the draft before showing.

5. **Show the draft** — Present the final draft. Mention any facts you could not verify and want the user to confirm.

## Common failure modes to avoid

These are the patterns drafts most often hit, and that this skill exists to prevent:

- **Restating the diff as prose** — "Adds X. Modifies Y. Changes Z." That's what the diff shows. State the behavior change and the one concept behind it.
- **Function-by-function walkthrough** — "In `foo()`, we now do A. In `bar()`, we adjust B." The reviewer reads the diff for that. Collapse into the single mechanism.
- **Enumerating touched files, classes, or call sites** — "Adopt it in `Foo`, `Bar`, `Baz`, `Qux`, and `Quux`." or "Updated across N call sites." The diff is the source of truth for scope. Naming the touched symbols eats space without informing the reader. Exception: name a specific file only when its role in the change is not obvious from the title (e.g., the test file that needed expectation updates, or the one production file the rest of the diff supports).
- **Long lists embedded in a sentence** — any comma-separated enumeration of more than ~3 items inside a sentence forces the reader to parse a list while tracking the surrounding clause. Examples: a list of test scenarios ("covers nulls, empty input, single-row, all-nulls, mixed-encoding, dictionary, constant, ..."), a list of operator variants. Fix by either (a) breaking the list out into its own bullet sub-list, or (b) summarizing as a category ("the main edge cases for null handling"). The bullet form scales with item count; the summary form is right when individual items don't carry weight.
- **Missing big picture** — Diving into internal symbols in paragraph 1. Lead with user-visible behavior; descend into mechanism in paragraph 2.
- **Reasoning scaffolding** — "We considered X but chose Y because Z." Belongs in design docs or PR threads, not the commit log.
- **Hard-wrapped paragraphs** — Hard line breaks at ~70/80 columns render as ragged short lines wherever the message is reflowed.
- **Pass-counting test plans** — "All 47 tests pass." CI says that. State *what was covered*, not that it succeeded.

## When NOT to invoke

- Pure typo fixes to an already-approved message.
- The user provides the full message themselves and asks you to commit it verbatim.

For all other Velox commit-message work — drafting, revising, or rewriting — invoke this skill.
