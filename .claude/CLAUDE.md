# CLAUDE.md

Guidance for Claude Code when working in the Velox repository.

## PR Review

When asked to review a PR (via `/pr-review`), always use the /pr-review skill.

## Queries

When asked a question about the PR or codebase (via `/query`), use the /query skill.

## Overview

Velox is an open source C++ library for composable data processing and
query execution. Licensed under Apache 2.0. Requires C++20, GCC 11+ or
Clang 15+.

## Build

```bash
make debug    # debug build
make release  # optimized build
```

## Testing

```bash
make unittest                    # run all tests
cd _build/debug && ctest -j 8   # run all tests in parallel
ctest -R ExprTest                # run tests matching a pattern
```

Test files live in `tests/` subdirectories alongside source.

### Grouped tests

Four test suites use `velox_add_grouped_tests` to reduce link times on Linux CI
by batching source files into shared binaries:
- `velox/exec/tests` (`velox_exec_test`, `velox_exec_util_test`)
- `velox/functions/prestosql/aggregates/tests`
- `velox/common/caching/tests`
- `velox/serializers/tests`

All other test suites use individual binaries on all platforms.

On macOS, grouping is off by default (`VELOX_ENABLE_GROUPED_TESTS=OFF`) and each
test file gets its own binary (e.g., `ValuesTest.cpp` → `velox_exec_test_ValuesTest`).
On Linux CI, grouping is on (`velox_exec_test_group0` through `_group7`).
Override with `-DVELOX_ENABLE_GROUPED_TESTS=ON/OFF`.

### Common test workflows

```bash
# Run all test binaries whose ctest name matches a regex.
# On Linux this matches velox_exec_test_group0 … _group7.
# On macOS this matches velox_exec_test_ValuesTest,
# velox_exec_test_HashJoinTest, etc.
cd _build/debug && ctest -R velox_exec

# Run a specific test file (macOS — individual binary)
_build/debug/velox/exec/tests/velox_exec_test_ValuesTest --gtest_filter="ValuesTest.*"

# Run a specific test case (Linux — grouped binary)
_build/debug/velox/exec/tests/velox_exec_test_group3 --gtest_filter="ValuesTest.empty"
```

**Re-running a CI failure locally:** CI reports a failure in
`velox_exec_test_group3` with `ValuesTest.empty`. On Linux, run the grouped
binary directly. On macOS, the grouped binary does not exist — use the
per-file binary instead: `velox_exec_test_ValuesTest --gtest_filter="ValuesTest.empty"`.

**Adding a new test to a grouped suite:** Add the source file to the `SOURCES`
list in the relevant `velox_add_grouped_tests()` call in `CMakeLists.txt`. It
is automatically assigned to a group on Linux and gets its own binary on macOS.

**Creating a new test suite:** Use `velox_add_grouped_tests` for suites with
many test files (10+) that link against large libraries like velox core — each
individual binary pays the full link cost, so grouping them into shared binaries
significantly reduces total CI build time. For suites with only a few test files
or lightweight dependencies, use `add_executable` / `add_test`.

## Formatting

```bash
make format  # format all changed files
```

## Coding Style

Read [CODING_STYLE.md](../CODING_STYLE.md) for the complete guide. Key rules
are summarized below.

### Comments

- Use `///` for public API documentation (classes, public methods, public members).
- Use `//` for private/protected members and comments inside code blocks.
- Start comments with active verbs, not "This class…" or "This method…".
  - ❌ `/// This class builds query plans.`
  - ✅ `/// Builds query plans.`
- Comments should be full English sentences starting with a capital letter and ending with a period.
- Comment every class, every non-trivial method, every member variable.
- Do not restate the variable name. Either explain the semantic meaning or omit the comment.
  - ❌ `// A simple counter.` above `size_t count_{0};`
- Avoid redundant comments that repeat what the code already says. Comments should explain *why*, not *what*.
- Use `// TODO: Description.` for future work. Do not include author's username.
- Do not duplicate comments between `.h` and `.cpp`. Document the function in the header; the implementation should not repeat the same comment. Duplicated comments diverge over time.

### Naming Conventions

- **PascalCase** for types and file names.
- **camelCase** for functions, member and local variables.
- **camelCase_** for private and protected member variables.
- **snake_case** for namespace names and build targets.
- **UPPER_SNAKE_CASE** for macros.
- **kPascalCase** for static constants and enumerators.
- Do not abbreviate. Use full, descriptive names. Well-established abbreviations (`id`, `url`, `sql`, `expr`) are acceptable.
- Prefer `numXxx` over `xxxCount` (e.g. `numRows`, `numKeys`).
- Never name a file or class `*Utils`, `*Helpers`, or `*Common`. These generic
  names attract unrelated functions over time and lose cohesion. Name files and
  classes after the concept they represent. Use a class with static methods to
  group related operations, and shorten method names since the class name
  provides context.

### Asserts and CHECKs

- Use `VELOX_CHECK_*` for internal errors, `VELOX_USER_CHECK_*` for user errors.
- Prefer two-argument forms: `VELOX_CHECK_LT(idx, size)` over `VELOX_CHECK(idx < size)`.
- Use `VELOX_FAIL()` / `VELOX_USER_FAIL()` to throw unconditionally.
- Use `VELOX_UNREACHABLE()` for impossible branches, `VELOX_NYI()` for unimplemented paths.
- Put runtime information (names, values, types) at the **end** of error messages, after the static description.
  - ❌ `VELOX_USER_FAIL("Column '{}' is ambiguous", name);`
  - ✅ `VELOX_USER_FAIL("Column is ambiguous: {}", name);`

### Variables

- Prefer value types, then `std::optional`, then `std::unique_ptr`.
- Prefer `std::string_view` over `const std::string&` for function parameters.
- Use uniform initialization: `size_t size{0}` over `size_t size = 0`.
- Declare variables in the smallest scope, as close to usage as possible.
- Use digit separators (`'`) for numeric literals with 4 or more digits: `10'000`, not `10000`.
- Use trailing commas in multi-line initializer lists, enum definitions, and
  function-call argument lists that span multiple lines. This produces cleaner
  diffs when items are added or reordered.

### API Design

- Keep the public API surface small.
- Prefer free functions in `.cpp` (anonymous namespace) over private/static class methods.
- Define free functions close to where they are used, not grouped together at the top or bottom of the file.
- Keep method implementations in `.cpp` except for trivial one-liners.
- Avoid default arguments when all callers can pass values explicitly.
- Never use `friend`, `FRIEND_TEST`, or any friend declarations. If a test needs access to private members, redesign the API or test through public methods instead.

### Tests

- Place new tests next to related existing tests, not at the end of the file. Group tests by topic (e.g., place `tryCast` next to `types`, `notBetween` next to `ifClause` which uses `between`).

Use gtest container matchers (`testing::ElementsAre`, etc.) for verifying collections:

```cpp
// ❌ Avoid - multiple individual assertions
EXPECT_EQ(result.size(), 3);
EXPECT_EQ(result[0], "a");
EXPECT_EQ(result[1], "b");
EXPECT_EQ(result[2], "c");

// ✅ Prefer - single matcher assertion
EXPECT_THAT(result, testing::ElementsAre("a", "b", "c"));
```

Common matchers:
- `ElementsAre(...)` - exact ordered match
- `UnorderedElementsAre(...)` - exact unordered match
- `Contains(...)` - at least one element matches
- `IsEmpty()` - collection is empty
- `SizeIs(n)` - collection has n elements

Requires `#include <gmock/gmock.h>`.

## Common Mistakes

These are frequently violated rules. Check every new or modified line against
this list before finishing.

- **Bug fixes without a failing test first.** Write the test first, confirm it fails, then fix. A test that passes with and without the fix proves nothing.
- **`///` vs `//` wrong comment style.** `///` is only for public API in headers. Everything else uses `//`.
- **One-letter and abbreviated variable names.** Use full, descriptive names. Only loop indices (`i`, `j`) are acceptable.
- **Undocumented APIs in headers.** Every class, method, and member variable in a `.h` file must have a comment.
- **Non-trivial implementations in headers.** If a method body has more than one statement, it belongs in the `.cpp` file.
- **`goto` statements.** Never use `goto`. Use early returns, helper functions, or duplicated code paths.
- **Fitting tests to buggy code.** Never update test expectations to match buggy output without verifying correctness first.
- **Generic file and class names.** Never name a file or class `*Utils`, `*Helpers`, or `*Common`.
- **Verify causation before asserting it.** Do not attribute failures to a commit based on its message alone. Verify empirically.
- **Silently simplifying an approved plan.** If a step is harder than expected, say so and get approval before reducing scope.
- **Working around infrastructure bugs.** Do not silently work around bugs in shared infrastructure. Report and discuss.

## Design Documents

Design (including proposals) live in `docs/designs/`.  When creating new
designs, place them there with a descriptive filename (e.g.,
`column-extraction-pushdown.md`).
