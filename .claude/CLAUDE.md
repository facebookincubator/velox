# CLAUDE.md

Guidance for Claude Code when working in the Velox repository.

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

### Naming Conventions

- **PascalCase** for types and file names.
- **camelCase** for functions, member and local variables.
- **camelCase_** for private and protected member variables.
- **snake_case** for namespace names and build targets.
- **UPPER_SNAKE_CASE** for macros.
- **kPascalCase** for static constants and enumerators.
- Do not abbreviate. Use full, descriptive names. Well-established abbreviations (`id`, `url`, `sql`, `expr`) are acceptable.
- Prefer `numXxx` over `xxxCount` (e.g. `numRows`, `numKeys`).

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

### API Design

- Keep the public API surface small.
- Prefer free functions in `.cpp` (anonymous namespace) over private/static class methods.
- Keep method implementations in `.cpp` except for trivial one-liners.
- Avoid default arguments when all callers can pass values explicitly.

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
