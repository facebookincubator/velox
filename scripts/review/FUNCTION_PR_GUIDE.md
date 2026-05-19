# Function PR Review Guide

Additional checklist for PRs that add or modify scalar functions, aggregate
functions, or special forms. Use alongside `SELF_REVIEW.md`.

## Documentation

- [ ] Doc entry exists in the correct `.rst` file under `docs/functions/`
      (e.g., `docs/functions/spark/aggregate.rst`,
      `docs/functions/presto/math.rst`).
- [ ] Doc entry uses the correct directive (e.g., `.. spark:function::`,
      `.. function::`).
- [ ] Signature matches the implementation: argument types, return type,
      nullability.
- [ ] Behavior is described precisely: edge cases, null handling, error
      conditions, valid input ranges.
- [ ] If the function mirrors a function in another engine (Spark, Presto),
      link to the canonical spec rather than copying its description.

## Registration

- [ ] Registration name matches the engine's canonical name.
- [ ] Prefix handling is correct (`prefix + "function_name"`).

## Implementation

- [ ] Use `SimpleFunction` API for scalar functions when possible.
- [ ] Use `SimpleAggregateAdapter` for aggregate functions when possible.
      Before concluding it doesn't support your use case, check the actual
      header — it supports `HashStringAllocator`, custom `destroy()`, variable-size
      accumulators, and external memory.
- [ ] `default_null_behavior_` is set correctly. Functions that produce
      non-null output for null inputs (e.g., `IS NULL`, aggregate functions
      returning default values) must set this to `false`.
- [ ] Input validation uses the [non-throwing error path](https://velox-lib.io/blog/optimize-try-more#non-throwing-simple-functions)
      (`Status` / `setError`) so functions work correctly inside `TRY`.
      Use `VELOX_CHECK_*` only for internal invariants that indicate bugs.
- [ ] For vector functions and special forms: `EvalCtx::moveOrCopyResult`
      is used when the function may be called with a pre-existing result
      vector (e.g., inside `IF` / `CASE WHEN`). Do not unconditionally
      replace `result`.

## Aggregate functions

- [ ] All aggregation steps work correctly: partial, intermediate, final,
      and single.
- [ ] `combine` (merge) handles null intermediate states.
- [ ] Memory allocated through `HashStringAllocator` is freed in
      `destroy()`.
- [ ] `use_external_memory_` is set to `true` if the accumulator allocates
      through `HashStringAllocator`.
- [ ] `is_fixed_size_` is set correctly.

## Tests

- [ ] Test file exists in the `tests/` subdirectory alongside the source.
- [ ] Test file is added to `CMakeLists.txt`.
- [ ] Tests cover: basic operation, null inputs, empty input, edge cases,
      error cases (invalid input with `VELOX_ASSERT_THROW`).
- [ ] For aggregate functions: tests cover `testAggregations` (exercises
      all aggregation modes), group-by, global aggregation, and an
      end-to-end test with upstream functions if applicable.
- [ ] For bug fixes: an integration test with the actual expression context
      (e.g., `SWITCH`, `IF`, `TRY`) that triggered the bug, not just
      direct function calls.
- [ ] Expected values are derived from the Spark/Presto spec or reference
      implementation, not hand-computed from the code being tested.

## Special forms

- [ ] Type resolution works through the standard `resolveType` path.
- [ ] If type resolution requires constant argument values, discuss the
      approach with the reviewer before modifying the framework.
- [ ] `constructSpecialForm` is implemented correctly.

## Naming

- [ ] File names match: `FunctionName.cpp`, `FunctionName.h`,
      `FunctionNameTest.cpp`.
- [ ] Class name matches the function: `FunctionNameFunction` or
      `FunctionNameAggregate`.
- [ ] Test fixture name matches: `FunctionNameTest`.
- [ ] Enum values for function variants use `kPascalCase`.
