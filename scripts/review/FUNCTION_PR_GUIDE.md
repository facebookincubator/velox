# Function PR Review Guide

Additional checklist for PRs that add or modify scalar functions, aggregate
functions, or special forms. Use alongside `SELF_REVIEW.md`.

## PR title

- Adding: `feat(presto): Add abs scalar function`
- Adding: `feat(spark): Add bitmap_or_agg aggregate function`
- Fixing: `fix(presto): Fix abs function for negative zero`
- Fixing: `fix(spark): Fix bitmap_or_agg for all-null inputs`

## Documentation

- [ ] Doc entry exists in the correct `.rst` file under `docs/functions/`
      (e.g., `docs/functions/spark/aggregate.rst`,
      `docs/functions/presto/math.rst`). Functions must appear in
      alphabetical order.
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
- [ ] Input validation uses the [non-throwing error path](https://velox-lib.io/blog/optimize-try-more#non-throwing-simple-functions)
      (`Status` / `setError`) so functions work correctly inside `TRY`.
      Use `VELOX_CHECK_*` only for internal invariants that indicate bugs.
- [ ] For vector functions and special forms: `EvalCtx::moveOrCopyResult`
      is used when the function may be called with a pre-existing result
      vector (e.g., inside `IF` / `CASE WHEN`). Do not unconditionally
      replace `result`.

