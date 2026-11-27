The new `divide_decimal` matches `decimalBinaryOperation` closely in how it takes the output type and dispatches on widths. It differs in two places that matter to Velox: how the output precision is given, and how failures are reported.

`decimalBinaryOperation` is Velox's overflow-checked path for ADD, SUB, MUL and MOD; DIV goes through Velox's separate `decimalDivide`. It is declared in `DecimalExpressionKernels.h` and implemented in `DecimalExpressionKernelsGpu.cu`:

```104:111:/home/seves/work/velox/velox/velox/experimental/cudf/expression/DecimalExpressionKernels.h
template <typename Lhs, typename Rhs>
std::unique_ptr<cudf::column> decimalBinaryOperation(
    const Lhs& lhs,
    const Rhs& rhs,
    cudf::binary_operator op,
    cudf::data_type outputType,
    int32_t outputPrecision,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr);
```

## Where they match

- **Overloads.** Both have column/column, column/scalar and scalar/column versions.
- **Output type.** Both take a `cudf::data_type` that sets the output's width and scale.
- **Widths.**
  - Both dispatch independently on the lhs, rhs and output widths, using three nested switches.
  - Both compute at a width covering all three. Velox uses `int64_t` only when all three are `DECIMAL64`, otherwise `__int128_t`. Ours uses the widest of the three, so it is the same for 64/128-bit and also handles `DECIMAL32`, which Velox doesn't use.
  - Neither needs the operands cast to a common width first. Velox's `decimalDivide` does need that cast today, so our API would let it drop the cast.
- **Scales.**
  - Velox's ADD/SUB/MOD/MUL path derives its rescale from the lhs, rhs and output scales, and `divide_decimal` does the same.
  - Velox's `decimalDivide` instead takes a precomputed `aRescale >= 0`. Our exponent is the same value, but it can also be negative, so output scales coarser than `lhs` work too.
- **Overflow checks.**
  - The rescale is checked for overflow in both.
  - The result is bounded by a power of ten: Velox calls `cudf::detail::ops::check_precision(value, outPrecision)`, and ours checks `quotient <= 10^digits10 - 1`.
  - The implied limits are the same: 18 digits for `DECIMAL64` and 38 for `DECIMAL128`, matching Velox's `kMaxPrecisionFor`.
- **Nulls.** A null input row gives a null output row, and an invalid scalar gives an all-null column, in both.
- **Rounding.** Velox only rounds HALF_UP (away from zero). Our default is the same, and HALF_EVEN is also available.

## Where they don't

1. **Precision is an argument in Velox, implied in ours.**
   - Velox passes `outputPrecision` separately and checks that it fits the output width. A `decimal(10,2)` result stored as `DECIMAL64` is therefore limited to 10 digits.
   - Our API always uses the width's maximum, 18 for `DECIMAL64`, so an 11- to 18-digit quotient would be returned rather than rejected.
   - For divide specifically, that is still stricter than Velox's own `decimalDivide` today. It only checks 10³⁸ and then narrows with an unchecked `cudf::cast`.
   - It isn't the same contract as `decimalBinaryOperation`, though. Matching it would mean an optional `int32_t precision` that defaults to the width's maximum, implemented with the same `check_precision`. That's a small change; you chose the implied precision earlier, so this would revisit that decision.

2. **Velox needs errors; ours gives nulls.**
   - `decimalBinaryOperationWithOverflow` returns `pair<column, DecimalBinaryOpStatus>`. That status is one flag word per launch, which keeps divide-by-zero and overflow apart, and the wrapper raises "Division by zero" or "Decimal overflow".
   - Our API turns both into nulls. Velox can't tell them apart without extra passes:
     - check the divisor for zeros to spot divide-by-zero;
     - compare the output null count with `bitmask_and` of the inputs to spot overflow.

     Each pass needs a host round trip.
   - The fix is the error/status reporting the plan listed as out of scope. Either add a variant that also returns a `cudf::errc` summary, or an option to throw. Velox's status bits are already numbered to match `cudf::errc`.

3. **Smaller differences that need no API change.**
   - Velox reads scalar values and validity on the host, which syncs. Ours reads them on the device.
   - Velox precomputes the null mask from the inputs before launching. Ours builds it afterwards from a per-row validity buffer. The results are the same.

## Bottom line

With an explicit precision argument and failure reporting, `divide_decimal` could serve as the DIV counterpart of `decimalBinaryOperation`. Velox could then delete its own divide kernel, the operand casts, and the final unchecked narrowing.

As it stands, the new API is consistent with `decimalBinaryOperation` in shape (output type, independent widths, scale handling, null rules). Velox would still need its own fail-fast error checks around it, and its own precision check for results narrower than the full width.

For DIV alone, `cudf::divide_decimal` is close to a drop-in replacement for Velox's `decimalDivide`. The precision mismatch from my previous answer doesn't arise, because Velox's DIV path never passes a precision. The arithmetic gives identical results, and for `DECIMAL64` outputs ours is safer. The one real gap is that Velox fails the batch where we return null, and that can be bridged in Velox with a cheap check after the call.

## What Velox's DIV path does today

The evaluator (`ExpressionEvaluator.cpp`, lines 660–687):
1. **Working type.** It picks `DECIMAL128` if any of lhs, rhs or the result is 128-bit, otherwise `DECIMAL64`, using the result scale.
2. **Casts up.** It casts any `DECIMAL64` operand up to `DECIMAL128` when the working type is 128-bit.
3. **Rescale factor.** It computes `aRescale = outScale − lhsScale + rhsScale` and calls `decimalDivide`.
4. **Casts down.** It casts the result to the plan's type with `finalizeDecimalDivision`, which is an unchecked `cudf::cast`.

The kernel (`decimalDivideImpl`, `DecimalExpressionKernelsGpu.cu` lines 160–207):
- multiplies the dividend by `10^aRescale` in `unsigned __int128`, with an overflow check;
- divides and rounds HALF_UP away from zero;
- checks `|q| ≤ 10³⁸ − 1`, then does `static_cast<OutT>`.

Failures set a flag for each launch. The host then raises "Division by zero" or "Decimal overflow in divide".

## What matches, row by row

- **Rescale factor.** Our `exponent = lhs.scale − rhs.scale − out.scale` is exactly `aRescale`; cudf's scales are just the negated digit counts.
- **Rounding.** Velox rounds up when `remainder > (d − 1) / 2`, and ours when `remainder >= d − remainder`. Both mean 2·remainder ≥ d for every divisor d, so the results are bit-identical.
- **Compute width.** Velox always computes in `unsigned __int128`. Ours uses `__uint128_t` whenever an operand or the output is 64- or 128-bit, so overflow happens at the same boundary.
- **Nulls.** A null input gives a null output, and an invalid scalar gives an all-null column, in both. Velox's branch masking ANDs the branch mask into the inputs' null masks. Our kernel skips null rows, so that keeps working.
- **Signed edge cases.** `MIN / -1` and similar fail in both.

## Where ours is better for Velox

- **No casts.**
  - The operands can have different widths, so the casts up to `DECIMAL128` go away. That saves one full column copy per `DECIMAL64` operand.
  - The output is written straight at the plan's type, so `finalizeDecimalDivision` goes away too.
- **Overflow check for `DECIMAL64` outputs.**
  - Velox only checks against 10³⁸, and then narrows to `int64_t` unchecked, both inside the kernel and in the final cast. A `DECIMAL64` quotient between 10¹⁸ and 10³⁸ therefore wraps silently today.
  - Ours makes it null, because it checks against 10¹⁸ − 1.
- **No host reads of scalars.** We read scalar values and validity on the device. Velox's scalar path reads them on the host first, which syncs.
- **Rounding mode.** HALF_EVEN is also available if it's ever needed.

## The gaps, and how Velox would bridge them

1. **Failures are nulls, not errors.** This is the only real gap. Velox can recover the fail-fast behaviour with no new cudf API:

   ```cpp
   auto result = cudf::divide_decimal(lhs, rhs, type_, cudf::rounding_method::HALF_UP, stream, mr);
   auto const input_nulls = cudf::bitmask_and(cudf::table_view{{lhs, rhs}}, stream).second;
   if (result->null_count() > input_nulls) {
     // Rare path: some row failed. Report a zero divisor among rows with both inputs valid,
     // otherwise overflow.
   }
   ```

   - **Common path.** It costs one count of the input null masks. Velox already computes that mask today, and it's free when neither input has nulls. Our result's null count is already known on the host, because `bools_to_mask` syncs once, much as Velox's per-launch flag read does.
   - **Error path.** Only here does Velox need a second pass, to tell divide-by-zero from overflow, and that keeps divide-by-zero taking precedence. The check has to ignore rows where the dividend is null, because Velox's kernel doesn't raise for those.
   - **Scalar divisors.** Velox's existing host check for a zero scalar divisor can stay as it is.

2. **A rescale exponent above 38.**
   - Velox rejects the whole call on the host when `aRescale > 38`.
   - Ours makes nonzero dividends null and returns 0 for a zero dividend, which is mathematically correct.
   - For exact parity, Velox can keep its one-line host check. A negative `aRescale` can't occur under Velox's result-scale rule, though ours would handle it.

3. **Precision.** No gap for DIV. Velox doesn't pass a precision here; it only checks against 10³⁸. Our width-based limit is the same for `DECIMAL128` and stricter, and correct, for `DECIMAL64`.

## Bottom line

Replacing Velox's DIV with `cudf::divide_decimal(lhs, rhs, type_, HALF_UP, stream, mr)` gives the same values for every row that succeeds. It also removes the operand and result casts, and fixes the unchecked narrowing to `DECIMAL64`. Velox would add the null-count comparison above to keep failing fast, and could optionally keep its `aRescale ≤ 38` host check. Velox can then delete its divide kernel and `decimalDivide`.

A precision argument would only matter if Velox later wanted DIV to enforce the plan's precision the way `decimalBinaryOperation` does for its operators.

The DIV path in Velox now calls `cudf::divide_decimal`, and the custom divide code it replaced has been removed. In total that's 164 lines added and 658 removed across five files, all under `velox/experimental/cudf/expression/`. Nothing has been built or committed.

**What changed:**
- **`ExpressionEvaluator.cpp`:**
  - The three DIV branches (column/column, column/scalar, scalar/column) each pass their operands and the result type straight to `decimalDivide(...)`. Velox no longer casts between widths or computes the rescale factor itself.
  - `decimalDivisionWorkingType` and `finalizeDecimalDivision` are gone.
- **`DecimalExpressionKernels.h` / `.cpp`:**
  - The `decimalDivide` overloads no longer take `aRescale`. Each calls `cudf::divide_decimal` with `HALF_UP`, which rounds half away from zero like the CPU path.
  - `cudf::divide_decimal` returns null for a failed row. If the result has more nulls than the inputs, a row failed, and the batch then fails with Velox's errors. "Division by zero" wins if any row with both inputs valid has a zero divisor; otherwise the error is "Decimal overflow in divide".
  - The rescale limit (at most 38) is still checked on the host first, so that error matches the CPU path.
  - `makeAllNullDecimalColumn`, `checkDecimalDivideTypes` and the `DecimalUtil.h` and `column_factories.hpp` includes were removed.
- **`DecimalExpressionKernelsGpu.cu` / `.h`:**
  - All of Velox's own divide code is removed: the int128 and long-decimal constants, the helpers (`absToUnsigned`, `signedFromUnsigned`, `fitsRepresentableInt128`, `decimalDivideImpl`), the three divide functors, the `ValidDecimalDivideStorageTypes` concept, and the three `detail::decimalDivide*` kernels with their declarations.
  - The `<concepts>` and `type_dispatcher.hpp` includes went with them, and comments that described the code as shared with divide were updated.

**Deliberately kept:**
- The ADD/SUB/MUL/MOD path, which still uses the status bits, `isRowActive`, `checkDecimalBinaryOpStatus` and `kDivisionByZero`.
- `getDecimalScalarValue`, which that path uses, and which the new code uses to check a scalar divisor for zero.
- `decimalScalarIsZero` in the evaluator. It isn't redundant: with a zero scalar divisor and an all-null column, it still fails with "Division by zero", where the new failure check alone would not.

**Behaviour changes to watch for once you can build:**
- A DECIMAL64 result that overflows now raises "Decimal overflow in divide". The old path's narrowing cast let it wrap silently.
- A result now also fails when it exceeds its type's precision (18 digits for DECIMAL64), not only DECIMAL128's 38-digit limit.
- Mixed-width operands are handled inside cudf, so nothing is cast to DECIMAL128 beforehand.

The tests in `DecimalExpressionTest.cpp` don't call these functions directly, so none needed changing. The divide tests there are the ones to run first.
