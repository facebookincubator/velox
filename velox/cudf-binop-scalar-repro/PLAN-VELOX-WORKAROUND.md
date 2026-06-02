# Velox workaround: cuDF column-vs-scalar fixed_point binary ops

**Status (2026-06):** Implemented. Stash popped; all changes below are in the working tree.

## Problem summary

cuDF’s compiled binary-op path wraps scalars as 1-row `column_view` objects but reinterprets the scalar’s **1-byte device `bool` validity** as a **column bitmask**. Null-aware kernels (`NULL_MAX`, `NULL_MIN`, `NULL_EQUALS`, …) call `column_device_view::is_valid()`, which uses `bit_is_set` and reads a **4-byte word** — causing OOB reads under compute-sanitizer and undefined behavior at runtime.

Velox originally hit this via `greatest` / `least` with decimal literals (`NULL_MAX` / `NULL_MIN` column-vs-scalar). Stream corruption from those OOB reads could also cause downstream divide tests to fail.

**Root cause is in cuDF** — see `PLAN-CUDF-UPSTREAM-FIX.md` and `velox/cudf-binop-scalar-repro/`.

---

## Implemented workaround

Broadcast the scalar to a full column via `cudf::make_column_from_scalar`, then call column-vs-column `cudf::binary_operation`. This avoids the broken scalar-as-column validity path entirely.

### New files

| File | Role |
|------|------|
| `velox/experimental/cudf/expression/CudfBinaryOpUtils.h` | Shared API |
| `velox/experimental/cudf/expression/CudfBinaryOpUtils.cpp` | Implementation |

**API** (`facebook::velox::cudf_velox`):

- `castDecimalScalar(src, targetType, stream, mr)` — decimal scale/precision alignment (moved from `ExpressionEvaluator.cpp`)
- `binaryOpColumnWithBroadcastScalar(col, scalar, op, outType, stream, mr)`
- `binaryOpScalarWithBroadcastColumn(scalar, col, op, outType, stream, mr)`

### Build integration

`CudfBinaryOpUtils.cpp` added to `velox/experimental/cudf/expression/CMakeLists.txt` in the `velox_cudf_expression` library.

### Call sites updated

#### 1. `BinaryFunction::eval` (`ExpressionEvaluator.cpp`)

For **all** `fixed_point` column-vs-scalar paths (except divide, which uses custom kernels):

| Op category | Path |
|-------------|------|
| Comparisons (`LESS`, `GREATER`, `EQUAL`, …) | Scale-align via `castDecimalScalar` + broadcast |
| `ADD`, `SUB`, `MOD` | Cast column to output type if needed + broadcast |
| `MUL` | DECIMAL128 promotion + broadcast |
| Other fixed_point ops | Broadcast fallback |

**Unchanged:**

- Column-vs-column → direct `cudf::binary_operation`
- `DIV` → `decimalDivide` custom kernel (column/column or column/scalar overload)
- Non-decimal types → direct `cudf::binary_operation`
- Divide-by-zero on scalar divisor → `decimalScalarIsZero` + throw before kernel

#### 2. `BetweenFunction::eval` (`ExpressionEvaluator.cpp`)

When `minLiteral_` or `maxLiteral_` is present and the value column is `fixed_point`:

- `GREATER_EQUAL` / `LESS_EQUAL` via `binaryOpColumnWithBroadcastScalar`
- Final `LOGICAL_AND` remains column-vs-column

#### 3. `GreatestLeastFunction::eval` (`ExpressionEvaluator.cpp`)

- Column-only reduction loop: column-vs-column (unchanged)
- Final folded literal step: cast lhs/scalar to `type_` if needed, then `binaryOpColumnWithBroadcastScalar` with `NULL_MAX` / `NULL_MIN`

This is the **primary** sanitizer trigger (`decimalGreatestLeastMixed`).

#### 4. `scatterNullsAtZeroDivisor` (`DecimalExpressionKernels.cpp`)

Zero-check uses broadcast for `EQUAL` column-vs-scalar:

```cpp
auto divisorIsZero = binaryOpColumnWithBroadcastScalar(
    divisor, *zeroScalar, cudf::binary_operator::EQUAL, ...);
```

`EQUAL` is not null-aware in the cuDF kernel and likely did not need this for correctness, but broadcast avoids any risk and sidesteps stream corruption when null-aware ops ran earlier in the same expression.

### Repro target

`velox/cudf-binop-scalar-repro/` — minimal cuDF repro + sanitizer harness. Wired in `velox/CMakeLists.txt` when `VELOX_ENABLE_CUDF=ON`.

---

## Design note: breadth vs narrow fix

Analysis showed only **null-aware** cuDF ops (`NULL_MAX`, `NULL_MIN`, `NULL_EQUALS`, …) actually read scalar validity in the kernel. `ADD`/`SUB`/`MUL`/`MOD`/comparisons/`EQUAL` are safe on the native column-vs-scalar path.

The current implementation broadcasts **all** decimal column-vs-scalar binops for simplicity and defensive coverage. A future refinement could:

1. Restrict broadcast to null-aware ops only (`GreatestLeastFunction` is mandatory).
2. Revert `BinaryFunction` arithmetic/comparison paths to native cuDF once upstream is fixed.
3. Replace `scatterNullsAtZeroDivisor` broadcast with a tiny custom CUDA kernel (`out_null[i] |= (divisor[i] == 0)`).

---

## Verification

### cuDF repro (confirms upstream bug)

```bash
cmake --build /opt/velox-build/release --target cudf_nullmax_scalar_repro -j

compute-sanitizer --tool memcheck \
  /opt/velox-build/release/velox/cudf-binop-scalar-repro/cudf_nullmax_scalar_repro \
  --only null_max
```

Expect OOB on `null_max` / `null_min` / `null_equals`; clean on `equal` / `add`.

### Velox expression tests (confirms workaround)

```bash
velox_cudf_decimal_expression_test --gtest_filter='CudfDecimalTest.decimalGreatestLeast*'

compute-sanitizer --tool memcheck \
  velox_cudf_decimal_expression_test \
  --gtest_filter='CudfDecimalTest.decimalGreatestLeastMixed'
```

Additional tests that were failing before the workaround:

```bash
velox_cudf_decimal_expression_test --gtest_filter='\
CudfDecimalTest.decimalBinaryNullPropagation:\
CudfDecimalTest.decimalDivide*:\
CudfDecimalTest.decimalArithmeticWithScalar*:\
CudfDecimalBinaryTest.cpuGpuMatch/div_*'
```

---

## Long-term cleanup (after cuDF fix)

1. Drop broadcast wrappers for ops that cuDF fixes in `scalar_as_column_view`.
2. Keep `castDecimalScalar` — still needed for Velox decimal scale alignment.
3. Re-run sanitizer on repro + full `velox_cudf_decimal_expression_test`.
4. Optionally narrow broadcast scope per [Design note](#design-note-breadth-vs-narrow-fix).

---

## File inventory (current tree)

| Artifact | Status |
|----------|--------|
| `CudfBinaryOpUtils.h` | Present |
| `CudfBinaryOpUtils.cpp` | Present, in CMakeLists |
| `ExpressionEvaluator.cpp` | Uses broadcast helpers; no duplicate `castDecimalScalar` |
| `DecimalExpressionKernels.cpp` | `scatterNullsAtZeroDivisor` uses broadcast |
| `cudf-binop-scalar-repro/` | Present, in `velox/CMakeLists.txt` |

**Velox cuDF pin:** `VELOX_cudf_COMMIT d09d10d` (26.06) — see `CMake/resolve_dependency_modules/cudf.cmake`.

---

## References

- Sanitizer log: `vcdet.log` (repo root) — `NullMax`, `fixed_point`, 4-byte read, 1-byte allocation
- cuDF bug site: `rapidsai/cudf/cpp/src/binaryop/compiled/binary_ops.cu` (`scalar_as_column_view`)
- Upstream fix plan: `PLAN-CUDF-UPSTREAM-FIX.md`
