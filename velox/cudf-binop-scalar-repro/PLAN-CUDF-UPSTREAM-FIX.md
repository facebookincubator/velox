# cuDF upstream fix: scalar validity vs column bitmask in compiled binops

**Status (2026-06):** Root cause identified; minimal repro in Velox tree; Velox workaround implemented; upstream patch not yet filed.

## Summary

There is a **real cuDF bug** in the compiled binary-operation path when one operand is a **scalar** and the operator is **null-aware**. Scalar validity is stored as a single device `bool`; the scalar-as-column wrapper exposes it as a column null **bitmask**, and kernels call `bit_is_set`, which reads 32 bits from a 1-byte allocation.

This is **not** a Velox scalar construction bug. cuDF’s own column-vs-scalar tests for `ADD`/`DIV`/`MOD`/comparisons pass because those kernels **do not** call `is_valid()` on operands inside the device loop.

Velox ships a workaround (scalar broadcast) — see `PLAN-VELOX-WORKAROUND.md`.

---

## Root cause

### 1. Scalar validity representation

`cudf::scalar` stores validity as:

```cpp
// cpp/include/cudf/scalar/scalar.hpp
cudf::detail::device_scalar<bool> _is_valid;
```

`validity_data()` returns `bool*` (1 byte on device).

### 2. Scalar wrapped as column_view

`scalar_as_column_view` in `cpp/src/binaryop/compiled/binary_ops.cu`:

```cpp
auto col_v = column_view(s.type(),
                         1,
                         h_scalar_type_view.data(),
                         reinterpret_cast<bitmask_type const*>(s.validity_data()),
                         !s.is_valid(stream));
```

The `reinterpret_cast<bool* → bitmask_type*` is incorrect: column validity is a **bitmask**; scalar validity is a **single bool**.

### 3. Null-aware kernels read validity per row

`ops_wrapper` in `cpp/src/binaryop/compiled/binary_ops.cuh` for `NullMax`, `NullMin`, `NullEquals`, etc.:

```cpp
lhs.is_valid(is_lhs_scalar ? 0 : i),
rhs.is_valid(is_rhs_scalar ? 0 : i),
```

Even with index `0` for scalars, `is_valid_nocheck` → `bit_is_set(_null_mask, offset() + 0)` reads a **full `bitmask_type` word** (typically 4 bytes).

### 4. compute-sanitizer signature

From Velox `vcdet.log`:

```text
Invalid __global__ read of size 4 bytes
... ops_wrapper ... NullMax ... fixed_point<long, ...>
Access at ... is out of bounds
and is inside the nearest allocation at ... of size 1 bytes
```

Matches: 4-byte `bit_is_set` read from 1-byte scalar validity buffer.

### 5. Entry point sets scalar flag correctly

Vector-scalar path in `binary_ops.cu` is correct **except** for the validity representation:

```cpp
auto [rhsv, aux] = scalar_to_column_view(rhs, stream);
operator_dispatcher(out, lhs, rhsv, false, true, op, stream);  // is_rhs_scalar = true
```

The bug is **not** a missing `is_rhs_scalar` flag; it is **invalid null-mask plumbing**.

---

## Affected vs unaffected ops

| Operator class | Reads operand validity in kernel? | Column-vs-scalar fixed_point |
|----------------|-----------------------------------|------------------------------|
| `NULL_MAX`, `NULL_MIN` | Yes | **Broken** |
| `NULL_EQUALS`, `NULL_NOT_EQUALS` | Yes | **Broken** |
| `NULL_LOGICAL_AND`, `NULL_LOGICAL_OR` | Yes | **Broken** (if used) |
| `ADD`, `SUB`, `MUL`, `MOD`, `DIV`, … | No | OK (cuDF tests, 1000 rows) |
| `EQUAL`, `NOT_EQUAL`, `LESS`, … | No (non-null-aware `ops::Equal`) | OK |
| `GREATER_EQUAL`, `LESS_EQUAL`, … | No | OK |

### Test gap in cuDF

`cpp/tests/binaryop/binop-compiled-fixed_point-test.cpp` has:

- Column/column `NULL_MAX` / `NULL_MIN`
- Column/scalar `ADD`, `DIV`, `MOD`, comparisons (multi-row)

It does **not** appear to have column/scalar `NULL_MAX` / `NULL_MIN` on fixed_point with **N > 1** rows — the exact Velox `greatest(a, literal, b)` pattern.

---

## Proposed cuDF fixes (pick one)

### Option A — Fix `scalar_as_column_view` (recommended)

Build a **proper 1-element null mask** instead of casting `bool*`:

```cpp
// Pseudocode
if (s.is_valid(stream)) {
  // null_mask = nullptr, null_count = 0  → nullable() == false, is_valid() short-circuits true
} else {
  // allocate bitmask with bit 0 clear, null_count = 1
}
```

For **invalid** scalars, must satisfy `column_view` invariants (`null_count > 0` requires non-null `null_mask`).

Simplest robust approach: **`make_column_from_scalar(s, 1, stream, mr)`** and use the returned column’s view (same strategy as the Velox workaround). Slight allocation cost, correct semantics.

### Option B — Branch in `ops_wrapper` on `is_*_scalar`

When `is_rhs_scalar`, read validity via `scalar_device_view` / single `bool` load instead of `column_device_view::is_valid()`.

Requires passing scalar device views into the kernel or a side channel — larger API change.

### Option C — Host-only validity for scalars

For null-aware ops with one scalar operand, compute output null mask on host (similar to non-null-aware ops today) and use a kernel variant that skips validity reads.

More invasive; duplicates logic.

**Recommendation:** Option A in `scalar_as_column_view` — localized, matches existing `make_column_from_scalar` behavior, fixes all null-aware ops at once.

---

## Minimal reproduction (Velox tree)

Location:

```text
velox/cudf-binop-scalar-repro/
├── main.cpp
├── CMakeLists.txt
├── README.md
├── PLAN-VELOX-WORKAROUND.md
└── PLAN-CUDF-UPSTREAM-FIX.md   (this file)
```

### Build in-tree

```bash
cmake --build /opt/velox-build/release --target cudf_nullmax_scalar_repro -j
```

### Build standalone

```bash
cd velox/cudf-binop-scalar-repro
cmake -S . -B build \
  -DCMAKE_PREFIX_PATH=/opt/velox-build/release \
  -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build build -j
```

### Run sanitizer

```bash
compute-sanitizer --tool memcheck ./build/cudf_nullmax_scalar_repro
compute-sanitizer --tool memcheck ./build/cudf_nullmax_scalar_repro --only null_max
```

### Expected results

| `--only` | Op | Sanitizer |
|----------|-----|-----------|
| `null_max` | `NULL_MAX` | **OOB** (bug) |
| `null_min` | `NULL_MIN` | **OOB** |
| `null_equals` | `NULL_EQUALS` | **OOB** |
| `equal` | `EQUAL` | Clean (control) |
| `add` | `ADD` | Clean (control) |

---

## Suggested upstream issue / PR outline

**Title:** `Compiled binops: OOB read when null-aware ops use column-vs-scalar fixed_point`

**Body:**

1. Link to repro (`velox/cudf-binop-scalar-repro/`) or inline gtest added to `binop-compiled-fixed_point-test.cpp`.
2. Explain `bool*` vs `bitmask_type*` in `scalar_as_column_view`.
3. Stack: `apply_binary_op<NullMax>` → `ops_wrapper` → `is_valid(0)` → `bit_is_set`.
4. Proposed fix: Option A above.
5. Add test:

```cpp
// FixedPointCompiledTest, new case
auto col = fp_wrapper<RepType>({40, 30, 20}, scale_type{-2});
auto scalar = cudf::make_fixed_point_scalar<decimalXX>(500, scale_type{-2});
auto result = cudf::binary_operation(col, *scalar, cudf::binary_operator::NULL_MAX, type);
// + compute-sanitizer in CI if available
```

**Velox cuDF pin:** `VELOX_cudf_COMMIT d09d10d` (26.06) — verify on current `rapidsai/cudf` main as well.

**Local cuDF checkout for inspection:** `/home/seves/work/rapidsai/cudf`

---

## Verification after upstream fix

1. cuDF: new gtest + sanitizer clean on repro cases.
2. Velox: drop scalar broadcast workaround where no longer needed; keep `castDecimalScalar`.
3. Velox: `compute-sanitizer` on `decimalGreatestLeastMixed` and full `velox_cudf_decimal_expression_test`.

---

## Related Velox work

See `PLAN-VELOX-WORKAROUND.md` for the implemented Velox-side workaround:

- `CudfBinaryOpUtils.{h,cpp}` — broadcast helpers
- `ExpressionEvaluator.cpp` — `BinaryFunction`, `BetweenFunction`, `GreatestLeastFunction`
- `DecimalExpressionKernels.cpp` — `scatterNullsAtZeroDivisor`
