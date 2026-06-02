# cuDF column-vs-scalar fixed_point binary op repro

Minimal standalone program that triggers the cuDF bug in
`scalar_as_column_view` (`cpp/src/binaryop/compiled/binary_ops.cu`): scalar
validity is a single device `bool`, but null-aware compiled kernels treat it as
a column bitmask and `bit_is_set` reads 4 bytes from a 1-byte allocation.

This matches the failure seen in Velox `greatest`/`least` with decimal literals
(`NULL_MAX` / `NULL_MIN` column-vs-scalar).

See also:

- `PLAN-CUDF-UPSTREAM-FIX.md` — root cause and upstream fix proposal
- `PLAN-VELOX-WORKAROUND.md` — Velox broadcast workaround (implemented)

## Build

### In-tree (recommended)

From an existing Velox release build with `VELOX_ENABLE_CUDF=ON` (reconfigure
if this directory was added after the last `cmake` run):

```bash
cmake --build /opt/velox-build/release --target cudf_nullmax_scalar_repro -j
```

Binary path:

```text
/opt/velox-build/release/velox/cudf-binop-scalar-repro/cudf_nullmax_scalar_repro
```

### Standalone (optional)

Configure from this directory; point `CMAKE_PREFIX_PATH` at a Velox build tree
that already fetched cuDF:

```bash
cd velox/cudf-binop-scalar-repro
cmake -S . -B build \
  -DCMAKE_PREFIX_PATH=/opt/velox-build/release \
  -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build build -j
./build/cudf_nullmax_scalar_repro
```

## Run

```bash
/opt/velox-build/release/velox/cudf-binop-scalar-repro/cudf_nullmax_scalar_repro
```

Run a single case:

```bash
.../cudf_nullmax_scalar_repro --only null_max
```

Available `--only` values: `null_max`, `null_min`, `null_equals`, `equal`, `add`.

## compute-sanitizer

Null-aware cases should report an invalid global read; controls should be clean:

```bash
compute-sanitizer --tool memcheck \
  /opt/velox-build/release/velox/cudf-binop-scalar-repro/cudf_nullmax_scalar_repro

compute-sanitizer --tool memcheck \
  /opt/velox-build/release/velox/cudf-binop-scalar-repro/cudf_nullmax_scalar_repro \
  --only null_max
```

Expected sanitizer signature (abridged):

```text
Invalid __global__ read of size 4 bytes
... NullMax ... fixed_point ...
Access ... is out of bounds
and is inside the nearest allocation ... of size 1 bytes
```

## Cases

| Case | Op | Sanitizer |
|------|-----|-----------|
| `null_max` | `NULL_MAX` column vs scalar | OOB (bug) |
| `null_min` | `NULL_MIN` column vs scalar | OOB (bug) |
| `null_equals` | `NULL_EQUALS` column vs scalar | OOB (bug) |
| `equal` | `EQUAL` column vs scalar | Clean (control) |
| `add` | `ADD` column vs scalar | Clean (control) |

## Upstream fix direction

In cuDF, avoid casting `scalar::validity_data()` (`bool*`) to `bitmask_type*`
when building the temporary `column_view` for compiled binops; materialize a
real 1-bit null mask or branch on `is_*_scalar` in `ops_wrapper`.
