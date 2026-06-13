# Remote Function Execution Enhancements

*2026-03-14*

## Motivation

Velox supports remote function execution ([#5288][issue]) — a framework that
allows scalar functions to be evaluated in a separate process via Thrift RPC.
This enables language interoperability (e.g., calling Java SparkSQL functions
from C++ Velox), process isolation, and resource separation.

The core framework (client, server, Thrift IDL, serialization) was delivered
in [#4595][pr4595] with subsequent improvements for Unix domain sockets
([#5482][pr5482]), named serializers ([#5370][pr5370]), and open-source build
([#5730][pr5730]).

Several items remain open from the original issue:

1. **Active-row serialization** — Currently, all rows in a batch are serialized
   and sent to the remote server, even when only a subset is selected (e.g.,
   inside an `IF` or `CASE WHEN`). This wastes network bandwidth and server
   compute.

2. **Non-deterministic and non-default null behavior** — The metadata fields
   `deterministic` and `defaultNullBehavior` are inherited but have no test
   coverage, making it unclear whether they work correctly with remote
   functions.

3. **Error re-throwing semantics** — Remote errors are deserialized and
   re-thrown as `std::runtime_error`, losing the `VeloxUserError` type that
   callers expect. A stale TODO in the Thrift IDL suggests the error format
   was never finalized.

This design document describes the approach for addressing all three items.

[issue]: https://github.com/facebookincubator/velox/issues/5288
[pr4595]: https://github.com/facebookincubator/velox/pull/4595
[pr5482]: https://github.com/facebookincubator/velox/pull/5482
[pr5370]: https://github.com/facebookincubator/velox/pull/5370
[pr5730]: https://github.com/facebookincubator/velox/pull/5730

## Background

### Current Architecture

```
Client (RemoteVectorFunction)           Server (RemoteFunctionServiceHandler)
─────────────────────────────           ─────────────────────────────────────
1. Wrap args into RowVector              1. Deserialize RowVector from IOBuf
2. Serialize RowVector → IOBuf           2. Build ExprSet with CallTypedExpr
3. Build RemoteFunctionRequest           3. Evaluate expression
4. Thrift RPC ──────────────────────>    4. Serialize result RowVector → IOBuf
5. Deserialize result IOBuf          <── 5. Return RemoteFunctionResponse
6. Extract result vector                 6. (Optional) Serialize error payload
```

### Key Types

| Type | Location | Purpose |
|------|----------|---------|
| `RemoteVectorFunction` | `client/RemoteVectorFunction.h` | Abstract base; serializes input, invokes remote, deserializes output |
| `RemoteThriftFunction` | `client/Remote.cpp` | Concrete subclass adding Thrift transport |
| `RemoteVectorFunctionMetadata` | `client/RemoteVectorFunction.h` | Extends `VectorFunctionMetadata` with `serdeFormat`, `preserveEncoding` |
| `RemoteFunctionServiceHandler` | `server/RemoteFunctionService.h` | Thrift service handler; evaluates functions via `ExprSet` |
| `PageFormat` | `if/RemoteFunction.thrift` | Enum: `PRESTO_PAGE` or `SPARK_UNSAFE_ROW` |

### Serialization Path

Both `rowVectorToIOBuf()` and `rowVectorToIOBufBatch()` accept an `IndexRange`
parameter specifying which rows to serialize. Currently, the client always
passes `{0, rows.end()}` — the full batch.

The serializer APIs (`IterativeVectorSerializer::append()` and
`BatchVectorSerializer::serialize()`) natively support
`folly::Range<const IndexRange*>`, allowing non-contiguous row ranges in a
single call.

## Design

### 1. Active-Row Serialization

**Problem**: When a remote function is called inside a conditional expression
(e.g., `IF(c0 > 2, remote_fn(c0), c0)`), the expression engine passes a
`SelectivityVector` indicating which rows are active. The current
implementation ignores this and serializes all rows.

**Approach**: Convert the `SelectivityVector` into contiguous `IndexRange`
runs and pass them to the serialization API.

#### Client-Side Changes (`RemoteVectorFunction.cpp`)

**Serialization phase**:

```cpp
// Convert SelectivityVector to IndexRange runs.
std::vector<IndexRange> activeRanges = toIndexRanges(rows);
auto rangesView = folly::Range<const IndexRange*>(
    activeRanges.data(), activeRanges.size());

// Serialize using the ranges.
streamGroup->append(remoteRowVector, rangesView, scratch);
```

The helper `toIndexRanges()` iterates set bits via `bits::forEachSetBit()`
and merges contiguous positions into `{begin, size}` pairs. For example,
a `SelectivityVector` with bits `{0, 0, 1, 1, 1}` produces
`[{begin=2, size=3}]`.

**Row count**: `requestInputs->rowCount()` is set to the number of *selected*
rows (sum of range sizes), not the full batch size.

**Deserialization phase**: The server returns a compacted result with N' rows
(where N' <= N). The client must scatter these back to original positions:

```cpp
if (!allSelected) {
  auto fullResult = BaseVector::create(outputType, rows.end(), pool);
  vector_size_t compactIdx = 0;
  rows.applyToSelected([&](vector_size_t origRow) {
    fullResult->copy(compactedResult.get(), origRow, compactIdx, 1);
    ++compactIdx;
  });
  result = fullResult;
}
```

**Error mapping**: When `throwOnError` is false and the server returns an error
payload, error indices are similarly mapped from compacted to original
positions using the same `applyToSelected` iteration.

**Fast path**: When `rows.isAllSelected()`, the existing single-range path is
used with no scatter overhead.

#### Server-Side Changes

None required. The server receives fewer rows, evaluates them, and returns a
correspondingly smaller result. The `rowCount` field correctly reflects the
compacted count.

#### Performance Characteristics

| Scenario | Before | After |
|----------|--------|-------|
| All rows selected | Serialize N rows | Same (fast path) |
| K of N rows selected | Serialize N rows | Serialize K rows |
| Scatter overhead | None | O(K) copies |

The net effect is positive whenever K < N, which is the common case for
conditional expressions.

### 2. Non-Deterministic and Non-Default Null Behavior

**Problem**: `RemoteVectorFunctionMetadata` inherits `deterministic` and
`defaultNullBehavior` from `VectorFunctionMetadata`, and
`registerRemoteFunction()` passes the metadata through to
`registerStatefulVectorFunction()`. The plumbing works, but there is no test
coverage.

**Approach**: No production code changes needed. The fix is purely test
coverage:

- **`nonDefaultNullBehavior` test**: Register a server-side function
  (`NullableDoubleFunction`) that uses `callNullable` to handle null inputs
  explicitly (returns -1 for null, doubles the value otherwise). Register the
  remote wrapper with `metadata.defaultNullBehavior = false`. Verify that null
  inputs produce the sentinel value, not null.

- **`nonDeterministic` test**: Register a remote function with
  `metadata.deterministic = false`. Verify correct execution (smoke test,
  since determinism is an optimizer hint).

### 3. Error Re-Throwing Semantics

**Problem**: When `throwOnError` is false and the server captures errors,
they are serialized as VARCHAR strings in `errorPayload`. On the client side,
these are re-thrown as `std::runtime_error`:

```cpp
// Before:
throw std::runtime_error(std::string(errorsVector->valueAt(i)));
```

This loses the `VeloxUserError` type, which callers (e.g., `TRY()`) may
depend on for proper error classification.

**Approach**: Use `VELOX_USER_FAIL` instead:

```cpp
// After:
VELOX_USER_FAIL("{}", errorsVector->valueAt(i));
```

This produces a `VeloxUserError` with proper error code and type, consistent
with how other Velox functions report user-facing errors.

Additionally, the stale TODO comment in `RemoteFunction.thrift` (which
suggests the error format needs to be defined) is replaced with documentation
describing the implemented behavior.

**Future work**: For full fidelity, structured error propagation could carry
the original error code, type, file, and line from the server. This would
require a new Thrift struct and is deferred as a separate effort. The current
string-based approach is sufficient for the `TRY()` use case.

## Alternatives Considered

### Active-Row Serialization

**Alternative A: Row-indices-based serialization** — Instead of `IndexRange`
runs, pass individual row indices via `append(vector, rows, scratch)`. This
was rejected because the `IterativeVectorSerializer` base class returns
`VELOX_UNSUPPORTED` for this overload, meaning not all serde implementations
support it.

**Alternative B: Copy selected rows into a compact vector, then serialize** —
Create a new RowVector containing only the selected rows, then serialize it
as a single contiguous range. This was rejected because it requires an
extra copy step and additional memory allocation, while the `IndexRange`
approach achieves the same result with zero-copy.

### Error Re-Throwing

**Alternative: Structured error Thrift struct** — Define a new Thrift struct
carrying error code, type, message, file, and line. This provides full
error fidelity but adds complexity to the Thrift IDL and both client/server
implementations. Deferred to a future change since the string-based approach
works for current use cases.

## Testing

### New Test Cases

| Test | Fixture | Description |
|------|---------|-------------|
| `nonDefaultNullBehavior` | `RemoteFunctionTest` (parameterized) | Nullable inputs with `defaultNullBehavior=false` |
| `nonDeterministic` | `RemoteFunctionTest` (parameterized) | Smoke test with `deterministic=false` |
| `partialSelection` | `RemoteFunctionTest` (parameterized) | End-to-end with `if(c0 > 2, remote_plus(...), c0)` |
| `activeRowsSerialization` | `MockRemoteFunctionTest` | Verifies request `rowCount` matches selected rows |

All parameterized tests run with both `PRESTO_PAGE` and `SPARK_UNSAFE_ROW`
serialization formats.

### Verification

```bash
cmake -B _build/debug -DCMAKE_BUILD_TYPE=Debug -DVELOX_ENABLE_REMOTE_FUNCTIONS=ON
cmake --build _build/debug --target velox_functions_remote_test
cd _build/debug && ctest -R RemoteFunctionTest
```

## Files Changed

| File | Change |
|------|--------|
| `velox/functions/remote/client/RemoteVectorFunction.cpp` | Active-row serialization, scatter, error re-throw |
| `velox/functions/remote/client/tests/RemoteFunctionTest.cpp` | 4 new tests, `NullableDoubleFunction` UDF |
| `velox/functions/remote/if/RemoteFunction.thrift` | Replace stale TODO with accurate docs |
| `velox/docs/develop/remote-functions.rst` | New developer documentation |
| `velox/docs/develop.rst` | Toctree entry for new doc |
