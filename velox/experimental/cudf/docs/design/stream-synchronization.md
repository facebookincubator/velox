# Stream Synchronization in velox-cudf

This document describes stream synchronization patterns and best practices for the velox-cudf integration. It is intended for developers familiar with CUDA streams and RMM.

## Table of Contents

1. [Overview](#overview)
2. [Core Concepts](#core-concepts)
3. [Synchronization Primitives](#synchronization-primitives)
4. [Common Patterns](#common-patterns)
5. [CPU vs GPU Synchronization](#cpu-vs-gpu-synchronization)
6. [Anti-Patterns](#anti-patterns)
7. [Design Decisions](#design-decisions)
8. [Reference Files](#reference-files)

---

## Overview

In velox-cudf, operators execute on multiple CUDA streams. Each `CudfVector` tracks the stream on which it was created. When data flows between operators or is concatenated from multiple sources, proper stream synchronization is required to avoid race conditions.

**Key challenges:**
- Tables may arrive from different drivers/operators on different streams
- Concatenation reads from multiple input streams, writes to an output stream
- Input tables must not be deallocated while the GPU is still reading from them

---

## Core Concepts

### Stream-Ordered Memory (RMM)

RMM provides stream-ordered allocation semantics:

- Memory allocated on stream A is valid for use on stream A immediately
- Using memory on a different stream requires explicit synchronization (via CUDA events)
- Deallocation is stream-ordered: `deallocate(ptr, stream)` enqueues the free on `stream`

**Implication:** With stream-ordered memory resources (pool, arena, async), memory won't actually be reused until pending work on that stream completes. This provides implicit safety for deallocation ordering.

### CudfVector Stream Tracking

Each `CudfVector` stores its creation stream:

```cpp
class CudfVector : public RowVector {
  rmm::cuda_stream_view stream_;  // Stream on which table was created
public:
  rmm::cuda_stream_view stream() const { return stream_; }
};
```

Operations that consume `CudfVector`s must respect this stream for proper synchronization.

---

## Synchronization Primitives

### CUDA Events

Events coordinate work between streams without blocking the CPU:

```
cudaEventRecord(event, streamA)     // Places marker in streamA's queue
cudaStreamWaitEvent(streamB, event) // streamB waits for marker
```

Timeline visualization:
```
Stream A:  [Kernel1]──[Kernel2]──[EventRecord]──[Kernel3]──►
                                      │
                                      │ (dependency)
                                      ▼
Stream B:  [KernelX]──────────────[WaitEvent]──[KernelY]──►
                                       │
                                       └─► KernelY waits until
                                           EventRecord completes
```

### CudaEvent Wrapper

`Utilities.h` provides a RAII wrapper:

```cpp
CudaEvent event(cudaEventDisableTiming);  // Timing disabled for efficiency
event.recordFrom(stream);   // Record marker on stream
event.waitOn(otherStream);  // Make otherStream wait for marker
```

### streamsWaitForStream

Makes multiple streams wait for a single source stream:

```cpp
void streamsWaitForStream(
    CudaEvent& event,
    const std::vector<rmm::cuda_stream_view>& streams,  // Streams that will wait
    rmm::cuda_stream_view stream);                       // Source stream
```

Implementation:
1. Records event on `stream`
2. Each stream in `streams` waits on that event

This is more efficient than calling `join_streams` in a loop (single event creation, single record).

---

## Common Patterns

### Multi-Stream Concatenation

When concatenating tables from multiple streams (`getConcatenatedTable`, `getConcatenatedTableBatched`):

```cpp
std::unique_ptr<cudf::table> getConcatenatedTable(
    std::vector<CudfVectorPtr>&& tables,  // Rvalue: takes ownership
    const TypePtr& tableType,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {

  // 1. Collect input streams and table views
  std::vector<rmm::cuda_stream_view> inputStreams;
  std::vector<cudf::table_view> tableViews;
  for (const auto& table : tables) {
    tableViews.push_back(table->getTableView());
    inputStreams.push_back(table->stream());
  }

  // 2. Make output stream wait for all inputs before reading
  cudf::detail::join_streams(inputStreams, stream);

  // 3. Perform concatenation on output stream
  auto output = cudf::concatenate(tableViews, stream, mr);

  // 4. Make input streams wait for output stream before deallocation
  //    (ensures concatenate finishes reading before inputs are freed)
  CudaEvent event(cudaEventDisableTiming);
  streamsWaitForStream(event, inputStreams, stream);

  // 5. Input tables deallocated here when 'tables' goes out of scope
  //    Deallocation is stream-ordered on each input's stream
  return output;
}
```

**Why both directions of synchronization?**
- Step 2 (`join_streams`): Output stream waits for inputs to be ready before reading
- Step 4 (`streamsWaitForStream`): Input streams wait for output before deallocation

### Rvalue Reference for Input Ownership

When a function consumes and deallocates inputs:

```cpp
// Function signature takes rvalue reference
[[nodiscard]] std::unique_ptr<cudf::table> getConcatenatedTable(
    std::vector<CudfVectorPtr>&& tables,  // Takes ownership
    ...);

// Caller transfers ownership with std::exchange (for member variables)
auto result = getConcatenatedTable(std::exchange(inputs_, {}), type, stream, mr);
// inputs_ is now explicitly empty
```

**Why `std::exchange` instead of `std::move` for member variables?**

`std::move(member_)` leaves the member in a valid but *unspecified* state. While the moved-from vector is typically empty, this is not guaranteed by the standard. Using `std::exchange(member_, {})` explicitly resets the member to an empty state, making the code's intent clear and avoiding subtle bugs.

```cpp
// ❌ Avoid: leaves inputs_ in unspecified state
getConcatenatedTable(std::move(inputs_), ...);

// ✅ Prefer: explicitly resets inputs_ to empty
getConcatenatedTable(std::exchange(inputs_, {}), ...);
```

**Note:** For local variables, `std::move` is fine since the variable goes out of scope immediately.

**Benefits:**
- Encapsulates synchronization logic in the function
- Prevents callers from prematurely clearing buffers (race condition)
- Makes ownership transfer explicit
- `std::exchange` ensures member is in a known state after the call

---

## CPU vs GPU Synchronization

| Operation | Semantics | Blocks CPU? |
|-----------|-----------|-------------|
| `join_streams(inputs, target)` | Target stream waits for all input streams | No |
| `streamsWaitForStream(event, targets, source)` | All target streams wait for source | No |
| `stream.synchronize()` | CPU blocks until all work on stream completes | **Yes** |

### When is stream.synchronize() needed?

**Generally not needed** with stream-ordered memory resources. The `streamsWaitForStream` pattern ensures proper GPU-side ordering for deallocation.

**May be needed** when:
- Callers might not use stream-ordered memory resources
- You need CPU-side visibility of results immediately
- Interoperating with code that expects synchronous completion

**Trade-off:** `stream.synchronize()` is simpler but blocks the CPU, potentially reducing pipeline parallelism.

---

## Anti-Patterns

### ❌ Loop-based stream waiting

```cpp
// Inefficient: creates a new event on each iteration
auto const outputStreams = std::vector<rmm::cuda_stream_view>{stream};
for (auto& s : inputStreams) {
  cudf::detail::join_streams(outputStreams, s);
}
```

### ✅ Use streamsWaitForStream

```cpp
// Efficient: single event creation, single record, multiple waits
CudaEvent event(cudaEventDisableTiming);
streamsWaitForStream(event, inputStreams, stream);
```

---

### ❌ Clearing buffer after async operation

```cpp
auto tables = getConcatenatedTable(buffer_, ...);  // Takes lvalue ref
buffer_.clear();  // Race condition! GPU may still be reading from buffer_
```

### ✅ Rvalue ownership transfer with std::exchange

```cpp
auto tables = getConcatenatedTable(std::exchange(inputs_, {}), ...);
// Function handles sync and deallocation; inputs_ is explicitly empty
```

---

### ❌ Ignoring input stream differences

```cpp
// Assumes all inputs are on the same stream
auto stream = inputs_.front()->stream();
// ... use stream for all operations ...
```

### ✅ Collect and synchronize all input streams

```cpp
std::vector<rmm::cuda_stream_view> inputStreams;
for (const auto& input : inputs_) {
  inputStreams.push_back(input->stream());
}
cudf::detail::join_streams(inputStreams, outputStream);
```

---

## Design Decisions

### close() Method Synchronization

**Decision:** Most cuDF operators do not explicitly synchronize in `close()`.

**Rationale:**
- **Normal operation:** `getOutput()` synchronizes before returning, so pending work completes before `close()` is called
- **Cancellation:** When cancelled, `close()` is called with potentially pending GPU work. This relies on stream-ordered memory resources for safety—deallocations are ordered on each input's stream
- **Consistency:** This approach is used across cuDF operators (CudfOrderBy, CudfHashJoinProbe, CudfToVelox, etc.)

**Risk:** If not using stream-ordered memory resources, cancellation could cause use-after-free. This is an accepted trade-off given the typical deployment uses RMM pool/arena allocators.

### Event Lifetime

**Decision:** Create local `CudaEvent` in utility functions rather than passing as parameter.

**Rationale:**
- Simpler API for callers
- Event creation overhead is negligible compared to `stream.synchronize()` or actual GPU work
- For hot-path operations called repeatedly (e.g., `CudfTopN::mergeTopK`), member variable events can be used

---

## Reference Files

| File | Contents |
|------|----------|
| `velox/experimental/cudf/exec/Utilities.h` | `CudaEvent`, `streamsWaitForStream`, `getConcatenatedTable` declarations |
| `velox/experimental/cudf/exec/Utilities.cpp` | Implementation of concatenation patterns |
| `velox/experimental/cudf/vector/CudfVector.h` | `CudfVector` class with `stream_` member |
| `velox/experimental/cudf/exec/CudfHashJoin.cpp` | Multi-stream join patterns |
| `velox/experimental/cudf/exec/CudfHashAggregation.cpp` | Aggregation stream handling |
