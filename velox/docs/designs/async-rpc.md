# Async RPC Functions

A SQL function that calls a remote model breaks what a vectorized engine
assumes: the call takes seconds rather than microseconds, the service meters it
and can refuse, and a fraction of calls fail for reasons unrelated to the query.

That forces three decisions, and most of what follows is mechanism for them:

- **How many calls may be outstanding.** Too few wastes a service you pay for,
  too many collapses it, and capacity is shared with callers you cannot see.
- **How many rows go into one call.** Services meter rows, bytes or tokens, not
  calls.
- **What a row that never came back evaluates to.** Failing a million-row query
  because eleven rows timed out is usually the wrong answer.

This document describes the shape the pieces form;
`develop/async-rpc-functions.rst` is the how-to for writing a function.

## Layering

Four responsibilities, separated by what each is in a position to know:

- **The engine boundary** owns vectors, the driver thread and memory, and knows
  nothing about remote calls — `RPCOperator` and `RPCState`.
- **Admission** decides how many calls may be outstanding, and knows nothing
  about what is in them — `CongestionController` and `RPCRateLimiter`.
- **Meaning** knows what a completion or an embedding is: the request shape and
  the parameters that apply to it — `AsyncRPCFunction`.
- **Reach** knows one backend service: its IDL, its retry semantics, its limits
  — the transports.

Six components implement those four. The useful column below is the last one:
what a component structurally *cannot* say is checkable, where a description of
what it does is not.

| Component | Is | In / Out | Cannot express, and by what mechanism |
| --- | --- | --- | --- |
| `RPCOperator` | A Velox `Operator` that turns input rows into RPCs and responses into output rows. | *In* `addInput(RowVectorPtr)`, `noMoreInput()`<br>*Out* `getOutput()` → `RowVectorPtr`; `isBlocked(ContinueFuture*)` → `BlockingReason` | That this is model inference, or which backend. It holds no transport and forwards `admissionKey_` without parsing it. |
| `RPCState` | The per-driver ledger of outstanding RPCs, and the boundary between the driver thread and transport threads. | *In* `addPendingRow(...)`, `addPendingBatch(...)`, `onUnitSample(rttNs)`, `onUnitError()`, `storeInputBatch(...)`<br>*Out* `tryClaimReady()` → `optional<ReadyRow>`; `tryPollReady()` → `optional<ReadyBatch>`; `numInFlight()`; `dispatchHeadroom()`; `isFinished()` | A request. It stores a row id, a `RowLocation` and a future; there is no field a request could go in. |
| `CongestionController` | A latency-gradient law turning round-trip samples into one integer window. | *In* `onSample(rttNs)`, `onError()`<br>*Out* `limit()` → `int64_t`; `baselineRttNs()`; `numShrinks()` | A backend, a row, or another driver. It has no admission key and no handle to shared state. |
| `RPCRateLimiter` | A per-backend admission semaphore whose capacity adapts to the congestion verdict. One instance per admission key, from a process-scoped `RPCRateLimiterRegistry`. | *In* `configure(Config)`, `onOutcome(Outcome, units)`<br>*Out* `tryAcquireUpTo(want)` → `vector<Token>`; `admitOrWait()` → `Admission`; `available()`; `stats()`<br>The admission key is a registry lookup, not a parameter — one instance per key. | A row, a kind, or a model. Its input surface is an outcome enum and a count. |
| `AsyncRPCFunction` | The interface a SQL function implements to build requests, dispatch them, and turn responses into a column. The extension point. | *In* `initialize(queryConfig, inputTypes, constantInputs, mode)`; optionally `prepareInputForAdmissionAndDispatch(rows, args)`; then `dispatchPerRow(rows, args)` **or** `accumulateBatch(rows, args)` + `flushBatch(maxRows)`<br>*Out* futures of `RPCResponse`; `buildOutput(responses, pool)` → `VectorPtr`; `evaluateCongestion(responses)` → `CongestionSignal`. Declares `resultType()` and `admissionKey()`. | The driver thread, spill, or the congestion window. It is handed rows and returns futures, and never sees `RPCState`. |
| Transport | A role, not a base class: one adapter per backend service, each with its own IDL and retry semantics. | *In* `scheduleAsync(rowId, isNull, MethodInvoker)`<br>*Out* `SemiFuture<RPCResponse>` carrying a `RPCErrorKind` | A kind or a SQL type. The closure captures an already-built typed request. |

Adding a backend takes a new adapter and an enum value. If a new backend has to
reach into admission or the operator, the separation is decorative.

## Threads, and why payloads are plain C++

A response is produced on a transport's executor thread and consumed on the
driver thread. Velox memory allocation must happen on the driver thread, so a
payload cannot be a Velox vector: it would have to be allocated where the
response arrives.

Payloads are therefore plain C++ objects — `std::vector<float>` for an
embedding, `std::string` for text. `buildOutput()` is the crossing: it runs on
the driver thread, and it is the only place a `VectorPtr` is built.

This is why `buildOutput()` takes a `memory::MemoryPool*` and returns a vector
rather than filling one supplied by the caller.

## What a response carries

The framework reads three things from a response: the row id, whether it
failed, and the typed cause. Everything a backend actually returns lives in a
function-owned payload the framework moves but never opens.

A response is a payload or an error, never both and never neither. `RPCResponse`
holds the outcome behind `setPayload()` / `setError()`. A default-constructed
response reads as the defensive `kUnset` error; the operator rejects that
sentinel if it reaches congestion evaluation or output construction.

`responseAs<T>()` reads the payload back as the concrete type. The cast is
checked in every build: a transport helper shared by several functions writes
one payload type for all of them, so a function does not always read back the
type it wrote.

## Execution mode and dispatch path

Two decisions turn what a query asked for into a call, and they are made in
different places because they need different knowledge.

**Execution mode** (`RPCStreamingMode`: `kPerRow`, `kBatch`) is the
coordinator's. `RpcExecutionPolicy` resolves the requested objective, and a
policy may inspect table statistics. The OSS default maps `AUTOMATIC` to
`PER_ROW`. The resolved mode crosses to the worker on the plan node.

**Dispatch path** (`RpcDispatchPath`: `kPerRow`, `kNativeBatch`, `kAsyncJob`) is
the function's. Choosing it needs to know the backend: "batch" is not one thing,
since some backends answer a multi-row request on the same call while others run
an offline job — submit, poll, fetch — whose round trip measures queue time
rather than load.

The function derives the path from the backend it has resolved, so the two
cannot disagree: a backend settled lazily, after `initialize()`, changes the
path with it. Nothing in the framework branches on the
path; a function expresses the consequences of its own choice through the other
hooks it implements. Concrete functions may retain the path internally for
backend bounds, metrics, or congestion semantics.

## The flush boundary

In batch mode the operator accumulates rows and the function flushes them. The
row-id protocol at that boundary is the part that is easy to get wrong.

1. `accumulateBatch(rows, args)` returns the indices it accepted. The operator
   records one `RowLocation` and one **global** row id per accepted row.
2. `flushBatch(maxRows)` returns responses whose `rowId` is the **batch
   position** — the 0-based index within the flushed batch, not the global id.
3. Responses may arrive in any order. The operator sorts them by batch position,
   then overwrites each `rowId` with the global id for that position.

Step 3 is `scatterIntoBatchOrder()`. It enforces three invariants, each of
which indicates the function broke the protocol: the response count equals the
row count; every batch position is in range; and no position appears twice.

Violating one fails the query. Once response count or position is malformed,
the operator cannot determine which response belongs to which row; degrading
the flush to NULL would hide a framework contract violation. Backend and
transport failures that preserve row correspondence remain typed row errors
and continue through the function's configured error policy.

## Congestion

`evaluateCongestion(responses)` is how a function tells the framework what a
completed unit means on its backend. It returns one of five signals:

- `kSuccess` — feed the round trip to the latency gradient, and recover
  admission capacity.
- `kSuccessNoLatency` — recover admission capacity without feeding the round
  trip to the latency gradient.
- `kOverloaded` — the backend pushed back. Both scopes back off.
- `kNonOverloadError` — the unit failed without explicit evidence of overload.
  **Neither scope backs off.**
- `kNone` — nothing to learn, including an all-null-input unit.

The `kNonOverloadError` case is the one worth stating. A deterministic client
error fails identically at any concurrency, so backing off cannot help — and
because admission capacity is shared, treating it as overload degrades every
other query on that backend. A drain that is mostly invalid requests must not
shrink either window.

Runtime stats distinguish the two backoff mechanisms.
`rpcCongestionOverloadShrinks` counts observable per-driver window reductions
caused by explicit overload. `rpcCongestionShrinks` also includes
latency-gradient reductions. `rpcRateLimiterMinCap` records the shared
controller's low-water capacity.

An async-job path returns `kSuccessNoLatency` after a successful unit: its round
trip is queue and scheduling time, which says nothing about backend load, but
the success must still recover shared admission capacity.

### Why admission has two scopes

A per-driver window and a per-backend cap look redundant until you ask what each
learns from.

Round-trip time is interpretable only against the request mix that produced it.
Average a driver sending small prompts with one sending large ones and the
gradient signal is destroyed — so the latency estimator must be per driver.

Rejections are produced by the aggregate. No single driver's refusal count
reveals the backend's capacity, only the sum does — so the occupancy bound must
be per backend.

Neither signal is derivable from the other. A per-driver window also cannot go
below one, so under per-driver control alone the least load a worker can offer
is one call per active driver, and there are many drivers per worker.

## Types

Two type relationships hold, and only one of them is checked.

`resultType()` against the plan **is checked**. `RPCOperator` compares the
function's `resultType()` with the type the plan declares for the result column,
at operator initialization. The plan node separately checks its call expression
against that column, but neither knows what the registered function returns, so
this check is where the two meet.

The payload type against `resultType()` **is convention**. Nothing verifies that
a function whose `resultType()` is `ARRAY(REAL)` produces a payload carrying
floats. What it gets instead is a runtime failure by name: `responseAs<T>()`
rejects a payload of the wrong type in every build, rather than reinterpreting
its bytes. A function's `buildOutput()` and its payload type have to be written
to agree.

## Invariants

Each is checkable against the code.

1. **The admission controller never inspects row contents.** A function may
   inspect selected rows before the operator binds admission, solely to resolve
   a stable admission identity and configuration or to prove that every row
   completes locally. Once bound, shared control sees only an opaque key, unit
   counts and outcome signals.
2. **The driver thread never blocks on a network call.** A correctness property,
   not a performance one: a parked driver thread can be reused, a blocked one
   cannot.
3. **A backend is named in exactly one layer.** If the shared machinery has to
   know which service this is, it needs changing for the next one.
4. **The framework never opens a payload.** It moves it from the transport to
   `buildOutput()`; only the owning function interprets it.
