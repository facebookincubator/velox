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

## Flow-control policy

`rpc.flow_control_policy` selects how a query's rate limiters adapt their
window: `WINDOW_AIMD`, the default, or experimental `CAPACITY_PROBING`.
`CAPACITY_PROBING` takes effect only for adaptive `PER_ROW` execution of a
function that returns capacity retries; any other query runs `WINDOW_AIMD`.
Limiters are keyed by policy, so a `CAPACITY_PROBING` query gets its own limiter
per backend, with its own ceiling. A worker that runs both policies against one
backend can therefore admit up to two ceilings, and the two share whatever the
backend serves; pilot `CAPACITY_PROBING` on backends without default traffic.

`CAPACITY_PROBING` is `WINDOW_AIMD` changed only where `WINDOW_AIMD`'s window
is too small for the backend. It starts faster, and it answers rejections the
function names as a lack of serving capacity differently; its window never
grows past twice the most calls the worker has had in flight, a simpler form of
Linux TCP's rule that grows only a window that limits sending:

- Startup. Both policies start at the smaller of 200 and the ceiling.
  `CAPACITY_PROBING` doubles the window every round until the first overload,
  since nothing has yet shown where the backend's limit is. A backend that
  queues calls instead of refusing them may never overload, so doubling also
  stops once a round's fastest response is an eighth slower than the fastest
  so far, as in TCP's HyStart; the window then grows one call per round.
  Until the first successful response, each acquisition takes at most four
  calls, so a backend's first rejections arrive while the first burst is still
  going out.
- The floor. `WINDOW_AIMD` never cuts below `rpc.ratelimiter.min_limit` (50).
  When a worker's share of a backend is smaller, it keeps overloading the
  backend and rows fail after their retries. A capacity rejection may cut
  `CAPACITY_PROBING` below that floor and below one call, where the limiter
  paces admissions as it does for any window below one. The probe floor still
  admits one call per `rpc.ratelimiter.max_probe_interval_ms`.
- The cut. A capacity rejection keeps 80% of the calls the backend held when it
  refused, rather than half the window: the calls in flight besides the
  rejected ones, or the window if more were in flight. TCP cuts from the same
  flight size, and TCP BIC keeps the same 80%.
  Rejections of calls admitted before a cut belong to the round that cut
  answered and do not cut again, as under `WINDOW_AIMD`.
- Regrowth. `WINDOW_AIMD` grows one call per round, so with calls of 2-4 s a
  window halved from 200 takes minutes to recover. After a capacity rejection,
  `CAPACITY_PROBING` grows back to the calls the backend held, by half the
  remaining distance each round. Past that level it grows by a step that
  starts at one and doubles each round, up to twice the most calls in flight.
  Both are capped at 16 calls per round. This is TCP BIC's binary search and
  probing. A round is a window of successes.

Anything else answers as `WINDOW_AIMD` does. A rejection not named as a lack of
capacity, such as a rate quota, a rejection without a retry hint, a timeout or
an aggregate congestion verdict, halves the window down to the configured
floor, and the window then grows one call per round until the next capacity
rejection.

For example, a worker alone on a backend with room for 100 calls and no queue
cuts to 80 when it first exceeds 100, climbs back through 90, 95 and 97 in
three rounds, reaches 100 a few rounds later and cuts again when it tries 101:
about 94% of the slots stay busy. `WINDOW_AIMD` saws between 50 and 100 in
cycles of 50 rounds, unless its fixed driver window of 100 happens to match the
backend.

The rules give these properties. Workers do not talk to each other; each sees
only how its own calls end, and the backend refuses a call promptly when it is
full.

- Safety. The window stays between the probe floor and the ceiling, and never
  grows past twice the most calls the worker has had in flight. A cut never
  raises it. After the first capacity rejection, one round of growth exceeds
  the backend by at most 16 calls; slow start before it can overshoot by up to
  a window, as in TCP.
- Progress. A worker with work always has a call in flight or admits one
  within a pacing interval, at most `rpc.ratelimiter.max_probe_interval_ms`: a
  window of at least one call admits whenever fewer calls are in flight, and a
  smaller window admits one call per interval. So no worker is shut out, and a worker that
  arrives at a full backend gets room as soon as a cut frees it.
- Fairness, in expectation. When the backend is full, a slot freed by a
  completion goes to whichever call arrives first, so a worker's calls are
  rejected in proportion to its window, and each rejection removes 20% of
  that window. Its expected loss per round therefore grows with the square of
  its window, while its growth is at most 16 calls per round whatever its
  size. So the gap between two workers' windows shrinks in expectation, as in
  Chiu and Jain's argument for AIMD. A worker that arrives late at a full
  backend loses 20% of its window each round until it fits, then grows toward
  a share instead of waiting for others to yield.
- Recovery. During an outage every round's rejections cut, so the window falls
  to the probe floor and a paced call reaches the backend at least once per
  interval. After the first success the window doubles each service time back
  to one call, then grows by the doubling step, so it is back to a window W in
  about log2(16) + W/16 rounds; `WINDOW_AIMD` needs about W/2 rounds from its
  floor.
- Bounded rejections. After the first capacity rejection, each round of
  growth exceeds the backend by at most 16 calls, and the rejections of calls
  admitted before the cut that follows cost no further cut. A worker alone on
  a backend causes one such burst per cycle of about eight rounds.

Under `CAPACITY_PROBING` the shared limiter owns the backoff of capacity
rejections: it can always cut further or pace, so a retried capacity rejection
does not also halve the driver window. `WINDOW_AIMD`, and `CAPACITY_PROBING`
for any other rejection or for a row out of retries, halve the driver window
when overloaded calls come back, once for those that come back together. A `PER_ROW` driver window starts at its ceiling,
100 by default; it shrinks on overloads and rising latency and regrows through
its latency gradient, never past the ceiling. Under `CAPACITY_PROBING` that
ceiling rises to the limiter's ceiling, so the shared limiter, not a fixed
driver window, sets how many calls a worker keeps in flight. It does not rise
past a ceiling the function declares, which sizes the function's backend even
when a session limit takes precedence for the limiter, and it does not rise at
all when the session sets `rpc.congestion.max_window`. When only the 1,000,000
safety ceiling of `rpc.ratelimiter.hard_limit=0` bounds the limiter, it rises
to 200, where an adaptive limiter starts, so a driver never holds an unbounded
number of calls.

Each driver reports `rpcFlowControlRequestedPolicy` and
`rpcFlowControlEffectivePolicy` (0 for `WINDOW_AIMD`, 1 for
`CAPACITY_PROBING`), and the runtime stats sum them across drivers; read the
max.

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
| `RPCRateLimiter` | A per-backend admission controller whose window adapts to the congestion verdict. One instance per admission key and resolved policy, from a process-scoped `RPCRateLimiterRegistry`. | *In* `initializeOrMatch(Config policy)`, `onSuccessForEpoch(...)`, `handleOverloadForEpoch(...)`<br>*Out* `tryAcquireUpTo(want)` → `vector<Token>`; `admitOrWait()` → `Admission`; `makeLease(...)`; `available()`; `stats()`<br>The admission key and policy are a registry lookup, not parameters — one instance per key and policy. Integral windows bound in-flight units; fractional windows pace discrete admissions. | A row, a kind, or a model. Its input surface is an outcome, an epoch, a count and an observed service duration. |
| `AsyncRPCFunction` | The interface a SQL function implements to build requests, dispatch them, and turn responses into a column. The extension point. | *In* `initialize(queryConfig, inputTypes, constantInputs, mode)`; optionally `prepareInputForAdmissionAndDispatch(rows, args)`; then `dispatchPerRow(rows, args)` **or** `accumulateBatch(rows, args)` + `flushBatch(maxRows)`<br>*Out* futures of `RPCResponse`; `buildOutput(responses, pool)` → `VectorPtr`; `evaluateCongestion(responses)` → `CongestionSignal`. Declares `resultType()`, `admissionKey()`, and stable admission bounds. | The driver thread, spill, or mutable congestion-controller state. It proposes backend constraints but never sees `RPCState` or mutates a limiter. |
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
other query on that backend that runs the same policy. A drain that is mostly invalid requests must not
shrink either window.

Runtime stats distinguish the two backoff mechanisms.
`rpcCongestionOverloadShrinks` counts observable per-driver window reductions
caused by explicit overload. `rpcCongestionShrinks` also includes
latency-gradient reductions. `rpcRateLimiterMinCap` records the low-water
capacity of the controller the query ran under, which every query on that
backend with the same policy shares.

Three counters describe the shared controller from a driver's first admission
request until the stats sample or the driver's close.
`rpcRateLimiterAttachedNanos` is the length of that span.
`rpcRateLimiterPacingNanos` is the part of it during which the controller paced
admission. `rpcRateLimiterLimitMilliUnitSeconds` is the committed window, the
one `rpcRateLimiterLimitMilli` shows, integrated over the span, in thousandths
of a unit times seconds; recovery growth counts once the limiter commits it to
the window. Each driver reports its own counts, so their sums across drivers and
tasks stay meaningful on Prestissimo and on Presto-on-Spark. The sums are driver
time, not wall time. Divide the sums to read them. Pacing time over attached
time is the share of attached time during which the controller paced. It is an
upper bound on pacing delay, because it includes time a driver was idle or
waiting on upstream. The window integral over attached seconds is the average
window, in thousandths of a unit.

`rpcStaleSuccessUnits` counts each driver's successful units whose recovery
credit the controller discarded because an overload, including one in the same
completion, arrived after their admission. PER_ROW counts error-free rows; BATCH
counts the admission units of a flush with at least one successful row as a
whole, as the controller credits them.

An async-job path returns `kSuccessNoLatency` after a successful unit: its round
trip is queue and scheduling time, which says nothing about backend load, but
the success must still recover shared admission capacity.

A transport that retries inside one RPC hides those retries from congestion
control, which sees only the final outcome. The transport can report in
`RPCResponse::numOverloadedAttempts` how many attempts before the final one it
counted as overloaded, such as rate-limited, timed-out or shed attempts. Model
API also counts errors that carry no error code, which include its load
shedding. The operator counts them for PER_ROW rows in two stats, which stay
zero in BATCH mode and for transports that do not report:

- `rpcOverloadedAttempts` sums the reported overloaded attempts, failed rows
  included.
- `rpcSuccessesAfterOverload` counts successes that needed at least one.

The counts do not cut the driver window or the shared limiter the way an
overload error does.

### Why admission has two scopes

A per-driver window and a per-backend cap look redundant until you ask what each
learns from.

Round-trip time is interpretable only against the request mix that produced it.
Average a driver sending small prompts with one sending large ones and the
gradient signal is destroyed — so the latency estimator must be per driver.

Rejections are produced by the aggregate. No single driver's refusal count
reveals the backend's capacity, only the sum does — so the occupancy bound must
be per backend. Queries that run different policies against one backend each
get a bound of their own today, so together they can admit the sum of their
ceilings until a backend-level cap nests them.

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
