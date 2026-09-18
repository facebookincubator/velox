# Stage 2 Handover: ABFS Runtime, Admission, and Lifetime

Date: 2026-07-20
Base SHA: `06dec49a333b7e39f73a49ba9dd2c4ee12e4fd4e`
Status: **C2 PASS - SUPERSEDED BY STAGE 3 HANDOVER**

## Completion notice

C2 passed after the active/fiber-bound increment and the complete ASAN/TSAN
matrix. See [Stage 2 checkpoint](STAGE_2_CHECKPOINT.md) for the final verdict
and retained evidence. Continue with [Stage 3 handover](STAGE_3_HANDOVER.md).

The pending-work and evidence-matrix sections below are retained as the
historical continuation contract that led to C2. Their missing/pending labels
are superseded by the checkpoint and must not be used as current status.

## Purpose and boundary

Stage 1 and checkpoint C1 are complete. Stage 2 now includes bounded-runtime
and transport-pool lifecycle increments, but C2 has not passed. This handover
is for a fresh agent continuing Stage 2 from the current uncommitted
workspace.

Read these documents before editing:

- [Authoritative transport design](../ABFS_PREADV_ASYNC_FIBER_TRANSPORT.md)
- [Stage 1 checkpoint](STAGE_1_CHECKPOINT.md)
- [Stage 1 focused test evidence](results/stage1-main-06dec49a/tests/focused-tests.txt)
- [Stage 1 source integrity](results/stage1-main-06dec49a/source-integrity.txt)

Stage 2 turns the proven transport spike into bounded connector
infrastructure. It must not expose native ABFS reads yet. Do not add provider,
`AbfsFileSystem`, `AbfsReadFile`, dual-client, or `preadvAsync` wiring before
C2 passes. Those changes belong to Stage 3.

## Workspace topology

- `C:\velox` is the Windows staging and editor checkout on `main`.
- Build and execute only in the WSL Ubuntu-24.04 ext4 checkout at
  `~/src/velox`.
- The focused Debug build directory is `~/src/velox/abfs-stage1-debug`.
- Source changes are uncommitted. Preserve every existing modified or
  untracked ABFS artifact; do not reset or recreate the worktree.
- No branch or commit was created.

The Stage 0 benchmark block in `tests/CMakeLists.txt` remains frozen. Its
LF-normalized SHA-256 is:

```text
7b4af22ec2ddc0dcb812dc9d59edceb48262f31b9594a8a4453e1e230ba50e0d
```

Keep reports and command output free of credentials, SAS values,
authorization headers, account IDs, real endpoints, usernames, hostnames, and
private absolute paths.

## Completed Stage 2 increment

The following files were added:

- `AbfsAsyncRuntime.h`
- `AbfsAsyncRuntime.cpp`
- `tests/AbfsAsyncRuntimeTest.cpp`

The isolated `velox_abfs_stage2_runtime_test` target was added to
`tests/CMakeLists.txt`. No Stage 2 source was added to the production
`velox_abfs` target.

`AbfsAsyncRuntime` currently provides:

- Config-scoped `ScopedEventBaseThread` and `FiberManager` shards.
- A global `maxActiveRequests` bound on scheduled and executing fibers.
- A global bounded inactive queue controlled by `maxQueuedRequests`.
- Nonblocking submission that returns a ready failed future on overload.
- Stable endpoint-key-to-shard assignment for the life of a runtime.
- A configurable prototype fiber stack size, currently 256 KiB by default.
- Thread-safe admission, queue, completion, cancellation, and peak metrics.
- Cooperative shutdown cancellation for active tasks.
- Deterministic failure of queued tasks during shutdown.
- Completion and shutdown linearized under the runtime-state mutex.
- Safe final-owner destruction from a runtime fiber by moving blocking teardown
  to a detached helper thread.
- Runtime-owned endpoint states keyed and assigned to one runtime shard.
- Lazy EventBase-local construction of each endpoint's channel factory,
  transport, and connection pool.
- Rejection of conflicting endpoint options for an existing key.
- EventBase-local destruction of queued request state and endpoint transports
  after active requests complete.
- A dedicated bounded synchronous resolver executor, with one worker by
  default and an explicit pending-resolution queue bound.
- A bounded LRU endpoint and DNS cache with lazy positive and negative expiry.
- Deterministic single-flight for concurrent endpoint misses.
- Rejection when cache capacity is occupied entirely by in-flight resolutions.
- Resolver-queue cancellation and exactly-once request completion during
  shutdown.
- Injectable resolver and monotonic clock interfaces for deterministic tests.
- DNS cache, resolver, expiry, eviction, rejection, and failure metrics.
- Global active and inactive queue bounds proven across three runtime shards.
- Configured fiber stack size and peak active stack-capacity metrics.
- Optional Folly stack sampling with a measured high-watermark metric.
- Rejection of configurations whose active stack capacity would overflow.

The tests cover:

- Three active slots on three runtime shards, two global queued slots, and
  deterministic rejection of the next submission.
- Configured peak fiber stack capacity, sampled stack high-water use, and
  rejection of an unrepresentable active stack capacity.
- Shutdown with one active and one queued request.
- Idempotent explicit shutdown.
- Releasing the last public runtime owner from a runtime fiber.
- Execution inside Folly fibers and stable endpoint sharding across two
  EventBase threads.
- Reuse of one runtime-owned transport by concurrent requests for an endpoint.
- Public runtime destruction while private request state is queued, waiting for
  the endpoint's only connection, and holding an unread response body.
- Request-state destruction on the assigned runtime thread and peer-observed
  closure of the abandoned response-body connection.
- Resolver execution outside both the submitting and runtime EventBase threads.
- Shared DNS resolution, positive and negative expiry, cache capacity, resolver
  queue capacity, failure caching, and shutdown cancellation.

The existing Stage 1 transport files now also provide:

- A distinct bounded `connectionAcquire` timeout, capped by the remaining
  transaction deadline.
- A positive `connectionIdle` deadline for reusable pooled connections.
- One pool-owned `AsyncTimeout` tracking the earliest FIFO idle deadline.
- EventBase-affine idle eviction, socket destruction, timer cancellation,
  waiter handoff, and pool teardown.
- An `idleConnectionEvictions` metric in the pool snapshot.

The transport tests additionally cover:

- Idle eviction with peer-observed socket closure and zero final live
  connection counters.
- Reusable connection return on both sides of the waiter-timeout boundary.
- Exactly-once waiter completion when return and timeout share a deadline.
- Public transport destruction across leased, waiting, and final idle state.
- Idle pool and timer destruction initiated from outside the owning
  EventBase.

## Validation completed

The following checks passed in the WSL ext4 checkout:

- `velox_abfs_stage2_runtime_test`: 4 of 4 tests passed.
- The runtime binary passed 100 in-process repetitions of all four tests.
- `velox_abfs_stage1_transport_test`: complete suite passed after the pool
  lifecycle increment.
- Five focused idle, timeout-boundary, waiter-return, and teardown tests
  passed 100 in-process repetitions.
- The runtime-owned endpoint lifetime test passed 100 in-process repetitions.
- The complete Stage 1 transport and Stage 2 runtime suites passed after the
  endpoint ownership increment.
- The expanded runtime suite passes all 11 tests and passed 100 in-process
  repetitions, including all DNS cache and shutdown tests.
- The complete Stage 1 transport suite passes after the DNS increment.
- The expanded runtime suite passes all 12 tests after the active/fiber-bound
  increment.
- The focused multi-shard bound and capacity-overflow tests passed 100
  in-process repetitions.
- The measured Debug stack high watermark was 4,944 bytes with a configured
  131,072-byte stack and 393,216-byte peak capacity across three active fibers.
  See [active/fiber-bound evidence](results/stage2-main-06dec49a/tests/active-fiber-bounds.txt).
- Repository-pinned clang-format v21.1.2 check: passed.
- Git whitespace checks and editor diagnostics: clean.
- All files touched by this increment use LF endings.
- The Stage 0 benchmark block hash remains unchanged.

Representative commands:

```bash
cd ~/src/velox
cmake --build abfs-stage1-debug \
  --target velox_abfs_stage2_runtime_test -j2
ctest --test-dir abfs-stage1-debug \
  -R '^velox_abfs_stage2_runtime_test$' --output-on-failure
abfs-stage1-debug/velox/connectors/hive/storage_adapters/abfs/tests/\
velox_abfs_stage2_runtime_test \
  --gtest_repeat=100 --gtest_break_on_failure

abfs-stage1-debug/velox/connectors/hive/storage_adapters/abfs/tests/\
velox_abfs_stage1_transport_test \
  --gtest_filter='FollyHttpTransportTest.EvictsIdleConnectionAtTimeout:\
FollyHttpTransportTest.ConnectionReturnAtWaiterTimeoutBoundary:\
FollyHttpTransportTest.PoolAcquireTimeoutRemovesWaiter:\
FollyHttpTransportTest.DestroysLeasedWaitingAndIdlePoolState:\
FollyHttpTransportTest.DestroysIdlePoolOnOwningEventBase' \
  --gtest_repeat=100 --gtest_break_on_failure --gtest_brief=1

cmake --build abfs-stage1-debug \
  --target velox_abfs_stage1_transport_test -j2
ctest --test-dir abfs-stage1-debug \
  -R '^velox_abfs_stage1_transport_test$' --output-on-failure
```

The pinned formatter used for the check is available through the repository's
pre-commit environment. Locate it rather than assuming `clang-format` is on
`PATH`; the validated version is 21.1.2.

## Current runtime contract and limitations

The original runtime task type remains transport-neutral:

```cpp
folly::Function<void(const folly::CancellationToken&)>
```

The caller supplies an endpoint key and receives a
`folly::SemiFuture<folly::Unit>`. A new overload accepts a pre-resolved
`AbfsAsyncEndpointOptions` and a private task receiving the endpoint's
runtime-owned `FollyHttpTransport`. The endpoint state lazily owns its channel
factory, transport, and pool on its assigned shard. No Azure client or
connector-facing API is owned by the runtime yet.

Important current constraints:

- Shutdown is cooperative. It waits for active tasks to return after
  cancellation, so a task that ignores its token can delay shutdown.
- Explicit `shutdown()` from a runtime thread is rejected to prevent a
  self-wait. Destructor teardown from that context uses an off-thread helper.
- Admission and metrics are global across shards, not per-shard.
- The inactive queue is global FIFO. A completing shard may dispatch the next
  request to a different request's assigned shard.
- `std::hash<std::string>` assignment is suitable only for process-local
  affinity. Do not persist or externally expose its result.
- Runtime overload and shutdown currently use `std::runtime_error`; no public
  connector error contract has been introduced.
- No source is production-linked, and no public configuration keys are parsed.
- Endpoint and DNS expiry is lazy: an expired entry is replaced on its next
  lookup, not by a background timer.
- A synchronous resolver call already executing on a resolver worker cannot be
  preempted. Shutdown cancels queued resolution jobs but waits for an active
  resolver call to return.
- Exact fiber stack sampling is optional because painting and scanning sampled
  stacks adds overhead. Folly disables the measured watermark under ASAN, so
  ASAN reports zero while configured capacity remains available.
- Pool timers and functional boundary tests are complete, but no Stage 2 ASAN
  or TSAN result exists yet.

Do not mistake the functional unit tests for C2 completion.

## Existing Stage 1 transport state

Stage 1 already implements and tests:

- Plaintext and verified TLS asynchronous channels.
- Incremental HTTP/1.1 serialization and parsing.
- Content-Length, chunked, close-delimited, informational, and no-body
  framing.
- Bounded ingress and parser limits.
- Pull-based Azure `BodyStream` response streaming.
- A bounded EventBase-affine HTTP/1.1 connection pool.
- Consumed-body connection reuse and abandoned-body connection close.
- Pool acquisition timeout and FIFO fiber waiters.
- Real Azure `BlobClient::Download` parsing and C1 concurrency.

Do not redesign these layers without a failing Stage 2 test that identifies a
specific contract defect. Extend the nearest owning abstraction and preserve
the C1 tests.

## Remaining C2 work

### 1. Integrate the completed transport pool lifecycle

Idle eviction, timeout-boundary behavior, destruction ordering, and their
functional tests are complete in the isolated Stage 1 transport target. The
pool is under runtime-owned endpoint state. The remaining C2 work for this
surface is to pass ASAN and TSAN. Preserve the current rules:

- A fully consumed reusable response may return its connection to the pool.
- An unread or abandoned body closes its connection; do not drain it in the
  background.
- A non-reusable or failed connection decrements physical pool accounting
  exactly once.
- A waiter is completed exactly once by connection handoff, retry, timeout, or
  shutdown.
- Idle deadlines, timer callbacks, pool mutation, connection destruction, and
  waiter completion remain on the owning EventBase.

Do not replace the focused functional tests. Extend them only where
runtime-owned endpoint teardown or sanitizer evidence exposes a specific gap.

### 2. Connect endpoint ownership to runtime shards

The initial endpoint ownership and private request-state lifetime increment is
complete. The runtime now owns one lazily constructed transport and pool per
pre-resolved endpoint, and destroys them on the selected EventBase after
queued and active request state is gone.

Move from a string-only shard choice to runtime-owned endpoint state without
adding connector wiring. Each endpoint's channel factory, transport, pool,
socket, timers, and callbacks must remain affine to one EventBase for their
entire lifetime.

Keep the ownership graph directed:

```text
AbfsAsyncRuntime
  -> shared RuntimeState
  -> runtime shards and endpoint states
  -> admitted request state
  -> transport response body and connection lease
```

`RuntimeState` must never point to an `AbfsReadFile::Impl`. A future Stage 3
request may retain the runtime and file implementation, but Stage 2 must not
create that integration or a cycle.

The private request-state test covers runtime/public-owner destruction while a
request is queued, active, waiting for a connection, and holding a response
body. Preserve this coverage under ASAN and TSAN. Bounded DNS cache entries now
share these endpoint states without adding an owning reference from resolver
work back to `RuntimeState`.

### 3. Bounded DNS ownership and cache

The bounded resolver executor and endpoint cache increment is functionally
complete. DNS occurs once per endpoint miss and is shared. The default blocking
resolver executes only on the dedicated resolver worker, never an EventBase
thread.

The cache has explicit capacity, positive and negative TTLs, deterministic
single-flight, LRU eviction of completed entries, rejection when all entries
are resolving, shutdown cancellation, and metrics. Resolver jobs retain only
endpoint and completion state; the runtime owns and joins the executor.

Tests use an injectable resolver, injected monotonic clock, and condition
variables. They do not use public DNS, real Azure, or wall-clock sleeps. ASAN
and TSAN coverage for this increment remains pending.

### 4. Configured bounds complete

The runtime test now holds one active request on each of three shards, proves
the global active and inactive queue limits, and rejects the next submission.
Metrics report configured stack size, configured peak capacity as
`peakActiveRequests * fiberStackBytes`, and a separate Folly-sampled stack high
watermark. Overflowing capacity configurations are rejected. The measured
value is evidence from the focused Debug run, not inferred actual use for other
workloads. See [active/fiber-bound evidence](results/stage2-main-06dec49a/tests/active-fiber-bounds.txt).

### 5. Run sanitizer gates

Create separate Debug ASAN and TSAN build directories in WSL ext4. Do not
reuse or mutate the existing non-sanitized build. At minimum, repeatedly run:

- Queue saturation and shutdown.
- Runtime creation and destruction.
- Last-owner release from a runtime fiber.
- Abandoned response bodies.
- Idle eviction and timeout/completion boundary tests.
- Concurrent enqueue, completion, connection return, and shutdown.
- DNS miss, shared resolution, expiry, failure, and shutdown.

ASAN must report no use-after-free or leak. TSAN must report no race in
enqueue, completion, cancellation, timer handling, connection return, DNS
completion, or teardown. Retain sanitized commands and summaries under a new
`design/results/stage2-main-06dec49a/` directory.

## C2 evidence matrix

Do not write a passing Stage 2 checkpoint until every row has executable
evidence.

| C2 requirement | Current state | Required evidence |
| --- | --- | --- |
| Queue full | Covered | Focused test remains green under TSAN |
| Shutdown with queued and active requests | Covered | Repeated ASAN and TSAN runs |
| Stable endpoint assignment | Functionally covered | Runtime-owned transport identity and shard-affinity tests; TSAN pending |
| File/runtime destruction | Functionally covered | Private request-state lifetime test; ASAN and TSAN pending |
| Abandoned bodies | Functionally covered | Peer-observed Stage 2 runtime-owned transport lifetime test; sanitizers pending |
| Idle eviction | Functionally covered | Focused test and 100x repeat pass; ASAN and TSAN pending |
| Timer/completion races | Functionally covered | Both boundary orderings and 100x repeat pass; TSAN pending |
| Bounded DNS cache | Functionally covered | Capacity, expiry, single-flight, failure, and shutdown tests; sanitizers pending |
| No EventBase DNS blocking | Functionally covered | Injectable resolver thread-affinity test; TSAN pending |
| Active/fiber memory bound | Covered | Three-shard peak counters, configured capacity, sampled high watermark, and 100x focused repeat |
| ASAN clean | Missing | Repeated startup/shutdown and transport lifetime run |
| TSAN clean | Missing | Admission, completion, pool, timer, and DNS run |

## Recommended next increment

Create separate Debug ASAN and TSAN builds and run the full C2 sanitizer
matrix, including the DNS cache tests. Do not enable fiber stack sampling as a
required ASAN signal because Folly disables its watermark there; retain the
configured-capacity assertions and use ASAN for memory-safety and leak proof.

Before editing, read `AbfsAsyncRuntime.h`, `AbfsAsyncRuntime.cpp`, and
`tests/AbfsAsyncRuntimeTest.cpp`. Preserve the runtime-owned transport identity,
private request-state lifetime, DNS lifecycle, and active/fiber-bound tests.
Do not move provider or file-system ownership into Stage 2.

## Workflow and scope controls

- Use small, test-first increments. A bug fix needs a test that fails without
  the fix.
- After the first substantive edit, immediately compile and run the narrowest
  affected target.
- Keep `velox_abfs_stage1_transport_test` and
  `velox_abfs_stage2_runtime_test` isolated until C2 passes.
- Rerun the complete Stage 1 transport suite after each pool or connection
  lifecycle change.
- Preserve the existing test CA and loopback-only fixtures.
- Use EventBase timers and injectable clocks/hooks where appropriate; do not
  use sleeping production threads.
- Never execute a blocking resolver, Azure request, socket read/write, mutex
  wait, or backoff sleep on an EventBase thread.
- Preserve bounded response ingress and pull-based streaming.
- Keep SDK retries out of Stage 2 user-facing behavior. Retry preservation is
  a later checkpoint.
- Do not add OAuth, dynamic SAS, proxy, io_uring, or Fizz work in Stage 2.
- Do not modify Stage 0 benchmark sources or rerun its benchmark unless a
  measured regression question requires it.
- Do not create a branch, commit, or revert existing uncommitted files.

## Stage 2 checkpoint deliverables

When C2 passes, add `design/STAGE_2_CHECKPOINT.md` and sanitized evidence under
`design/results/stage2-main-06dec49a/`. Record:

- Base SHA and exact source scope.
- Runtime, endpoint, pool, timer, and DNS ownership behavior.
- Queue, active, connection, idle, waiter, DNS, and cancellation counters.
- Focused and regression test commands and results.
- ASAN and TSAN tool versions, commands, repeats, and summaries.
- Fiber stack and active-request bound evidence.
- Remaining risks and the exact Stage 3 entry condition.

The Stage 3 entry condition is a complete C2 pass. Until then, the provider
and `AbfsReadFile::preadvAsync` surfaces remain deferred.
