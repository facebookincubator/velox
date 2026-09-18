# Stage 2 checkpoint: ABFS runtime, admission, and lifetime

Date: 2026-07-20
Base SHA: `06dec49a333b7e39f73a49ba9dd2c4ee12e4fd4e`
Verdict: **C2 PASS**

## Scope

Stage 2 turns the isolated Stage 1 transport into bounded connector
infrastructure without exposing native ABFS reads. The runtime and transport
remain in isolated test targets. No provider, `AbfsFileSystem`,
`AbfsReadFile`, dual-client, configuration-key, or `preadvAsync` integration is
present.

Stage 2 adds and proves:

- Config-scoped EventBase/FiberManager shards.
- A global active-fiber limit and bounded inactive FIFO queue.
- Ready overload failures and deterministic queued shutdown failures.
- Cooperative active cancellation and terminal completion accounting.
- Safe final public-owner release from a runtime fiber.
- Runtime-owned endpoint transports and connection pools with fixed shard
  affinity and EventBase-local construction/destruction.
- Bounded connection acquisition, idle eviction, waiter completion, and pool
  teardown.
- Unread-body connection close and fully consumed keep-alive reuse.
- A dedicated bounded blocking resolver executor.
- Bounded LRU endpoint/DNS state with positive/negative TTLs, single-flight,
  capacity rejection, expiry, eviction, and shutdown cancellation.
- Configured and measured active fiber-stack bounds.

## Ownership

The validated ownership direction is:

```text
AbfsAsyncRuntime
  -> shared RuntimeState
  -> runtime shards and endpoint states
  -> admitted request state
  -> transport response body and connection lease
```

Endpoint channels, sockets, connection pools, timers, and callbacks remain on
one EventBase for their lifetime. Resolver jobs retain endpoint/completion
state but do not own `RuntimeState`. Stage 2 contains no pointer to a future
`AbfsReadFile::Impl`.

## Bounds and metrics

The runtime snapshot covers configured shards, active and queued limits,
current/peak requests, accepted/overloaded/completed/cancelled requests,
configured fiber stack size, estimated peak active stack capacity, endpoint
count, DNS active/queued/peak work, cache hits/misses/expiry/eviction/rejection,
and resolution/failure totals.

A deterministic three-shard test proved:

- 3 simultaneous active fibers on 3 runtime threads.
- 2 accepted inactive requests in one global queue.
- Immediate rejection of the next submission.
- 131072 configured bytes per fiber.
- 393216 bytes configured peak active stack capacity.
- 4944 bytes measured ordinary Debug stack high watermark.

The measured value is not inferred from configured capacity and is not a
production sizing recommendation.

## Functional validation

Final ordinary Debug results:

- `velox_abfs_stage2_runtime_test`: 12 of 12 passed.
- `velox_abfs_stage1_transport_test`: 59 of 59 passed.
- Focused active/fiber bounds: 100 of 100 repetitions passed.
- Existing pool/timer and endpoint-lifetime focused stress remains green.
- Repository-pinned clang-format 21.1.2, whitespace, LF, mirror, frozen Stage 0
  hash, and editor-diagnostic checks passed.

## Sanitizer gates

Separate Clang 20.1.2 Debug build trees use bundled Folly compiled with the
same sanitizer as the ABFS targets.

ASAN with leak detection:

- Complete runtime suite: PASS.
- Complete transport/C1 suite: PASS.
- Runtime C2 matrix: 100 repetitions, 11 tests each, PASS.
- Transport C2 matrix: 100 repetitions, 12 tests each, PASS.
- No use-after-free or leak remains.

TSAN:

- Complete runtime suite: PASS.
- Complete transport/C1 suite: PASS.
- Runtime C2 matrix: 100 repetitions, 11 tests each, PASS.
- Transport C2 matrix: 100 repetitions, 12 tests each, PASS.
- No race report.

The first valid ASAN transport run found a dangling-reference defect in
`PoolCapSuspendsFiberUntilBodyRelease`: a nested test fiber captured an
outer-fiber-local transport and request by reference. The nested fiber now owns
the transport and constructs its own request from a copied URL. ASAN, TSAN, and
ordinary Debug suites pass after the repair. Production behavior did not
change for this finding.

## Evidence

- [Environment](results/stage2-main-06dec49a/environment.txt)
- [Commands](results/stage2-main-06dec49a/commands.txt)
- [Source integrity](results/stage2-main-06dec49a/source-integrity.txt)
- [Ordinary Debug suites](results/stage2-main-06dec49a/tests/debug-suites.txt)
- [Active/fiber bounds](results/stage2-main-06dec49a/tests/active-fiber-bounds.txt)
- [ASAN](results/stage2-main-06dec49a/sanitizers/asan.txt)
- [TSAN](results/stage2-main-06dec49a/sanitizers/tsan.txt)

## Remaining risks

- Shutdown is cooperative; a task that ignores cancellation can delay it.
- An executing synchronous resolver call cannot be preempted.
- Endpoint/DNS expiry is lazy.
- Admission and metrics are global, not per shard.
- The inactive queue is global FIFO and may dispatch across shards.
- Endpoint shard assignment uses process-local `std::hash`.
- Errors are still private runtime `std::runtime_error` contracts.
- Stage 2 sources are not production-linked and no configuration keys exist.
- WSL loopback sanitizer evidence is correctness evidence, not native-Linux or
  real-Azure performance evidence.

## Stage 3 entry

C2 is complete. Stage 3 may begin with the source-compatible provider-pair
contract, built-in Shared Key/fixed-SAS fiber clients, optional runtime
ownership in `AbfsFileSystem`, and direct `AbfsReadFile::preadvAsync` tests.

Stage 3 must preserve the disabled synchronous path, null-gap scatter
semantics, one contiguous range request, safe file destruction, and truthful
`hasPreadvAsync()`. OAuth, dynamic SAS, cooperative retry preservation,
Parquet scheduler changes, io_uring, and Fizz remain later checkpoints unless
the authoritative design explicitly assigns a Stage 3 prerequisite.
