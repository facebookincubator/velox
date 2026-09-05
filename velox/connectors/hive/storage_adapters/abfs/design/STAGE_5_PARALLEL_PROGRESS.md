# Stage 5 parallel progress: dynamic SAS refresh safety

Date: 2026-07-22
Base SHA: `06dec49a333b7e39f73a49ba9dd2c4ee12e4fd4e`
Status: **DYNAMIC SAS COMPLETE; OAUTH AND C4 STILL OPEN**

## Boundary

This report completes only the dynamic-SAS work authorized by the parallel
handover. It does not mark C4 or C5 complete, expose native async as supported,
or remove either retry safety gate. OAuth, proxy implementation, io_uring/Fizz,
performance tuning, HTTP/2, and downstream Parquet scheduling remain outside
this slice.

The local Azure retry-delay callback remains an unreleased prototype. Native
async construction still requires
`fs.azure.async-read.disable-retries-for-test=true`, and the fiber Blob client
still sets `Retry.MaxRetries = 0`.

## Implementation

`AbfsAsyncRuntime` now owns one shared `AbfsAsyncAuthService` for its complete
lifetime. The service provides:

- One blocking authentication worker by default, configurable to one or two.
- A bounded queue of distinct refresh keys, configurable with a positive bound.
- A dedicated account, filesystem, path, and operation key type.
- One in-flight callback per complete key and shared fiber waiters.
- Token and exception fan-out without holding the service lock while invoking
  callbacks, waiting on Batons, or posting completion.
- Prompt waiter removal on caller cancellation.
- Removal of an unstarted queued callback after its final waiter cancels.
- Queued-refresh cancellation and executing-callback join during shutdown.
- Thread-safe worker, queue, in-flight, waiter, completion, overload, and
  cancellation metrics.

The new `AzureAsyncReadContext` gives providers only caller-supplied Blob
options and the narrow auth service. Existing providers remain source
compatible. Providers overriding only `getReadFileClientWithOptions` are
reached through default delegation, while a null context invokes no async
method. Shared Key and fixed SAS continue to receive the exact supplied Blob
options. OAuth still returns null through the existing unsupported path.

`DynamicSasTokenClientProvider` retains its synchronous client unchanged and
adds a distinct fiber client. The fiber client:

- Does not call the token provider during construction or `getUrl()`.
- Returns a stable unsigned Blob URL outside a runtime fiber.
- Honors the configured Blob endpoint.
- Obtains initial and renewed tokens through the bounded auth service.
- Copies and reapplies `BlobClientOptions` on every Blob client recreation.
- Keeps token expiry and client state EventBase-affine.
- Rechecks cached state after a shared refresh before recreating a client.

`AbfsFileSystem` forwards the bounded auth settings
`fs.azure.async-read.auth-threads` and
`fs.azure.async-read.max-queued-auth-refreshes` into the runtime. The worker
count rejects zero and values above two; the queue bound rejects zero.

## Executable behavior

The runtime matrix proves:

- A blocked provider callback runs off both the submitting and EventBase
  threads while an unrelated request completes on the one-thread runtime.
- Sixty-four same-key fibers invoke one callback and receive one shared token.
- Two workers plus two queued distinct keys enforce active and queue bounds;
  the next distinct key fails promptly without blocking an EventBase sibling.
- Callback exceptions reach all waiters.
- Cancelling one waiter preserves delivery to another waiter.
- Cancelling every waiter for a queued key prevents its callback from starting.
- Shutdown settles queued and active request futures, then waits for an already
  executing callback to return.
- Completion, future cancellation, and shutdown race for 100 iterations with
  one terminal request result and empty final auth state.

The dynamic-SAS matrix proves lazy first refresh, unsigned stable `getUrl()`,
custom endpoint handling, transport option retention, 64-read forced-refresh
single-flight, provider exception fan-out, direct `AbfsReadFile::preadvAsync`
through Azurite, one-thread sibling read progress during a blocked callback,
file/filesystem destruction during refresh, and caller cancellation distinct
from runtime shutdown. The existing synchronous URL refresh test remains
unchanged and passes in the complete suite.

## Validation

Final ordinary Debug suites:

- `velox_abfs_test`: 42 of 42 passed.
- `velox_abfs_registration_test`: 25 of 25 passed.
- `velox_abfs_stage1_transport_test`: 59 of 59 passed.
- `velox_abfs_stage2_runtime_test`: 28 of 28 passed.

The complete ordinary run took 87.04 seconds with zero failures.

ASAN uses Clang/Compiler-rt 20.1.2, matching bundled Folly instrumentation,
Folly's generated library-ASAN marker set to `1`, leak detection, and halt on
first error. The five-test auth lifecycle matrix passed 100 in-process
repetitions, the four-test dynamic/lifetime matrix passed 100 repetitions, and
all four complete suites passed. No AddressSanitizer or LeakSanitizer finding
was reported.

TSAN uses Clang/Compiler-rt 20.1.2 and launches every test process through
`setarch x86_64 -R`. The same focused matrices passed 100 repetitions, all
four complete suites passed, and no ThreadSanitizer race was reported.

Repository-pinned clang-format 21.1.2, Git whitespace checks in both worktrees,
LF/final-newline checks, exact Windows/WSL source mirroring, editor diagnostics,
and the frozen Stage 0 benchmark hash all pass. The frozen hash remains
`7b4af22ec2ddc0dcb812dc9d59edceb48262f31b9594a8a4453e1e230ba50e0d`.

## Evidence

- [Environment](results/stage5-parallel-main-06dec49a/environment.txt)
- [Commands](results/stage5-parallel-main-06dec49a/commands.txt)
- [Ordinary suites](results/stage5-parallel-main-06dec49a/tests/debug-suites.txt)
- [Auth metrics](results/stage5-parallel-main-06dec49a/auth-metrics.txt)
- [ASAN](results/stage5-parallel-main-06dec49a/sanitizers/asan.txt)
- [TSAN](results/stage5-parallel-main-06dec49a/sanitizers/tsan.txt)
- [Source integrity](results/stage5-parallel-main-06dec49a/source-integrity.txt)

## Remaining blockers

- C4 still requires an upstream Azure retry-delay hook in a released SDK, a
  normal dependency pin update, and complete revalidation against that release.
- OAuth still requires its separate fiber-safe forced-refresh implementation
  and matrix.
- Native async remains a test-gated prototype and is not ready for supported or
  default enablement.
- WSL loopback and Azurite are correctness evidence, not native-Linux or
  real-Azure performance evidence.
