# Stage 5 parallel handover: dynamic SAS refresh safety

Date: 2026-07-22
Base SHA: `06dec49a333b7e39f73a49ba9dd2c4ee12e4fd4e`
Status: **DYNAMIC SAS COMPLETE; OAUTH AND C4 STILL OPEN**

## Purpose and boundary

The local Stage 4 cooperative-retry prototype passes its behavior and
sanitizer gates, but C4 cannot pass until Azure releases the upstream retry
delay hook and Velox consumes that release. This handover defines work that can
proceed without that release: make dynamic-SAS token refresh fiber-safe behind
the existing native-async and provider capability gates.

This is parallel development, not checkpoint advancement. Do not mark C4 or
C5 complete. Do not expose native async as a supported setting. Keep both retry
safety gates unchanged:

- `fs.azure.async-read.disable-retries-for-test=true` remains required.
- `fiberOptions.Retry.MaxRetries = 0` remains set for the fiber client.

OAuth, proxy implementation, io_uring/Fizz, performance tuning, HTTP/2, and
downstream Parquet scheduling are outside this handover.

## Completion update

The dynamic-SAS parallel slice defined below is complete. The implementation
adds the bounded runtime-owned authentication service, keyed single-flight,
source-compatible async provider context, option-preserving dynamic-SAS fiber
client, direct Azurite integration, forced-refresh concurrency, cancellation,
shutdown, and file-lifetime coverage.

The final ordinary, ASAN, and TSAN matrices pass. Focused auth-runtime and
dynamic-SAS lifetime matrices pass 100 in-process repetitions under both
sanitizers. See [the Stage 5 parallel progress report](STAGE_5_PARALLEL_PROGRESS.md)
and its sanitized evidence under `results/stage5-parallel-main-06dec49a/`.

This completion does not advance C4 or C5. OAuth remains unsupported, native
async remains behind both retry safety gates, and the instructions below are
retained as the execution contract used for this completed slice.

## Required reading

Read these files before editing:

1. [Authoritative design](../ABFS_PREADV_ASYNC_FIBER_TRANSPORT.md), especially
   dynamic SAS providers, runtime ownership, Stage 5, and authentication tests.
2. [Stage 4 handover](STAGE_4_HANDOVER.md).
3. [Upstream Azure proposal](STAGE_4_UPSTREAM_AZURE_PROPOSAL.md).
4. [Stage 3 checkpoint](STAGE_3_CHECKPOINT.md) for the provider-pair and file
   ownership contracts.
5. `AbfsAsyncRuntime.h/.cpp` and `tests/AbfsAsyncRuntimeTest.cpp`.
6. `AzureClientProvider.h`, `AzureClientProviderFactories.h/.cpp`, and their
   tests.
7. `DynamicSasTokenClientProvider.h/.cpp` and
   `tests/DynamicSasTokenClientProviderTest.cpp`.
8. `AbfsReadFile.cpp` and the native-async tests in
   `tests/AbfsFileSystemTest.cpp`.

## Workspace and preservation rules

- `C:\velox` is the Windows staging/editor checkout.
- Build and execute only in the WSL Ubuntu ext4 checkout at `~/src/velox`.
- Synchronize only an explicit reviewed file list between the two checkouts and
  verify the mirror before validation.
- Preserve every existing modified and untracked ABFS artifact.
- Preserve the local Azure SDK prototype under
  `deps-download/azure-sdk-for-cpp`; do not reset, rebase, commit, or mix it
  with stale Azure component libraries.
- Do not create a branch or commit.
- Keep credentials, SAS values, authorization headers, account identifiers,
  real endpoints, usernames, hostnames, and private absolute paths out of
  source, commands, logs, and reports.

The frozen Stage 0 benchmark CMake block LF-normalized SHA-256 remains:

```text
7b4af22ec2ddc0dcb812dc9d59edceb48262f31b9594a8a4453e1e230ba50e0d
```

## Completed foundation

Do not redesign these passing layers without a failing focused test:

- Stage 1: fiber-backed Azure HTTP transport, verified TLS, incremental Beast
  parsing, bounded streaming, pooling, and one-thread concurrency proof.
- Stage 2: bounded runtime admission, endpoint ownership, DNS executor/cache,
  cancellation, shutdown, metrics, and ASAN/TSAN lifetime coverage.
- Stage 3: source-compatible provider pairing, Shared Key and fixed-SAS fiber
  clients, config-scoped runtime ownership, truthful `hasPreadvAsync`, and
  shared scatter behavior.
- Stage 4 prototype: cooperative retry timer, future cancellation, timer races,
  stock/callback parity, and no-`sleep_for` runtime stack.

The current synchronous dynamic-SAS client lazily invokes
`SasTokenProvider::getSasToken` from `getBlobClient()`. This callback is
arbitrary synchronous user code. Invoking it on the ABFS EventBase thread can
block all fibers on that runtime shard.

The existing optional provider method accepts only `BlobClientOptions`. That
is sufficient for Shared Key and fixed SAS, but it cannot provide a
runtime-owned bounded auth service to a dynamic-SAS client. Do not work around
this by dynamic-casting the transport, using global state, or giving the
provider unrestricted access to the complete runtime.

## Local hypothesis and first check

Falsifiable hypothesis:

> A config-scoped, runtime-owned auth service can execute a blocking SAS token
> callback on one bounded worker while the requesting fiber yields, allowing
> an unrelated request to complete on the same one-thread runtime.

The cheapest discriminating check is a new
`AbfsAsyncRuntimeTest` that:

1. Starts a token refresh whose callback blocks on a test condition variable.
2. Waits until the callback is executing on the auth worker.
3. Submits an unrelated request to the same one-thread runtime.
4. Proves the unrelated request completes before the token callback is
   released.
5. Releases the callback and proves the original request settles once.
6. Proves the callback thread differs from the submitting and EventBase
   threads.

Write this failing test first. The first substantive implementation edit must
be followed immediately by the focused Stage 2 runtime build and test. Do not
touch provider or file wiring until this check passes.

## Required architecture

### Narrow async construction context

Keep these existing provider methods unchanged:

- `getReadFileClient` remains pure virtual and preserves synchronous behavior.
- `getReadFileClientWithOptions` remains optional and source-compatible for
  registered providers that already opt into custom Blob options.

Add a narrow async construction context and a new optional provider method.
The structure should be equivalent to:

```cpp
class AbfsAsyncAuthService;

struct AzureAsyncReadContext {
  const Azure::Storage::Blobs::BlobClientOptions& clientOptions;
  std::shared_ptr<AbfsAsyncAuthService> authService;
};

virtual std::unique_ptr<AzureBlobClient> getReadFileClientForAsync(
    const std::shared_ptr<AbfsPath>& path,
    const config::ConfigBase& config,
    const AzureAsyncReadContext& context) {
  return getReadFileClientWithOptions(
      path, config, context.clientOptions);
}
```

Names may be adjusted to match the final owning abstraction, but preserve the
semantics:

- Existing registered providers require no source changes.
- Existing Shared Key and fixed-SAS overrides continue to receive the supplied
  Blob options through the default delegation.
- Dynamic SAS overrides the new method because it also needs the narrow auth
  service.
- OAuth remains null/unsupported until its separate forced-refresh checkpoint.
- Disabled mode does not construct or pass an async context.

Do not add a default argument. Document every public type, method, and member
in the header according to Velox style.

### Runtime-owned auth service

`AbfsAsyncRuntime` owns one shared auth service for its complete lifetime. The
service owns:

- One blocking auth worker by default, configurable to at most two.
- A bounded queue of distinct refresh jobs.
- One in-flight refresh per account, filesystem, path, and operation key.
- Fiber waiters sharing the same result or exception.
- Shutdown and cancellation state needed to settle each waiter once.

Use a dedicated key type with the full semantic identity. Do not concatenate
an ambiguous string key.

The service API may synchronously return a token to the dynamic Blob wrapper
because it is called from the synchronous Azure stack, but its implementation
must suspend only the current fiber while the callback runs on the auth
worker. It must reject calls outside an active request fiber.

Locks may protect queue and single-flight state transitions only. Never hold a
lock while:

- Calling `SasTokenProvider::getSasToken`.
- Waiting on a fiber Baton.
- Posting completion to a waiter.
- Settling a promise or exception.

An executing provider callback cannot be preempted. Cancellation removes or
interrupts the requesting waiter promptly, but the worker may finish the
callback later. Runtime shutdown cancels queued refreshes and waits for an
already executing callback to return. Tests must release blocking callbacks
before teardown.

The ownership direction must remain acyclic:

```text
AbfsFileSystem
  -> AbfsAsyncRuntime
  -> RuntimeState
  -> AbfsAsyncAuthService
  -> bounded worker and in-flight refresh state

AbfsReadFile::Impl
  -> dynamic-SAS fiber client
  -> shared AbfsAsyncAuthService
```

Neither the auth service nor runtime state may point to `AbfsReadFile::Impl`,
an Azure client, or a provider instance.

### Dynamic-SAS fiber client

Preserve the existing synchronous dynamic-SAS client unchanged. Add a distinct
fiber client that:

- Retains a copy of the caller-supplied `BlobClientOptions`.
- Retains the narrow shared auth service, not the full file implementation.
- Uses the auth service for initial and renewed tokens.
- Applies the supplied Blob options every time it recreates the Azure
  `BlobClient`; otherwise refresh would silently fall back to the default
  blocking transport.
- Keeps token and expiry state EventBase-affine.
- Rechecks cached state after a shared refresh completes so resumed fibers do
  not create unnecessary client instances.
- Never logs or stores a token in metrics, errors, or retained evidence.

`AbfsReadFile` calls `fiberFileClient_->getUrl()` during construction to derive
the transport endpoint, before any runtime request fiber exists. The fiber
dynamic-SAS client's `getUrl()` must therefore return the stable token-free
Blob URL without invoking the token provider or waiting on the auth service.
Do not change the existing synchronous client's observable URL-refresh tests.

## Implementation sequence

Use small test-first increments and validate after each one:

1. Add the one-thread sibling-progress runtime test described above.
2. Add the bounded auth worker and one-key refresh path.
3. Add a same-key single-flight test with many fiber waiters and one callback.
4. Add distinct-key queue-bound and overload tests.
5. Add callback exception fan-out, future cancellation, queued cancellation,
   and shutdown tests.
6. Add the source-compatible async provider context and factory tests.
7. Add the dynamic-SAS fiber client, stable token-free `getUrl`, and option
   retention tests.
8. Add a direct `AbfsReadFile::preadvAsync` dynamic-SAS integration test with a
   controlled local endpoint.
9. Add forced-expiry concurrency coverage on one runtime thread.
10. Run ordinary, ASAN, and TSAN regressions and retain sanitized evidence.

Do not open provider wiring before the runtime service tests are green. Do not
open OAuth work before the dynamic-SAS matrix is complete.

## Required tests

### Runtime auth service

- A blocking callback runs off the EventBase and submitting threads.
- An unrelated request completes while one fiber waits for refresh.
- Sixty-four same-key waiters invoke the provider exactly once.
- Different keys respect the configured active and queued bounds.
- Queue overload fails without blocking the EventBase.
- All waiters receive the same token or exception.
- Cancelling one waiter does not cancel successful delivery to other waiters.
- Cancelling all queued waiters prevents an unstarted callback when practical.
- Shutdown fails queued waiters and does not touch destroyed waiter state.
- Callback completion racing cancellation or shutdown settles every request
  exactly once.

Use conditions, Batons, injectable clocks, and synthetic tokens. Do not use
wall-clock sleeps to coordinate concurrency.

### Provider compatibility

- A legacy provider that implements only the existing pure virtual methods
  remains source-compatible.
- A provider overriding only `getReadFileClientWithOptions` is reached through
  the new default async-context delegation.
- Null async context never invokes an async provider method.
- Shared Key and fixed SAS continue to receive exactly the supplied options.
- OAuth remains explicitly unsupported with its existing auth context.
- The dynamic-SAS sync client follows its existing path and behavior.

### Dynamic-SAS integration

- Fiber-client construction does not call the token provider.
- Fiber `getUrl()` is stable, token-free, and callable outside a runtime fiber.
- The first data operation obtains one token through the auth worker.
- A near-expiry token triggers one shared refresh under concurrent reads.
- Every recreated Blob client retains the fiber transport and retry callback.
- A blocking provider does not block sibling reads on the EventBase.
- Provider failure reaches every affected future without exposing the token.
- File and filesystem destruction during refresh are safe.
- Caller cancellation during refresh is distinct from runtime shutdown.

Use a deterministic local server or test transport. Do not retain generated SAS
values or signed URLs in logs.

## Validation

Run all builds and tests from the WSL ext4 checkout. Reuse the existing build
trees only after synchronizing and verifying the exact changed file list.

Ordinary Debug:

```text
cmake --build <ordinary-build> --target \
  velox_abfs_test velox_abfs_registration_test \
  velox_abfs_stage1_transport_test velox_abfs_stage2_runtime_test -j 2

ctest --test-dir <ordinary-build> --output-on-failure \
  -R '^(velox_abfs_test|velox_abfs_registration_test|velox_abfs_stage1_transport_test|velox_abfs_stage2_runtime_test)$'
```

Run the focused auth-runtime and dynamic-SAS filters repeatedly before the
complete suites. Record exact final test names rather than retaining planned
names in evidence.

ASAN:

- Use the existing Clang 20 Debug ASAN tree with bundled Folly under matching
  instrumentation.
- After any CMake regeneration, verify Folly's generated library-ASAN marker is
  `1` before execution.
- Enable leak detection and halt on the first finding.
- Run complete four-target suites and at least 100 focused in-process repeats
  of single-flight, cancellation, shutdown, and file-lifetime tests.

TSAN:

- Use the existing Clang 20 Debug TSAN tree with bundled Folly.
- Run every binary through `setarch x86_64 -R` on this WSL kernel.
- Run complete four-target suites and at least 100 focused repeats.
- Treat any race in queueing, single-flight state, waiter completion, provider
  callback completion, cancellation, or teardown as a blocker.

After source changes, run:

- Repository-pinned clang-format 21.1.2 on changed C++ lines/files.
- `git diff --check` in both the Velox and nested Azure SDK worktrees.
- LF and final-newline checks.
- Windows/WSL mirror comparison for every changed source.
- Editor diagnostics for every changed file.
- Frozen Stage 0 benchmark block hash verification.
- A safety-gate check proving both retry gates remain present.

## Evidence and status reporting

Retain sanitized evidence under:

```text
velox/connectors/hive/storage_adapters/abfs/design/results/
  stage5-parallel-main-06dec49a/
```

Record:

- Environment and toolchain versions.
- Exact commands with private roots replaced by placeholders.
- Focused and complete ordinary test results.
- ASAN and TSAN configuration, repeats, and results.
- Auth worker, queue, single-flight, cancellation, and shutdown counters.
- Source integrity, mirror, scope, and frozen-hash checks.
- Remaining OAuth and C4 blockers.

Do not create `STAGE_5_CHECKPOINT.md` or claim C5 PASS. When this dynamic-SAS
slice is complete, update this handover status and add a
`STAGE_5_PARALLEL_PROGRESS.md` report stating **DYNAMIC SAS COMPLETE; OAUTH AND
C4 STILL OPEN**.

## Stop conditions

Stop and document the first failed condition if:

- The token callback executes on an EventBase thread.
- Single-flight requires holding an ordinary mutex while a fiber waits.
- A client refresh loses the supplied Blob transport or retry callback.
- The async client must call the token provider from construction-time
  `getUrl()`.
- Queue, worker, fiber, or retained refresh state is unbounded.
- Cancellation or shutdown can settle a waiter twice or touch destroyed state.
- Dynamic-SAS support requires changing the existing synchronous provider
  behavior or breaking legacy registered providers.
- ASAN or TSAN reports a finding that cannot be repaired within this slice.

Do not silently simplify the approved architecture or move the callback onto a
generic unbounded executor to make tests pass.

## Remaining next actions

1. Keep `fs.azure.async-read.disable-retries-for-test=true` required and keep
  native `fiberOptions.Retry.MaxRetries = 0`.
2. Land and consume a released Azure retry-delay hook before advancing C4.
3. Implement and force-refresh-test OAuth in its separate checkpoint.
4. Rerun the complete authentication and retry matrices before claiming C5.
