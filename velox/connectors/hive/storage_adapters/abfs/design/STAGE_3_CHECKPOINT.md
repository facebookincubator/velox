# Stage 3 checkpoint: dual clients and native ABFS reads

Date: 2026-07-21
Base SHA: `06dec49a333b7e39f73a49ba9dd2c4ee12e4fd4e`
Verdict: **C3 PASS**

## Scope

Stage 3 production-links the Stage 1 transport and Stage 2 bounded runtime,
then exposes the standalone native `AbfsReadFile::preadvAsync` contract. It
adds:

- A source-compatible optional provider method for caller-supplied Blob client
  options.
- One provider-factory operation that creates paired sync and fiber clients
  from the same provider instance.
- Shared Key and fixed-SAS option propagation.
- Config-scoped optional runtime ownership in `AbfsFileSystem`.
- Truthful `hasPreadvAsync()` and direct `preadvAsync()`.
- One shared scatter/discard implementation for sync and fiber clients.
- Explicit configuration and unsupported-provider failures.
- Filesystem cache isolation across complete connector configurations.

OAuth, dynamic SAS, cooperative retries, Parquet scheduler integration,
proxy parity, io_uring, and Fizz remain outside C3.

## Provider and configuration contract

`AzureClientProvider::getReadFileClient` remains unchanged and pure virtual.
The distinctly named `getReadFileClientWithOptions` is non-pure and returns
null by default, preserving source compatibility for registered providers.

`AzureClientProviderFactories::getReadFileClients`:

- Selects the provider factory once.
- Instantiates one provider.
- Always creates the sync client through the existing method.
- Requests the fiber client only when options are present.
- Preserves provider-local state between both calls.
- Returns the actual selected provider context and a stable unsupported reason.

Registered factories continue to take precedence over configured auth. Error
messages report `registered provider` for that case even when a conflicting
auth key exists.

`AbfsFileSystem` parses native-async settings once. Disabled configurations do
not construct a runtime or request a fiber client. Enabled configurations must
also set the Stage 3 retry-disable test gate. Cached filesystem identity now
uses the complete sorted connector configuration, preventing catalogs with
different credentials, endpoints, provider inputs, async mode, or runtime
limits from sharing an instance.

## Read and ownership contract

The ownership direction is:

```text
AbfsFileSystem
  -> shared AbfsAsyncRuntime
  -> runtime shards and endpoint transports

AbfsReadFile
  -> shared Impl
  -> paired Azure clients
  -> shared AbfsAsyncRuntime

submitted request
  -> shared Impl
```

Submitted work captures `shared_ptr<Impl>`, never raw `this`. Runtime state does
not own `Impl`, so there is no ownership cycle. The fiber Azure client delegates
through a fiber-local binding to the endpoint transport selected by the Stage 2
runtime.

Both sync and async vectored reads use one scatter helper and one contiguous
Blob range. It preserves null gaps, exact logical return values, fragmented
body handling, and the 256 KiB discard-buffer bound. Native async treats a
short body as an error delivered through the future. Disabled sync retains the
base revision's legacy short-body behavior.

When native async is enabled, synchronous data reads submit through the same
async core and wait only on the external caller thread. A runtime-thread check
fails an accidental reentrant synchronous call instead of deadlocking the
EventBase.

## Functional validation

Final ordinary Debug suites:

- `velox_abfs_test`: 32 of 32 passed.
- `velox_abfs_registration_test`: 25 of 25 passed.
- `velox_abfs_stage1_transport_test`: 59 of 59 passed.
- `velox_abfs_stage2_runtime_test`: 12 of 12 passed.

The direct C3 matrix covers:

- Legacy provider compatibility and one provider instance per client pair.
- Null and non-null fiber options and stable unsupported behavior.
- Shared Key and fixed-SAS option forwarding.
- Shared Key and fixed-SAS sync/async Azurite parity.
- Disabled runtime/client behavior and complete filesystem cache isolation.
- Zero length, one destination, multiple destinations, null gaps, and all-null
  ranges.
- One recorded contiguous download and exact logical lengths.
- Fragmented body chunks crossing destination and gap boundaries.
- Transport, timeout, and short-body failures through the future.
- A pending future before a blocked response is released.
- Safe file destruction immediately after submission.
- Four pending reads on one runtime thread.
- Explicit OAuth and custom-provider failure context.
- Runtime-thread synchronous-wait rejection.

No Parquet scheduler source changed. The standalone Stage 0 Parquet benchmark
block remains hash-frozen and is not a C3 dependency.

## Sanitizer gates

Separate Clang 20.1.2 Debug trees use bundled Folly compiled with matching
sanitizer flags.

ASAN with leak detection:

- All four complete suites passed.
- The focused eight-test C3 lifetime, scatter, concurrency, and error matrix
  passed 100 in-process repetitions.
- No AddressSanitizer or LeakSanitizer finding remains.

TSAN:

- All four complete suites passed through `setarch x86_64 -R`.
- The focused eight-test C3 matrix passed 100 in-process repetitions.
- No ThreadSanitizer race report remains.

The first valid full TSAN run found a test-only race in
`multipleThreadsWithReadFile`: ten workers shared a local random engine and
distribution by reference. Delay selection now happens serially before each
thread starts, and workers capture only the selected delay. The focused test,
complete TSAN suites, ASAN suites, and ordinary suites pass after the repair.
Production behavior did not change for this finding.

An earlier ASAN attempt was invalid because CMake regeneration reset bundled
Folly's generated library-ASAN marker after it had been changed. That attempt
aborted in Folly's fiber invariant and is excluded. The valid build set the
marker after regeneration, rebuilt the affected graph, and passed all gates.

## Source integrity

- Base SHA is unchanged.
- Pinned clang-format 21.1.2 passes for all changed C++ sources. The hash-frozen
  Stage 0 benchmark remains intentionally untouched.
- Git whitespace and LF checks pass.
- Windows staging and WSL ext4 copies match after line-ending normalization.
- The frozen Stage 0 benchmark CMake block SHA-256 remains
  `7b4af22ec2ddc0dcb812dc9d59edceb48262f31b9594a8a4453e1e230ba50e0d`.
- All source changes remain under the ABFS adapter directory.
- Editor diagnostics for the ABFS tree are clean.

## Evidence

- [Environment](results/stage3-main-06dec49a/environment.txt)
- [Commands](results/stage3-main-06dec49a/commands.txt)
- [Source integrity](results/stage3-main-06dec49a/source-integrity.txt)
- [Ordinary suites](results/stage3-main-06dec49a/tests/debug-suites.txt)
- [ASAN](results/stage3-main-06dec49a/sanitizers/asan.txt)
- [TSAN](results/stage3-main-06dec49a/sanitizers/tsan.txt)

## Remaining risks and boundaries

- Azure retries are disabled behind a test-only gate. Native async is not ready
  for user-facing enablement until C4 preserves retries cooperatively.
- OAuth and dynamic SAS explicitly fail native async initialization.
- Caller buffers must outlive the returned future.
- Dropping a future does not cancel the submitted request.
- Enabled synchronous reads still occupy one external caller while waiting.
- WSL loopback and Azurite are correctness evidence, not native-Linux or
  real-Azure performance evidence.

## Stage 4 entry

C3 is complete. Stage 4 may replace the test-only retry gate with cooperative
preservation of Azure SDK retry classification, request reset, jitter,
`Retry-After`, delay, and attempt semantics. No async setting may be presented
as user-facing until C4 passes.
