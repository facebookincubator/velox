# Stage 3 handover: dual clients and native ABFS reads

Date: 2026-07-20
Base SHA: `06dec49a333b7e39f73a49ba9dd2c4ee12e4fd4e`
Status: **C3 PASS - SUPERSEDED BY STAGE 4 HANDOVER**

## Completion notice

C3 passed after the provider-pair, native read, cache-isolation, and complete
ordinary/ASAN/TSAN gates. See [Stage 3 checkpoint](STAGE_3_CHECKPOINT.md) for
the final verdict and retained evidence. Continue with
[Stage 4 handover](STAGE_4_HANDOVER.md).

The implementation instructions below are retained as the historical contract
that led to C3. Their pending language is superseded by the checkpoint and must
not be used as current status.

## Entry condition

[Stage 2 checkpoint C2](STAGE_2_CHECKPOINT.md) passed. Runtime admission,
endpoint ownership, connection-pool lifetime, DNS bounds, active fiber bounds,
ASAN, and TSAN are complete. Stage 3 may now wire these isolated components to
the connector.

Read before editing:

- [Authoritative design](../ABFS_PREADV_ASYNC_FIBER_TRANSPORT.md), especially
  provider changes, async submission, Stage 3, and `AbfsReadFile` tests.
- [Stage 2 checkpoint](STAGE_2_CHECKPOINT.md).
- [Stage 2 sanitizer environment](results/stage2-main-06dec49a/environment.txt).
- [Stage 2 source integrity](results/stage2-main-06dec49a/source-integrity.txt).

Build and execute only in the WSL ext4 checkout. Preserve all uncommitted ABFS
artifacts. Do not create a branch or commit. Keep credentials, account IDs,
real endpoints, authorization headers, SAS values, usernames, hostnames, and
private absolute paths out of reports and commands.

## Stage 3 objective

Exercise the standalone native `AbfsReadFile::preadvAsync` contract while
preserving synchronous behavior. Stage 3 owns:

- A source-compatible optional provider method for fiber-client construction.
- One factory call that returns paired sync/fiber clients and an unsupported
  reason.
- Built-in Shared Key and fixed-SAS option propagation.
- Optional runtime ownership in `AbfsFileSystem`.
- Truthful `hasPreadvAsync()` and direct `preadvAsync()`.
- Shared scatter/discard logic for sync and fiber clients.
- Safe file destruction after async submission.
- Direct local/Azurite contract tests with concurrency exceeding runtime thread
  count.

Do not add Parquet scheduler integration in Stage 3.

## Recommended first slice

Implement only the provider-pair contract first.

1. Add a distinctly named, non-pure virtual method to
   `AzureClientProvider`:

```cpp
virtual std::unique_ptr<AzureBlobClient> getReadFileClientWithOptions(
    const std::shared_ptr<AbfsPath>& path,
    const config::ConfigBase& config,
    const Azure::Storage::Blobs::BlobClientOptions& options) {
  return nullptr;
}
```

Keep the existing pure virtual `getReadFileClient` unchanged. The distinct name
avoids overload hiding and preserves source compatibility for registered
providers.

2. Add the exact paired result to `AzureClientProviderFactories`:

```cpp
struct AzureReadClients {
  std::unique_ptr<AzureBlobClient> sync;
  std::unique_ptr<AzureBlobClient> fiber;
  std::string asyncUnsupportedReason;
};
```

Add `getReadFileClients(path, config, fiberOptions)` with these rules:

- Invoke the registered provider factory exactly once.
- Always create `sync` through the existing method.
- Create `fiber` through the new method only when `fiberOptions` is non-null.
- Preserve provider-local state across both calls.
- Return a stable unsupported reason when the optional method returns null.
- Do not choose fallback behavior in the factory.

3. Add tests next to the existing factory tests proving:

- A legacy custom provider still compiles and returns a sync client.
- Null fiber options do not invoke the optional method.
- Non-null options invoke the optional method once.
- The provider factory is invoked once for the pair.
- Unsupported custom providers produce null `fiber` plus a stable reason.
- Existing registration precedence and account routing remain unchanged.

The nearest owning files are:

- `AzureClientProvider.h`
- `AzureClientProviderFactories.h/.cpp`
- `tests/AzureClientProviderFactoriesTest.cpp`

The narrow first validation target is `velox_abfs_registration_test`.

After this contract passes, add Shared Key and fixed-SAS overrides that forward
`BlobClientOptions` into every relevant Blob client constructor. Test option
propagation before touching `AbfsFileSystem` or `AbfsReadFile`.

## Later Stage 3 slices

### Runtime and configuration

Parse `fs.azure.async-read.enabled` once in `AbfsFileSystem`. When disabled,
do not construct a runtime and do not request a fiber client. When enabled,
construct one config-scoped runtime and pass it to every read file.

Do not add silent fallback. If async is enabled but the selected provider does
not return a fiber client, throw one user-facing construction error containing
the relevant provider/auth context and stable unsupported reason.

### Read file contract

`AbfsReadFile::Impl` owns paired clients and the optional runtime. Async request
state retains `shared_ptr<Impl>`; never capture raw `this`.

Extract the existing `preadvInternal` scatter/discard behavior into one helper
usable with either client. Preserve:

- One contiguous Blob range request.
- Null-buffer gap semantics.
- Exact logical return value.
- Fragmented-body behavior across buffers and gaps.
- Existing 256 KiB bounded discard buffer.
- Existing synchronous results and errors when async is disabled.

`hasPreadvAsync()` is true only when runtime and fiber client are both present.
`preadvAsync()` must return a pending future before a delayed response begins
and settle errors through that future.

### Direct tests

Checkpoint tests must cover:

- Disabled mode uses the original default-transport sync path.
- Shared Key and fixed SAS sync/async parity.
- Zero-length, one buffer, multiple buffers, and null gaps.
- One contiguous range request and identical sync/async bytes.
- Fragmented response chunks crossing destination/gap boundaries.
- Transport, short-body, and timeout errors through the future.
- Destroying `AbfsReadFile` immediately after submission.
- Multiple outstanding futures exceeding runtime thread count.
- Truthful `hasPreadvAsync()`.

## Boundaries

Do not begin these in Stage 3:

- OAuth native async enablement.
- Dynamic SAS callback offload or async token APIs.
- Cooperative Azure retry-delay preservation beyond the design's test-only
  Stage 3 switch.
- Parquet/reader scheduler integration.
- io_uring, Fizz, proxy, or backend-selection work.
- A generic Velox HTTP framework.

Do not production-link Stage 2 sources until the provider/read-file wiring
requires each one and its disabled-path regression is executable.

## C3 gate

C3 passes only when:

- Existing ABFS tests pass unchanged with async disabled.
- Shared Key and fixed SAS pass direct sync and async tests.
- Sync/async scatter and null-gap cases are byte-identical and issue one
  contiguous range request.
- File destruction after submission is safe.
- Multiple pending futures exceed runtime thread count.
- Provider/configuration failure behavior is explicit and truthful.
- No Parquet scheduler change is required.
- Focused ASAN/TSAN lifetime coverage remains green after production wiring.

When C3 passes, write `design/STAGE_3_CHECKPOINT.md` and a handover for Stage 4
retry preservation. Until then, do not claim native async ABFS is ready for
user-facing enablement.
