# Stage 4 upstream Azure retry-delay proposal

Date: 2026-07-21
Status: Draft only; not posted

## Verified upstream state

- Azure SDK `origin/main` at
  `c66422a7db6e04a0680fdd8798830a2d0f5338d3` has no retry-delay callback and
  still calls `std::this_thread::sleep_for` in the retry loop.
- Stable Azure Core 1.16.4 at
  `92e86b48dcb6bb54654d4cf5290c46ea61128039` adds `RetryPolicyBase`, but its
  retry loop still sleeps.
- Stable Data Lake 12.16.0 at
  `1c32f92de06445c467715b55860083202194aee4` also contains the blocking wait.
- No matching open Azure SDK issue or pull request was found.

## Issue draft

### Title

Allow callers to customize how RetryPolicy waits between attempts

### Body

Azure Core's retry policy selects retry eligibility and delay correctly, then
waits with `std::this_thread::sleep_for`. A custom synchronous HTTP transport
may run the Azure pipeline on a cooperative scheduler, such as a fiber or event
loop. Sleeping the current thread in that environment blocks unrelated work on
the same scheduler.

Disabling the built-in retries loses Azure's status and transport-error
classification, request and body reset behavior, `Retry-After` parsing, jitter,
attempt accounting, and terminal behavior. Implementing a separate policy
above the transport requires duplicating that logic and reviewing the copy on
every Azure SDK update.

I propose adding an optional wait callback to `RetryOptions`:

```cpp
std::function<void(
    std::chrono::milliseconds,
    Azure::Core::Context const&)>
    RetryDelayCallback;
```

The retry policy would invoke the callback with the delay it already selected
and the original request context. If the callback is absent, the current
`std::this_thread::sleep_for` path remains unchanged. Exceptions from the
callback propagate and stop retry processing, allowing cooperative
cancellation or scheduler shutdown to interrupt a pending wait.

This changes only how the selected delay is waited. Retry classification,
header precedence and parsing, exponential delay and jitter, request reset,
policy order, and final response or exception behavior remain owned by Azure
Core.

`RetryPolicyBase` does not remove this need for service clients that construct
the default retry policy from client options. A caller can set built-in retries
to zero and add another policy, but that requires copying the complete retry
algorithm this proposal is intended to preserve.

The API is additive and defaults to the existing behavior. It does add state to
`RetryOptions`; all Azure components in a packaged release must therefore be
built together. Maintainer guidance is welcome if a different public extension
point is preferred for compatibility reasons.

A working prototype includes deterministic tests for:

- Capped exponential delay.
- Integer-second `Retry-After`.
- `retry-after-ms`.
- `x-ms-retry-after-ms`.
- Callback invocation count, selected duration, original context, and attempt
  count.
- Callback exception propagation without another transport attempt.
- Stock-versus-callback parity for 408, configured and default 429 behavior,
  500, 502, 503, 504, 400, 501, transport failures, multiple retries, terminal
  failures, body rewind, retry-scoped query and header reset, retry counters,
  and policy order.

All 15 Azure RetryPolicy tests pass. The changed retry test translation unit
also compiles with warnings treated as errors.

## Pull request draft

### Title

Add an optional RetryPolicy delay callback

### Summary

- Add `RetryOptions::RetryDelayCallback` as an optional callback receiving the
  selected delay and original request context.
- Preserve the existing thread-sleep behavior when no callback is supplied.
- Propagate callback exceptions to support interruptible cooperative waits.
- Add deterministic delay-selection, failure, and stock-path parity tests.
- Explicitly initialize the new final aggregate member in existing retry tests
  so warnings-as-errors builds remain clean.

### Test plan

```text
cmake --build <azure-core-test-build> --target azure-core-test -j 2
ctest --test-dir <azure-core-test-build> --output-on-failure \
  -R '^azure-core\.RetryPolicy\.'
```

Result: 15 of 15 RetryPolicy tests pass.

The changed retry test translation unit also passes an isolated build with
`WARNINGS_AS_ERRORS=ON`. A complete warnings-as-errors build of the older
prototype base reaches an unrelated existing pessimizing-move warning in
`nullable_test.cpp`; that result is not attributed to this change.

### Compatibility and review notes

- The callback is the final `RetryOptions` member and is empty by default.
- Existing behavior and timing remain unchanged when it is empty.
- The callback receives the delay after Azure Core has applied classification,
  header parsing, exponential backoff, jitter, and maximum-delay rules.
- Exceptions are intentionally terminal and documented.
- The public options layout changes, so consumers must not mix newly rebuilt
  Core with stale Azure component libraries.
- The contribution should target current Azure SDK `main`, add the current
  Azure Core changelog entry, and pass API review before release.

## Velox consumption after release

1. Update `AZURE_SDK_VERSION` through Velox's normal Data Lake monorepo pin only
   after a stable tag contains the callback.
2. Rebuild Azure Core, Identity, Storage Common, Storage Blobs, and Storage Data
   Lake together.
3. Set the ABFS fiber client's retry callback to
   `AbfsAsyncRuntime::waitForRetryDelay` and restore the normal Azure retry
   count.
4. Remove `fs.azure.async-read.disable-retries-for-test` only after the complete
   ordinary, ASAN, and TSAN C4 matrices pass against the released SDK.
5. Retain the callback parity, cancellation, shutdown, timer-race, and runtime
   stack evidence.
