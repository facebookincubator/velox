# Stage 4 handover: cooperative Azure retry preservation

Date: 2026-07-21
Base SHA: `06dec49a333b7e39f73a49ba9dd2c4ee12e4fd4e`
Status: **C4 IN PROGRESS; SAFETY GATE RETAINED**

## Entry condition

[Stage 3 checkpoint C3](STAGE_3_CHECKPOINT.md) passed. Paired clients,
config-scoped runtime ownership, direct native `preadvAsync`, disabled-path
compatibility, cache isolation, and focused ASAN/TSAN lifetime coverage are
complete.

Read before editing:

- [Authoritative design](../ABFS_PREADV_ASYNC_FIBER_TRANSPORT.md), especially
  retry delay, Stage 4, and the retry test matrix.
- [Stage 3 checkpoint](STAGE_3_CHECKPOINT.md).
- [Stage 3 sanitizer environment](results/stage3-main-06dec49a/environment.txt).
- [Stage 3 source integrity](results/stage3-main-06dec49a/source-integrity.txt).

Build and execute only in the WSL ext4 checkout. Preserve all uncommitted ABFS
artifacts. Do not create a branch or commit. Keep credentials, account IDs,
real endpoints, authorization headers, SAS values, usernames, hostnames, and
private absolute paths out of reports and commands.

## Progress on 2026-07-21

The preferred-route development prototype now has three independently tested
pieces:

- The exact pinned Azure SDK source revision
  `224332197dcd405f67445e2e2a7105ea62f01496` has an optional final
  `RetryOptions::RetryDelayCallback` member. The callback receives the selected
  `std::chrono::milliseconds` delay and original request `Context`. An absent
  callback still executes the existing `std::this_thread::sleep_for` path.
- The Azure retry-policy test proves callback use for capped exponential delay,
  integer-second `Retry-After`, `retry-after-ms`, and
  `x-ms-retry-after-ms`. Each case verifies one callback, the selected duration,
  the original context, and two total attempts. A callback-failure test also
  proves that an exception stops retry processing without another transport
  attempt. All 15 Azure `RetryPolicy` tests pass.
- `AbfsAsyncRuntime::waitForRetryDelay` uses a fiber Baton timer and a Folly
  cancellation callback. A one-thread runtime test proves that an unrelated
  request completes during a 30-second delay and shutdown interrupts the
  delayed request promptly with one terminal future result.
- A second one-thread runtime test executes Azure's real `RetryPolicy`: the
  first attempt returns 408, the selected delay runs through the cooperative
  callback, an unrelated request completes during backoff, and the second
  attempt returns 200.
- A dual-run Azure matrix compares the stock sleep path with the callback path
  for 408, 429, 500, 502, 503, 504, 400, 501, transport failures, multiple
  retries, and terminal response and transport failures. Attempts, final
  result, retry counters, body rewind, retry-scoped query/header reset, and
  policy order match. Pinned Core does not retry 429 by default; the matrix
  proves both that default and retry after explicitly adding 429 to
  `StatusCodes`.
- Runtime tests cover normal timer expiry and 100 expiry-versus-shutdown
  collisions. Every collision produces one terminal future result and one
  completed request under ordinary, ASAN, and TSAN builds.
- Cancelling a returned `SemiFuture` now removes queued requests or signals an
  active request's cancellation source. Cancellation interrupts a 30-second
  retry delay promptly, propagates through `AbfsReadFile::preadvAsync`, and is
  distinct from runtime shutdown. A 100-iteration cancellation-versus-expiry-
  versus-shutdown race settles each request exactly once under ordinary, ASAN,
  and TSAN builds.
- A sanitized GDB stack captured at the Baton `timed_wait` line proceeds from
  Azure `RetryPolicy::Send` through the delay callback and
  `AbfsAsyncRuntime::waitForRetryDelay`. It contains no
  `std::this_thread::sleep_for` frame.

`AbfsReadFile` supplies the cooperative callback through the fiber client
options. The callback weakly captures the runtime, checks Azure context
cancellation before and after waiting, and does not add a runtime-to-file
ownership edge. `fiberOptions.Retry.MaxRetries` remains zero and the Stage 3
configuration gate remains required. The hook is wired for development, but
native retries are not enabled.

The Azure change is only a local build-tree prototype. Adding the callback
changes the public `RetryOptions` layout, so every linked Azure component must
be rebuilt together. The development validation rebuilt Azure Core, Identity,
Storage Common, Storage Blobs, and Storage Data Lake before relinking Velox.
Mixing rebuilt Core with stale component libraries caused a deterministic
crash and is not a valid test environment.

Validation after the complete SDK rebuild:

- Azure `RetryPolicy`: 15 of 15 pass.
- Azure Core complete run: 317 tests pass; two unrelated external libcurl/CRL
  tests fail in this environment and two auxiliary test executables were not
  built by the focused target.
- ABFS: 33 of 33 pass.
- Registration and provider compatibility: 25 of 25 pass.
- Stage 1 transport: 59 of 59 pass.
- Stage 2 runtime: 19 of 19 pass.
- The new callback-wiring, delayed-shutdown, and real-policy integration tests
  pass under focused ASAN and TSAN runs. The delayed-shutdown test also passes
  100 in-process repetitions under each sanitizer.
- The complete four-target ordinary, ASAN, and TSAN matrices pass. The ASAN
  run verified Folly's generated library sanitizer marker before execution;
  TSAN ran through `setarch x86_64 -R`.

All local C4 behavior and validation gates pass. C4 is not complete because the
hook exists only in the local Azure build-tree prototype. Before changing
either retry gate, consume a released upstream hook through the normal pin
update and rerun the complete validation against that release.

An upstream availability check on 2026-07-21 confirmed the external blocker:

- Azure SDK `origin/main` at
  `c66422a7db6e04a0680fdd8798830a2d0f5338d3` still calls
  `std::this_thread::sleep_for` in the retry loop and exposes no delay callback.
- The latest stable Azure Core tag is `azure-core_1.16.4` at
  `92e86b48dcb6bb54654d4cf5290c46ea61128039`. It adds
  `RetryPolicyBase`, but the retry loop still sleeps and a Blob client cannot
  inject a cooperative wait through `BlobClientOptions`.
- The latest stable Data Lake tag is
  `azure-storage-files-datalake_12.16.0` at
  `1c32f92de06445c467715b55860083202194aee4`. Its exact monorepo commit also
  retains the blocking sleep and has no callback member.
- Searches of open Azure SDK issues and pull requests found no existing retry
  delay callback proposal.

The pinned prototype was refined for upstream submission: callback exceptions
are documented and tested as terminal, all existing `RetryOptions` aggregate
initializers in the retry suite explicitly initialize the new final member,
and the retry test translation unit compiles with warnings treated as errors.
The complete 15-test RetryPolicy suite passes. A full pinned-Core
warnings-as-errors build reaches an unrelated pre-existing pessimizing-move
warning in `nullable_test.cpp`; that failure is not used as retry-hook evidence.
The ready-to-post issue and pull request text is retained in
[the Stage 4 upstream Azure proposal](STAGE_4_UPSTREAM_AZURE_PROPOSAL.md). It
has not been posted externally.

While the upstream hook is evaluated, Velox-contained dynamic-SAS refresh work
may proceed behind all existing gates using the
[Stage 5 parallel handover](STAGE_5_PARALLEL_HANDOVER.md). This does not advance
C4 or C5 status.

## Current boundary

The repository pins Azure SDK version `12.8.0` in `scripts/setup-versions.sh`.
Both the pinned retry policy and the latest stable Data Lake 12.16.0 monorepo
commit use `std::this_thread::sleep_for` for backoff. That is not legal on an
ABFS EventBase thread because one delayed request would stall all fibers on its
runtime shard.

C3 therefore requires both:

- `fs.azure.async-read.disable-retries-for-test=true` during construction.
- `fiberOptions.Retry.MaxRetries = 0` for the native client.

This is a prototype safety gate, not production behavior. Do not remove the
gate, expose native async as user-facing, or claim retry parity until C4 passes.

## Stage 4 objective

Preserve Azure SDK retry behavior without blocking a runtime thread. Keep the
SDK's existing:

- Retryable status and transport-error classification.
- Request and body reset behavior.
- Policy order and attempt accounting.
- Exponential delay and jitter.
- `Retry-After`, `retry-after-ms`, and `x-ms-retry-after-ms` handling.
- Final response and exception behavior.

The only intended semantic change is how the selected delay waits: an ABFS
runtime fiber must yield cooperatively and resume from an EventBase timer.

## Preferred route

1. Prepare a backward-compatible Azure Core change adding an optional delay
   callback to `RetryOptions`.
2. Keep the absent-callback default identical to the current
   `std::this_thread::sleep_for` path.
3. Add Azure SDK tests proving unchanged default behavior and callback use for
   exponential and all `Retry-After` forms.
4. Land or consume an Azure SDK release containing the hook and update Velox's
   pin through the normal dependency process.
5. Supply a callback from the ABFS fiber client that waits on a fiber Baton and
   EventBase timer.
6. Make cancellation and runtime shutdown interrupt a pending delay and settle
   the request exactly once.
7. Remove the Stage 3 retry-disable gate only after the complete C4 matrix is
   green.

Do not maintain a permanent local patch to an installed dependency as the
final Velox change. A build-tree patch is acceptable only for developing and
validating the upstream API before a released dependency is available.

## Fallback route

Use this only after explicit Velox maintainer approval:

- Set Azure's built-in retry count to zero.
- Add a private cooperative policy above the transport.
- Port the pinned SDK algorithm exactly, including classification, reset,
  header precedence and parsing, jitter, limits, and terminal behavior.
- Link every parity test to the corresponding pinned Azure source/test.
- Treat every Azure SDK upgrade as a mandatory retry-parity review.

The fallback duplicates SDK behavior and is not preferred.

## Recommended first slice

Start outside `AbfsReadFile` by proving the delay hook against the pinned Azure
retry policy:

1. Record the exact pinned policy implementation and existing retry tests.
2. Add an optional callback with default behavior unchanged.
3. Add deterministic Azure-level tests for one exponential delay, one standard
   `Retry-After`, and both millisecond headers.
4. Prove callback invocation count, selected duration, attempt count, and
   unchanged default behavior.
5. Only then wire a fiber callback into `BlobClientOptions`.

The first discriminating check is an Azure policy test showing the callback
receives the same delay the stock policy would pass to `sleep_for`.

## Required retry matrix

C4 tests must compare the cooperative path with the stock SDK path for:

- HTTP 408 and 429.
- Retryable 5xx responses, including 500, 502, 503, and 504.
- Non-retry controls such as 400 and 501.
- Transport exceptions.
- Integer-second `Retry-After`.
- `retry-after-ms`.
- `x-ms-retry-after-ms`.
- Header-free exponential delay and jitter.
- Success after one or more retries.
- Terminal failure after the configured attempt limit.
- Request-body reset and policy invocation order.

Use a deterministic clock/random source where the SDK permits it. For the
small real-timer integration set, state a timer tolerance rather than asserting
an exact wall-clock duration.

## Runtime and lifetime gates

On a one-thread ABFS runtime:

- Start one request in a multi-second retry backoff.
- Complete an unrelated request on the same runtime thread before the delayed
  request resumes.
- Cancel a request during delay and prove one terminal future result.
- Shut down the runtime during delay and prove prompt interruption with no
  timer callback touching destroyed state.
- Race timer expiry with cancellation and shutdown under ASAN and TSAN.
- Capture a runtime stack during backoff and prove it contains no
  `std::this_thread::sleep_for`.

The runtime owns timer and cancellation state. Do not introduce an ownership
edge from runtime state back to `AbfsReadFile::Impl`.

## Regression boundary

Preserve all C3 contracts:

- Async disabled uses the original client and stock SDK behavior.
- Shared Key and fixed SAS retain sync/async parity.
- Registered providers remain source-compatible and explicitly unsupported
  unless they opt in.
- OAuth and dynamic SAS remain explicitly unsupported for native async.
- One contiguous range, null-gap scatter, fragmented body handling, and future
  error settlement remain unchanged.
- Filesystem cache identity continues to include the complete connector config.
- No Parquet scheduler, proxy, io_uring, Fizz, or generic HTTP framework work.

## C4 validation

Run:

- Upstream/default Azure retry-policy tests.
- Deterministic cooperative callback tests.
- The complete C3 ABFS, registration, Stage 1 transport, and Stage 2 runtime
  suites.
- Focused retry, cancellation, shutdown, and timer-race matrices under ASAN and
  TSAN.
- Pinned formatting, whitespace, LF, mirror, scope, and frozen Stage 0 hash
  checks.

Retain sanitized evidence under `design/results/stage4-main-06dec49a/`.

## C4 exit

C4 passes only when:

- Retry classification, attempts, reset, jitter, and every delay/header form
  match the stock SDK path within documented tolerance.
- Another request progresses on the same runtime thread during backoff.
- Cancellation and shutdown interrupt pending delay exactly once.
- No ABFS runtime stack contains `std::this_thread::sleep_for`.
- Complete ordinary, ASAN, and TSAN regressions pass.

Only after C4 may native async ABFS be exposed as an opt-in user setting. Stage
5 then owns OAuth and dynamic-SAS fiber-safety work.
