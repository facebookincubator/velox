# Stage 1 checkpoint: ABFS Azure fiber transport

Date: 2026-07-20
Base SHA: `06dec49a333b7e39f73a49ba9dd2c4ee12e4fd4e`
Verdict: **C1 PASS**

## Scope

Stage 1 proves the Azure synchronous-SDK-to-fiber bridge over the reference
Folly EventBase backend. It remains an isolated spike and test target. No
production `velox_abfs` source list, provider, `AbfsReadFile`, Stage 2 runtime,
retry behavior, OAuth, dynamic SAS, proxy, DNS, or io_uring path was added.

The implementation includes:

- Backend-neutral resolved endpoint and HTTP contracts.
- Plaintext `AsyncSocket` and verified `AsyncSSLSocket` reference channels.
- Incremental Boost.Beast HTTP/1.1 serialization and parsing.
- Content-Length, chunked, close-delimited, informational, and no-body framing.
- Bounded ingress, request body, buffered body, status, header, and timeout
  limits.
- Pull-based streaming through Azure `BodyStream` without full Download
  buffering.
- A bounded EventBase-affine HTTP/1.1 pool with cooperative FIFO fiber waiters.
- Real Azure `BlobClient::Download` response parsing and range requests.
- One-thread, 64-download C1 concurrency and memory harness.

## Files

Added Stage 1 source and tests:

- `velox/connectors/hive/storage_adapters/abfs/AsyncChannelFactory.h`
- `velox/connectors/hive/storage_adapters/abfs/HttpConnection.h`
- `velox/connectors/hive/storage_adapters/abfs/EventSocketChannelFactory.h`
- `velox/connectors/hive/storage_adapters/abfs/EventSocketChannelFactory.cpp`
- `velox/connectors/hive/storage_adapters/abfs/FollyHttpConnection.h`
- `velox/connectors/hive/storage_adapters/abfs/FollyHttpConnection.cpp`
- `velox/connectors/hive/storage_adapters/abfs/FollyHttpTransport.h`
- `velox/connectors/hive/storage_adapters/abfs/FollyHttpTransport.cpp`
- `velox/connectors/hive/storage_adapters/abfs/FollyResponseBodyStream.h`
- `velox/connectors/hive/storage_adapters/abfs/FollyResponseBodyStream.cpp`
- `velox/connectors/hive/storage_adapters/abfs/tests/FollyHttpTransportTest.cpp`
- `velox/connectors/hive/storage_adapters/abfs/tests/FollyHttpTransportC1Test.cpp`
- Deterministic loopback-only TLS fixtures under
  `velox/connectors/hive/storage_adapters/abfs/tests/data/`.

Modified only the isolated test target in
`velox/connectors/hive/storage_adapters/abfs/tests/CMakeLists.txt`. The Stage 0
benchmark block is byte-for-byte unchanged after LF normalization.

## Delegation and review

All implementation increments were delegated to GPT-5.6 Luna high reasoning.
Fifteen fresh Luna passes covered recovery contracts, plaintext sockets, HTTP,
TLS, Azure adapters, pooling, and C1. The parent reviewed every pass for scope,
lifetime, concurrency, parser semantics, security, and evidence quality, sent
fresh correction passes for every finding, and ran the narrow executable test
after each increment. No Terra or Sol escalation was needed after workspace
recovery.

The parent corrections included pinned Folly/Boost/Azure API alignment,
callback ownership and wakeup fixes, parser spill and EOF handling, mandatory
OpenSSL hostname verification, valid deterministic TLS fixtures, Azure stream
lifetime, EventBase-affine pool teardown, waiter deadline/accounting, and C1
evidence hardening. Final sources pass the repository-pinned clang-format
v21.1.2 check and contain one source unit each.

## Focused validation

The final complete Stage 1 CTest entry passed in 7.15 seconds. The functional
matrix passed 54 of 54 tests. It covers real plaintext and TLS sockets, HTTP
framing and limits, timeout/abandon behavior, pool reuse and backpressure,
Azure request/response adaptation, and real Blob SDK parsing.

Final TLS stress ran 20 iterations of the positive, unknown-CA,
hostname-mismatch, and handshake-timeout matrix: 80 of 80 executions passed.

Existing ABFS compatibility passed:

- `velox_abfs_test`: PASS, 59.11 seconds.
- `velox_abfs_registration_test`: PASS, 0.05 seconds.

The Stage 0 benchmark was not rerun.

## C1 result

`FollyHttpTransportC1Test.SixtyFourDownloadsOneEventBase` passed three
consecutive normal post-format runs, a GDB-instrumented run, and a
strace-instrumented run.

Every run proved:

- 64 real `BlobClient::Download` calls completed through Azure response parsing.
- Exactly 32 physical connections and at least 32 simultaneous active requests.
- All returned futures remained pending after 32 requests were active and
  before any response header was released.
- One EventBase/FiberManager runtime OS thread executed every Azure call, and
  it differed from the submitting thread.
- All 64 responses streamed 4 MiB through fixed-size buffers with exact byte
  counts and deterministic checksums.
- Pool leases and waiters returned to zero after completion.

Representative normal-run metrics:

```text
C1_METRICS requests=64 connections=32 peak_active=32 runtime_threads=1 body_bytes=268435456 body_bytes_each=4194304 rss_baseline_kib=20580 rss_peak_kib=28284 rss_growth_kib=7704 modeled_bound_kib=18432 full_buffer_kib=262144
```

Across the three normal repeats, RSS growth was 7,704 KiB, 3,588 KiB, and
3,588 KiB. This is below the 18,432 KiB modeled stack-plus-ingress bound and
far below the 262,144 KiB full-response-buffering counterfactual.

## Profiler evidence

GDB stopped the named `abfs-c1-event` runtime thread in
`folly::fibers::Baton::wait()`. The stack continued through
`EventSocketChannelFactory::connect`, the connection pool,
`FollyHttpTransport::Send`, Azure Core transport/log/activity/retry/telemetry
policies, and Azure Storage policies. C1 then continued and passed.

Final per-thread strace showed on `abfs-c1-event`:

- 32 of 32 connects returned `EINPROGRESS`.
- 15,806 of 15,806 receives used `MSG_DONTWAIT`.
- 64 of 64 sends used `MSG_DONTWAIT` and `MSG_NOSIGNAL`.
- 2,563 `epoll_wait` calls handled readiness.
- Zero `poll` or `ppoll` calls occurred on the runtime thread.
- Maximum traced connect, receive, and send durations were 0.465 ms, 0.545 ms,
  and 0.272 ms respectively.

Blocking `poll` appeared only on the separate deterministic server thread.

## Evidence

All evidence paths are repository-relative and sanitized:

- `velox/connectors/hive/storage_adapters/abfs/design/results/stage1-main-06dec49a/environment.txt`
- `velox/connectors/hive/storage_adapters/abfs/design/results/stage1-main-06dec49a/commands.txt`
- `velox/connectors/hive/storage_adapters/abfs/design/results/stage1-main-06dec49a/source-integrity.txt`
- `velox/connectors/hive/storage_adapters/abfs/design/results/stage1-main-06dec49a/tests/focused-tests.txt`
- `velox/connectors/hive/storage_adapters/abfs/design/results/stage1-main-06dec49a/tests/existing-abfs.txt`
- `velox/connectors/hive/storage_adapters/abfs/design/results/stage1-main-06dec49a/c1/c1-runs.txt`
- `velox/connectors/hive/storage_adapters/abfs/design/results/stage1-main-06dec49a/c1/gdb-baton-stack.txt`
- `velox/connectors/hive/storage_adapters/abfs/design/results/stage1-main-06dec49a/c1/strace-summary.txt`

Raw traces were not persisted because they contain deterministic test request
bytes. The retained profiler summaries contain no credentials, query values,
private paths, usernames, account IDs, or real endpoints.

## Unresolved risks

- This is a Stage 1 spike, not production connector wiring.
- SDK retries are disabled only in the isolated Blob tests. Cooperative retry
  preservation remains a later checkpoint.
- Endpoint addresses are pre-resolved; asynchronous DNS and proxy parity are
  not implemented.
- Authentication coverage is fixed-SAS-style/credential-free local pipeline
  use only; OAuth and dynamic SAS safety gates remain later work.
- Evidence is WSL loopback correctness data, not native-Linux or real-Azure
  performance evidence.
- The reference EventBase backend is proven. No io_uring or Fizz data-path
  claim is made.
- ASAN and TSAN runtime-lifetime stress belongs to Stage 2 C2.

## Stage 2 entry

The exact Stage 2 entry condition is satisfied: C1 is complete and passed.
Stage 2 may begin with the config-scoped bounded runtime, admission queue,
shutdown/lifetime graph, endpoint sharding, DNS ownership, and associated
ASAN/TSAN tests. Stage 3 provider and `AbfsReadFile::preadvAsync` wiring must
remain deferred until C2 passes.