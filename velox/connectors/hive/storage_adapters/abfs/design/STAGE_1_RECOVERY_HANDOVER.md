# Stage 1 Recovery Handover: ABFS Fiber Transport

Date: 2026-07-20
Base SHA: `06dec49a333b7e39f73a49ba9dd2c4ee12e4fd4e`
Status: **BLOCKED BEFORE C1; CURRENT STAGE 1 SOURCES ARE UNTRUSTED**

## Purpose

This document supersedes the implementation-state portions of
`STAGE_1_HANDOVER.md`. It does not supersede the architecture or acceptance
criteria in the authoritative specification or Stage 1 prompt.

The previous implementation attempts did not produce a valid Stage 1
transport. The immediate next task is to restore a clean editor/build state,
then restart Stage 1 from the contracts and a narrow executable test. Do not
try to repair the concatenated source files in place, do not run C1 against
the current tree, and do not begin Stage 2.

## Mandatory reading order

Before searching or editing, read these files completely:

1. `velox/connectors/hive/storage_adapters/abfs/ABFS_PREADV_ASYNC_FIBER_TRANSPORT.md`
2. `velox/connectors/hive/storage_adapters/abfs/design/STAGE_0_CHECKPOINT.md`
3. `velox/connectors/hive/storage_adapters/abfs/design/preflight/stage-0-preflight.txt`
4. `velox/connectors/hive/storage_adapters/abfs/design/STAGE_1_AGENT_PROMPT.md`
5. `velox/connectors/hive/storage_adapters/abfs/design/STAGE_1_HANDOVER.md`
6. This recovery handover

The Stage 1 prompt remains authoritative when this document is silent.

## Repository topology

- Windows staging/editor checkout: `C:\velox`, branch `main`, at the base SHA.
- Linux implementation/build checkout: WSL Ubuntu-24.04 at `~/src/velox`,
  detached at the same SHA on ext4.
- Build only in WSL ext4. Never build under `/mnt/c`.
- No branch or commit has been created.
- There is currently no VS Code workspace open. This caused `apply_patch`
  whole-file deletion/recreation to behave incorrectly in delegated passes.

### Workspace prerequisite

Before any source recovery, open one of these as the active VS Code workspace:

1. Preferred: open `~/src/velox` through VS Code Remote - WSL and edit/build
   directly in the ext4 checkout.
2. Fallback: open `C:\velox`, edit there with `apply_patch`, and synchronize
   only an explicit reviewed Stage 1 file list into WSL before each build.

Do not resume with no workspace open. If using the fallback, verify every
copied source with a hash or `cmp` and do not copy the complete ABFS directory.

## State that must be preserved

Stage 0 is **PASS** and must remain intact. Preserve these intentional Windows
changes and untracked artifacts:

- Modified `velox/connectors/hive/storage_adapters/abfs/tests/CMakeLists.txt`.
  It contains the authoritative Stage 0 benchmark block as well as the failed
  Stage 1 insertion. Remove only the Stage 1 insertion during recovery.
- `velox/connectors/hive/storage_adapters/abfs/tests/AbfsParquetBenchmark.cpp`.
- `velox/connectors/hive/storage_adapters/abfs/ABFS_PREADV_ASYNC_FIBER_TRANSPORT.md`.
- Everything under `velox/connectors/hive/storage_adapters/abfs/design/`.

The Stage 0 benchmark block starts with:

```cmake
if(
  VELOX_ENABLE_BENCHMARKS
  AND VELOX_ENABLE_PARQUET
  AND VELOX_ENABLE_HIVE_CONNECTOR)
```

It defines `velox_abfs_parquet_benchmark`. Preserve that block byte for byte.

Generated WSL directories `abfs-main-release`, `abfs-main-debug`,
`abfs-stage1-debug`, `deps-install`, and `conda` are not source work.

## Current dirty state

### Windows staging checkout

Tracked state:

```text
M velox/connectors/hive/storage_adapters/abfs/tests/CMakeLists.txt
```

Intentional Stage 0/spec/design untracked state is listed above. The following
untracked files are failed Stage 1 output and must not be treated as accepted
implementation:

```text
velox/connectors/hive/storage_adapters/abfs/AsyncChannelFactory.h
velox/connectors/hive/storage_adapters/abfs/HttpConnection.h
velox/connectors/hive/storage_adapters/abfs/EventSocketChannelFactory.h
velox/connectors/hive/storage_adapters/abfs/EventSocketChannelFactory.cpp
velox/connectors/hive/storage_adapters/abfs/FollyHttpConnection.h
velox/connectors/hive/storage_adapters/abfs/FollyHttpConnection.cpp
velox/connectors/hive/storage_adapters/abfs/FollyHttpTransport.h
velox/connectors/hive/storage_adapters/abfs/FollyHttpTransport.cpp
velox/connectors/hive/storage_adapters/abfs/FollyResponseBodyStream.h
velox/connectors/hive/storage_adapters/abfs/FollyResponseBodyStream.cpp
velox/connectors/hive/storage_adapters/abfs/tests/FollyHttpTransportTest.cpp
```

Integrity audit of the Windows files:

| File | Copyright blocks | Final namespace closes | Lines | Assessment |
|---|---:|---:|---:|---|
| `AsyncChannelFactory.h` | 1 | 1 | 35 | Unreviewed candidate |
| `HttpConnection.h` | 1 | 1 | 74 | Unreviewed candidate |
| `EventSocketChannelFactory.h` | 1 | 1 | 25 | Unreviewed candidate |
| `EventSocketChannelFactory.cpp` | 1 | 1 | 85 | Unreviewed candidate |
| `FollyHttpConnection.h` | 1 | 1 | 40 | Unreviewed candidate |
| `FollyHttpConnection.cpp` | 4 | 4 | 1418 | Corrupted/concatenated |
| `FollyHttpTransport.h` | 1 | 1 | 41 | Unreviewed candidate |
| `FollyHttpTransport.cpp` | 1 | 1 | 78 | Unreviewed candidate |
| `FollyResponseBodyStream.h` | 1 | 1 | 39 | Unreviewed candidate |
| `FollyResponseBodyStream.cpp` | 1 | 1 | 52 | Unreviewed candidate |
| `tests/FollyHttpTransportTest.cpp` | 3 | 3 | 185 | Corrupted/concatenated |

Even files with one source unit are not accepted. They were produced alongside
the corrupted files and have not passed the required functional review.

### WSL build checkout

The WSL checkout contains a different synchronized snapshot. Its current
integrity counts are:

```text
FollyHttpConnection.cpp: copyright=2, namespace markers=4, lines=538
tests/FollyHttpTransportTest.cpp: copyright=2, namespace markers=4, lines=99
```

The WSL checkout also contains untracked files with names ending in
`:sec.endpointdlp`. These were copied Windows security metadata streams, are
not source, and should be removed as part of recovery after verifying the
exact suffix match. Do not add them to CMake or evidence.

WSL `velox/connectors/hive/storage_adapters/abfs/CMakeLists.txt` is marked
modified because the Windows file was copied into WSL. Its diff is line-ending
only; no Stage 1 source is intentionally part of the production `velox_abfs`
target. Normalize that file only after confirming there is no semantic diff.

## Failed Stage 1 CMake insertion

The failed Stage 1 block in `tests/CMakeLists.txt` defines:

```text
velox_abfs_stage1_transport_test_lib
velox_abfs_stage1_transport_test
```

It currently compiles these sources into an isolated static test library:

```text
../EventSocketChannelFactory.cpp
../FollyHttpConnection.cpp
../FollyHttpTransport.cpp
../FollyResponseBodyStream.cpp
```

Keeping Stage 1 isolated from production `velox_abfs` is the correct boundary,
but the current source list and test are not an accepted implementation. During
recovery, remove this Stage 1 block while preserving the Stage 0 benchmark
block, then add back the smallest coherent target as the first new edit.

## Latest executable result

The most recent focused command was:

```bash
cd ~/src/velox
PATH=~/.local/bin:$PATH \
CMAKE_PREFIX_PATH=~/src/velox/deps-install \
cmake --build abfs-stage1-debug \
  --target velox_abfs_stage1_transport_test --parallel 4
```

Result: **FAIL before link or test execution**.

First actionable errors:

```text
tests/FollyHttpTransportTest.cpp:82: duplicate license/source unit begins
tests/FollyHttpTransportTest.cpp:85: parser sees comment text as code
FollyHttpConnection.cpp:319: duplicate license/source unit begins
FollyHttpConnection.cpp:322: parser sees comment text as code
```

No current focused transport test passes. No deterministic socket test, real
Azure `BlobClient::Download`, TLS matrix, profiler run, or C1 execution has
passed. There is no `STAGE_1_CHECKPOINT.md` and no valid Stage 1 result
directory. Do not infer C1 progress from the earlier smoke-test result.

## What happened

### Initial local hypothesis and cheap check

The initial hypothesis was that Stage 1 could remain in new private
transport/test files plus ABFS test CMake, without provider or `AbfsReadFile`
changes. The cheap check was an isolated test target containing a fiber Baton
post-before-wait test.

An early scaffold compiled and that trivial Baton test passed. This validated
only the target boundary. It did not validate HTTP, sockets, TLS, Azure, or C1.

### Delegation history

The Stage 1 prompt required implementation by a GPT-5.6 Luna high-reasoning
subagent followed by parent review and fresh correction passes.

- Luna produced declarations and a full-buffering parser scaffold. Parent
  review rejected it for missing implementations and incorrect semantics.
- Fresh Luna correction passes introduced compile errors, then produced one
  compiling target with only two trivial tests. Parent full review rejected
  its transport and lifetime design.
- Later Luna passes repeatedly appended old source units instead of replacing
  files, creating the current concatenation.
- Per user direction, Terra was attempted after Luna. Terra failed its own
  duplicate-content check and produced no syncable candidate.
- Sol was attempted after Terra. With no workspace open, `apply_patch` delete
  and add operations did not replace the absolute-path files; Sol appended a
  fourth `FollyHttpConnection.cpp` unit and stopped.

Do not repeat these agents against absolute paths with no workspace open. If
delegation is used after workspace recovery, use the required order: Luna,
then Terra if Luna fails, then Sol if Terra fails. Review and validate each
pass before invoking the next one.

## Parent review findings that remain open

Any fresh implementation must explicitly close all of these findings:

1. Streaming responses must not use Beast `string_body` or buffer the full
   Download before `Send` returns.
2. Persistent Beast parser state must consume exactly the returned byte count
   and retain parser spill.
3. A one-shot Baton cannot be reused in read loops without a correct reset or
   generation protocol.
4. Socket read callbacks must be transaction-owned and cleared before
   destruction; stack callback pointers caused a UAF risk.
5. Every `noexcept` callback must catch allocation and exception construction
   failures, record state, and post exactly once.
6. Connect, TLS, write, first-byte, body-idle, and relevant total timeouts need
   race-safe completion and wake-once behavior.
7. Valid close-delimited EOF must call Beast `put_eof`; Content-Length and
   chunked early EOF must fail.
8. One or more informational responses must be discarded before returning the
   final response head.
9. HEAD, 204, and 304 skip-body behavior must be configured before body parse.
10. Serialize requests with Beast while preserving Azure signed header values;
    do not concatenate protocol lines manually.
11. Enforce independent configurable status, header, request-body, buffered
    response-body, and ingress limits. Ingress starts at no more than 64 KiB.
12. Preserve duplicate response fields and the actual HTTP version.
13. One connection has one active response. Fully consumed reusable bodies
    return to the correct EventBase-affine pool; abandoned/unread/error bodies
    close and never enter the pool.
14. Validate request scheme, host, and port against the configured endpoint.
15. `Send` must require the scheduled EventBase fiber, not merely its OS thread.
16. `FollyResponseBodyStream` must expose known length when available and map
    body failures to Azure `TransportException`.
17. Do not run synchronous DNS on the EventBase thread. Stage 1 can require a
    pre-resolved numeric loopback address while retaining the TLS server name.
18. Explicitly load system trust roots, require TLS 1.2+, set SNI, and verify
    both chain and hostname. Do not add a verification bypass.
19. Confirm the AsyncSSLSocket callback used represents completed and verified
    TLS negotiation.
20. Tests must exercise real sockets and the real Azure SDK path, not only
    structs, memory transactions, or a Baton.

## Verified dependency/API facts

These facts were checked against the pinned WSL dependencies and can prevent
repeating earlier compile failures:

- Azure Storage Data Lake SDK: `12.8.0`.
- Folly/Fizz: `v2026.01.05.00`.
- Boost: `1.84.0`.
- Azure `Url::GetUrlWithoutQuery(bool)` is private. Public URL APIs include
  `GetRelativeUrl()` and `GetAbsoluteUrl()`.
- Azure `RawResponse` supports `SetBody()` and `SetBodyStream()`.
- Azure Blob `Download()` uses an unbuffered response and exposes a body stream.
- `BlobClientOptions.Transport.Transport` accepts a shared `HttpTransport`.
- Include `<folly/io/async/AsyncSocketException.h>` before calling
  `AsyncSocketException::what()`.
- `AsyncSocketException::what()` returns `const char*`; it has no
  `toStdString()` method.
- `folly::AsyncSSLSocket` takes `Options&&`; pass `std::move(options)`.
- Folly provides `folly::fibers::onFiber()` and
  `folly::fibers::getFiberManager(EventBase&)`.
- Fiber Baton supports post-before-wait and timed waits.
- Boost 1.84 serializer uses
  `serializer.next(error, visitor)` and `serializer.consume(bytes)` inside the
  visitor. Do not call a nonexistent/incorrect `serializer.get()` path.
- Beast `buffer_body` is available for incremental streaming.
- System trust roots require invoking OpenSSL default verification paths on
  the underlying `SSL_CTX` in addition to Folly peer/name authentication.

## Required recovery sequence

### 1. Establish a real workspace

Open the WSL checkout as the workspace if possible. Confirm:

```bash
cd ~/src/velox
git rev-parse HEAD
git status --short --branch
stat -f -c 'filesystem=%T' .
```

Expected SHA is the base SHA and the filesystem is Linux-native ext4
(`ext2/ext3` may be the reported type name).

### 2. Preserve and separate Stage 0

Before deleting anything, inspect `tests/CMakeLists.txt` and identify the Stage
0 benchmark block. Record hashes for the benchmark source, specification, and
design reports. Do not remove or overwrite them.

### 3. Remove only failed Stage 1 output

Treat the eleven failed Stage 1 source/test files listed above as disposable.
After the workspace is open and paths are inside it, remove them with explicit
`apply_patch` delete operations. Remove only the Stage 1 library/test block
from `tests/CMakeLists.txt`. Preserve the Stage 0 benchmark block.

In WSL, also remove only files matching the exact suffix
`:sec.endpointdlp`. Normalize the line-ending-only production
`abfs/CMakeLists.txt` change after verifying it has no semantic delta.

After cleanup, both checkouts should contain no Stage 1 transport source and
the Stage 0 state should match the checkpoint.

### 4. Re-establish the smallest testable boundary

Before the first new edit, state a falsifiable local hypothesis and its cheap
disproof. Recommended hypothesis:

> A private Stage 1 test library can compile backend-neutral channel and HTTP
> contracts without changing production `velox_abfs` or Stage 0 behavior.

First edit:

- Add clean `AsyncChannelFactory.h` and `HttpConnection.h` contracts.
- Add one isolated Stage 1 target.
- Add a Baton post-before-wait smoke test.

Immediately configure/build/run only that target. Do not add further layers
until this check is green.

### 5. Implement in validated increments

Use this order, validating the focused target after each increment:

1. EventBase plaintext AsyncSocket channel and delayed loopback server.
2. Persistent incremental Beast headers and bounded streaming body.
3. Request serialization and framing/error/limit tests.
4. Transaction ownership, timers, abandon, reuse, and pool tests.
5. AsyncSSLSocket with deterministic test CA and negative TLS tests.
6. Azure transport/body-stream adapters.
7. Real Azure `BlobClient::Download` through Azure response parsing.
8. One-thread FiberManager concurrency and pending-submission proof.
9. C1 memory counters and profiler-ready harness.

Do not open multiple unvalidated implementation layers at once.

### 6. Required delegation/review policy

All actual implementation changes remain subject to the Stage 1 prompt:

1. Delegate implementation to GPT-5.6 Luna high reasoning.
2. Parent reviews the complete resulting code for correctness, lifetime,
   concurrency, security, and scope.
3. Send a fresh Luna correction pass with every finding.
4. If Luna fails, try Terra. If Terra fails, try Sol.
5. After every pass, run duplicate-content checks and the narrow executable
   test before accepting or escalating.

Never allow a subagent to report success without a nonempty changed-file list,
`git diff --check`, source-integrity counts, and the exact tests it ran or left
for the parent.

### 7. Full C1 authorization

Do not run expensive full C1 or profiling until the parent review loop is
clean and all focused tests pass. C1 still requires:

- 64 real Azure `BlobClient::Download` calls through Azure response parsing.
- At least 32 simultaneous active requests.
- A pending future returned before delayed response headers.
- One EventBase/FiberManager runtime OS thread.
- Bounded RSS consistent with ingress buffers and fiber stacks, not full
  response size multiplied by request count.
- TLS positive, unknown-CA, and hostname-mismatch results.
- Profiler evidence of fiber Baton waits and no EventBase thread blocking in
  socket poll, read, or write.

Recommended C1 test name:

```text
FollyHttpTransportC1Test.SixtyFourDownloadsOneEventBase
```

## Build commands

Use the existing separate WSL build or recreate it with:

```bash
cd ~/src/velox
export PATH=~/.local/bin:$PATH
export CMAKE_PREFIX_PATH=~/src/velox/deps-install

cmake -S . -B abfs-stage1-debug -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_PREFIX_PATH=~/src/velox/deps-install \
  -DVELOX_ENABLE_ABFS=ON \
  -DVELOX_ENABLE_PARQUET=ON \
  -DVELOX_ENABLE_HIVE_CONNECTOR=ON \
  -DVELOX_ENABLE_EXEC=ON \
  -DVELOX_ENABLE_BENCHMARKS=ON \
  -DVELOX_BUILD_TESTING=ON

cmake --build abfs-stage1-debug \
  --target velox_abfs_stage1_transport_test --parallel 4

ctest --test-dir abfs-stage1-debug --output-on-failure \
  -R '^velox_abfs_stage1_transport_test$'
```

After focused review is clean, run the existing ABFS tests. Do not rerun the
Stage 0 benchmark unless regression validation specifically requires it.

## Security and evidence rules

Do not expose or record credentials, Shared Key values, SAS tokens,
authorization headers, account IDs, real service endpoints, usernames,
hostnames, or private absolute home paths. Test certificates and private keys
must be deterministic, clearly test-only loopback fixtures with no external
use.

Sanitize all Stage 1 evidence under:

```text
velox/connectors/hive/storage_adapters/abfs/design/results/stage1-main-06dec49a/
```

Write `design/STAGE_1_CHECKPOINT.md` only after a real C1 run. State `C1 PASS`
only if every criterion passes. Otherwise record the first failed criterion
and stop. The exact Stage 2 entry condition remains a complete C1 pass.

## Immediate next action

Open the WSL checkout as the active workspace, verify the preservation
boundary, remove only the failed Stage 1 output and copied metadata streams,
then recreate the two contracts and the narrow Baton test as the first
validated edit. Do not salvage the concatenated files and do not start from
their current apparent implementation depth.
