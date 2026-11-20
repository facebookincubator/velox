# Stage 1 Handover: ABFS Fiber Transport

## Status and boundary

Stage 0 is **PASS**. Stage 1 is next. The current main SHA is
`06dec49a333b7e39f73a49ba9dd2c4ee12e4fd4e`. No branch or commit was created,
and no production transport code exists yet. This handover is for a fresh
implementation agent.

The standalone native ABFS async work does not depend on downstream reader
changes. Downstream integration is deferred and is not a Stage 1 gate. Do not
use any downstream pull request as a dependency or direction for this work.
This task has no WAVE dependency.

Read the full authoritative specification before searching or editing:
[ABFS_PREADV_ASYNC_FIBER_TRANSPORT.md](../ABFS_PREADV_ASYNC_FIBER_TRANSPORT.md).

## Workspace topology and preserved state

- `C:\velox` is the Windows staging/editor checkout on `main`.
- The actual Linux implementation and build checkout is WSL Ubuntu-24.04 at
  `~/src/velox`, detached at the same SHA, on ext4.
- Build only in WSL ext4. Do not build under `/mnt/c`.
- Intentional source artifacts are the authoritative spec, this Stage 0
  report and evidence, `tests/AbfsParquetBenchmark.cpp`, and the modified
  `tests/CMakeLists.txt`.
- WSL directories `abfs-main-release`, `abfs-main-debug`, and `deps-install`,
  plus `conda`, are generated or dependency state, not source changes.
- The Windows worktree has modified `tests/CMakeLists.txt` and untracked
  `ABFS_PREADV_ASYNC_FIBER_TRANSPORT.md`, `design/`, and
  `tests/AbfsParquetBenchmark.cpp`. Preserve them. Do not revert them.

There is no username, private absolute home path, credential, account ID, or
service endpoint in this handover. Keep all future reports similarly sanitized.

## Stage 0 result

Release and Debug configurations were created and the three relevant tests
passed in both configurations: `velox_abfs_test`,
`velox_abfs_registration_test`, and `velox_dwio_parquet_reader_test`. The
benchmark is a nine-row, synchronous Azure SDK default-transport run over
local Azurite. Mean wall times were 47.32 ms, 38.82 ms, and 34.51 ms for IO
executor sizes 1, 2, and 8. It makes no async, fiber, or io_uring claim.

Evidence and report:

- [Stage 0 checkpoint](STAGE_0_CHECKPOINT.md)
- [Stage 0 preflight](preflight/stage-0-preflight.txt)
- [Stage 0 result directory](results/stage0-main-06dec49a/)
- [Benchmark summary](results/stage0-main-06dec49a/abfs-parquet-baseline-summary.json)
- [Benchmark CSV](results/stage0-main-06dec49a/abfs-parquet-baseline.csv)
- [Build summary](results/stage0-main-06dec49a/tests/build-summary.txt)
- [Release CTest](results/stage0-main-06dec49a/tests/release-ctest.txt)
- [Debug CTest](results/stage0-main-06dec49a/tests/debug-ctest.txt)

## Environment and dependencies

- Ubuntu 24.04.4 WSL2; kernel `6.18.33.2-microsoft-standard-WSL2`.
- CMake `3.30.4`; GCC `13.3.0`; OpenSSL `3.0.13`.
- Azure Data Lake SDK `12.8.0` under `~/src/velox/deps-install`.
- Folly/Fizz `v2026.01.05.00`.
- Node `22.23.1`; Azurite `3.36.0`.
- `io_uring_disabled=0`; `ulimit -n` is `10240`.

Stage 1 uses the reference EventBase backend, not io_uring. Do not claim
io_uring merely because the kernel permits it.

## Stage 1 objective and scope

Prove the Azure fiber bridge over the reference backend with the real
synchronous Azure SDK. Add only the Stage 1 spike and test surfaces, and
private helper layers named by the specification:

- Backend-neutral `AsyncChannelFactory` and `HttpConnection` contracts.
- Reference `EventSocketChannelFactory` using `AsyncSocket` and
  `AsyncSSLSocket`.
- Beast-based HTTP/1.1 connection with incremental parsing and bounded
  response buffering.
- `FollyHttpTransport` as the Azure adapter.
- `FollyResponseBodyStream` for fiber-aware streaming reads.
- A one-thread `FiberManager` harness.
- A deterministic delayed HTTP/TLS server and test CA.

Do not implement Stage 2 runtime admission/lifetime, Stage 3 ABFS provider or
`preadvAsync` wiring, or later retry, authentication, proxy, or io_uring work.
Do not change the Stage 0 harness.

## Architecture rules

Azure remains responsible for request construction, signing, authentication,
retry classification, Blob parsing, and storage errors. Do not sign REST
requests directly. The adapter must use the synchronous `Send` and `OnRead`
interfaces with a fiber `Baton`; it must not wrap blocking Azure I/O in an
executor. Socket callbacks must never block. Use Boost.Beast incrementally,
not an ad hoc parser, and keep response ingress bounded.

Use the `AsyncSocket`/`AsyncSSLSocket` EventBase reference backend. Preserve
backend-neutral boundaries from day one. TLS requires system roots, SNI,
hostname and chain verification, TLS 1.2 or newer, and no verification bypass.
Retries may be disabled only in this isolated Stage 1 spike and never in a
user-facing implementation. OAuth and dynamic SAS are not Stage 1. Make no
io_uring claim.

Required transport behavior includes fragmented status, headers, chunks, and
bodies; Content-Length, chunked, and close-delimited framing; informational
responses; bounded status/header/body limits; early EOF and malformed framing
errors; keep-alive and close handling; abandoned-body connection close;
consumed-body connection reuse; and timeout wake-once behavior.

## C1 gate

C1 is executable, not an inspection claim. At least 64 real
`BlobClient::Download` calls must complete through Azure response parsing on
one runtime thread, with at least 32 simultaneous requests. Submission must
return a pending future before response headers arrive. RSS growth must match
bounded body buffers plus active fiber stacks, not full response size times
request count. TLS positive and negative chain/hostname tests must pass. A
profiler must prove that waits occur in fiber `Baton` waits and that no
EventBase thread blocks in socket poll, read, or write.

The Stage 1 tests must cover fragmented status/header/chunk/body, framing
variants relevant to this stage, early EOF and malformed framing, response
limits, abandoned bodies closing their connections, consumed-body reuse, TLS
positive and negative cases, timeout wake-once, Baton post-before-wait, and a
one-thread concurrency proof. Keep the implementation narrow, but do not
weaken C1.

Stop and write a failed Stage 1 report if full-body buffering is required,
callbacks cannot resume the correct `FiberManager`, or the SDK calls the
transport outside the scheduled fiber. If C1 fails, do not begin Stage 2.

## Local workflow

Before the first edit, form one falsifiable local hypothesis and name the
cheapest check that could disconfirm it. After the first substantive edit,
immediately run a narrow compile or test for the touched slice. Actual
implementation changes must be delegated to a GPT-5.6 Luna high-reasoning
subagent. The parent agent reviews the full resulting code, sends a fresh Luna
correction pass for every finding, and repeats until clean. Expensive builds,
benchmarks, profiling, and the full C1 run are authorized only after that code
review loop. Preserve untracked and user files. Use `apply_patch` for manual
edits. Do not create branches or commits.

Configure or reuse a separate Stage 1 build directory; preserve the existing
flags and dependency prefix. A command template is:

```bash
cmake -S . -B <stage1-build> -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_PREFIX_PATH=~/src/velox/deps-install \
  -DVELOX_ENABLE_ABFS=ON \
  -DVELOX_ENABLE_PARQUET=ON \
  -DVELOX_ENABLE_HIVE_CONNECTOR=ON \
  -DVELOX_ENABLE_EXEC=ON \
  -DVELOX_ENABLE_BENCHMARKS=ON \
  -DVELOX_BUILD_TESTING=ON
```

Use `PATH=~/.local/bin:$PATH`. Add a narrow new transport test target first,
then run relevant existing tests. Do not rerun the Stage 0 benchmark unless
needed for regression validation.

## Artifacts and checkpoint

Keep raw logs under `design/results/stage1-<sha>/`, with separate sanitized
command, environment, test, profiler, and measurement artifacts. Never record
credentials, SAS or authorization material, account IDs, endpoints, usernames,
hostnames, or private absolute paths. The report belongs at
`design/STAGE_1_CHECKPOINT.md` and records SHA, commands, results, measured
counters, unresolved risks, and the next entry condition. A successful report
must state the exact C1 evidence and permit Stage 2 only when every C1 item
passes. A failed report must state the first failed criterion and stop.

## First actions

1. Read the entire authoritative spec, this handover, the entire Stage 0
   checkpoint, and the preflight/evidence artifacts before search or edit.
2. Verify the Windows and WSL checkout SHA/state and keep implementation/build
   work under `~/src/velox` on ext4.
3. Identify the smallest existing Folly, Azure, and Velox test surfaces needed
   for the reference EventBase fiber spike.
4. Delegate implementation to GPT-5.6 Luna, review the full code, and iterate
   with fresh Luna correction passes before expensive validation.
5. Run the narrow post-edit compile/test, then execute the C1 gate and write
   the sanitized checkpoint report.

## Do not

- Do not modify production ABFS transport code outside Stage 1 scope.
- Do not implement Stage 2 or Stage 3, downstream reader integration, or any
  downstream pull request dependency work.
- Do not direct work to `/mnt/c`, revert Stage 0/user artifacts, create a
  branch, or commit.
- Do not hand-write REST signing, TLS, or a general-purpose HTTP parser.
- Do not block an EventBase thread, use an executor wrapper around Azure I/O,
  or make io_uring claims.
- Do not weaken C1 or continue to Stage 2 after a failed C1.