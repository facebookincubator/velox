# Stage 1 Implementation Prompt: ABFS Fiber Transport

You are the fresh implementation agent for Stage 1 of native standalone ABFS
`preadvAsync` over a fiber-backed Azure transport in the Windows Velox staging
checkout `C:\velox`. This is a staged implementation task. Execute exactly
Stage 1 and C1. Do not skip stages or combine Stage 1 with later stages.

## Required reading and state preservation

Before any search or edit, read the entire files below:

1. `velox/connectors/hive/storage_adapters/abfs/ABFS_PREADV_ASYNC_FIBER_TRANSPORT.md`
2. `velox/connectors/hive/storage_adapters/abfs/design/STAGE_0_CHECKPOINT.md`
3. `velox/connectors/hive/storage_adapters/abfs/design/preflight/stage-0-preflight.txt`
4. `velox/connectors/hive/storage_adapters/abfs/design/STAGE_1_HANDOVER.md`

Also inspect the current benchmark, CMake, evidence, and Git state as they
exist now. Current main SHA is
`06dec49a333b7e39f73a49ba9dd2c4ee12e4fd4e`. Do not assume prior-turn file
contents. Preserve the current dirty and untracked state: the Windows
checkout has the intentional modified ABFS `tests/CMakeLists.txt` and
untracked authoritative spec, design artifacts, and
`tests/AbfsParquetBenchmark.cpp`. Do not revert or overwrite user files.

`C:\velox` is the Windows staging/editor checkout on `main`. The actual Linux
implementation and build checkout is WSL Ubuntu-24.04 at `~/src/velox`,
detached at the same SHA on ext4. Build only in WSL ext4, never under
`/mnt/c`. Generated WSL state such as `abfs-main-release`, `abfs-main-debug`,
`deps-install`, and `conda` is not source work. Do not create a branch or
commit. Do not use any downstream pull request as a dependency or direction.
This task has no WAVE dependency.

Do not expose or record credentials, SAS tokens, authorization headers,
account IDs, endpoints, usernames, hostnames, or private absolute home paths.
Use sanitized `~/src/velox` and repository-relative paths in reports.

## Delegation and review loop

All actual implementation changes must be delegated to a GPT-5.6 Luna
high-reasoning subagent. After the subagent returns, the parent agent must
review the full code, identify every correctness, lifetime, concurrency,
security, and scope issue, and send a fresh Luna correction pass with that
feedback. Repeat the full-code review and fresh correction pass until clean.
Only then authorize expensive builds, benchmarks, profiling, and the complete
C1 run. A narrow compile or test immediately after the first edit is still
required; distinguish that cheap focused validation from the expensive full
validation.

Before the first edit, state one falsifiable local hypothesis and the cheapest
check that could disconfirm it. After the first substantive edit, immediately
run that narrow compile/test. Use `apply_patch` for manual edits.

## Exact Stage 1 scope

Prove the central Azure fiber bridge with the real synchronous Azure SDK over
the reference EventBase backend. Add only the Stage 1 spike and test surfaces:

- Backend-neutral `AsyncChannelFactory`.
- Protocol-neutral `HttpConnection` contract.
- Reference `EventSocketChannelFactory` using Folly `AsyncSocket` and
  `AsyncSSLSocket`.
- Beast incremental HTTP/1.1 connection and bounded response buffering.
- `FollyHttpTransport` Azure `HttpTransport` adapter.
- `FollyResponseBodyStream` fiber-aware response streaming adapter.
- One-thread EventBase-integrated `FiberManager` harness.
- Deterministic delayed HTTP/TLS server and test CA.

The Stage 1 spike may disable Azure SDK retries only in this isolated test
surface. It must not be presented as user-facing or production-ready.

Do not implement Stage 2 runtime admission/lifetime, Stage 3
`AbfsReadFile::preadvAsync` or provider wiring, retry preservation,
OAuth/dynamic-SAS support, proxy parity, io_uring/Fizz backend work, downstream
reader scheduling, or performance claims. Downstream integration is deferred
and is not a Stage 1 gate.

## Architecture constraints

Azure SDK remains responsible for request construction, signed headers,
Shared Key/SAS/auth policy, retry classification, Blob response parsing, and
storage errors. Never construct or sign Blob REST requests directly. Adapt the
synchronous `Send` and response `BodyStream::OnRead` contracts with a fiber
`folly::fibers::Baton`.

`Send` is legal only on the scheduled ABFS fiber. Socket callbacks are
`noexcept`, capture failures in transaction state, and post the Baton. A post
before the fiber waits must not lose the wakeup. No callback or socket path
may block an EventBase thread, and no executor wrapper may hide blocking Azure
I/O. Use Boost.Beast incremental parsing, not delimiter-based production
parsing. Keep decoded ingress and buffered response bodies bounded.

Use the EventBase `AsyncSocket`/`AsyncSSLSocket` reference backend. Keep the
HTTP and Azure layers independent of the socket/TLS engine. TLS must enforce
TLS 1.2 or newer, SNI, system trust roots, certificate chain verification, and
hostname verification, with no production verification bypass. Do not make
io_uring claims. OAuth and dynamic SAS are outside Stage 1.

Cover GET and HEAD request forms needed by the spike, request bodies where
relevant to the transport contract, Content-Length, chunked and
connection-close response framing, informational responses, fragmented
status/headers/chunks/body, keep-alive and `Connection: close`, configurable
status/header/body limits, parser errors, and premature EOF. A body abandoned
before EOF must close its connection; a fully consumed reusable body may
return it to the correct pool.

## C1 acceptance gate

Run an executable C1, not a code-inspection substitute:

- Complete 64 real `BlobClient::Download` calls through Azure response parsing
  on one runtime thread.
- Reach at least 32 simultaneous requests.
- Prove the submitting thread returns a pending future before response headers
  arrive.
- Prove RSS is bounded by configured body buffers and active fiber stacks,
  rather than full response size multiplied by request count.
- Pass TLS positive and negative chain/hostname verification tests.
- Use profiler evidence to prove waits occur in fiber Baton waits and no
  EventBase thread blocks in socket poll, read, or write.

Required focused tests include fragmented status/header/chunk/body, relevant
framing variants, early EOF, malformed framing, header/body limits, abandoned
body close, consumed-body reuse, TLS positive and negative cases, timeout
wake-once, Baton post-before-wait, and a one-thread concurrency proof. Do not
weaken C1 to make the implementation pass.

Stop and write a failed Stage 1 checkpoint report if full-body buffering is
required, callbacks cannot resume the correct FiberManager, or the SDK calls
the transport outside the scheduled fiber. If C1 fails, stop; do not start
Stage 2.

## Build and validation workflow

Use a separate Stage 1 WSL build directory or an explicitly clean equivalent,
preserving the project flags and dependency prefix. A representative command
is:

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

Use `PATH=~/.local/bin:$PATH`. Add and compile a narrow transport test target
first, then run relevant existing ABFS tests. Do not rerun the Stage 0
benchmark unless regression validation specifically requires it.

Keep raw sanitized artifacts under
`velox/connectors/hive/storage_adapters/abfs/design/results/stage1-<sha>/`.
Write `design/STAGE_1_CHECKPOINT.md` with the SHA, commands, results,
concurrency and memory counters, profiler evidence, unresolved risks, and the
exact next-stage entry condition. The report must say C1 PASS only when every
criterion passes; otherwise record the first failed criterion and stop. The
exact Stage 2 entry condition is a complete C1 pass. No credentials or
machine-private paths may appear in source, commands, logs, or reports.

## Final response

Return a concise report containing:

1. Files changed and the Stage 1 scope implemented.
2. The review-loop result, including the fresh Luna correction passes.
3. Narrow post-edit validation results and the full C1 results.
4. Sanitized artifact paths, measured concurrency/memory/profiler evidence,
   and unresolved risks.
5. Whether C1 passed and, only if it passed, the precise Stage 2 entry
   condition.

Stop at C1. Do not implement or claim any later stage.